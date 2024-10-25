#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <immintrin.h>
#include <png.h>
#include <sched.h>

#include <boost/sort/spreadsort/spreadsort.hpp>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#if MULTITHREADED == 1
#include <pthread.h>
#endif
#if MULTITHREADED == 2
#include <omp.h>
#endif
#ifdef MPI_ENABLED
#include <mpi.h>
#endif

#ifdef PROFILING
#include <nvtx3/nvtx3.hpp>
#define NVTX_RANGE_START(arg) \
    nvtxRangePushA(#arg);
#define NVTX_RANGE_END() \
    nvtxRangePop();
#else
#define NVTX_RANGE_START(arg) \
    {}
#define NVTX_RANGE_END() \
    {}
#endif  // PROFILING

const int max_num_procs = 48;
const int max_num_cpus = 12;
const long long min_tasks_per_process = (long long)500 * 1920 * 1080;
const int min_batch_size_per_thread = 1000;
const int max_buffer_size = max_num_cpus * min_batch_size_per_thread * 100;

class Solver {
   public:
    Solver() {}
    ~Solver() {};
    int solve(int argc, char** argv);

   private:
    struct SharedData {
        Solver* solver;
        int num_threads;
        int batch_size;
        int* pixels;
        int start_pixel;
        int end_pixel;
        int shared_pixel;
        int* buffer;
#if MULTITHREADED == 1
        pthread_mutex_t mutex;
#endif
    };
    struct ThreadData {
        SharedData* shared;
    };
    struct PNGFillThreadData {
        const Solver* solver;
        png_bytep row;
        int start_pixel;
        int end_pixel;
        const int* buffer;
    };
#ifdef MPI_ENABLED
    struct MPITask {
        int start_pixel;
        int end_pixel;
    };
    struct PartialBuffer {
        MPITask task;
        int buffer[max_buffer_size];
    };
    struct MasterSharedData {
        Solver* solver;
        int num_procs;
        int batch_size;
        int* buffer;
        MPITask shared_task;
        pthread_mutex_t mutex;
    };
    struct MasterThreadData {
        MasterSharedData* shared;
        int rank;
        MPITask init_task;
    };
    struct WorkerSharedData {
        Solver* solver;
    };
    struct WorkerThreadData {
        WorkerSharedData* shared;
        MPITask init_task;
    };
#endif

    // Arguments
    char* filename;
    int iters;
    double left;
    double right;
    double lower;
    double upper;
    int width;
    int height;

    int num_cpus;
    int world_rank;
    int world_size;

    void random_choices(int* buffer, int buffer_size, int seed, int chunk_size);
    void mandelbrot();
#ifdef MPI_ENABLED
    void mandelbrot_mpi();
#endif
    void partial_mandelbrot(int* pixels, int num_pixels, int* buffer);
#if MULTITHREADED == 1
    static void* pthreads_partial_mandelbrot_thread(void* arg);
#endif
    void partial_mandelbrot_thread(ThreadData* thread_data);
    void partial_mandelbrot_single_thread(int* pixels, int num_pixels, int* buffer);
    void write_png(const int* buffer) const;
#if MULTITHREADED == 1
    static void* pthreads_write_png_fill_rows_thread(void* arg);
#endif
    void write_png_fill_rows(png_bytep* rows, png_bytep row, const int* buffer) const;
    void write_png_fill_rows_thread(PNGFillThreadData* thread_data) const;
};

int main(int argc, char** argv) {
    Solver solver;
    return solver.solve(argc, argv);
}

int Solver::solve(int argc, char** argv) {
    NVTX_RANGE_START(solve_all)
    std::ios::sync_with_stdio(0);
    std::cin.tie(0);
    std::cout.tie(0);
    if (argc != 9) {
        std::cerr << "must provide exactly 8 arguments!\n";
        return 1;
    }

#ifdef MPI_ENABLED
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        std::cerr << "MPI does not support MPI_THREAD_MULTIPLE, provided level: " << provided << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
#else
    world_size = 1;
    world_rank = 0;
#endif
    // detect how many CPUs are available
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    num_cpus = CPU_COUNT(&cpu_set);

    // argument parsing
    filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

#ifdef MPI_ENABLED
    if (world_size == 1 || (long long)iters * width * height <= min_tasks_per_process) {
        mandelbrot();
    } else {
        mandelbrot_mpi();
    }
#else
    mandelbrot();
#endif

#ifdef MPI_ENABLED
#ifndef NO_FINALIZE
    NVTX_RANGE_START(MPI_Finalize)
    MPI_Finalize();
    NVTX_RANGE_END()
#endif
#endif
    NVTX_RANGE_END()
    return 0;
}

void Solver::random_choices(int* buffer, int buffer_size, int seed, int chunk_size) {
    int num_chunks = std::ceil((double)buffer_size / chunk_size);
    int* chunks = (int*)malloc(num_chunks * sizeof(int));
    for (int i = 0; i < num_chunks; i++) {
        chunks[i] = i;
    }
    // Shuffle the chunks, note to exclude the last chunk
    srand(seed);
    for (int i = num_chunks - 2; i > 0; i--) {
        int j = rand() % (i + 1);
        std::swap(chunks[i], chunks[j]);
    }
    // Apply chunks
    for (int i = 0; i < buffer_size; i += chunk_size) {
        int chunk_id = chunks[i / chunk_size];
        int chunk_start = chunk_id * chunk_size;
        for (int j = 0; j < chunk_size && i + j < buffer_size; j++) {
            buffer[chunk_start + j] = i + j;
        }
    }
#ifndef NO_FINALIZE
    free(chunks);
#endif
}

void Solver::mandelbrot() {
    // allocate memory for image
    int* pixels = (int*)malloc(width * height * sizeof(int));
    for (int i = 0; i < width * height; i++) {
        pixels[i] = i;
    }
    int* buffer = (int*)malloc(width * height * sizeof(int));

    // compute partial mandelbrot set
    NVTX_RANGE_START(partial_mandelbrot)
    partial_mandelbrot(pixels, width * height, buffer);
    NVTX_RANGE_END()

    // draw and cleanup
    if (world_rank == 0) {
        NVTX_RANGE_START(write_png)
        write_png(buffer);
        NVTX_RANGE_END()
    }
    free(pixels);
    free(buffer);
}

#ifdef MPI_ENABLED
void Solver::mandelbrot_mpi() {
    int num_procs;
    int *pixels, *buffer, *tmp_buffer;
    int pivots[max_num_procs + 1];
    int pixels_per_proc;

    // setup tasks
    num_procs = world_size;
    pixels = (int*)malloc(width * height * sizeof(int));
    random_choices(pixels, width * height, 42, 2000);
    pixels_per_proc = std::ceil((double)width * height / num_procs);
    pivots[0] = 0;
    for (int i = 1; i < world_size; i++) {
        pivots[i] = std::min(pixels_per_proc + pivots[i - 1], width * height);
    }
    pivots[world_size] = width * height;
    // allocate memory for image
    tmp_buffer = (int*)malloc(width * height * sizeof(int));
    memset(tmp_buffer, 0, width * height * sizeof(int));
    buffer = (int*)malloc(width * height * sizeof(int));

    // compute partial mandelbrot set
    if (pivots[world_rank + 1] - pivots[world_rank] > 0) {
        NVTX_RANGE_START(partial_mandelbrot)
        partial_mandelbrot(pixels + pivots[world_rank], pivots[world_rank + 1] - pivots[world_rank], tmp_buffer);
        NVTX_RANGE_END()
    }

    MPI_Reduce(tmp_buffer, buffer, width * height, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // draw and cleanup
    if (world_rank == 0) {
        NVTX_RANGE_START(write_png)
        write_png(buffer);
        NVTX_RANGE_END()
    }
#ifndef NO_FINALIZE
    free(pixels);
    free(tmp_buffer);
    free(buffer);
#endif
}
#endif

void Solver::partial_mandelbrot(int* pixels, int num_pixels, int* buffer) {
    const int num_threads = num_cpus;
#if MULTITHREADED == 1 || MULTITHREADED == 2
    const int batch_size = std::min(1000, (int)std::ceil((double)width * height / num_threads));
#endif

    // Set up shared data
    if (num_threads > 1) {
#if MULTITHREADED == 1 || MULTITHREADED == 2
#if MULTITHREADED == 1
        pthread_t threads[max_num_cpus];
#endif
        SharedData shared_data;
        shared_data.solver = this;
        shared_data.num_threads = num_threads;
        shared_data.batch_size = batch_size;
        shared_data.pixels = pixels;
        shared_data.start_pixel = 0;
        shared_data.end_pixel = num_pixels;
        shared_data.shared_pixel = 0;
        shared_data.buffer = buffer;
#if MULTITHREADED == 1
        shared_data.mutex = PTHREAD_MUTEX_INITIALIZER;
#endif
        ThreadData thread_data_array[max_num_cpus];
        for (int i = 0; i < num_threads; i++) {
            thread_data_array[i].shared = &shared_data;
        }
#endif
#if MULTITHREADED == 1
        for (int i = 0; i < num_threads; i++) {
            pthread_create(&threads[i], NULL, pthreads_partial_mandelbrot_thread, (void*)&thread_data_array[i]);
        }
        for (int i = 0; i < num_threads; i++) {
            pthread_join(threads[i], NULL);
        }
#elif MULTITHREADED == 2
#pragma omp parallel num_threads(num_threads)
        {
            partial_mandelbrot_thread(&thread_data_array[omp_get_thread_num()]);
        }
#else
        partial_mandelbrot_single_thread(pixels, num_pixels, buffer);
#endif
    } else {
        partial_mandelbrot_single_thread(pixels, num_pixels, buffer);
    }
}

#if MULTITHREADED == 1
void* Solver::pthreads_partial_mandelbrot_thread(void* arg) {
    ThreadData* thread_data = (ThreadData*)arg;
    thread_data->shared->solver->partial_mandelbrot_thread(thread_data);
    return NULL;
}
#endif

void Solver::partial_mandelbrot_thread(ThreadData* thread_data) {
    SharedData* shared = thread_data->shared;
    Solver* solver = shared->solver;
    int batch_size = shared->batch_size;
    const int num_threads = shared->num_threads;
    const int end_pixel = shared->end_pixel;
    int* pixels = shared->pixels;
    int* buffer = shared->buffer;
#if MULTITHREADED == 1
    pthread_mutex_t* mutex = &shared->mutex;
#endif

    NVTX_RANGE_START(thread)
    while (true) {
        int curr_start_pixel;
        int curr_end_pixel;
#if MULTITHREADED == 2
#pragma omp critical
#endif
        {
#if MULTITHREADED == 1
            pthread_mutex_lock(mutex);
#endif
            curr_start_pixel = shared->shared_pixel;
            if (batch_size > 8 && end_pixel - curr_start_pixel <= 100 * num_threads * 1000) {
                shared->batch_size = 8;
                batch_size = shared->batch_size;
            }
            shared->shared_pixel += batch_size;
#if MULTITHREADED == 1
            pthread_mutex_unlock(mutex);
#endif
        }
        if (curr_start_pixel >= end_pixel) {
            break;
        }
        curr_end_pixel = std::min(curr_start_pixel + batch_size, end_pixel);
        solver->partial_mandelbrot_single_thread(pixels + curr_start_pixel, curr_end_pixel - curr_start_pixel, buffer);
    }
    NVTX_RANGE_END()
}

void Solver::partial_mandelbrot_single_thread(int* pixels, int num_pixels, int* buffer) {
    boost::sort::spreadsort::spreadsort(pixels, pixels + num_pixels);
    // mandelbrot set
    const double h_norm = (upper - lower) / height;
    const double w_norm = (right - left) / width;
    int pi = 0;
#if defined(__AVX512F__) && defined(SIMD_ENABLED)
    // Constants
    const int vec_8_size = 8;
    const __m256i vec_8_1_epi32 = _mm256_set1_epi32(1);
    const __m512d vec_8_2 = _mm512_set1_pd(2);
    const __m512d vec_8_4 = _mm512_set1_pd(4);
    const __m512d vec_8_width = _mm512_set1_pd(width);
    const __m512d vec_8_w_norm = _mm512_set1_pd(w_norm);
    const __m512d vec_8_h_norm = _mm512_set1_pd(h_norm);
    const __m512d vec_8_left = _mm512_set1_pd(left);
    const __m512d vec_8_lower = _mm512_set1_pd(lower);
    for (; pi + vec_8_size - 1 < num_pixels; pi += vec_8_size) {
        // Calculate pixel coordinates
        __m512d vec_p_offset = _mm512_cvtepi32_pd(_mm256_loadu_si256((__m256i*)&pixels[pi]));
        __m512d vec_j = _mm512_floor_pd(_mm512_div_pd(vec_p_offset, vec_8_width));
        __m512d vec_i = _mm512_floor_pd(_mm512_fnmadd_pd(vec_8_width, vec_j, vec_p_offset));

        // Calculate initial values
        __m512d vec_y0 = _mm512_fmadd_pd(vec_j, vec_8_h_norm, vec_8_lower);
        __m512d vec_x0 = _mm512_fmadd_pd(vec_i, vec_8_w_norm, vec_8_left);

        // Initialize variables
        __m256i vec_repeats = _mm256_set1_epi32(0);
        __m512d vec_x = _mm512_setzero_pd();
        __m512d vec_x_sq = _mm512_setzero_pd();
        __m512d vec_y = _mm512_setzero_pd();
        __m512d vec_y_sq = _mm512_setzero_pd();
        __m512d vec_length_squared = _mm512_setzero_pd();
        int repeats = 0;
        __mmask8 mask = 0xFF;
        while (repeats < iters && mask) {
            vec_y = _mm512_fmadd_pd(_mm512_mul_pd(vec_x, vec_y), vec_8_2, vec_y0);
            vec_x = _mm512_add_pd(_mm512_sub_pd(vec_x_sq, vec_y_sq), vec_x0);
            vec_y_sq = _mm512_mul_pd(vec_y, vec_y);
            vec_x_sq = _mm512_mul_pd(vec_x, vec_x);
            vec_length_squared = _mm512_fmadd_pd(vec_x, vec_x, vec_y_sq);
            vec_repeats = _mm256_mask_add_epi32(vec_repeats, mask, vec_repeats, vec_8_1_epi32);
            ++repeats;
            mask = _mm512_cmp_pd_mask(vec_length_squared, vec_8_4, _CMP_LT_OQ);
        }
        for (int i = 0; i < vec_8_size; i++) {
            buffer[pixels[pi + i]] = _mm256_extract_epi32(vec_repeats, i);
        }
    }
#endif
    for (; pi < num_pixels; ++pi) {
        int j = pixels[pi] / width;
        int i = pixels[pi] % width;
        double y0 = j * h_norm + lower;
        double x0 = i * w_norm + left;

        int repeats = 0;
        double x = 0;
        double x_sq = 0;
        double y = 0;
        double y_sq = 0;
        double length_squared = 0;
        while (repeats < iters && length_squared < 4) {
            y = 2 * x * y + y0;
            x = x_sq - y_sq + x0;
            y_sq = y * y;
            x_sq = x * x;
            length_squared = x_sq + y_sq;
            ++repeats;
        }
        buffer[pixels[pi]] = repeats;
    }
}

void Solver::write_png(const int* buffer) const {
    NVTX_RANGE_START(write_png_setup)
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 0);
    NVTX_RANGE_END()
    NVTX_RANGE_START(write_png_loop)
    size_t row_size = 3 * width * height * sizeof(png_bytep);
    png_bytep* rows = (png_bytep*)malloc(height * sizeof(png_bytep));
    png_bytep row = (png_bytep)malloc(row_size);
    write_png_fill_rows(rows, row, buffer);
    png_write_rows(png_ptr, rows, height);
    NVTX_RANGE_END()
    NVTX_RANGE_START(write_png_cleanup)
    png_write_end(png_ptr, NULL);
#ifndef NO_FINALIZE
    free(row);
    free(rows);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
#endif
    NVTX_RANGE_END()
}

#if MULTITHREADED == 1
void* Solver::pthreads_write_png_fill_rows_thread(void* arg) {
    PNGFillThreadData* thread_data = (PNGFillThreadData*)arg;
    thread_data->solver->write_png_fill_rows_thread(thread_data);
    return NULL;
}
#endif

/*
    A wrapper function to fill rows of the PNG image with multithreading.
 */
void Solver::write_png_fill_rows(png_bytep* rows, png_bytep row, const int* buffer) const {
    for (int y = 0; y < height; ++y) {
        rows[y] = row + y * 3 * width;
    }

    int pixels_per_thread = std::max(1000, (int)std::ceil((double)width * height / num_cpus));
    int num_threads = std::min(num_cpus, (int)std::ceil((double)width * height / pixels_per_thread));

    PNGFillThreadData thread_data_array[max_num_cpus];
    for (int i = 0; i < num_threads; i++) {
        thread_data_array[i].solver = this;
        thread_data_array[i].row = row;
        thread_data_array[i].start_pixel = std::min(height * width, i * pixels_per_thread);
        thread_data_array[i].end_pixel = std::min(height * width, (i + 1) * pixels_per_thread);
        thread_data_array[i].buffer = buffer;
    }
#if MULTITHREADED == 1
    pthread_t threads[max_num_cpus];
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, pthreads_write_png_fill_rows_thread, (void*)&thread_data_array[i]);
    }
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
#elif MULTITHREADED == 2
#pragma omp parallel num_threads(num_threads)
    {
        write_png_fill_rows_thread(&thread_data_array[omp_get_thread_num()]);
    }
#else
    for (int i = 0; i < num_threads; i++) {
        write_png_fill_rows_thread(&thread_data_array[i]);
    }
#endif
}

/*
    A function to fill rows of the PNG image with a single thread given a range of pixels.
 */
void Solver::write_png_fill_rows_thread(PNGFillThreadData* thread_data) const {
    int start_pixel = thread_data->start_pixel;
    int end_pixel = thread_data->end_pixel;
    const int* buffer = thread_data->buffer;
    png_bytep row = thread_data->row;
    for (int pixel = start_pixel; pixel < end_pixel; ++pixel) {
        int y = pixel / width;
        int x = pixel % width;
        int p = buffer[(height - 1 - y) * width + x];
        png_bytep color = row + pixel * 3;
        if (p != iters) {
            if (p & 16) {
                color[0] = 240;
                color[1] = color[2] = p % 16 * 16;
            } else {
                color[0] = p % 16 * 16;
                color[1] = color[2] = 0;
            }
        } else {
            color[0] = color[1] = color[2] = 0;
        }
    }
}