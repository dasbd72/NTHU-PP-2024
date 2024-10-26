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

    // Shared derived variables
    double h_norm;
    double w_norm;
#if defined(__AVX512F__) && defined(SIMD_ENABLED)
    __m256i vec_8_0_epi32 = _mm256_setzero_si256();
    __m512d vec_8_0 = _mm512_setzero_pd();
    __m256i vec_8_1_epi32 = _mm256_set1_epi32(1);
    __m512d vec_8_2 = _mm512_set1_pd(2);
    __m512d vec_8_4 = _mm512_set1_pd(4);
    __m512d vec_8_width;
    __m512d vec_8_inv_width;
    __m512d vec_8_w_norm;
    __m512d vec_8_h_norm;
    __m512d vec_8_left;
    __m512d vec_8_lower;
    __m256i vec_8_iters_epi32;
#endif

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

    // set up shared derived variables
    h_norm = (upper - lower) / height;
    w_norm = (right - left) / width;
#if defined(__AVX512F__) && defined(SIMD_ENABLED)
    vec_8_width = _mm512_set1_pd(width);
    vec_8_inv_width = _mm512_set1_pd(1.0 / width);
    vec_8_w_norm = _mm512_set1_pd(w_norm);
    vec_8_h_norm = _mm512_set1_pd(h_norm);
    vec_8_left = _mm512_set1_pd(left);
    vec_8_lower = _mm512_set1_pd(lower);
    vec_8_iters_epi32 = _mm256_set1_epi32(iters);
#endif

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
    const int batch_size = std::min(width * height >= 10000000 ? 2048 : 512, (int)std::ceil((double)width * height / num_threads));
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
        NVTX_RANGE_START(thread_critical)
#if MULTITHREADED == 2
#pragma omp critical
#endif
        {
#if MULTITHREADED == 1
            pthread_mutex_lock(mutex);
#endif
            curr_start_pixel = shared->shared_pixel;
            shared->shared_pixel += batch_size;
#if MULTITHREADED == 1
            pthread_mutex_unlock(mutex);
#endif
        }
        NVTX_RANGE_END()
        if (curr_start_pixel >= end_pixel) {
            break;
        }
        curr_end_pixel = std::min(curr_start_pixel + batch_size, end_pixel);
        NVTX_RANGE_START(partial_mandelbrot)
        solver->partial_mandelbrot_single_thread(pixels + curr_start_pixel, curr_end_pixel - curr_start_pixel, buffer);
        NVTX_RANGE_END()
    }
    NVTX_RANGE_END()
}

void Solver::partial_mandelbrot_single_thread(int* pixels, int num_pixels, int* buffer) {
    NVTX_RANGE_START(partial_mandelbrot_sort)
    boost::sort::spreadsort::integer_sort(pixels, pixels + num_pixels);
    NVTX_RANGE_END()
    // mandelbrot set
    int pi = 0;
#if defined(__AVX512F__) && defined(SIMD_ENABLED)
    NVTX_RANGE_START(partial_mandelbrot_pixels_vec_8)
    // Constants
    const int vec_8_size = 8;
    const int mini_iters = std::min(500, iters);
    // Declare variables
    // Coordinates
    __m256i vec_p;
    __m512d vec_p_offset;
    __m512d vec_j;
    __m512d vec_i;
    __m512d vec_y0;
    __m512d vec_x0;
    // Iteration variables
    __m256i vec_repeats = vec_8_0_epi32;
    __m512d vec_x = vec_8_0;
    __m512d vec_x_sq = vec_8_0;
    __m512d vec_y = vec_8_0;
    __m512d vec_y_sq = vec_8_0;
    __m512d vec_x_y, vec_length_squared;
    // Masks
    __mmask8 length_valid_mask;
    __mmask8 repeats_exceed_mask;
    __mmask8 mini_done_mask;
    __mmask8 done_mask;
#define PIXEL_COORDINATES()                                                      \
    vec_p_offset = _mm512_cvtepi32_pd(vec_p);                                    \
    vec_j = _mm512_floor_pd(_mm512_mul_pd(vec_p_offset, vec_8_inv_width));       \
    vec_i = _mm512_floor_pd(_mm512_fnmadd_pd(vec_8_width, vec_j, vec_p_offset)); \
    vec_y0 = _mm512_fmadd_pd(vec_j, vec_8_h_norm, vec_8_lower);                  \
    vec_x0 = _mm512_fmadd_pd(vec_i, vec_8_w_norm, vec_8_left);  // PIXEL_COORDINATES
#define INNER_LOOP_COMPUTATION()                                                                     \
    vec_x_y = _mm512_mul_pd(vec_x, vec_y);                                                           \
    vec_y = _mm512_fmadd_pd(vec_x_y, vec_8_2, vec_y0);                                               \
    vec_x = _mm512_add_pd(_mm512_sub_pd(vec_x_sq, vec_y_sq), vec_x0);                                \
    vec_y_sq = _mm512_mul_pd(vec_y, vec_y);                                                          \
    vec_x_sq = _mm512_mul_pd(vec_x, vec_x);                                                          \
    vec_length_squared = _mm512_fmadd_pd(vec_x, vec_x, vec_y_sq);                                    \
    vec_repeats = _mm256_mask_add_epi32(vec_repeats, length_valid_mask, vec_repeats, vec_8_1_epi32); \
    length_valid_mask = _mm512_cmp_pd_mask(vec_length_squared, vec_8_4, _CMP_LT_OQ);  // INNER_LOOP_COMPUTATION
    if (mini_iters * 10 >= iters) {
        // Static scheduling
        for (; pi + vec_8_size - 1 < num_pixels; pi += vec_8_size) {
            vec_p = _mm256_loadu_si256((__m256i*)&pixels[pi]);
            // Calculate pixel coordinates
            PIXEL_COORDINATES()
            // Initialize iteration variables
            vec_repeats = vec_8_0_epi32;
            vec_x = vec_8_0;
            vec_x_sq = vec_8_0;
            vec_y = vec_8_0;
            vec_y_sq = vec_8_0;
            // Initialize masks
            length_valid_mask = 0xFF;
            for (int r = 0; r < iters && length_valid_mask; r++) {
                INNER_LOOP_COMPUTATION()
            }

            // Store results
#define STATIC_STORE_RESULTS(i) \
    buffer[_mm256_extract_epi32(vec_p, i)] = _mm256_extract_epi32(vec_repeats, i);  // STATIC_STORE_RESULTS
            STATIC_STORE_RESULTS(0)
            STATIC_STORE_RESULTS(1)
            STATIC_STORE_RESULTS(2)
            STATIC_STORE_RESULTS(3)
            STATIC_STORE_RESULTS(4)
            STATIC_STORE_RESULTS(5)
            STATIC_STORE_RESULTS(6)
            STATIC_STORE_RESULTS(7)
        }
    } else if (pi + vec_8_size - 1 < num_pixels) {
        // Dynamic scheduling
        // Load first 8 pixels
        vec_p = _mm256_loadu_si256((__m256i*)&pixels[pi]);
        pi += vec_8_size;
        // Initialize masks
        length_valid_mask = 0xFF;
        mini_done_mask = 0xFF;
        done_mask = 0x0;
        while (done_mask != 0xFF) {
            // Initialize values for mini iterations done entries
            // Calculate pixel coordinates
            PIXEL_COORDINATES()
            // Initialize iteration variables
            vec_repeats = _mm256_mask_mov_epi32(vec_repeats, mini_done_mask, vec_8_0_epi32);
            vec_x = _mm512_mask_mov_pd(vec_x, mini_done_mask, vec_8_0);
            vec_x_sq = _mm512_mask_mov_pd(vec_x_sq, mini_done_mask, vec_8_0);
            vec_y = _mm512_mask_mov_pd(vec_y, mini_done_mask, vec_8_0);
            vec_y_sq = _mm512_mask_mov_pd(vec_y_sq, mini_done_mask, vec_8_0);
            // Initialize masks
            length_valid_mask |= mini_done_mask;
            for (int r = 0; r < mini_iters && length_valid_mask; r++) {
                INNER_LOOP_COMPUTATION()
            }
            // Clamp repeats to iters
            repeats_exceed_mask = _mm256_cmpge_epi32_mask(vec_repeats, vec_8_iters_epi32);
            vec_repeats = _mm256_mask_mov_epi32(vec_repeats, repeats_exceed_mask, vec_8_iters_epi32);
            mini_done_mask = (~length_valid_mask | repeats_exceed_mask) & ~done_mask & 0xFF;

            // Store results
#define DYNAMIC_STORE_RESULTS(i)                                                       \
    if (mini_done_mask & (1 << i)) {                                                   \
        buffer[_mm256_extract_epi32(vec_p, i)] = _mm256_extract_epi32(vec_repeats, i); \
        if (pi < num_pixels) {                                                         \
            vec_p = _mm256_insert_epi32(vec_p, pixels[pi++], i);                       \
        } else {                                                                       \
            done_mask |= 1 << i;                                                       \
        }                                                                              \
    }  // DYNAMIC_STORE_RESULTS
            DYNAMIC_STORE_RESULTS(0)
            DYNAMIC_STORE_RESULTS(1)
            DYNAMIC_STORE_RESULTS(2)
            DYNAMIC_STORE_RESULTS(3)
            DYNAMIC_STORE_RESULTS(4)
            DYNAMIC_STORE_RESULTS(5)
            DYNAMIC_STORE_RESULTS(6)
            DYNAMIC_STORE_RESULTS(7)
        }
    }
    NVTX_RANGE_END()
#endif
    NVTX_RANGE_START(partial_mandelbrot_pixels)
    for (; pi < num_pixels; ++pi) {
        int j = pixels[pi] / width;
        int i = pixels[pi] % width;
        double y0 = j * h_norm + lower;
        double x0 = i * w_norm + left;

        int repeats = 1;
        double x = x0;
        double y = y0;
        double x_sq = x * x;
        double y_sq = y * y;
        double length_squared = x_sq + y_sq;
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
    NVTX_RANGE_END()
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