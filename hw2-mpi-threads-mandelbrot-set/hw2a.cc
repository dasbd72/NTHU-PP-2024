#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <immintrin.h>
#include <png.h>
#include <sched.h>

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

class Solver {
   public:
    Solver() {}
    ~Solver() {};
    int solve(int argc, char** argv);

   private:
#if MULTITHREADED == 1
    struct SharedData {
        Solver* solver;
        int batch_size;
        int end_height;
        int shared_height;
        int* buffer;
        pthread_mutex_t mutex;
    };
    struct ThreadData {
        SharedData* shared;
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

    // Constants
    const long long min_tasks_per_process = (long long)500 * 1920 * 1080;
    const int min_height_per_process = 10;

    void mandelbrot(int* buffer);
#ifdef MPI_ENABLED
    void mandelbrot_mpi(int* buffer);
#endif
    void partial_mandelbrot(int start_height, int end_height, int* buffer);
    void partial_mandelbrot_single_thread(int start_height, int end_height, int* buffer);
#if MULTITHREADED == 1
    static void* pthreads_partial_mandelbrot(void* arg);
#endif
    void write_png(const int* buffer) const;
};

int main(int argc, char** argv) {
    Solver solver;
    return solver.solve(argc, argv);
}

int Solver::solve(int argc, char** argv) {
    std::ios::sync_with_stdio(0);
    std::cin.tie(0);
    std::cout.tie(0);
    if (argc != 9) {
        std::cerr << "must provide exactly 8 arguments!\n";
        return 1;
    }

#ifdef MPI_ENABLED
    MPI_Init(&argc, &argv);
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

    // allocate memory for image
    int* buffer = (int*)malloc(width * height * sizeof(int));

#ifdef MPI_ENABLED
    if (world_size == 1 || (long long)iters * width * height <= min_tasks_per_process) {
        mandelbrot(buffer);
    } else {
        mandelbrot_mpi(buffer);
    }
#else
    mandelbrot(buffer);
#endif

    // draw and cleanup
    if (world_rank == 0) {
        write_png(buffer);
    }
    free(buffer);

#ifdef MPI_ENABLED
    // MPI_Finalize();
#endif
    return 0;
}

void Solver::mandelbrot(int* buffer) {
    // compute partial mandelbrot set
    partial_mandelbrot(0, height, buffer);
}

#ifdef MPI_ENABLED
void Solver::mandelbrot_mpi(int* buffer) {
    // arguments
    const int height_per_process = std::max(min_height_per_process, (int)std::ceil((double)height / world_size));
    const int actual_world_size = std::ceil((double)height / height_per_process);
    int pivots[actual_world_size + 1] = {0};
    for (int i = 1; i < actual_world_size; i++) {
        pivots[i] = pivots[i - 1] + height_per_process;
    }
    pivots[actual_world_size] = height;
    if (world_rank >= actual_world_size) {
        return;
    }

    // compute partial mandelbrot set
    partial_mandelbrot(pivots[world_rank], pivots[world_rank + 1], buffer);

    // aggregate results
    if (world_rank == 0) {
        for (int i = 1; i < actual_world_size; i++) {
            MPI_Recv(buffer + pivots[i] * width, (pivots[i + 1] - pivots[i]) * width, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(buffer + pivots[world_rank] * width, (pivots[world_rank + 1] - pivots[world_rank]) * width, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}
#endif

void Solver::partial_mandelbrot(int start_height, int end_height, int* buffer) {
    const int num_threads = num_cpus;
    const int batch_size = std::min(10, (int)std::ceil((double)height / num_threads));

// mandelbrot set
#if MULTITHREADED == 1
    pthread_t threads[num_threads];
    SharedData shared_data;
    shared_data.solver = this;
    shared_data.batch_size = batch_size;
    shared_data.end_height = end_height;
    shared_data.shared_height = start_height;
    shared_data.buffer = buffer;
    shared_data.mutex = PTHREAD_MUTEX_INITIALIZER;
    ThreadData thread_data_array[num_threads];
    for (int i = 0; i < num_threads; i++) {
        thread_data_array[i].shared = &shared_data;
        pthread_create(&threads[i], NULL, pthreads_partial_mandelbrot, (void*)&thread_data_array[i]);
    }
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
#elif MULTITHREADED == 2
    int shared_height = start_height;
#pragma omp parallel num_threads(num_threads) shared(shared_height, buffer)
    {
        while (true) {
            int curr_start_height;
            int curr_end_height;
#pragma omp critical
            {
                curr_start_height = shared_height;
                shared_height += batch_size;
            }
            if (curr_start_height >= end_height) {
                break;
            }
            curr_end_height = std::min(curr_start_height + batch_size, end_height);
            partial_mandelbrot_single_thread(curr_start_height, curr_end_height, buffer);
        }
    }
#endif
}

void Solver::partial_mandelbrot_single_thread(int start_height, int end_height, int* buffer) {
    // mandelbrot set
    const double h_norm = (upper - lower) / height;
    const double w_norm = (right - left) / width;
#if defined(__AVX512F__) && defined(SIMD_ENABLED)
    // Constants
    const int vec_size = 8;
    const __m256i vec_1_epi32 = _mm256_set1_epi32(1);
    const __m512d vec_2 = _mm512_set1_pd(2);
    const __m512d vec_4 = _mm512_set1_pd(4);
    const __m512d vec_offset = _mm512_set_pd(7, 6, 5, 4, 3, 2, 1, 0);
    const __m512d vec_w_norm = _mm512_set1_pd(w_norm);
    const __m512d vec_left = _mm512_set1_pd(left);
    for (int j = start_height; j < end_height; ++j) {
        double y0 = j * h_norm + lower;
        __m512d vec_y0 = _mm512_set1_pd(y0);

        int i = 0;
        for (; i + vec_size - 1 < width; i += vec_size) {
            __m512d vec_x0 = _mm512_add_pd(_mm512_mul_pd(_mm512_add_pd(_mm512_set1_pd(i), vec_offset), vec_w_norm), vec_left);

            __m256i vec_repeats = _mm256_set1_epi32(0);
            __m512d vec_x = _mm512_setzero_pd();
            __m512d vec_x_sq = _mm512_setzero_pd();
            __m512d vec_y = _mm512_setzero_pd();
            __m512d vec_y_sq = _mm512_setzero_pd();
            __m512d vec_length_squared = _mm512_setzero_pd();

            int repeats = 0;
            __mmask8 mask = 0xFF;
            while (repeats < iters && mask) {
                __m512d vec_temp = _mm512_add_pd(_mm512_sub_pd(vec_x_sq, vec_y_sq), vec_x0);

                vec_y = _mm512_add_pd(_mm512_mul_pd(_mm512_mul_pd(vec_x, vec_y), vec_2), vec_y0);
                vec_y_sq = _mm512_mul_pd(vec_y, vec_y);
                vec_x = vec_temp;
                vec_x_sq = _mm512_mul_pd(vec_x, vec_x);
                vec_length_squared = _mm512_add_pd(vec_x_sq, vec_y_sq);
                vec_repeats = _mm256_mask_add_epi32(vec_repeats, mask, vec_repeats, vec_1_epi32);

                ++repeats;
                mask = _mm512_cmp_pd_mask(vec_length_squared, vec_4, _CMP_LT_OQ);
            }
            _mm256_storeu_epi32(&buffer[j * width + i], vec_repeats);
        }
        for (; i < width; ++i) {
            double x0 = i * w_norm + left;

            int repeats = 0;
            double x = 0;
            double y = 0;
            double length_squared = 0;
            while (repeats < iters && length_squared < 4) {
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            buffer[j * width + i] = repeats;
        }
    }
#else
    for (int j = start_height; j < end_height; ++j) {
        double y0 = j * h_norm + lower;
        for (int i = 0; i < width; ++i) {
            double x0 = i * w_norm + left;

            int repeats = 0;
            double x = 0;
            double y = 0;
            double length_squared = 0;
            while (repeats < iters && length_squared < 4) {
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            buffer[j * width + i] = repeats;
        }
    }
#endif
}

#if MULTITHREADED == 1
void* Solver::pthreads_partial_mandelbrot(void* arg) {
    ThreadData* thread_data = (ThreadData*)arg;
    SharedData* shared = thread_data->shared;
    Solver* solver = shared->solver;
    const int batch_size = shared->batch_size;
    const int end_height = shared->end_height;
    int* buffer = shared->buffer;
    pthread_mutex_t* mutex = &shared->mutex;

    while (true) {
        int curr_start_height;
        int curr_end_height;
        pthread_mutex_lock(mutex);
        curr_start_height = shared->shared_height;
        shared->shared_height += batch_size;
        pthread_mutex_unlock(mutex);
        if (curr_start_height >= end_height) {
            break;
        }
        curr_end_height = std::min(curr_start_height + batch_size, end_height);
        solver->partial_mandelbrot_single_thread(curr_start_height, curr_end_height, buffer);
    }
    return NULL;
}
#endif

void Solver::write_png(const int* buffer) const {
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
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}