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
#endif  // MULTITHREADED == 1
#if MULTITHREADED == 2
#include <omp.h>
#endif  // MULTITHREADED == 2
#ifdef MPI_ENABLED
#include <mpi.h>
#endif  // MPI_ENABLED

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

    FILE* fp;
    png_structp png_ptr;
    png_infop info_ptr;

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
#endif  // defined(__AVX512F__) && defined(SIMD_ENABLED)

    png_bytep pm_image;
    int* pm_pixels;
    char* pm_pixels_done;
    int* pm_buffer;
    int pm_num_threads;
    int pm_batch_size;
    int pm_start_pixel;
    int pm_end_pixel;
    int pm_shared_pixel;
#if MULTITHREADED == 1
    pthread_mutex_t pm_mutex;
#endif  // MULTITHREADED == 1

    void random_choices(int* buffer, int buffer_size, int seed, int chunk_size);
    void mandelbrot();
#ifdef MPI_ENABLED
    void mandelbrot_mpi();
#endif  // MPI_ENABLED
    void partial_mandelbrot(png_bytep image, int* pixels, char* pixels_done, int num_pixels, int* buffer);
#if MULTITHREADED == 1
    static void* pthreads_partial_mandelbrot_thread(void* arg);
#endif  // MULTITHREADED == 1
    void partial_mandelbrot_thread();
    void partial_mandelbrot_single_thread(int* pixels, int num_pixels, int* buffer);

    void pixels_to_image_single_thread(png_bytep image, int* pixels, char* pixels_done, int num_pixels, const int* buffer) const;
    static void* pthreads_write_png_controller_thread(void* arg);
    void write_png_controller_thread();
    void write_png_init();
    void write_png_rows(const png_bytep image, int y, int n) const;
    void write_png_end() const;
    void write_png_cleanup();
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
#endif  // MPI_ENABLED
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
#endif  // defined(__AVX512F__) && defined(SIMD_ENABLED)

#ifdef MPI_ENABLED
    if (world_size == 1 || (long long)iters * width * height <= min_tasks_per_process) {
        if (world_rank == 0) {
            mandelbrot();
        }
    } else {
        mandelbrot_mpi();
    }
#else
    mandelbrot();
#endif  // MPI_ENABLED

#ifdef MPI_ENABLED
#ifndef NO_FINALIZE
    NVTX_RANGE_START(MPI_Finalize)
    MPI_Finalize();
    NVTX_RANGE_END()
#endif  // NO_FINALIZE
#endif  // MPI_ENABLED
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
#endif  // NO_FINALIZE
}

void Solver::mandelbrot() {
    int *pixels, *buffer;
    char* pixels_done;
    png_bytep image;
    pthread_t thread_png;

    // allocate memory for image
    pixels = (int*)malloc(width * height * sizeof(int));
    for (int i = 0; i < width * height; i++) {
        pixels[i] = i;
    }
    pixels_done = (char*)malloc(width * height * sizeof(char));
    memset(pixels_done, 0, width * height * sizeof(char));
    buffer = (int*)malloc(width * height * sizeof(int));
    image = (png_bytep)malloc(height * width * 3);

    // Create a thread to write PNG
    pthread_create(&thread_png, NULL, pthreads_write_png_controller_thread, (void*)this);

    // compute partial mandelbrot set
    NVTX_RANGE_START(partial_mandelbrot)
    partial_mandelbrot(image, pixels, pixels_done, width * height, buffer);
    NVTX_RANGE_END()

    pthread_join(thread_png, NULL);
#ifndef NO_FINALIZE
    write_png_cleanup();
    free(pixels);
    free(buffer);
    free(image);
#endif  // NO_FINALIZE
}

#ifdef MPI_ENABLED
void Solver::mandelbrot_mpi() {
    int num_procs;
    int *pixels, *buffer;
    int pivots[max_num_procs + 1];
    int pixels_per_proc;
    png_bytep agg_image = NULL, image;

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
    buffer = (int*)malloc(width * height * sizeof(int));
    memset(buffer, 0, width * height * sizeof(int));
    image = (png_bytep)malloc(height * width * 3 * sizeof(png_byte));
    memset(image, 0, height * width * 3);
    if (world_rank == 0) {
        agg_image = (png_bytep)malloc(height * width * 3 * sizeof(png_byte));
    }

    // compute partial mandelbrot set
    if (pivots[world_rank + 1] - pivots[world_rank] > 0) {
        NVTX_RANGE_START(partial_mandelbrot)
        partial_mandelbrot(image, pixels + pivots[world_rank], NULL, pivots[world_rank + 1] - pivots[world_rank], buffer);
        NVTX_RANGE_END()
    }

    MPI_Reduce(image, agg_image, width * height * 3, MPI_UNSIGNED_CHAR, MPI_SUM, 0, MPI_COMM_WORLD);

    // draw and cleanup
    if (world_rank == 0) {
        NVTX_RANGE_START(write_png)
        write_png_init();
        write_png_rows(agg_image, 0, height);
        write_png_end();
        NVTX_RANGE_END()
    }
#ifndef NO_FINALIZE
    write_png_cleanup();
    free(pixels);
    free(buffer);
    free(image);
    if (world_rank == 0) {
        free(agg_image);
    }
#endif  // NO_FINALIZE
}
#endif  // MPI_ENABLED

void Solver::partial_mandelbrot(png_bytep image, int* pixels, char* pixels_done, int num_pixels, int* buffer) {
    const int num_threads = num_cpus;
#if MULTITHREADED == 1 || MULTITHREADED == 2
    const int batch_size = std::min(width * height >= 10000000 ? 2048 : 512, (int)std::ceil((double)width * height / num_threads));
#endif  // MULTITHREADED == 1 || MULTITHREADED == 2

    // Set up shared data
    if (num_threads > 1) {
#if MULTITHREADED == 1 || MULTITHREADED == 2
#if MULTITHREADED == 1
        pthread_t threads[max_num_cpus];
#endif  // MULTITHREADED == 1
        pm_num_threads = num_threads;
        pm_batch_size = batch_size;
        pm_start_pixel = 0;
        pm_pixels = pixels;
        pm_pixels_done = pixels_done;
        pm_end_pixel = num_pixels;
        pm_shared_pixel = 0;
        pm_buffer = buffer;
        pm_image = image;
#if MULTITHREADED == 1
        pm_mutex = PTHREAD_MUTEX_INITIALIZER;
#endif  // MULTITHREADED == 1
#endif  // MULTITHREADED == 1 || MULTITHREADED == 2
#if MULTITHREADED == 1
        for (int i = 0; i < num_threads; i++) {
            pthread_create(&threads[i], NULL, pthreads_partial_mandelbrot_thread, (void*)this);
        }
        for (int i = 0; i < num_threads; i++) {
            pthread_join(threads[i], NULL);
        }
#elif MULTITHREADED == 2
#pragma omp parallel num_threads(num_threads)
        {
            partial_mandelbrot_thread();
        }
#else
        partial_mandelbrot_single_thread(pixels, num_pixels, buffer);
        pixels_to_image_single_thread(image, pixels, pixels_done, num_pixels, buffer);
#endif  // MULTITHREADED
    } else {
        partial_mandelbrot_single_thread(pixels, num_pixels, buffer);
        pixels_to_image_single_thread(image, pixels, pixels_done, num_pixels, buffer);
    }
}

#if MULTITHREADED == 1
void* Solver::pthreads_partial_mandelbrot_thread(void* arg) {
    Solver* solver = (Solver*)arg;
    solver->partial_mandelbrot_thread();
    return NULL;
}
#endif

void Solver::partial_mandelbrot_thread() {
    NVTX_RANGE_START(thread)
    while (true) {
        int curr_start_pixel;
        int curr_end_pixel;
        NVTX_RANGE_START(thread_critical)
#if MULTITHREADED == 2
#pragma omp critical
#endif  // MULTITHREADED == 2
        {
#if MULTITHREADED == 1
            pthread_mutex_lock(&pm_mutex);
#endif  // MULTITHREADED == 1
            curr_start_pixel = pm_shared_pixel;
            pm_shared_pixel += pm_batch_size;
#if MULTITHREADED == 1
            pthread_mutex_unlock(&pm_mutex);
#endif  // MULTITHREADED == 1
        }
        NVTX_RANGE_END()
        if (curr_start_pixel >= pm_end_pixel) {
            break;
        }
        curr_end_pixel = std::min(curr_start_pixel + pm_batch_size, pm_end_pixel);
        NVTX_RANGE_START(partial_mandelbrot_single_thread)
        partial_mandelbrot_single_thread(pm_pixels + curr_start_pixel, curr_end_pixel - curr_start_pixel, pm_buffer);
        NVTX_RANGE_END()
        NVTX_RANGE_START(pixels_to_image_single_thread)
        pixels_to_image_single_thread(pm_image, pm_pixels + curr_start_pixel, pm_pixels_done, curr_end_pixel - curr_start_pixel, pm_buffer);
        NVTX_RANGE_END()
    }
    NVTX_RANGE_END()
}

void Solver::partial_mandelbrot_single_thread(int* pixels, int num_pixels, int* buffer) {
    NVTX_RANGE_START(partial_mandelbrot_pixel_translation)
    int* trans_pixels = (int*)malloc(num_pixels * sizeof(int));
#pragma GCC ivdep
    for (int i = 0; i < num_pixels; i++) {
        int x = pixels[i] % width;
        int y = pixels[i] / width;
        trans_pixels[i] = (height - 1 - y) * width + x;
    }
    NVTX_RANGE_END()
    NVTX_RANGE_START(partial_mandelbrot_sort)
    boost::sort::spreadsort::integer_sort(trans_pixels, trans_pixels + num_pixels);
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
#define PIXEL_COORDINATES()                                                \
    vec_p_offset = _mm512_cvtepi32_pd(vec_p);                              \
    vec_j = _mm512_floor_pd(_mm512_mul_pd(vec_p_offset, vec_8_inv_width)); \
    vec_i = _mm512_fnmadd_pd(vec_8_width, vec_j, vec_p_offset);            \
    vec_y0 = _mm512_fmadd_pd(vec_j, vec_8_h_norm, vec_8_lower);            \
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
#define STATIC_INITIALIZATION()  \
    vec_repeats = vec_8_0_epi32; \
    vec_x = vec_8_0;             \
    vec_x_sq = vec_8_0;          \
    vec_y = vec_8_0;             \
    vec_y_sq = vec_8_0;  // STATIC_INITIALIZATION
#define STATIC_STORE_RESULTS(i) \
    buffer[_mm256_extract_epi32(vec_p, i)] = _mm256_extract_epi32(vec_repeats, i);  // STATIC_STORE_RESULTS
#define DYNAMIC_INITIALIZATION()                                                     \
    vec_repeats = _mm256_mask_mov_epi32(vec_repeats, mini_done_mask, vec_8_0_epi32); \
    vec_x = _mm512_mask_mov_pd(vec_x, mini_done_mask, vec_8_0);                      \
    vec_x_sq = _mm512_mask_mov_pd(vec_x_sq, mini_done_mask, vec_8_0);                \
    vec_y = _mm512_mask_mov_pd(vec_y, mini_done_mask, vec_8_0);                      \
    vec_y_sq = _mm512_mask_mov_pd(vec_y_sq, mini_done_mask, vec_8_0);                \
    length_valid_mask |= mini_done_mask;  // DYNAMIC_INITIALIZATION
#define DYNAMIC_STORE_RESULTS(i)                                                       \
    if (mini_done_mask & (1 << i)) {                                                   \
        buffer[_mm256_extract_epi32(vec_p, i)] = _mm256_extract_epi32(vec_repeats, i); \
        if (pi < num_pixels) {                                                         \
            vec_p = _mm256_insert_epi32(vec_p, trans_pixels[pi++], i);                 \
        } else {                                                                       \
            done_mask |= 1 << i;                                                       \
        }                                                                              \
    }  // DYNAMIC_STORE_RESULTS
    if (mini_iters * 10 >= iters) {
        // Static scheduling
        for (; pi + vec_8_size - 1 < num_pixels; pi += vec_8_size) {
            vec_p = _mm256_loadu_si256((__m256i*)&trans_pixels[pi]);
            // Calculate pixel coordinates
            PIXEL_COORDINATES()
            // Initialize iteration variables
            STATIC_INITIALIZATION()
            // Initialize masks
            length_valid_mask = 0xFF;
            for (int r = 0; r < iters && length_valid_mask; r++) {
                INNER_LOOP_COMPUTATION()
            }

            // Store results
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
        vec_p = _mm256_loadu_si256((__m256i*)&trans_pixels[pi]);
        pi += vec_8_size;
        // Initialize masks
        length_valid_mask = 0xFF;
        mini_done_mask = 0xFF;
        done_mask = 0x0;
        while (done_mask != 0xFF) {
            // Initialize values for mini iterations done entries
            // Calculate pixel coordinates
            PIXEL_COORDINATES()
            // Initialize iteration variables & masks
            DYNAMIC_INITIALIZATION()
            for (int r = 0; r < mini_iters && length_valid_mask; r++) {
                INNER_LOOP_COMPUTATION()
            }
            // Clamp repeats to iters
            repeats_exceed_mask = _mm256_cmpge_epi32_mask(vec_repeats, vec_8_iters_epi32);
            vec_repeats = _mm256_mask_mov_epi32(vec_repeats, repeats_exceed_mask, vec_8_iters_epi32);
            mini_done_mask = (~length_valid_mask | repeats_exceed_mask) & ~done_mask & 0xFF;

            // Store results
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
#endif  // defined(__AVX512F__) && defined(SIMD_ENABLED)
    NVTX_RANGE_START(partial_mandelbrot_pixels)
    for (; pi < num_pixels; ++pi) {
        int j = trans_pixels[pi] / width;
        int i = trans_pixels[pi] % width;
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
        buffer[trans_pixels[pi]] = repeats;
    }
    NVTX_RANGE_END()
    free(trans_pixels);
}

/*
    A function to fill rows of the PNG image with a single thread given a range of pixels.
 */
void Solver::pixels_to_image_single_thread(png_bytep image, int* pixels, char* pixels_done, int num_pixels, const int* buffer) const {
    for (int pi = 0; pi < num_pixels; ++pi) {
        int pixel = pixels[pi];
        int y = pixel / width;
        int x = pixel % width;
        int p = buffer[(height - 1 - y) * width + x];
        png_bytep color = image + pixel * 3;
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
        if (pixels_done) {
            pixels_done[pixel] = 1;
        }
    }
}

void* Solver::pthreads_write_png_controller_thread(void* arg) {
    Solver* solver = (Solver*)arg;
    solver->write_png_controller_thread();
    return NULL;
}

void Solver::write_png_controller_thread() {
    int pi = 0;
    write_png_init();
    png_set_flush(png_ptr, 1);
    while (pi < width * height) {
        if (pm_pixels_done[pi] == 0) {
            usleep(100);
            continue;
        }
        pi++;
        if (pi % width == 0) {
            write_png_rows(pm_image, pi / width - 1, 1);
        }
    }
    write_png_end();
}

void Solver::write_png_init() {
    NVTX_RANGE_START(write_png_init)
    fp = fopen(filename, "wb");
    assert(fp);
    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_FILTER_NONE);
    png_set_compression_level(png_ptr, 0);
    png_init_io(png_ptr, fp);
    png_write_info(png_ptr, info_ptr);
    NVTX_RANGE_END()
}

void Solver::write_png_rows(const png_bytep image, int y, int n) const {
    NVTX_RANGE_START(write_png_rows)
    png_bytep* rows = (png_bytep*)malloc(n * sizeof(png_bytep));
    for (int i = 0; i < n; i++) {
        rows[i] = image + (y + i) * width * 3;
    }
    png_write_rows(png_ptr, rows, n);
    free(rows);
    NVTX_RANGE_END()
}

void Solver::write_png_end() const {
    NVTX_RANGE_START(write_png_end)
    png_write_end(png_ptr, NULL);
    NVTX_RANGE_END()
}

void Solver::write_png_cleanup() {
    NVTX_RANGE_START(write_png_cleanup)
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
    NVTX_RANGE_END()
}