#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>

#include <iostream>
#include <limits>

#ifdef DEBUG
#define DEBUG_MSG(str) std::cerr << str << "\\n";
#else
#define DEBUG_MSG(str)
#endif  // DEBUG

#ifdef TIMING
#include <ctime>
#define TIMING_START(arg)          \
    struct timespec __start_##arg; \
    clock_gettime(CLOCK_MONOTONIC, &__start_##arg);
#define TIMING_END(arg)                                                                       \
    {                                                                                         \
        struct timespec __temp_##arg, __end_##arg;                                            \
        double __duration_##arg;                                                              \
        clock_gettime(CLOCK_MONOTONIC, &__end_##arg);                                         \
        if ((__end_##arg.tv_nsec - __start_##arg.tv_nsec) < 0) {                              \
            __temp_##arg.tv_sec = __end_##arg.tv_sec - __start_##arg.tv_sec - 1;              \
            __temp_##arg.tv_nsec = 1000000000 + __end_##arg.tv_nsec - __start_##arg.tv_nsec;  \
                                                                                              \
        } else {                                                                              \
            __temp_##arg.tv_sec = __end_##arg.tv_sec - __start_##arg.tv_sec;                  \
            __temp_##arg.tv_nsec = __end_##arg.tv_nsec - __start_##arg.tv_nsec;               \
        }                                                                                     \
        __duration_##arg = __temp_##arg.tv_sec + (double)__temp_##arg.tv_nsec / 1000000000.0; \
        std::cerr << #arg << " " << rank << " took " << __duration_##arg << "s.\n";           \
        std::cerr.flush();                                                                    \
    }
#define TIMING_INIT(arg) double __duration_##arg = 0;
#define TIMING_ACCUM(arg)                                                                      \
    {                                                                                          \
        struct timespec __temp_##arg, __end_##arg;                                             \
        clock_gettime(CLOCK_MONOTONIC, &__end_##arg);                                          \
        if ((__end_##arg.tv_nsec - __start_##arg.tv_nsec) < 0) {                               \
            __temp_##arg.tv_sec = __end_##arg.tv_sec - __start_##arg.tv_sec - 1;               \
            __temp_##arg.tv_nsec = 1000000000 + __end_##arg.tv_nsec - __start_##arg.tv_nsec;   \
                                                                                               \
        } else {                                                                               \
            __temp_##arg.tv_sec = __end_##arg.tv_sec - __start_##arg.tv_sec;                   \
            __temp_##arg.tv_nsec = __end_##arg.tv_nsec - __start_##arg.tv_nsec;                \
        }                                                                                      \
        __duration_##arg += __temp_##arg.tv_sec + (double)__temp_##arg.tv_nsec / 1000000000.0; \
    }
#define TIMING_FIN(arg)                                          \
    std::cerr << #arg << " took " << __duration_##arg << "s.\n"; \
    std::cerr.flush();
#else
#define TIMING_START(arg)
#define TIMING_END(arg)
#define TIMING_INIT(arg)
#define TIMING_ACCUM(arg)
#define TIMING_FIN(arg)
#endif  // TIMING

typedef unsigned long long ull;
typedef long double ld;

const ull half_max_ull = std::numeric_limits<ull>::max() / 2;
int rank = 0, size = 1;

inline void partial_pixels(ull r, ull k, ull start, ull end, ull* pixels);
inline void finalize_pixels(ull k, ull square, ull& pixels);
void exec_seq(ull r, ull k);
void exec_mpi(ull r, ull k);

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(0);
    std::cin.tie(0);
    std::cout.tie(0);
    if (argc != 3) {
        std::cerr << "must provide exactly 2 arguments!\n";
        return 1;
    }
    ull r = atoll(argv[1]);
    ull k = atoll(argv[2]);
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size == 1) {
        if (rank != 0)
            return 0;
        exec_seq(r, k);
    } else {
        exec_mpi(r, k);
    }
    MPI_Finalize();
}

inline void partial_pixels(ull r, ull k, ull start, ull end, ull& pixels) {
    const ull sq_r = r * r;
    ull pxls = 0;
    ull x = start;
    ull y = ceil(sqrtl(sq_r - start * start));
    // Optimization: Unroll to reduce modular
    for (; x + 3 < end; x += 4) {
        ull diff1 = sq_r - x * x;
        ull diff2 = sq_r - (x + 1) * (x + 1);
        ull diff3 = sq_r - (x + 2) * (x + 2);
        ull diff4 = sq_r - (x + 3) * (x + 3);
        // Optimization: Find y by substracting instead of computing ceiling and square root
        while ((y - 1) * (y - 1) >= diff1) {
            y--;
        }
        pxls += y;
        while ((y - 1) * (y - 1) >= diff2) {
            y--;
        }
        pxls += y;
        while ((y - 1) * (y - 1) >= diff3) {
            y--;
        }
        pxls += y;
        while ((y - 1) * (y - 1) >= diff4) {
            y--;
        }
        pxls += y;
        if (pxls > half_max_ull)
            pxls %= k;
    }
    for (; x < end; x++) {
        ull diff = sq_r - x * x;
        while ((y - 1) * (y - 1) >= diff)
            y--;
        if (pxls > half_max_ull)
            pxls %= k;
        pxls += y;
    }
    pixels = pxls;
}

inline void finalize_pixels(ull k, ull square, ull& pixels) {
    if (pixels > half_max_ull)
        pixels %= k;
    pixels *= 2;
    if (pixels > half_max_ull)
        pixels %= k;
    pixels += square;
    if (pixels > half_max_ull)
        pixels %= k;
    pixels = (4 * pixels) % k;
}

void exec_seq(ull r, ull k) {
    TIMING_START(seq_all);
    const ull sq_r = r * r;
    const ull hf_x = ceil(sqrtl(sq_r / 2));
    const ull square = k - (hf_x * hf_x) % k;
    ull pixels = 0;
    partial_pixels(r, k, 0, hf_x, pixels);
    finalize_pixels(k, square, pixels);
    std::cout << pixels << "\n";
    TIMING_END(seq_all);
}

void exec_mpi(ull r, ull k) {
    TIMING_START(mpi_all);
    // Optimization: Compute first and reuse the square of radius
    const ull sq_r = r * r;
    const ull hf_x = ceil(sqrtl(sq_r / 2));    // End of all tasks
    const ull square = k - (hf_x * hf_x) % k;  // Value to be deducted after multipling pixels by 2
    // Optimization: Different task size for each process
    const double factor = 0.08;
    const double sum_weights = size * 1.0 + factor * (size - 1) * size / 2;  // Sum of weights
    ull pivots[size + 1] = {};
    pivots[0] = 0;
    for (int i = 1; i < size; i++)
        pivots[i] = pivots[i - 1] + ceil((1.0 + (size - i) * factor) / sum_weights * hf_x);
    pivots[size] = hf_x;
    const ull local_start = pivots[rank];
    const ull local_end = pivots[rank + 1];
    ull pixels = 0;
    ull* remote_pixels = nullptr;
    ull local_pixels = 0;
    TIMING_START(mpi_loop);
    partial_pixels(r, k, local_start, local_end, local_pixels);
    TIMING_END(mpi_loop);
    if (rank == 0)
        remote_pixels = (ull*)malloc(sizeof(ull) * size);  // Allocate buffer
    TIMING_START(mpi_gather);
    MPI_Gather(&local_pixels, 1, MPI_UNSIGNED_LONG_LONG, remote_pixels, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        TIMING_END(mpi_gather);
    }
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            if (pixels > k)
                pixels %= k;
            pixels += remote_pixels[i];
        }
        finalize_pixels(k, square, pixels);
        std::cout << pixels << "\n";
    }
    if (rank == 0) {
        TIMING_END(mpi_all);
    }
}