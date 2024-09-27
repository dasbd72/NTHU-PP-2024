#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>

#include <iostream>
#include <limits>

#ifdef DEBUG
#define DEBUG_MSG(str) std::cerr << str << "\n";
#else
#define DEBUG_MSG(str) \
    {}
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
#define TIMING_START(arg) \
    {}
#define TIMING_END(arg) \
    {}
#define TIMING_INIT(arg) \
    {}
#define TIMING_ACCUM(arg) \
    {}
#define TIMING_FIN(arg) \
    {}
#endif  // TIMING

typedef unsigned long long ull;
typedef long double ld;

class Solver {
   public:
    Solver() {}
    ~Solver();
    int solve(int argc, char** argv);

   private:
    const ull half_max_ull = std::numeric_limits<ull>::max() / 2;

    int rank = 0;
    int size = 1;
    ull r;
    ull k;
    ull sq_r;
    ull hf_x;
    ull* pivots;

    inline void exec_seq();
    inline void exec_mpi();
    inline void param_init();
    inline void partial_pixels(ull start, ull end, ull& pixels);
    inline void finalize_pixels(ull& pixels);
};

int main(int argc, char** argv) {
    Solver solver;
    return solver.solve(argc, argv);
}

Solver::~Solver() {
    if (size > 1) {
        delete[] pivots;
    }
}

int Solver::solve(int argc, char** argv) {
    std::ios::sync_with_stdio(0);
    std::cin.tie(0);
    std::cout.tie(0);
    if (argc != 3) {
        std::cerr << "must provide exactly 2 arguments!\n";
        return 1;
    }

    TIMING_START(mpi_init);
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    TIMING_END(mpi_init);

    r = atoll(argv[1]);
    k = atoll(argv[2]);
    TIMING_START(param_init);
    param_init();
    if (rank == 0) {
        TIMING_END(param_init);
    }

    if (size == 1 || r <= 21474) {
        if (rank == 0)
            exec_seq();
    } else {
        exec_mpi();
    }
    // TIMING_START(mpi_finallize);
    // MPI_Finalize();
    // TIMING_END(mpi_finallize);
    return 0;
}

inline void Solver::exec_seq() {
    TIMING_START(seq_all);
    ull pixels = 0;
    partial_pixels(hf_x, r, pixels);
    finalize_pixels(pixels);
    std::cout << pixels << "\n";
    TIMING_END(seq_all);
}

inline void Solver::param_init() {
    // Optimization: Compute first and reuse the square of radius
    sq_r = r * r;

    // Optimization: Reduce number of tasks to r - hf_x
    hf_x = ceil(sqrtl(sq_r / 2));  // End of all tasks

    // Optimization: Different task size for each process
    if (size > 1) {
        // The task separating problem
        // We have p,r > 0, r >= p
        // We want to separate p to r into x_0, x_1, x_{k+1} pivots
        // y_i = ceil(sqrtl(r^2-x_i^2)), j belongs to [0, k+1]
        // Where x_0 = p and x_{k+1} = r
        // Let A_j = (x_{j+1} - x_j) * (y_j - y_{j+1}), j belongs to [0, k]
        // Give the best way to let A_n ~= A_m for all n, m belongs to [0, k]

        pivots = new ull[sizeof(ull) * (size + 1)];
        const int method = 2;

        if (method == 0) {
            // Equal width method
            const ull width = (r - hf_x) / size;
            pivots[0] = hf_x;
            for (int i = 1; i < size; i++) {
                pivots[i] = hf_x + width * i;
            }
            pivots[size] = r;
        } else if (method == 1) {
            // Arithmetic sequence method
            const double factor = 0.62;
            const double sum_weights = size * 1.0 + factor * (size - 1) * size / 2;  // Sum of weights
            pivots[0] = hf_x;
            for (int i = 1; i < size; i++) {
                const double weight = 1.0 + (size - i) * factor;
                pivots[i] = pivots[i - 1] + ceil(weight / sum_weights * (r - hf_x));
            }
            pivots[size] = r;
        } else {
            // The purbula method
            const double a = -0.065;
            const double m = (double)size / 7;
            double w[size];
            double min_w = std::numeric_limits<double>::max();
            double sum_w = 0.0;
            for (int i = 0; i < size; i++) {
                w[i] = a * (i - m) * (i - m);
                min_w = std::min(min_w, w[i]);
            }
            for (int i = 0; i < size; i++) {
                w[i] = w[i] - min_w + 1;
                sum_w += w[i];
                if (rank == 0) {
                    DEBUG_MSG(i << " " << w[i]);
                }
            }
            pivots[0] = hf_x;
            for (int i = 1; i < size; i++) {
                pivots[i] = pivots[i - 1] + ceil(w[i - 1] / sum_w * (r - hf_x));
            }
            pivots[size] = r;
        }
    }
}

inline void Solver::exec_mpi() {
    TIMING_START(mpi_all);
    ull pixels = 0;
    ull* remote_pixels = nullptr;
    ull local_pixels = 0;
    TIMING_START(mpi_loop);
    partial_pixels(pivots[rank], pivots[rank + 1], local_pixels);
    TIMING_END(mpi_loop);
    if (rank == 0)
        remote_pixels = new ull[sizeof(ull) * size];
    TIMING_START(mpi_gather);
    MPI_Gather(&local_pixels, 1, MPI_UNSIGNED_LONG_LONG, remote_pixels, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        TIMING_END(mpi_gather);
    }
    if (rank == 0) {
        TIMING_START(mpi_summing);
        for (int i = 0; i < size; i++) {
            if (pixels > k)
                pixels %= k;
            pixels += remote_pixels[i];
        }
        finalize_pixels(pixels);
        std::cout << pixels << "\n";
        delete[] remote_pixels;
        TIMING_END(mpi_summing);
    }
    if (rank == 0) {
        TIMING_END(mpi_all);
    }
}

inline void Solver::partial_pixels(ull start, ull end, ull& pixels) {
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

inline void Solver::finalize_pixels(ull& pixels) {
    if (pixels > half_max_ull)
        pixels %= k;
    pixels *= 2;
    if (pixels > half_max_ull)
        pixels %= k;
    pixels += (hf_x * hf_x) % k;
    if (pixels > half_max_ull / 2)
        pixels %= k;
    pixels = (4 * pixels) % k;
}
