#include <mpi.h>

#include <boost/sort/spreadsort/spreadsort.hpp>
#include <cstdio>
#include <cstdlib>
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
#define TIMING_END(arg, i)                                                                    \
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
        std::cerr << #arg << " " << i << " took " << __duration_##arg << "s.\n";              \
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
#define TIMING_FIN(arg, i)                                                   \
    std::cerr << #arg << " " << i << " took " << __duration_##arg << "s.\n"; \
    std::cerr.flush();
#else
#define TIMING_START(arg) \
    {}
#define TIMING_END(arg, i) \
    {}
#define TIMING_INIT(arg) \
    {}
#define TIMING_ACCUM(arg) \
    {}
#define TIMING_FIN(arg, i) \
    {}
#endif  // TIMING

class Solver {
   public:
    Solver() {}
    ~Solver() {};
    int solve(int argc, char **argv);
    void odd_even_sort_seq();
    void odd_even_sort_mpi();

   private:
    const int MIN_SIZE_PER_PROC = 100000;
    int world_rank = 0;
    int world_size = 1;
    int array_size = 0;
    char *input_filename = nullptr;
    char *output_filename = nullptr;
    void merge_left(int n, float *&left, float *&right, float *&buffer);
    void merge_right(int n, float *&left, float *&right, float *&buffer);
};

int main(int argc, char **argv) {
    Solver solver;
    return solver.solve(argc, argv);
}

int Solver::solve(int argc, char **argv) {
    std::ios::sync_with_stdio(0);
    std::cin.tie(0);
    std::cout.tie(0);
    if (argc != 4) {
        std::cerr << "must provide exactly 3 arguments!\n";
        return 1;
    }

    // Initialize mpi
    TIMING_START(solve_all);
    TIMING_START(mpi_init);
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    TIMING_END(mpi_init, world_rank);

    array_size = std::stoi(argv[1]);
    input_filename = argv[2];
    output_filename = argv[3];

    if (array_size <= MIN_SIZE_PER_PROC) {
        if (world_rank == 0) {
            TIMING_START(odd_even_sort_seq);
            odd_even_sort_seq();
            TIMING_END(odd_even_sort_seq, world_rank);
        }
    } else {
        TIMING_START(odd_even_sort_mpi);
        odd_even_sort_mpi();
        TIMING_END(odd_even_sort_mpi, world_rank);
    }

    // Finalize mpi
    // TIMING_START(mpi_finalize);
    // MPI_Finalize();
    // TIMING_END(mpi_finalize, world_rank);
    TIMING_END(solve_all, world_rank);
    return 0;
}

void Solver::odd_even_sort_seq() {
    MPI_File input_file, output_file;
    float *buffer = new float[array_size];

    TIMING_START(mpi_read);
    MPI_File_open(MPI_COMM_SELF, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, 0, buffer, array_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);
    TIMING_END(mpi_read, world_rank);

    boost::sort::spreadsort::spreadsort(buffer, buffer + array_size);

    TIMING_START(mpi_write);
    MPI_File_open(MPI_COMM_SELF, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, 0, buffer, array_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);
    TIMING_END(mpi_write, world_rank);
    delete[] buffer;
}

void Solver::odd_even_sort_mpi() {
    MPI_File input_file, output_file;
    int local_size, local_start, local_end;
    int actual_local_size, actual_world_size;
    float *buffer, *local_data, *neighbor_data, *merge_buffer;
    int left_rank, right_rank;

    local_size = std::min(array_size, std::max((int)ceil((double)array_size / world_size), MIN_SIZE_PER_PROC));
    local_start = std::min(array_size, world_rank * local_size);
    local_end = std::min(array_size, local_start + local_size);
    actual_local_size = local_end - local_start;
    actual_world_size = std::min(world_size, (int)ceil((double)array_size / local_size));

    buffer = new float[local_size * 3];
    local_data = buffer;
    neighbor_data = buffer + local_size;
    merge_buffer = buffer + local_size * 2;
    if (world_rank == 0) {
        DEBUG_MSG("array_size: " << array_size);
        DEBUG_MSG("local_size: " << local_size);
        DEBUG_MSG("local_start: " << local_start);
        DEBUG_MSG("local_end: " << local_end);
        DEBUG_MSG("actual_local_size: " << actual_local_size);
        DEBUG_MSG("actual_world_size: " << actual_world_size);
    }

    // Read file into buffer and fill the rest with max value
    TIMING_START(mpi_read);
    MPI_File_open(MPI_COMM_SELF, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    if (world_rank < actual_world_size) {
        MPI_File_read_at(input_file, sizeof(float) * local_start, local_data, actual_local_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&input_file);
    for (int i = actual_local_size; i < local_size; i++) {
        local_data[i] = std::numeric_limits<float>::max();
    }
    TIMING_END(mpi_read, world_rank);

    // Calculate neighbor rank
    right_rank = world_rank + 1;
    left_rank = world_rank - 1;
    if (right_rank < 0 || right_rank >= actual_world_size || world_rank >= actual_world_size) {
        right_rank = MPI_PROC_NULL;
    }
    if (left_rank < 0 || left_rank >= actual_world_size || world_rank >= actual_world_size) {
        left_rank = MPI_PROC_NULL;
    }

    // === odd-even sort start ===
    // Sort local
    TIMING_START(local_sort);
    boost::sort::spreadsort::spreadsort(local_data, local_data + local_size);
    TIMING_END(local_sort, world_rank);
    TIMING_INIT(mpi_exchange_1);
    TIMING_INIT(mpi_exchange_2);
    TIMING_INIT(local_merge);
    // Initialize neighbor buffer
    for (int p = 0; p < actual_world_size + 1; p++) {
        // phase[even,odd] rank[even,odd] way[right,left]
        // 0 0 0
        // 0 1 1
        // 1 0 1
        // 1 1 0 -> ~(phase ^ rank) = way
        if ((p ^ world_rank) & 1) {
            // Communicate with left
            if (left_rank == MPI_PROC_NULL)
                continue;
            // Pre-check if the two ranks are well-sorted
            TIMING_START(mpi_exchange_1);
            MPI_Sendrecv(local_data, 1, MPI_FLOAT, left_rank, 0, neighbor_data + local_size - 1, 1, MPI_FLOAT, left_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            TIMING_ACCUM(mpi_exchange_1);
            if (*(neighbor_data + local_size - 1) <= *local_data) {
                // Skip since sorted
                continue;
            }
            // Exchange data
            if (local_size > 1) {
                TIMING_START(mpi_exchange_2);
                MPI_Sendrecv(local_data + 1, local_size - 1, MPI_FLOAT, left_rank, 0, neighbor_data, local_size - 1, MPI_FLOAT, left_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                TIMING_ACCUM(mpi_exchange_2);
            }
            // Merge
            TIMING_START(local_merge);
            merge_right(local_size, neighbor_data, local_data, merge_buffer);
            TIMING_ACCUM(local_merge);
        } else {
            // Communicate with right
            if (right_rank == MPI_PROC_NULL)
                continue;
            // Pre-check if the two ranks are well-sorted
            TIMING_START(mpi_exchange_1);
            MPI_Sendrecv(local_data + local_size - 1, 1, MPI_FLOAT, right_rank, 0, neighbor_data, 1, MPI_FLOAT, right_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            TIMING_ACCUM(mpi_exchange_1);
            if (*(local_data + local_size - 1) <= *neighbor_data) {
                // Skip since sorted
                continue;
            }
            // Exchange data
            if (local_size > 1) {
                TIMING_START(mpi_exchange_2);
                MPI_Sendrecv(local_data, local_size - 1, MPI_FLOAT, right_rank, 0, neighbor_data + 1, local_size - 1, MPI_FLOAT, right_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                TIMING_ACCUM(mpi_exchange_2);
            }
            // Merge
            TIMING_START(local_merge);
            merge_left(local_size, local_data, neighbor_data, merge_buffer);
            TIMING_ACCUM(local_merge);
        }
    }
    TIMING_FIN(mpi_exchange_1, world_rank);
    TIMING_FIN(mpi_exchange_2, world_rank);
    TIMING_FIN(local_merge, world_rank);
    // === odd-even sort end ===

    TIMING_START(mpi_write);
    MPI_File_open(MPI_COMM_SELF, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    if (world_rank < actual_world_size) {
        MPI_File_write_at(output_file, sizeof(float) * local_start, local_data, actual_local_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&output_file);
    TIMING_END(mpi_write, world_rank);
    delete[] buffer;
}

void Solver::merge_left(int n, float *&left, float *&right, float *&buffer) {
    int l = 0;
    int r = 0;
#pragma GCC unroll 8
    for (int i = 0; i < n; i++) {
        if (left[l] < right[r]) {
            buffer[i] = left[l];
            l++;
        } else {
            buffer[i] = right[r];
            r++;
        }
    }
    std::swap(left, buffer);
}

void Solver::merge_right(int n, float *&left, float *&right, float *&buffer) {
    int l = n - 1;
    int r = n - 1;
#pragma GCC unroll 8
    for (int i = n - 1; i >= 0; i--) {
        if (left[l] > right[r]) {
            buffer[i] = left[l];
            l--;
        } else {
            buffer[i] = right[r];
            r--;
        }
    }
    std::swap(right, buffer);
}