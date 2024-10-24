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
#endif  // TIMING

class Solver {
   public:
    Solver() {}
    ~Solver() {};
    int solve(int argc, char **argv);
    void odd_even_sort_seq();
    void odd_even_sort_mpi();

   private:
    // Optimization: Give a minimum size per process to reduce communication overhead
    const int MIN_SIZE_PER_PROC = 100000;
    int world_rank = 0;
    int world_size = 1;
    int array_size = 0;
    char *input_filename = nullptr;
    char *output_filename = nullptr;
    // Optimization: Merge only to the left or right to reduce number of elements to be copied
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
    NVTX_RANGE_START(mpi_init)
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    NVTX_RANGE_END()

    array_size = std::stoi(argv[1]);
    input_filename = argv[2];
    output_filename = argv[3];

    if (array_size <= MIN_SIZE_PER_PROC) {
        if (world_rank == 0) {
            NVTX_RANGE_START(odd_even_sort_seq)
            odd_even_sort_seq();
            NVTX_RANGE_END()
        }
    } else {
        NVTX_RANGE_START(odd_even_sort_mpi)
        odd_even_sort_mpi();
        NVTX_RANGE_END()
    }

    // Optimization: Return without finalizing mpi
#ifndef NO_FINALIZE
    // Finalize mpi
    NVTX_RANGE_START(mpi_finalize)
    MPI_Finalize();
    NVTX_RANGE_END()
#endif
    return 0;
}

void Solver::odd_even_sort_seq() {
    MPI_File input_file, output_file;
    float *buffer = new float[array_size];

    NVTX_RANGE_START(mpi_read)
    MPI_File_open(MPI_COMM_SELF, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, 0, buffer, array_size, MPI_FLOAT, MPI_STATUS_IGNORE);
#ifndef NO_FINALIZE
    MPI_File_close(&input_file);
#endif
    NVTX_RANGE_END()

    NVTX_RANGE_START(local_sort)
    boost::sort::spreadsort::float_sort(buffer, buffer + array_size);
    NVTX_RANGE_END()

    NVTX_RANGE_START(mpi_write)
    MPI_File_open(MPI_COMM_SELF, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, 0, buffer, array_size, MPI_FLOAT, MPI_STATUS_IGNORE);
#ifndef NO_FINALIZE
    MPI_File_close(&output_file);
#endif
    NVTX_RANGE_END()
#ifndef NO_FINALIZE
    delete[] buffer;
#endif
}

void Solver::odd_even_sort_mpi() {
    MPI_File input_file, output_file;
    int max_local_size, local_start, local_end, local_size, actual_world_size;
    // Optimization: Use one contiguous buffer
    // Optimization: Use pointers to represent each part of the buffer to enable swapping without copying
    float *buffer, *local_data, *neighbor_data, *merge_buffer;
    int left_rank, right_rank;

    max_local_size = std::min(array_size, std::max((int)ceil((double)array_size / world_size), MIN_SIZE_PER_PROC));
    actual_world_size = std::min(world_size, (int)ceil((double)array_size / max_local_size));
    local_start = std::min(array_size, world_rank * max_local_size);
    local_end = std::min(array_size, local_start + max_local_size);
    local_size = local_end - local_start;

    buffer = new float[max_local_size * 3];
    local_data = buffer;
    neighbor_data = buffer + max_local_size;
    merge_buffer = buffer + max_local_size * 2;
    if (world_rank == 0) {
        DEBUG_MSG("array_size: " << array_size);
        DEBUG_MSG("max_local_size: " << max_local_size);
        DEBUG_MSG("local_start: " << local_start);
        DEBUG_MSG("local_end: " << local_end);
        DEBUG_MSG("local_size: " << local_size);
        DEBUG_MSG("actual_world_size: " << actual_world_size);
    }

    // Calculate neighbor rank
    right_rank = world_rank + 1;
    left_rank = world_rank - 1;
    if (right_rank < 0 || right_rank >= actual_world_size || world_rank >= actual_world_size) {
        right_rank = MPI_PROC_NULL;
    }
    if (left_rank < 0 || left_rank >= actual_world_size || world_rank >= actual_world_size) {
        left_rank = MPI_PROC_NULL;
    }

    // Optimization: Use MPI_COMM_SELF to read and write files to avoid synchronization overhead
    // Read file into buffer and fill the rest with max value
    NVTX_RANGE_START(mpi_read)
    MPI_File_open(MPI_COMM_SELF, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    if (world_rank < actual_world_size) {
        MPI_File_read_at(input_file, sizeof(float) * local_start, local_data, local_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
#ifndef NO_FINALIZE
    MPI_File_close(&input_file);
#endif
    for (int i = local_size; i < max_local_size; i++) {
        local_data[i] = std::numeric_limits<float>::max();
    }
    NVTX_RANGE_END()

    // === odd-even sort start ===
    // Optimization: Use spreadsort to sort local data for better performance
    // Sort local
    NVTX_RANGE_START(local_sort)
    boost::sort::spreadsort::float_sort(local_data, local_data + local_size);
    NVTX_RANGE_END()
    // Initialize neighbor buffer
    for (int p = 0; p < actual_world_size; p++) {
        // Optimization: Compute to communicate with left or right rank instead of casing under odd or even phase
        // phase[even,odd] rank[even,odd] way[right,left]
        // 0 0 0
        // 0 1 1
        // 1 0 1
        // 1 1 0 -> ~(phase ^ rank) = way
        // Optimization: Skip if the two ranks are well-sorted before exchanging all data
        if ((p ^ world_rank) & 1) {
            // Communicate with left
            if (left_rank == MPI_PROC_NULL)
                continue;
            // Pre-check if the two ranks are well-sorted
            NVTX_RANGE_START(mpi_pre_exchange_right)
            MPI_Sendrecv(local_data, 1, MPI_FLOAT, left_rank, 0, neighbor_data + max_local_size - 1, 1, MPI_FLOAT, left_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            NVTX_RANGE_END()
            if (*(neighbor_data + max_local_size - 1) <= *local_data) {
                // Skip since sorted
                continue;
            }
            // Exchange data
            if (max_local_size > 1) {
                NVTX_RANGE_START(mpi_exchange_right)
                MPI_Sendrecv(local_data + 1, max_local_size - 1, MPI_FLOAT, left_rank, 0, neighbor_data, max_local_size - 1, MPI_FLOAT, left_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                NVTX_RANGE_END()
            }
            // Merge
            NVTX_RANGE_START(merge_right)
            merge_right(max_local_size, neighbor_data, local_data, merge_buffer);
            NVTX_RANGE_END()
        } else {
            // Communicate with right
            if (right_rank == MPI_PROC_NULL)
                continue;
            // Pre-check if the two ranks are well-sorted
            NVTX_RANGE_START(mpi_pre_exchange_left)
            MPI_Sendrecv(local_data + max_local_size - 1, 1, MPI_FLOAT, right_rank, 0, neighbor_data, 1, MPI_FLOAT, right_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            NVTX_RANGE_END()
            if (*(local_data + max_local_size - 1) <= *neighbor_data) {
                // Skip since sorted
                continue;
            }
            // Exchange data
            if (max_local_size > 1) {
                NVTX_RANGE_START(mpi_exchange_left)
                MPI_Sendrecv(local_data, max_local_size - 1, MPI_FLOAT, right_rank, 0, neighbor_data + 1, max_local_size - 1, MPI_FLOAT, right_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                NVTX_RANGE_END()
            }
            // Merge
            NVTX_RANGE_START(merge_left)
            merge_left(max_local_size, local_data, neighbor_data, merge_buffer);
            NVTX_RANGE_END()
        }
    }
    // === odd-even sort end ===

    NVTX_RANGE_START(mpi_write)
    MPI_File_open(MPI_COMM_SELF, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    if (world_rank < actual_world_size) {
        MPI_File_write_at(output_file, sizeof(float) * local_start, local_data, local_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
#ifndef NO_FINALIZE
    MPI_File_close(&output_file);
#endif
    NVTX_RANGE_END()
#ifndef NO_FINALIZE
    delete[] buffer;
#endif
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