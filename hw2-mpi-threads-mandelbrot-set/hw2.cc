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

#ifdef TIMING
#include <ctime>
double get_timestamp() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1000000000.0;
}
#define TIMING_START(arg) \
    double __start_##arg = get_timestamp();
#define TIMING_END(arg)                                              \
    {                                                                \
        double __end_##arg = get_timestamp();                        \
        double __duration_##arg = __end_##arg - __start_##arg;       \
        std::cerr << #arg << " took " << __duration_##arg << "s.\n"; \
        std::cerr.flush();                                           \
    }
#define TIMING_END_1(arg, i)                                                     \
    {                                                                            \
        double __end_##arg = get_timestamp();                                    \
        double __duration_##arg = __end_##arg - __start_##arg;                   \
        std::cerr << #arg << " " << i << " took " << __duration_##arg << "s.\n"; \
        std::cerr.flush();                                                       \
    }
#define TIMING_END_2(arg, i, j)                                                              \
    {                                                                                        \
        double __end_##arg = get_timestamp();                                                \
        double __duration_##arg = __end_##arg - __start_##arg;                               \
        std::cerr << #arg << " " << i << " " << j << " took " << __duration_##arg << "s.\n"; \
        std::cerr.flush();                                                                   \
    }
#define TIMING_INIT(arg) double __duration_##arg = 0;
#define TIMING_ACCUM(arg)                                \
    {                                                    \
        double __end_##arg = get_timestamp();            \
        __duration_##arg += __end_##arg - __start_##arg; \
    }
#define TIMING_FIN(arg)                                          \
    std::cerr << #arg << " took " << __duration_##arg << "s.\n"; \
    std::cerr.flush();
#else
#define TIMING_START(arg) \
    {}
#define TIMING_END(arg) \
    {}
#define TIMING_END_1(arg, i) \
    {}
#define TIMING_END_2(arg, i, j) \
    {}
#define TIMING_INIT(arg) \
    {}
#define TIMING_ACCUM(arg) \
    {}
#define TIMING_FIN(arg) \
    {}
#endif  // TIMING

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
        int start_pixel;
        int end_pixel;
        int shared_pixel;
        int* buffer;
        int buffer_offset;
#if MULTITHREADED == 1
        pthread_mutex_t mutex;
#endif
    };
    struct ThreadData {
        SharedData* shared;
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

    void mandelbrot();
#ifdef MPI_ENABLED
    void atomic_next_task(MasterSharedData* msd, MPITask& task);
    static void* pthreads_master(void* arg);
    static void* pthreads_worker_0(void* arg);
    static void* pthreads_worker(void* arg);
    void mandelbrot_mpi();
#endif
    void partial_mandelbrot(int start_pixel, int end_pixel, int* buffer, int buffer_offset);
#if MULTITHREADED == 1
    static void* pthreads_partial_mandelbrot_thread(void* arg);
#endif
    void partial_mandelbrot_thread(ThreadData* thread_data);
    void partial_mandelbrot_single_thread(int start_pixel, int end_pixel, int* buffer, int buffer_offset);
    void write_png(const int* buffer) const;
};

int main(int argc, char** argv) {
    Solver solver;
    return solver.solve(argc, argv);
}

int Solver::solve(int argc, char** argv) {
    TIMING_START(all);
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

    TIMING_START(mandelbrot);
#ifdef MPI_ENABLED
    if (world_size == 1 || (long long)iters * width * height <= min_tasks_per_process) {
        mandelbrot();
    } else {
        mandelbrot_mpi();
    }
#else
    mandelbrot();
#endif
    TIMING_END_1(mandelbrot, world_rank);

#ifdef MPI_ENABLED
    // TIMING_START(MPI_Finalize);
    // MPI_Finalize();
    // TIMING_END_1(MPI_Finalize, world_rank);
#endif
    TIMING_END_1(all, world_rank);
    return 0;
}

void Solver::mandelbrot() {
    // allocate memory for image
    int* buffer = (int*)malloc(width * height * sizeof(int));

    // compute partial mandelbrot set
    partial_mandelbrot(0, width * height, buffer, 0);

    // draw and cleanup
    if (world_rank == 0) {
        TIMING_START(write_png);
        write_png(buffer);
        TIMING_END_1(write_png, world_rank);
    }
    free(buffer);
}

#ifdef MPI_ENABLED
inline void Solver::atomic_next_task(MasterSharedData* msd, MPITask& task) {
    Solver* solver = msd->solver;
    MPITask* shared_task = &msd->shared_task;
    pthread_mutex_t* mutex = &msd->mutex;

    pthread_mutex_lock(mutex);
    task.start_pixel = shared_task->start_pixel;
    task.end_pixel = shared_task->end_pixel;
    if (task.start_pixel < task.end_pixel) {
        shared_task->start_pixel = shared_task->end_pixel;
        if (msd->batch_size > solver->num_cpus * min_batch_size_per_thread && shared_task->end_pixel - shared_task->start_pixel >= 24 * solver->num_cpus * min_batch_size_per_thread) {
            msd->batch_size = solver->num_cpus * min_batch_size_per_thread;
        }
        shared_task->end_pixel = std::min(shared_task->end_pixel + msd->batch_size, solver->width * solver->height);
    }
    pthread_mutex_unlock(mutex);
}

void* Solver::pthreads_master(void* arg) {
    MasterThreadData* mtd = (MasterThreadData*)arg;
    MasterSharedData* msd = mtd->shared;
    int rank = mtd->rank;

    Solver* solver = msd->solver;
    int* buffer = msd->buffer;

    PartialBuffer* pb = (PartialBuffer*)malloc(sizeof(PartialBuffer));
    MPITask prev_task;
    prev_task.start_pixel = mtd->init_task.start_pixel;
    prev_task.end_pixel = mtd->init_task.end_pixel;

    TIMING_START(master);
    while (prev_task.start_pixel < prev_task.end_pixel) {
        // receive results from worker
        MPI_Recv(pb, sizeof(MPITask) + (prev_task.end_pixel - prev_task.start_pixel) * sizeof(int), MPI_BYTE, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // send new task to worker
        MPITask curr_task;
        solver->atomic_next_task(msd, curr_task);
        MPI_Send(&curr_task, sizeof(curr_task), MPI_BYTE, rank, 0, MPI_COMM_WORLD);
        // copy results to buffer
        memcpy(buffer + pb->task.start_pixel, pb->buffer, (pb->task.end_pixel - pb->task.start_pixel) * sizeof(int));
        prev_task = curr_task;
    }
    TIMING_END_2(master, solver->world_rank, rank);
    free(pb);
    return NULL;
}

void* Solver::pthreads_worker_0(void* arg) {
    MasterSharedData* msd = (MasterSharedData*)arg;
    Solver* solver = msd->solver;
    int* buffer = msd->buffer;

    TIMING_START(worker);
    while (true) {
        MPITask curr_task;
        solver->atomic_next_task(msd, curr_task);
        if (curr_task.start_pixel >= curr_task.end_pixel) {
            break;
        }
        // compute partial mandelbrot set
        solver->partial_mandelbrot(curr_task.start_pixel, curr_task.end_pixel, buffer, 0);
    }
    TIMING_END_1(worker, solver->world_rank);
    return NULL;
}

void* Solver::pthreads_worker(void* arg) {
    WorkerThreadData* wtd = (WorkerThreadData*)arg;
    WorkerSharedData* wsd = wtd->shared;
    Solver* solver = wsd->solver;
    PartialBuffer* pb = (PartialBuffer*)malloc(sizeof(PartialBuffer));
    pb->task.start_pixel = wtd->init_task.start_pixel;
    pb->task.end_pixel = wtd->init_task.end_pixel;

    TIMING_START(worker);
    while (pb->task.start_pixel < pb->task.end_pixel) {
        // compute partial mandelbrot set
        solver->partial_mandelbrot(pb->task.start_pixel, pb->task.end_pixel, pb->buffer, pb->task.start_pixel);
        // send results to master
        MPI_Send(pb, sizeof(MPITask) + (pb->task.end_pixel - pb->task.start_pixel) * sizeof(int), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
        // receive new task
        MPI_Recv(&pb->task, sizeof(pb->task), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    TIMING_END_1(worker, solver->world_rank);
    free(pb);
    return NULL;
}

void Solver::mandelbrot_mpi() {
    const int batch_size = std::max(std::min(max_buffer_size, (int)std::ceil((double)width * height / world_size)), 32 * num_cpus * min_batch_size_per_thread);
    const int num_procs = std::min(world_size, (int)std::ceil((double)width * height / batch_size));

    MPITask init_tasks[max_num_procs];
    for (int i = 0; i < num_procs - 1; i++) {
        init_tasks[i].start_pixel = batch_size * i;
        init_tasks[i].end_pixel = std::min(batch_size * (i + 1), width * height);
    }

    // allocate memory for image
    int* buffer = NULL;
    if (world_rank == 0) {
        buffer = (int*)malloc(width * height * sizeof(int));
    }

    // create threads
    if (world_rank == 0) {
        MasterSharedData msd;
        msd.solver = this;
        msd.num_procs = num_procs;
        msd.batch_size = batch_size;
        msd.buffer = buffer;
        msd.shared_task.start_pixel = batch_size * (num_procs - 1);
        msd.shared_task.end_pixel = std::min(batch_size * num_procs, width * height);
        msd.mutex = PTHREAD_MUTEX_INITIALIZER;
        MasterThreadData mtd_array[max_num_procs];
        pthread_t master_threads[max_num_procs];
        pthread_t worker_thread;
        for (int i = 0; i < num_procs - 1; i++) {
            mtd_array[i].shared = &msd;
            mtd_array[i].rank = i + 1;
            mtd_array[i].init_task = init_tasks[i];
            pthread_create(&master_threads[i], NULL, pthreads_master, (void*)&mtd_array[i]);
        }
        pthread_create(&worker_thread, NULL, pthreads_worker_0, (void*)&msd);
        for (int i = 0; i < num_procs - 1; i++) {
            pthread_join(master_threads[i], NULL);
        }
        pthread_join(worker_thread, NULL);
    } else if (world_rank < num_procs) {
        WorkerSharedData wsd;
        wsd.solver = this;
        WorkerThreadData wtd;
        pthread_t worker_thread;
        wtd.shared = &wsd;
        wtd.init_task = init_tasks[world_rank - 1];
        pthread_create(&worker_thread, NULL, pthreads_worker, (void*)&wtd);
        pthread_join(worker_thread, NULL);
    }

    // draw and cleanup
    if (world_rank == 0) {
        TIMING_START(write_png);
        write_png(buffer);
        TIMING_END_1(write_png, world_rank);
        free(buffer);
    }
}
#endif

void Solver::partial_mandelbrot(int start_pixel, int end_pixel, int* buffer, int buffer_offset) {
#if MULTITHREADED == 1 || MULTITHREADED == 2
    const int num_threads = num_cpus;
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
        shared_data.start_pixel = start_pixel;
        shared_data.end_pixel = end_pixel;
        shared_data.shared_pixel = start_pixel;
        shared_data.buffer = buffer;
        shared_data.buffer_offset = buffer_offset;
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
        partial_mandelbrot_single_thread(start_pixel, end_pixel, buffer, buffer_offset);
#endif
    } else {
        partial_mandelbrot_single_thread(start_pixel, end_pixel, buffer, buffer_offset);
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
    int* buffer = shared->buffer;
    int buffer_offset = shared->buffer_offset;
#if MULTITHREADED == 1
    pthread_mutex_t* mutex = &shared->mutex;
#endif

    TIMING_START(thread);
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
        solver->partial_mandelbrot_single_thread(curr_start_pixel, curr_end_pixel, buffer, buffer_offset);
    }
#if MULTITHREADED == 1
    TIMING_END_1(thread, pthread_self());
#elif MULTITHREADED == 2
    if (solver->world_size == 0) {
        TIMING_END_1(thread, omp_get_thread_num());
    }
#endif
}

void Solver::partial_mandelbrot_single_thread(int start_pixel, int end_pixel, int* buffer, int buffer_offset) {
    // mandelbrot set
    const double h_norm = (upper - lower) / height;
    const double w_norm = (right - left) / width;
    int p = start_pixel;
#if defined(__AVX512F__) && defined(SIMD_ENABLED)
    // Constants
    const int vec_8_size = 8;
    const __m256i vec_8_1_epi32 = _mm256_set1_epi32(1);
    const __m512d vec_8_2 = _mm512_set1_pd(2);
    const __m512d vec_8_4 = _mm512_set1_pd(4);
    const __m512d vec_8_offset = _mm512_set_pd(7, 6, 5, 4, 3, 2, 1, 0);
    const __m512d vec_8_width = _mm512_set1_pd(width);
    const __m512d vec_8_w_norm = _mm512_set1_pd(w_norm);
    const __m512d vec_8_h_norm = _mm512_set1_pd(h_norm);
    const __m512d vec_8_left = _mm512_set1_pd(left);
    const __m512d vec_8_lower = _mm512_set1_pd(lower);
    for (; p + vec_8_size - 1 < end_pixel; p += vec_8_size) {
        // Calculate pixel coordinates
        __m512d vec_p_offset = _mm512_add_pd(_mm512_set1_pd(p), vec_8_offset);
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
        _mm256_storeu_epi32(&buffer[p - buffer_offset], vec_repeats);
    }
#endif
    for (; p < end_pixel; ++p) {
        int j = p / width;
        int i = p % width;
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
        buffer[p - buffer_offset] = repeats;
    }
}

void Solver::write_png(const int* buffer) const {
    TIMING_START(write_png_setup);
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
    TIMING_END(write_png_setup);
    TIMING_START(write_png_loop);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
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
        png_write_row(png_ptr, row);
    }
    TIMING_END(write_png_loop);
    TIMING_START(write_png_cleanup);
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
    TIMING_END(write_png_cleanup);
}