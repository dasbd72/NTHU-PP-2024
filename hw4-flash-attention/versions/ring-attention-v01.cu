#include <assert.h>
#include <cuda.h>
#include <float.h>
#include <omp.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <numeric>

#ifdef PROFILING
#include <nvtx3/nvtx3.hpp>
#define NVTX_RANGE_START(arg) \
    nvtxRangePushA(#arg);
#define NVTX_RANGE_END() \
    nvtxRangePop();
#define NVTX_RANGE_FUNC() \
    NVTX3_FUNC_RANGE()
#else
#define NVTX_RANGE_START(arg) \
    {}
#define NVTX_RANGE_END() \
    {}
#define NVTX_RANGE_FUNC() \
    {}
#endif  // PROFILING

#define CUDA_CHECK(call)                                                                               \
    do {                                                                                               \
        cudaError_t status = call;                                                                     \
        if (status != cudaSuccess) {                                                                   \
            fprintf(stderr, "CUDA Error: %s %d %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
            exit(1);                                                                                   \
        }                                                                                              \
    } while (0)

constexpr int bc = 32;
constexpr int br = 32;
constexpr int D = 64;

struct RingAttnDeviceParams {
    float *dev_Q, *dev_O;
    float *dev_K[2], *dev_V[2];  // Swapping buffer for odd and even ring phases
    float *dev_l, *dev_m;        // l and m are used to store the sum of pij and max of sij
    cudaStream_t *streams;       // Streams for each batch
    cudaEvent_t *copy_events;    // Events copying from previous device
};

int ncpus, ngpus;
int B, N, d;
float *Q, *K, *V, *O;

int setup(int argc, char *argv[]);
void input(char *input_filename);
void output(char *output_filename);
double get_time();

template <typename T>
void pad_buffer(T *&buf, int B, int N, int d, int B_pad, int N_pad, int d_pad);
template <typename T>
void unpad_buffer(T *&buf, int B, int N, int d, int B_pad, int N_pad, int d_pad);
template <typename T>
void convert_buffer(T *&buf, int B, int N, int d, int ngpus);
template <typename T>
void revert_buffer(T *&buf, int B, int N, int d, int ngpus);

__device__ void QKDotAndScalar(float *out, float *q, float *k, float scalar, int d);
__device__ void RowMax(float *out, float *in, int n);
__device__ void MinusMaxAndExp(float *out, float *in, float *mx);
__device__ void RowSum(float *out, float *in, int n);
__device__ void UpdateMiLiOi(float *mi, float *li, float *oi, float *mij, float *lij, float *pij, float *vj, int d, int n);
__global__ void block_flash_attention(float *q, float *k, float *v, float *o, float *l, float *m, int N_kv, int d, float scalar);
__global__ void init_float_array(float *arr, int size, float val);

int main(int argc, char *argv[]) {
    NVTX_RANGE_FUNC();

    if (setup(argc, argv)) {
        return 1;
    }

    // input
    input(argv[1]);

    // time recoding
    double start_time = get_time();

    // convert from (B, N, d) to (B, N_pad, d)
    int lcm_bc_br = std::lcm(bc, br);
    int N_pad = (int)ceil((float)N / (ngpus * lcm_bc_br)) * (ngpus * lcm_bc_br);
    pad_buffer(Q, B, N, d, B, N_pad, d);
    pad_buffer(K, B, N, d, B, N_pad, d);
    pad_buffer(V, B, N, d, B, N_pad, d);
    pad_buffer(O, B, N, d, B, N_pad, d);

    // convert from (B, N_pad, d) to (ngpus, B, N_pad / ngpus, d)
    convert_buffer(Q, B, N_pad, d, ngpus);
    convert_buffer(K, B, N_pad, d, ngpus);
    convert_buffer(V, B, N_pad, d, ngpus);

    // parameters for each device
    RingAttnDeviceParams params[ngpus];

#pragma omp parallel num_threads(ngpus)
    {
        int device_id = omp_get_thread_num();
        cudaSetDevice(device_id);  // Must be called before any runtime API calls

        size_t start_free_mem, end_free_mem, total_mem;
        cudaMemGetInfo(&start_free_mem, &total_mem);
        printf("Device %d: initial %f MB free of %f MB\n", device_id, (float)start_free_mem / 1024.0 / 1024.0, (float)total_mem / 1024.0 / 1024.0);

        int from_device_id = (device_id + ngpus - 1) % ngpus;
        int to_device_id = (device_id + 1) % ngpus;

        // Enable peer access
        int can_access_peer = 0;
        cudaDeviceCanAccessPeer(&can_access_peer, device_id, to_device_id);
        if (can_access_peer) {
            cudaDeviceEnablePeerAccess(to_device_id, 0);
        }

        int N_part = (int)ceil((float)N_pad / ngpus);  // For each device
        int N_part_last = N - N_part * (ngpus - 1);    // Different for last device
        int dev_offset = B * N_part * d * device_id;

        dim3 blk(bc, br);
        dim3 grid(1, (int)ceil((float)N_part / br));
        float scalar = 1.0 / sqrt(d);

        // Initialize device parameters
        params[device_id].streams = (cudaStream_t *)malloc(B * sizeof(cudaStream_t));
        params[device_id].copy_events = (cudaEvent_t *)malloc(ngpus * B * sizeof(cudaEvent_t));
        for (int i = 0; i < B; i++) {
            CUDA_CHECK(cudaStreamCreate(&params[device_id].streams[i]));
        }
        for (int i = 0; i < ngpus * B; i++) {
            CUDA_CHECK(cudaEventCreateWithFlags(&params[device_id].copy_events[i], cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaHostRegister(Q + dev_offset, B * N_part * d * sizeof(float), cudaHostRegisterDefault));
        CUDA_CHECK(cudaHostRegister(K + dev_offset, B * N_part * d * sizeof(float), cudaHostRegisterDefault));
        CUDA_CHECK(cudaHostRegister(V + dev_offset, B * N_part * d * sizeof(float), cudaHostRegisterDefault));
        CUDA_CHECK(cudaHostRegister(O + dev_offset, B * N_part * d * sizeof(float), cudaHostRegisterDefault));

        CUDA_CHECK(cudaMalloc(&params[device_id].dev_Q, B * N_part * d * sizeof(float)));
        for (int i = 0; i < 2; i++) {
            CUDA_CHECK(cudaMalloc(&params[device_id].dev_K[i], B * N_part * d * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&params[device_id].dev_V[i], B * N_part * d * sizeof(float)));
        }
        CUDA_CHECK(cudaMalloc(&params[device_id].dev_O, B * N_part * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&params[device_id].dev_l, B * N_part * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&params[device_id].dev_m, B * N_part * sizeof(float)));

        CUDA_CHECK(cudaMemset(params[device_id].dev_O, 0, B * N_part * d * sizeof(float)));
        CUDA_CHECK(cudaMemset(params[device_id].dev_l, 0, B * N_part * sizeof(float)));
        init_float_array<<<int(float(B * N_part) / (br * bc)), br * bc>>>(params[device_id].dev_m, B * N_part, -FLT_MAX);

        for (int step = 0; step < ngpus; step++) {
            int buffer_id = step & 1;
            int N_kv = (ngpus + step - device_id - 1) % ngpus == 0 ? N_part_last : N_part;
            if (step == 0) {
                // Asynchronous copy from host to device
                for (int i = 0; i < B; i++) {
                    int batch_offset = i * N_part * d;
                    CUDA_CHECK(cudaMemcpyAsync(params[device_id].dev_Q + batch_offset, Q + dev_offset + batch_offset, N_part * d * sizeof(float), cudaMemcpyHostToDevice, params[device_id].streams[i]));
                    CUDA_CHECK(cudaMemcpyAsync(params[device_id].dev_K[buffer_id] + batch_offset, K + dev_offset + batch_offset, N_part * d * sizeof(float), cudaMemcpyHostToDevice, params[device_id].streams[i]));
                    CUDA_CHECK(cudaMemcpyAsync(params[device_id].dev_V[buffer_id] + batch_offset, V + dev_offset + batch_offset, N_part * d * sizeof(float), cudaMemcpyHostToDevice, params[device_id].streams[i]));
                    // Record the copy event
                    cudaEventRecord(params[device_id].copy_events[step * B + i], params[device_id].streams[i]);
                }
            } else {
                // The most important barrier
                // Since we are now defining the "streams", this barrier does not blocks the kernel execution
                // Instead, it blocks to make the event and stream having the correct order
#pragma omp barrier
                // Pull from previous device
                for (int i = 0; i < B; i++) {
                    int batch_offset = i * N_part * d;
                    // Waits for the previous device to finish the kernel on the previous step
                    cudaStreamWaitEvent(params[device_id].streams[i], params[from_device_id].copy_events[(step - 1) * B + i], 0);
                    if (step > 1) {
                        // Waits for the next device to finish copying on the previous step
                        cudaStreamWaitEvent(params[device_id].streams[i], params[to_device_id].copy_events[(step - 1) * B + i], 0);
                    }
                    CUDA_CHECK(cudaMemcpyAsync(params[device_id].dev_K[buffer_id] + batch_offset, params[from_device_id].dev_K[buffer_id ^ 1] + batch_offset, N_part * d * sizeof(float), cudaMemcpyDeviceToDevice, params[device_id].streams[i]));
                    CUDA_CHECK(cudaMemcpyAsync(params[device_id].dev_V[buffer_id] + batch_offset, params[from_device_id].dev_V[buffer_id ^ 1] + batch_offset, N_part * d * sizeof(float), cudaMemcpyDeviceToDevice, params[device_id].streams[i]));
                    // Record the copy event
                    cudaEventRecord(params[device_id].copy_events[step * B + i], params[device_id].streams[i]);
                }
            }

            for (int i = 0; i < B; i++) {
                int batch_offset = i * N_part * d;
                block_flash_attention<<<grid, blk, 0, params[device_id].streams[i]>>>(
                    params[device_id].dev_Q + batch_offset,
                    params[device_id].dev_K[buffer_id] + batch_offset,
                    params[device_id].dev_V[buffer_id] + batch_offset,
                    params[device_id].dev_O + batch_offset,
                    params[device_id].dev_l + i * N_part,
                    params[device_id].dev_m + i * N_part,
                    N_kv, d, scalar);
            }
        }

        // Asynchronous copy from device to host
        for (int i = 0; i < B; i++) {
            int batch_offset = i * N_part * d;
            CUDA_CHECK(cudaMemcpyAsync(O + dev_offset + batch_offset, params[device_id].dev_O + batch_offset, N_part * d * sizeof(float), cudaMemcpyDeviceToHost, params[device_id].streams[i]));
        }

        // ring-attention completed
        // finalize
        for (int i = 0; i < B; i++) {
            CUDA_CHECK(cudaStreamSynchronize(params[device_id].streams[i]));
        }
#pragma omp barrier

        for (int i = 0; i < B; i++) {
            CUDA_CHECK(cudaStreamDestroy(params[device_id].streams[i]));
        }
        for (int i = 0; i < ngpus * B; i++) {
            CUDA_CHECK(cudaEventDestroy(params[device_id].copy_events[i]));
        }

        cudaMemGetInfo(&end_free_mem, &total_mem);
        printf("Device %d: after %f MB free of %f MB\n", device_id, (float)end_free_mem / 1024.0 / 1024.0, (float)total_mem / 1024.0 / 1024.0);
        printf("Device %d: peak %f MB used of %f MB\n", device_id, (float)(start_free_mem - end_free_mem) / 1024.0 / 1024.0, (float)total_mem / 1024.0 / 1024.0);

        cudaHostUnregister(Q + dev_offset);
        cudaHostUnregister(K + dev_offset);
        cudaHostUnregister(V + dev_offset);
        cudaHostUnregister(O + dev_offset);

        free(params[device_id].streams);
        free(params[device_id].copy_events);

        cudaFree(params[device_id].dev_Q);
        for (int i = 0; i < 2; i++) {
            cudaFree(params[device_id].dev_K[i]);
            cudaFree(params[device_id].dev_V[i]);
        }
        cudaFree(params[device_id].dev_O);
        cudaFree(params[device_id].dev_l);
        cudaFree(params[device_id].dev_m);
    }

    // convert from (ngpus, B, N_pad / ngpus, d) back to (B, N_pad, d)
    revert_buffer(O, B, N_pad, d, ngpus);
    unpad_buffer(O, B, N, d, B, N_pad, d);

    // time recording
    double end_time = get_time();
    printf("Time: %fs\n", end_time - start_time);

    // output
    output(argv[2]);

    return 0;
}

int setup(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }

    cpu_set_t cpuset;
    sched_getaffinity(0, sizeof(cpu_set_t), &cpuset);
    ncpus = CPU_COUNT(&cpuset);

    cudaGetDeviceCount(&ngpus);

    printf("ncpus = %d\n", ncpus);
    printf("ngpus = %d\n", ngpus);

    return 0;
}

void input(char *input_filename) {
    FILE *file = fopen(input_filename, "rb");

    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);

    printf("B = %d, N = %d, d = %d\n", B, N, d);

    Q = (float *)malloc(B * N * d * sizeof(float));
    K = (float *)malloc(B * N * d * sizeof(float));
    V = (float *)malloc(B * N * d * sizeof(float));
    O = (float *)malloc(B * N * d * sizeof(float));

    for (int i = 0; i < B; ++i) {
        fread(Q + i * N * d, sizeof(float), N * d, file);
        fread(K + i * N * d, sizeof(float), N * d, file);
        fread(V + i * N * d, sizeof(float), N * d, file);
    }

    memset(O, 0x00, B * N * d * sizeof(float));

    fclose(file);
}

void output(char *output_filename) {
    FILE *file = fopen(output_filename, "wb");

    fwrite(O, sizeof(float), B * N * d, file);

    free(Q);
    free(K);
    free(V);
    free(O);

    fclose(file);
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/**
 * Pad buffer from (B, N, d) to (B_pad, N_pad, d_pad)
 **/
template <typename T>
void pad_buffer(T *&buf, int B, int N, int d, int B_pad, int N_pad, int d_pad) {
    assert(B_pad >= B && N_pad >= N && d_pad >= d);
    T *buffer_pad = (T *)malloc(B_pad * N_pad * d_pad * sizeof(T));
    for (int i = 0; i < B_pad; i++) {
        for (int j = 0; j < N_pad; j++) {
            for (int k = 0; k < d_pad; k++) {
                if (i < B && j < N && k < d) {
                    buffer_pad[i * N_pad * d_pad + j * d_pad + k] = buf[i * N * d + j * d + k];
                } else {
                    buffer_pad[i * N_pad * d_pad + j * d_pad + k] = 0;
                }
            }
        }
    }
    free(buf);
    buf = buffer_pad;
}

/**
 * Unpad buffer from (B_pad, N_pad, d_pad) to (B, N, d)
 **/
template <typename T>
void unpad_buffer(T *&buf, int B, int N, int d, int B_pad, int N_pad, int d_pad) {
    assert(B_pad >= B && N_pad >= N && d_pad >= d);
    T *buffer_unpad = (T *)malloc(B * N * d * sizeof(T));
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < d; k++) {
                buffer_unpad[i * N * d + j * d + k] = buf[i * N_pad * d_pad + j * d_pad + k];
            }
        }
    }
    free(buf);
    buf = buffer_unpad;
}

/**
 * Convert buffer from (B, N, d) to (ngpus, B, N / ngpus, d)
 *
 * Asserts that N is a multiple of ngpus.
 **/
template <typename T>
void convert_buffer(T *&buf, int B, int N, int d, int ngpus) {
    assert(N % ngpus == 0);
    T *buffer_reorder = (T *)malloc(B * N * d * sizeof(T));
    int N_part = N / ngpus;
    for (int i = 0; i < ngpus; i++) {
        for (int j = 0; j < B; j++) {
            for (int k = 0; k < N_part; k++) {
                for (int l = 0; l < d; l++) {
                    buffer_reorder[i * B * N_part * d + j * N_part * d + k * d + l] = buf[j * N * d + i * N_part * d + k * d + l];
                }
            }
        }
    }
    free(buf);
    buf = buffer_reorder;
}

/**
 * Convert buffer from (ngpus, B, N / ngpus, d) to (B, N, d)
 *
 * Asserts that N is a multiple of ngpus.
 **/
template <typename T>
void revert_buffer(T *&buf, int B, int N, int d, int ngpus) {
    assert(N % ngpus == 0);
    T *buffer_revert = (T *)malloc(B * N * d * sizeof(T));
    int N_part = N / ngpus;
    for (int i = 0; i < ngpus; i++) {
        for (int j = 0; j < B; j++) {
            for (int k = 0; k < N_part; k++) {
                for (int l = 0; l < d; l++) {
                    buffer_revert[j * N * d + i * N_part * d + k * d + l] = buf[i * B * N_part * d + j * N_part * d + k * d + l];
                }
            }
        }
    }
    free(buf);
    buf = buffer_revert;
}

__device__ void QKDotAndScalar(float *out, float *q, float *k, float scalar, int d) {
    // use br*bc threads.
    int i = threadIdx.y, j = threadIdx.x;
    float reg_out = 0.0F;
#pragma unroll
    for (int t = 0; t < d; t++) {
        reg_out += q[i * d + t] * k[j * d + t];
    }
    out[i * bc + j] = reg_out * scalar;
}

__device__ void RowMax(float *out, float *in, int n) {
    int i = threadIdx.y;
    if (threadIdx.x == 0) {
        out[i] = in[i * bc];
#pragma unroll
        for (int j = 1; j < n; j++) {
            out[i] = fmaxf(out[i], in[i * bc + j]);
        }
    }
}

__device__ void MinusMaxAndExp(float *out, float *in, float *mx) {
    int i = threadIdx.y, j = threadIdx.x;
    out[i * bc + j] = expf(in[i * bc + j] - mx[i]);
}

__device__ void RowSum(float *out, float *in, int n) {
    int i = threadIdx.y;
    if (threadIdx.x == 0) {
        out[i] = in[i * bc];
#pragma unroll
        for (int j = 1; j < n; j++) {
            out[i] += in[i * bc + j];
        }
    }
}

__device__ void UpdateMiLiOi(float *mi, float *li, float *oi, float *mij, float *lij, float *pij, float *vj, int d, int n) {
    int i = threadIdx.y;

    float mi_val = mi[i];
    float mij_val = mij[i];
    float li_val = li[i];

    // Compute mi_new_i and li_new_i
    float mi_new_i = fmaxf(mi_val, mij_val);
    float exp_mi_diff = expf(mi_val - mi_new_i);
    float exp_mij_diff = expf(mij_val - mi_new_i);
    float li_new_i = exp_mi_diff * li_val + exp_mij_diff * lij[i];

    for (int j = threadIdx.x; j < d; j += bc) {
        float pv = 0.0F;
        for (int t = 0; t < n; t++) {
            pv += pij[i * bc + t] * vj[t * d + j];
        }
        oi[i * d + j] = (li_val * exp_mi_diff * oi[i * d + j] + exp_mij_diff * pv) / li_new_i;
    }
    if (threadIdx.x == 0) {
        mi[i] = mi_new_i;
        li[i] = li_new_i;
    }
}

__global__ void block_flash_attention(float *q, float *k, float *v, float *o, float *l, float *m, int N_kv, int d, float scalar) {
    __shared__ float kj[bc * D];    // kj = [bc, d]
    __shared__ float vj[bc * D];    // vj = [bc, d]
    __shared__ float qi[br * D];    // qi = [br, d]
    __shared__ float oi[br * D];    // oi = [br, d]
    __shared__ float li[br];        // li = [br, 1]
    __shared__ float mi[br];        // mi = [br, 1]
    __shared__ float sij[br * bc];  // sij = [br, bc]
    __shared__ float pij[br * bc];  // pij = [br, bc]
    __shared__ float mij[br];       // mij = [br, 1]
    __shared__ float lij[br];       // lij = [br, 1]

    int tc = (int)ceil((float)N_kv / bc);

    if (threadIdx.x == 0) {
        li[threadIdx.y] = l[blockIdx.y * br + threadIdx.y];
        mi[threadIdx.y] = m[blockIdx.y * br + threadIdx.y];
    }
    for (int h = threadIdx.x; h < d; h += bc) {
        qi[threadIdx.y * d + h] = q[blockIdx.y * br * d + threadIdx.y * d + h];
        oi[threadIdx.y * d + h] = o[blockIdx.y * br * d + threadIdx.y * d + h];
    }

    for (int j = 0; j < tc; ++j) {
        // n is for the last sequence of K or V that may not be divisible by bc
        int n = min(bc, N_kv - j * bc);
        if (threadIdx.x < n) {
            for (int h = threadIdx.y; h < d; h += br) {
                kj[threadIdx.x * d + h] = k[j * bc * d + threadIdx.x * d + h];
                vj[threadIdx.x * d + h] = v[j * bc * d + threadIdx.x * d + h];
            }
        }
        __syncthreads();
        QKDotAndScalar(sij, qi, kj, scalar, d);
        __syncthreads();
        RowMax(mij, sij, n);
        __syncthreads();
        MinusMaxAndExp(pij, sij, mij);
        __syncthreads();
        RowSum(lij, pij, n);
        __syncthreads();
        UpdateMiLiOi(mi, li, oi, mij, lij, pij, vj, d, n);
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        l[blockIdx.y * br + threadIdx.y] = li[threadIdx.y];
        m[blockIdx.y * br + threadIdx.y] = mi[threadIdx.y];
    }
    for (int h = threadIdx.x; h < d; h += bc) {
        o[blockIdx.y * br * d + threadIdx.y * d + h] = oi[threadIdx.y * d + h];
    }
}

__global__ void init_float_array(float *arr, int size, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        arr[idx] = val;
    }
}