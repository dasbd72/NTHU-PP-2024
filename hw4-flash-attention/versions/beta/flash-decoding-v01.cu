#include <cuda.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <unistd.h>

#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

#define CUDA_CHECK(condition)                                                                                     \
    if ((condition) != cudaSuccess) {                                                                             \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__); \
        exit(1);                                                                                                  \
    }

constexpr float FLOAT_MIN = -FLT_MAX;
constexpr int NumWarps = 8;
constexpr int ThreadsPerWarps = 32;

struct Data {
    char *input_filename;
    char *output_filename;
    FILE *input_file;
    FILE *output_file;
    int B, N, d;
    float *O;
};

template <typename T>
void cuda_init_array(T *arr, size_t size, T val, cudaStream_t stream);
template <typename T>
__global__ void cuda_init_array_kernel(T *arr, size_t size, T val);

namespace flash_attention {
void flash_attention_switch(Data *data);
template <int bc, int br, int cr>
void flash_attention(Data *data);
template <int bc, int br, int cr>
__global__ void flash_attention_kernel(float *O, float *Q, float *K, float *V, float *L, int Nq, int Nkv, int d);
template <int bc, int br>
__device__ __forceinline__ void qk_dot_and_scalar(float *out, float *q, float *k, int nq, int nkv, int d, float scalar);
template <int bc, int br>
__device__ __forceinline__ void row_max(float *mij1, float *sij, float *mij0, int nq, int nkv);
template <int bc, int br>
__device__ __forceinline__ void minus_max_and_exp(float *pij, float *sij, float *mij1, int nq, int nkv);
template <int bc, int br>
__device__ __forceinline__ void row_sum(float *lij1, float *pij, float *lij0, float *mij0, float *mij1, int nq, int nkv);
template <int bc, int br>
__device__ __forceinline__ void inner_update_o(float *oi, float *pij, float *vj, float *mij0, float *mij1, int nq, int nkv, int d);
template <int bc, int br>
__device__ __forceinline__ void outer_update_lo(float *lij1, float *oi, float *mij0, float *lij0, int nq, int d);
__global__ void flash_attention_merge_kernel(float *O, float *O_N, float *L, float *L_N, int N, int num_kv, int d);
};  // namespace flash_attention

int main(int argc, char *argv[]) {
    NVTX_RANGE_FUNC();
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }

    Data data;

    data.input_filename = argv[1];
    data.output_filename = argv[2];

    flash_attention::flash_attention_switch(&data);

    return 0;
}

template <typename T>
void cuda_init_array(T *arr, size_t size, T val, cudaStream_t stream) {
    cuda_init_array_kernel<<<(int)ceil((float)size / 1024), 1024, 0, stream>>>(arr, size, val);
}

template <typename T>
__global__ void cuda_init_array_kernel(T *arr, size_t size, T val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        arr[idx] = val;
    }
}

namespace flash_attention {
void flash_attention_switch(Data *data) {
    data->input_file = fopen(data->input_filename, "rb");
    fread(&data->B, sizeof(int), 1, data->input_file);
    fread(&data->N, sizeof(int), 1, data->input_file);
    fread(&data->d, sizeof(int), 1, data->input_file);
    if (data->d <= 64) {
        flash_attention<32, 32, 1>(data);
    }
    data->output_file = fopen(data->output_filename, "wb");
    fwrite(data->O, sizeof(float), data->B * data->N * data->d, data->output_file);

    fclose(data->input_file);
    fclose(data->output_file);

    cudaFreeHost(data->O);
}

template <int bc, int br, int cr>
void flash_attention(Data *data) {
    NVTX_RANGE_FUNC();
    int B = data->B;
    int N = data->N;
    int d = data->d;
    fprintf(stderr, "B: %d, N: %d, d: %d\n", B, N, d);
    int size_kv = min(N, 128);
    int num_kv = (int)ceil((float)N / size_kv);
    fprintf(stderr, "size_kv: %d, num_kv: %d\n", size_kv, num_kv);

    // Create a CUDA stream for asynchronous operations
    int num_streams_Q = B;
    int num_streams_KV = B * num_kv;
    int num_streams_merge = B;
    int num_streams = num_streams_Q + num_streams_KV + num_streams_merge;
    cudaStream_t streams[num_streams];
    cudaStream_t *streams_Q = streams;
    cudaStream_t *streams_KV = streams + num_streams_Q;
    cudaStream_t *streams_merge = streams + num_streams_Q + num_streams_KV;
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    int num_events_Q = B;
    int num_events_KV = B * num_kv;
    int num_events = num_events_Q + num_events_KV;
    cudaEvent_t events[num_events];
    cudaEvent_t *events_Q = events;
    cudaEvent_t *events_KV = events + num_events_Q;
    for (int i = 0; i < num_events; i++) {
        cudaEventCreate(&events[i]);
    }

    float *Q, *K, *V, *O;
    cudaMallocHost(&Q, B * N * d * sizeof(float));
    cudaMallocHost(&K, B * N * d * sizeof(float));
    cudaMallocHost(&V, B * N * d * sizeof(float));
    cudaMallocHost(&O, B * N * d * sizeof(float));
    data->O = O;
    for (int i = 0; i < B; i++) {
        fread(Q + i * N * d, sizeof(float), N * d, data->input_file);
        fread(K + i * N * d, sizeof(float), N * d, data->input_file);
        fread(V + i * N * d, sizeof(float), N * d, data->input_file);
    }

    float *d_Q, *d_K, *d_V, *d_O;
    float *d_L;
    float *d_O_N, *d_L_N;
    cudaMalloc(&d_Q, B * N * d * sizeof(float));
    cudaMalloc(&d_K, B * N * d * sizeof(float));
    cudaMalloc(&d_V, B * N * d * sizeof(float));
    cudaMalloc(&d_O, B * N * d * sizeof(float));
    cudaMalloc(&d_L, B * N * sizeof(float));
    cudaMalloc(&d_O_N, B * num_kv * N * d * sizeof(float));
    cudaMalloc(&d_L_N, B * num_kv * N * sizeof(float));

    // Kernel launch
    const int smem_size = (br * d +
                           br * d +
                           bc * d +
                           bc * d +
                           br +
                           br +
                           br +
                           br +
                           br * bc +
                           br * bc) *
                          sizeof(float);

    NVTX_RANGE_START(flash_attention_execute);
    NVTX_RANGE_START(flash_attention_declare);
    for (int i = 0; i < B; i++) {
        int offset_q = i * N * d;
        cudaMemcpyAsync(d_Q + offset_q, Q + offset_q, N * d * sizeof(float), cudaMemcpyHostToDevice, streams_Q[i]);
        cudaMemsetAsync(d_O + offset_q, 0, N * d * sizeof(float), streams_Q[i]);
        cudaMemsetAsync(d_L + i * N, 0, N * sizeof(float), streams_Q[i]);
        cudaEventRecord(events_Q[i], streams_Q[i]);

        for (int j = 0; j < num_kv; j++) {
            int _size_kv = min(size_kv, N - j * size_kv);
            // Asynchronous memory copy and initialization
            int offset_o = i * num_kv * N * d + j * N * d;
            int offset_kv = i * N * d + j * size_kv * d;
            int offset_l = i * num_kv * N + j * N;
            cudaStreamWaitEvent(streams_KV[i * num_kv + j], events_Q[i], 0);
            cudaMemcpyAsync(d_K + offset_kv, K + offset_kv, _size_kv * d * sizeof(float), cudaMemcpyHostToDevice, streams_KV[i * num_kv + j]);
            cudaMemcpyAsync(d_V + offset_kv, V + offset_kv, _size_kv * d * sizeof(float), cudaMemcpyHostToDevice, streams_KV[i * num_kv + j]);

            // Kernel launch
            dim3 grid((int)ceilf((float)N / (br * cr)), 1);
            dim3 block(NumWarps * ThreadsPerWarps);
            flash_attention_kernel<bc, br, cr><<<grid, block, smem_size, streams_KV[i * num_kv + j]>>>(
                d_O_N + offset_o,
                d_Q + offset_q,
                d_K + offset_kv,
                d_V + offset_kv,
                d_L_N + offset_l,
                N, _size_kv, d);
            cudaEventRecord(events_KV[i * num_kv + j], streams_KV[i * num_kv + j]);
            cudaStreamWaitEvent(streams_merge[i], events_KV[i * num_kv + j], 0);
        }

        // TODO: Merge the results
        flash_attention_merge_kernel<<<1, 1, 0, streams_merge[i]>>>(
            d_O + offset_q,
            d_O_N + i * num_kv * N * d,
            d_L + offset_q,
            d_L_N + i * num_kv * N,
            N, num_kv, d);

        // Asynchronous memory copy back to host
        cudaMemcpyAsync(O + offset_q, d_O + offset_q, N * d * sizeof(float), cudaMemcpyDeviceToHost, streams_merge[i]);
    }
    CUDA_CHECK(cudaPeekAtLastError());
    NVTX_RANGE_END();  // flash_attention_declare

    // Synchronize the stream to make sure all operations complete
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    CUDA_CHECK(cudaPeekAtLastError());
    NVTX_RANGE_END();  // flash_attention_execute

    // Clean up
    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    for (int i = 0; i < num_events; i++) {
        cudaEventDestroy(events[i]);
    }

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_L);
    cudaFree(d_O_N);
    cudaFree(d_L_N);

    cudaFreeHost(Q);
    cudaFreeHost(K);
    cudaFreeHost(V);
}

template <int bc, int br, int cr>
__global__ void flash_attention_kernel(float *O, float *Q, float *K, float *V, float *L, int Nq, int Nkv, int d) {
    // Thread and block index
    const int tx = threadIdx.x;
    const int tc = (int)ceilf((float)Nkv / bc);

    // Shared memory allocation
    extern __shared__ float shared_mem[];
    float *oi = shared_mem;      // (br, d)
    float *qi = oi + br * d;     // (br, d)
    float *kj = qi + br * d;     // (bc, d)
    float *vj = kj + bc * d;     // (bc, d)
    float *lij0 = vj + bc * d;   // (br)
    float *lij1 = lij0 + br;     // (br)
    float *mij0 = lij1 + br;     // (br)
    float *mij1 = mij0 + br;     // (br)
    float *sij = mij1 + br;      // (br, bc)
    float *pij = sij + br * bc;  // (br, bc)

    float *tmpptr;

    // Pointer to global memory
    float *o = O + blockIdx.x * cr * br * d;  // (cr, br, d)
    float *q = Q + blockIdx.x * cr * br * d;  // (cr, br, d)
    float *k = K;                             // (N, d)
    float *v = V;                             // (N, d)
    float *l = L + blockIdx.x * cr * br;      // (cr, br)

    int nq = min(Nq - blockIdx.x * cr * br, br);
    float scalar = 1.0 / sqrtf(d);

    // Load O, Q, l, m to shared memory
    for (int x = tx; x < nq * d; x += blockDim.x) {
        oi[x] = 0;
        qi[x] = q[x];
    }
    if (tx < nq) {
        lij0[tx] = 0;
        mij0[tx] = FLOAT_MIN;
    }
    for (int j = 0; j < tc; j++) {
        int nkv = min(Nkv - j * bc, bc);
        // Load K and V to shared memory
        for (int x = tx; x < nkv * d; x += blockDim.x) {
            kj[x] = k[j * bc * d + x];
            vj[x] = v[j * bc * d + x];
        }
        __syncthreads();
        qk_dot_and_scalar<bc, br>(sij, qi, kj, nq, nkv, d, scalar);
        __syncthreads();
        row_max<bc, br>(mij1, sij, mij0, nq, nkv);
        __syncthreads();
        minus_max_and_exp<bc, br>(pij, sij, mij1, nq, nkv);
        __syncthreads();
        row_sum<bc, br>(lij1, pij, lij0, mij0, mij1, nq, nkv);
        __syncthreads();
        inner_update_o<bc, br>(oi, pij, vj, mij0, mij1, nq, nkv, d);
        tmpptr = mij0;
        mij0 = mij1;
        mij1 = tmpptr;
        tmpptr = lij0;
        lij0 = lij1;
        lij1 = tmpptr;
        __syncthreads();
    }
    outer_update_lo<bc, br>(lij1, oi, mij0, lij0, nq, d);
    __syncthreads();
    // Save O, l, m back to global memory
    for (int x = tx; x < nq * d; x += blockDim.x) {
        o[x] = oi[x];
    }
    if (tx < nq) {
        l[tx] = lij1[tx];
    }
}

template <int bc, int br>
__device__ __forceinline__ void qk_dot_and_scalar(float *out, float *q, float *k, int nq, int nkv, int d, float scalar) {
    int tx = threadIdx.x % NumWarps;
    int ty = threadIdx.x / NumWarps;
    for (int y = ty; y < nq; y += ThreadsPerWarps) {
        for (int x = tx; x < nkv; x += NumWarps) {
            float sum = 0.0F;
            for (int t = 0; t < d; t++) {
                sum += q[y * d + t] * k[x * d + t];
            }
            out[y * bc + x] = sum * scalar;
        }
    }
}

template <int bc, int br>
__device__ __forceinline__ void row_max(float *mij1, float *sij, float *mij0, int nq, int nkv) {
    int tx = threadIdx.x % NumWarps;
    int ty = threadIdx.x / NumWarps;
    if (tx == 0) {
        for (int y = ty; y < nq; y += ThreadsPerWarps) {
            float mx = mij0[y];
            for (int t = 0; t < nkv; t++) {
                mx = fmaxf(mx, sij[y * bc + t]);
            }
            mij1[y] = mx;
        }
    }
}

template <int bc, int br>
__device__ __forceinline__ void minus_max_and_exp(float *pij, float *sij, float *mij1, int nq, int nkv) {
    int tx = threadIdx.x % NumWarps;
    int ty = threadIdx.x / NumWarps;
    for (int y = ty; y < nq; y += ThreadsPerWarps) {
        for (int x = tx; x < nkv; x += NumWarps) {
            pij[y * bc + x] = expf(sij[y * bc + x] - mij1[y]);
        }
    }
}

template <int bc, int br>
__device__ __forceinline__ void row_sum(float *lij1, float *pij, float *lij0, float *mij0, float *mij1, int nq, int nkv) {
    int tx = threadIdx.x % NumWarps;
    int ty = threadIdx.x / NumWarps;
    if (tx == 0) {
        for (int y = ty; y < nq; y += ThreadsPerWarps) {
            float sum = expf(mij0[y] - mij1[y]) * lij0[y];
            for (int t = 0; t < nkv; t++) {
                sum += pij[y * bc + t];
            }
            lij1[y] = sum;
        }
    }
}

template <int bc, int br>
__device__ __forceinline__ void inner_update_o(float *oi, float *pij, float *vj, float *mij0, float *mij1, int nq, int nkv, int d) {
    int tx = threadIdx.x % NumWarps;
    int ty = threadIdx.x / NumWarps;
    for (int y = ty; y < nq; y += ThreadsPerWarps) {
        float val0 = expf(mij0[y] - mij1[y]);

        for (int x = tx; x < d; x += NumWarps) {
            float sum = 0.0F;
            for (int t = 0; t < nkv; t++) {
                sum += pij[y * bc + t] * vj[t * d + x];
            }
            oi[y * d + x] = val0 * oi[y * d + x] + sum;
        }
    }
}

template <int bc, int br>
__device__ __forceinline__ void outer_update_lo(float *lij1, float *oi, float *mij0, float *lij0, int nq, int d) {
    int tx = threadIdx.x % NumWarps;
    int ty = threadIdx.x / NumWarps;
    for (int y = ty; y < nq; y += ThreadsPerWarps) {
        for (int x = tx; x < d; x += NumWarps) {
            oi[y * d + x] /= lij0[y];
        }
        if (tx == 0) {
            lij1[y] = mij0[y] + logf(lij0[y]);
        }
    }
}

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0 / (1.0 + expf(-x));
}

__global__ void flash_attention_merge_kernel(float *O, float *O_N, float *L, float *L_N, int N, int num_kv, int d) {
    // int y = blockIdx.x * blockDim.x + threadIdx.x;
    // if (y >= N) {
    //     return;
    // }
    // for (int i = 0; i < num_kv; i++) {
    //     float block_lse = L_N[i * N + y];
    //     float lse = L[y];
    //     for (int x = 0; x < d; x++) {
    //         float block_out = O_N[i * N * d + y * d + x];
    //         float out = O[y * d + x];
    //         O[y * d + x] = out - sigmoid(block_lse - lse) * (out - block_out);
    //     }
    //     __syncthreads();
    //     L[y] = lse - logf(sigmoid(lse - block_lse));
    // }

    for (int y = 0; y < N; y++) {
        float block_lse = L_N[y];
        float lse = L[y];
        for (int x = 0; x < d; x++) {
            float block_out = O_N[y * d + x];
            float out = O[y * d + x];
            O[y * d + x] = out - sigmoid(block_lse - lse) * (out - block_out);
        }
        L[y] = lse - logf(sigmoid(lse - block_lse));
    }
}
};  // namespace flash_attention