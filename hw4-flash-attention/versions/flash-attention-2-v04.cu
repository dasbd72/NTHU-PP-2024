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
template <int ac, int ar, int bc, int br, int bd, int num_warps, int threads_per_warp>
void flash_attention(Data *data);
template <int ac, int ar, int bc, int br, int bd, int num_warps, int threads_per_warp>
__global__ void flash_attention_kernel(float *O, float *QKV, float *L, int N, int d);
template <int ac, int ar, int bc, int br, int bd, int num_warps, int threads_per_warp>
__device__ __forceinline__ void qk_dot_and_scalar(float *out, float *q, float *k, int d, float scalar);
template <int ac, int ar, int bc, int br, int bd, int num_warps, int threads_per_warp>
__device__ __forceinline__ void row_max(float *mij1, float *sij, float *mij0, int n);
template <int ac, int ar, int bc, int br, int bd, int num_warps, int threads_per_warp>
__device__ __forceinline__ void minus_max_and_exp(float *pij, float *sij, float *mij1);
template <int ac, int ar, int bc, int br, int bd, int num_warps, int threads_per_warp>
__device__ __forceinline__ void row_sum(float *lij1, float *pij, float *lij0, float *mij0, float *mij1, int n);
template <int ac, int ar, int bc, int br, int bd, int num_warps, int threads_per_warp>
__device__ __forceinline__ void inner_update_o(float *oi, float *pij, float *vj, float *mij0, float *mij1, int n, int d);
template <int ac, int ar, int bc, int br, int bd, int num_warps, int threads_per_warp>
__device__ __forceinline__ void outer_update_lo(float *lij1, float *oi, float *mij0, float *lij0, int d);
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

#ifdef PROFILING
    timespec ts;
    double start, end;
    clock_gettime(CLOCK_REALTIME, &ts);
    start = ts.tv_sec + ts.tv_nsec * 1e-9;
#endif  // PROFILING
    flash_attention::flash_attention_switch(&data);
#ifdef PROFILING
    clock_gettime(CLOCK_REALTIME, &ts);
    end = ts.tv_sec + ts.tv_nsec * 1e-9;
    fprintf(stderr, "took: %lf\n", end - start);
#endif  // PROFILING

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
    data->output_file = fopen(data->output_filename, "wb");
    fread(&data->B, sizeof(int), 1, data->input_file);
    fread(&data->N, sizeof(int), 1, data->input_file);
    fread(&data->d, sizeof(int), 1, data->input_file);
    if (data->d <= 32) {
        flash_attention<32, 32, 37, 32, 37, 8, 16>(data);
    } else if (data->d <= 64) {
        flash_attention<32, 32, 37, 32, 69, 8, 32>(data);
    }

    fclose(data->input_file);
    fclose(data->output_file);

#ifndef NO_FINALIZE
    cudaFreeHost(data->O);
#endif  // NO_FINALIZE
}

template <int ac, int ar, int bc, int br, int bd, int num_warps, int threads_per_warp>
void flash_attention(Data *data) {
    NVTX_RANGE_FUNC();
    int B = data->B;
    int N = data->N;
    int d = data->d;
#ifdef PROFILING
    fprintf(stderr, "B: %d, N: %d, d: %d\n", B, N, d);
#endif  // PROFILING
    int bb = (int)ceilf((float)B / 64);

    // Create a CUDA stream for asynchronous operations
    int num_streams = (int)ceil((float)B / bb);
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    float *QKV, *O;
    cudaMallocHost(&QKV, B * N * d * 3 * sizeof(float));
    cudaMallocHost(&O, B * N * d * sizeof(float));
    data->O = O;

    float *d_QKV, *d_O;
    float *d_L;
    cudaMalloc(&d_QKV, B * N * d * 3 * sizeof(float));
    cudaMalloc(&d_O, B * N * d * sizeof(float));
    cudaMalloc(&d_L, B * N * sizeof(float));

    // Kernel launch
    const int smem_size = (br * bd +
                           br * bd +
                           bc * bd +
                           bc * bd +
                           br +
                           br +
                           br +
                           br +
                           br * bc +
                           br * bc) *
                          sizeof(float);

    NVTX_RANGE_START(flash_attention_execute);
    NVTX_RANGE_START(flash_attention_declare);
    for (int i = 0; i < num_streams; i++) {
        int num_batches = min(bb, B - i * bb);

        // Load data to host memory
        fread(QKV + i * bb * N * d * 3, sizeof(float), num_batches * N * d * 3, data->input_file);

        // Asynchronous memory copy and initialization
        cudaMemcpyAsync(d_QKV + i * bb * N * d * 3, QKV + i * bb * N * d * 3, num_batches * N * d * 3 * sizeof(float), cudaMemcpyHostToDevice, streams[i]);

        // Kernel launch
        dim3 grid((int)ceilf((float)N / ar), num_batches);
        dim3 block(num_warps * threads_per_warp);
        flash_attention_kernel<ac, ar, bc, br, bd, num_warps, threads_per_warp><<<grid, block, smem_size, streams[i]>>>(
            d_O + i * bb * N * d,
            d_QKV + i * bb * N * d * 3,
            d_L + i * bb * N,
            N, d);

        // Asynchronous memory copy back to host
        cudaMemcpyAsync(O + i * bb * N * d, d_O + i * bb * N * d, num_batches * N * d * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }
    NVTX_RANGE_END();  // flash_attention_declare

    // Synchronize the stream to make sure all operations complete
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        fwrite(data->O + i * bb * N * d, sizeof(float), bb * N * d, data->output_file);
    }
    NVTX_RANGE_END();  // flash_attention_execute

    // Clean up
    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }

#ifndef NO_FINALIZE
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_L);

    cudaFreeHost(Q);
    cudaFreeHost(K);
    cudaFreeHost(V);
#endif  // NO_FINALIZE
}

template <int ac, int ar, int bc, int br, int bd, int num_warps, int threads_per_warp>
__global__ void flash_attention_kernel(float *O, float *QKV, float *L, int N, int d) {
    // Thread and block index
    const int tx = threadIdx.x % num_warps;
    const int ty = threadIdx.x / num_warps;
    const int tc = (int)ceilf((float)N / ac);

    // Shared memory allocation
    extern __shared__ float shared_mem[];
    float *oi = shared_mem;      // (br, bd)
    float *qi = oi + br * bd;    // (br, bd)
    float *kj = qi + br * bd;    // (bc, bd)
    float *vj = kj + bc * bd;    // (bc, bd)
    float *lij0 = vj + bc * bd;  // (br)
    float *lij1 = lij0 + br;     // (br)
    float *mij0 = lij1 + br;     // (br)
    float *mij1 = mij0 + br;     // (br)
    float *sij = mij1 + br;      // (br, bc)
    float *pij = sij + br * bc;  // (br, bc)

    float *tmpptr;

    // Pointer to global memory
    float *o = O + blockIdx.y * N * d + blockIdx.x * ar * d;        // (ar, d)
    float *q = QKV + blockIdx.y * N * d * 3 + blockIdx.x * ar * d;  // (ar, d)
    float *k = QKV + blockIdx.y * N * d * 3 + N * d * 1;            // (N, d)
    float *v = QKV + blockIdx.y * N * d * 3 + N * d * 2;            // (N, d)
    float *l = L + blockIdx.y * N + blockIdx.x * ar;                // (ar)

    float scalar = 1.0 / sqrtf(d);

    // Load O, Q, l, m to shared memory
    for (int y = ty; y < ar; y += threads_per_warp) {
        for (int x = tx; x < d; x += num_warps) {
            oi[y * bd + x] = 0;
            qi[y * bd + x] = q[y * d + x];
        }
    }
    if (threadIdx.x < ar) {
        lij0[threadIdx.x] = 0;
#ifndef NO_ROWMAX
        mij0[threadIdx.x] = FLOAT_MIN;
#endif  // NO_ROWMAX
    }
    for (int j = 0; j < tc; j++) {
        int n = min(N - j * ac, ac);
        // Load K and V to shared memory
        for (int x = tx; x < n; x += num_warps) {
            for (int y = ty; y < d; y += threads_per_warp) {
                kj[x * bd + y] = k[j * ac * d + x * d + y];
                vj[x * bd + y] = v[j * ac * d + x * d + y];
            }
        }
        __syncthreads();
        qk_dot_and_scalar<ac, ar, bc, br, bd, num_warps, threads_per_warp>(sij, qi, kj, d, scalar);
#ifndef NO_ROWMAX
        __syncthreads();
        row_max<ac, ar, bc, br, bd, num_warps, threads_per_warp>(mij1, sij, mij0, n);
#endif  // NO_ROWMAX
        __syncthreads();
        minus_max_and_exp<ac, ar, bc, br, bd, num_warps, threads_per_warp>(pij, sij, mij1);
        __syncthreads();
        row_sum<ac, ar, bc, br, bd, num_warps, threads_per_warp>(lij1, pij, lij0, mij0, mij1, n);
        inner_update_o<ac, ar, bc, br, bd, num_warps, threads_per_warp>(oi, pij, vj, mij0, mij1, n, d);
#ifndef NO_ROWMAX
        tmpptr = mij0;
        mij0 = mij1;
        mij1 = tmpptr;
#endif  // NO_ROWMAX
        tmpptr = lij0;
        lij0 = lij1;
        lij1 = tmpptr;
        __syncthreads();
    }
    outer_update_lo<ac, ar, bc, br, bd, num_warps, threads_per_warp>(lij1, oi, mij0, lij0, d);
    __syncthreads();
    // Save O, l, m back to global memory
    for (int y = ty; y < ar; y += threads_per_warp) {
        for (int x = tx; x < d; x += num_warps) {
            o[y * d + x] = oi[y * bd + x];
        }
    }
    if (threadIdx.x < ar) {
        l[threadIdx.x] = lij1[threadIdx.x];
    }
}

template <int ac, int ar, int bc, int br, int bd, int num_warps, int threads_per_warp>
__device__ __forceinline__ void qk_dot_and_scalar(float *out, float *q, float *k, int d, float scalar) {
    int tx = threadIdx.x % num_warps;
    int ty = threadIdx.x / num_warps;
    for (int y = ty; y < ar; y += threads_per_warp) {
        for (int x = tx; x < ac; x += num_warps) {
            float sum = 0.0F;
            for (int t = 0; t < d; t++) {
                sum += q[y * bd + t] * k[x * bd + t];
            }
            out[y * bc + x] = sum * scalar;
        }
    }
}

template <int ac, int ar, int bc, int br, int bd, int num_warps, int threads_per_warp>
__device__ __forceinline__ void row_max(float *mij1, float *sij, float *mij0, int n) {
    for (int y = threadIdx.x; y < ar; y += blockDim.x) {
        float mx = mij0[y];
        for (int t = 0; t < n; t++) {
            mx = fmaxf(mx, sij[y * bc + t]);
        }
        mij1[y] = mx;
    }
}

template <int ac, int ar, int bc, int br, int bd, int num_warps, int threads_per_warp>
__device__ __forceinline__ void minus_max_and_exp(float *pij, float *sij, float *mij1) {
    int tx = threadIdx.x % num_warps;
    int ty = threadIdx.x / num_warps;
    for (int y = ty; y < ar; y += threads_per_warp) {
        float mx = mij1[y];
        for (int x = tx; x < ac; x += num_warps) {
#ifndef NO_ROWMAX
            pij[y * bc + x] = expf(sij[y * bc + x] - mx);
#else
            pij[y * bc + x] = expf(sij[y * bc + x]);
#endif  // NO_ROWMAX
        }
    }
}

template <int ac, int ar, int bc, int br, int bd, int num_warps, int threads_per_warp>
__device__ __forceinline__ void row_sum(float *lij1, float *pij, float *lij0, float *mij0, float *mij1, int n) {
    for (int y = threadIdx.x; y < ar; y += blockDim.x) {
#ifndef NO_ROWMAX
        float sum = expf(mij0[y] - mij1[y]) * lij0[y];
#else
        float sum = lij0[y];
#endif  // NO_ROWMAX
        for (int t = 0; t < n; t++) {
            sum += pij[y * bc + t];
        }
        lij1[y] = sum;
    }
}

template <int ac, int ar, int bc, int br, int bd, int num_warps, int threads_per_warp>
__device__ __forceinline__ void inner_update_o(float *oi, float *pij, float *vj, float *mij0, float *mij1, int n, int d) {
    int tx = threadIdx.x % num_warps;
    int ty = threadIdx.x / num_warps;
    for (int y = ty; y < ar; y += threads_per_warp) {
#ifndef NO_ROWMAX
        float val0 = expf(mij0[y] - mij1[y]);
#else
        float val0 = 1.0F;
#endif  // NO_ROWMAX

        for (int x = tx; x < d; x += num_warps) {
            float sum = 0.0F;
            for (int t = 0; t < n; t++) {
                sum += pij[y * bc + t] * vj[t * bd + x];
            }
            oi[y * bd + x] = val0 * oi[y * bd + x] + sum;
        }
    }
}

template <int ac, int ar, int bc, int br, int bd, int num_warps, int threads_per_warp>
__device__ __forceinline__ void outer_update_lo(float *lij1, float *oi, float *mij0, float *lij0, int d) {
    int tx = threadIdx.x % num_warps;
    int ty = threadIdx.x / num_warps;
    for (int y = ty; y < ar; y += threads_per_warp) {
        for (int x = tx; x < d; x += num_warps) {
            oi[y * bd + x] /= lij0[y];
        }
        if (tx == 0) {
#ifndef NO_ROWMAX
            lij1[y] = mij0[y] + logf(lij0[y]);
#else
            lij1[y] = logf(lij0[y]);
#endif  // NO_ROWMAX
        }
    }
}
};  // namespace flash_attention