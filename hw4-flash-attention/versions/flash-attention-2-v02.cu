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
template <int bc, int br, int cr, int bb>
void flash_attention(Data *data);
template <int bc, int br, int cr>
__global__ void flash_attention_kernel(float *O, float *Q, float *K, float *V, float *L, int N, int d);
template <int bc, int br>
__device__ void qk_dot_and_scalar(float *pij, float *sij, float *q, float *k, int d, float scalar);
template <int bc, int br>
__device__ void row_sum(float *lij1, float *pij, float *lij0, int n);
template <int bc, int br>
__device__ void inner_update_o(float *oi, float *pij, float *vj, int n, int d);
template <int bc, int br>
__device__ void outer_update_lo(float *lij1, float *oi, float *lij0, int d);
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
        flash_attention<32, 32, 1, 20>(data);
    }
    data->output_file = fopen(data->output_filename, "wb");
    fwrite(data->O, sizeof(float), data->B * data->N * data->d, data->output_file);

    fclose(data->input_file);
    fclose(data->output_file);

    cudaFreeHost(data->O);
}

template <int bc, int br, int cr, int bb>
void flash_attention(Data *data) {
    NVTX_RANGE_FUNC();
    int B = data->B;
    int N = data->N;
    int d = data->d;

    // Create a CUDA stream for asynchronous operations
    int num_streams = (int)ceil((float)B / bb);
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
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
    cudaMalloc(&d_Q, B * N * d * sizeof(float));
    cudaMalloc(&d_K, B * N * d * sizeof(float));
    cudaMalloc(&d_V, B * N * d * sizeof(float));
    cudaMalloc(&d_O, B * N * d * sizeof(float));
    cudaMalloc(&d_L, B * N * sizeof(float));

    // Kernel launch
    const int smem_size = (br * d +
                           br * d +
                           bc * d +
                           bc * d +
                           br +
                           br +
                           br * bc +
                           br * bc) *
                          sizeof(float);

    NVTX_RANGE_START(flash_attention_execute);
    NVTX_RANGE_START(flash_attention_declare);
    for (int i = 0; i < num_streams; i++) {
        int num_batches = min(bb, B - i * bb);

        // Asynchronous memory copy and initialization
        cudaMemcpyAsync(d_Q + i * bb * N * d, Q + i * bb * N * d, num_batches * N * d * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_K + i * bb * N * d, K + i * bb * N * d, num_batches * N * d * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_V + i * bb * N * d, V + i * bb * N * d, num_batches * N * d * sizeof(float), cudaMemcpyHostToDevice, streams[i]);

        // Kernel launch
        dim3 grid((int)ceilf((float)N / (br * cr)), num_batches);
        dim3 block(bc, br);
        flash_attention_kernel<bc, br, cr><<<grid, block, smem_size, streams[i]>>>(
            d_O + i * bb * N * d,
            d_Q + i * bb * N * d,
            d_K + i * bb * N * d,
            d_V + i * bb * N * d,
            d_L + i * bb * N,
            N, d);

        // Asynchronous memory copy back to host
        cudaMemcpyAsync(O + i * bb * N * d, d_O + i * bb * N * d, num_batches * N * d * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }
    NVTX_RANGE_END();  // flash_attention_declare

    // Synchronize the stream to make sure all operations complete
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    NVTX_RANGE_END();  // flash_attention_execute

    // Clean up
    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_L);

    cudaFreeHost(Q);
    cudaFreeHost(K);
    cudaFreeHost(V);
}

template <int bc, int br, int cr>
__global__ void flash_attention_kernel(float *O, float *Q, float *K, float *V, float *L, int N, int d) {
    // Thread and block index
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int tc = (int)ceilf((float)N / bc);

    // Shared memory allocation
    extern __shared__ float shared_mem[];
    float *oi = shared_mem;      // (br, d)
    float *qi = oi + br * d;     // (br, d)
    float *kj = qi + br * d;     // (bc, d)
    float *vj = kj + bc * d;     // (bc, d)
    float *lij0 = vj + bc * d;   // (br)
    float *lij1 = lij0 + br;     // (br)
    float *sij = lij1 + br;      // (br, bc)
    float *pij = sij + br * bc;  // (br, bc)

    float *tmpptr;

    // Pointer to global memory
    float *o = O + blockIdx.y * N * d + blockIdx.x * cr * br * d;  // (cr, br, d)
    float *q = Q + blockIdx.y * N * d + blockIdx.x * cr * br * d;  // (cr, br, d)
    float *k = K + blockIdx.y * N * d;                             // (N, d)
    float *v = V + blockIdx.y * N * d;                             // (N, d)
    float *l = L + blockIdx.y * N + blockIdx.x * cr * br;          // (cr, br)

    float scalar = 1.0 / sqrtf(d);

    // Load O, Q, l, m to shared memory
    for (int x = tx; x < d; x += bc) {
        oi[ty * d + x] = 0;
        qi[ty * d + x] = q[ty * d + x];
    }
    if (tx == 0) {
        lij0[ty] = 0;
    }
    for (int j = 0; j < tc; j++) {
        int n = min(N - j * bc, bc);
        // Load K and V to shared memory
        for (int y = ty; y < d; y += br) {
            kj[tx * d + y] = k[j * bc * d + tx * d + y];
            vj[tx * d + y] = v[j * bc * d + tx * d + y];
        }
        __syncthreads();
        qk_dot_and_scalar<bc, br>(pij, sij, qi, kj, d, scalar);
        __syncthreads();
        row_sum<bc, br>(lij1, pij, lij0, n);
        __syncthreads();
        inner_update_o<bc, br>(oi, pij, vj, n, d);
        tmpptr = lij0;
        lij0 = lij1;
        lij1 = tmpptr;
        __syncthreads();
    }
    outer_update_lo<bc, br>(lij1, oi, lij0, d);
    // Save O, l, m back to global memory
    for (int x = tx; x < d; x += bc) {
        o[ty * d + x] = oi[ty * d + x];
    }
    if (tx == 0) {
        l[ty] = lij1[ty];
    }
}

template <int bc, int br>
__device__ void qk_dot_and_scalar(float *pij, float *sij, float *q, float *k, int d, float scalar) {
    const int y = threadIdx.y;
    const int x = threadIdx.x;
    float sum = 0.0F;
    for (int t = 0; t < d; t++) {
        sum += q[y * d + t] * k[x * d + t];
    }
    sum *= scalar;
    sij[y * bc + x] = sum;
    pij[y * bc + x] = expf(sum);
}

template <int bc, int br>
__device__ void row_sum(float *lij1, float *pij, float *lij0, int n) {
    if (threadIdx.x == 0) {
        const int y = threadIdx.y;
        float sum = lij0[y];
        for (int t = 0; t < n; t++) {
            sum += pij[y * bc + t];
        }
        lij1[y] = sum;
    }
}

template <int bc, int br>
__device__ void inner_update_o(float *oi, float *pij, float *vj, int n, int d) {
    const int y = threadIdx.y;

    for (int x = threadIdx.x; x < d; x += bc) {
        float sum = 0.0F;
        for (int t = 0; t < n; t++) {
            sum += pij[y * bc + t] * vj[t * d + x];
        }
        oi[y * d + x] = oi[y * d + x] + sum;
    }
}

template <int bc, int br>
__device__ void outer_update_lo(float *lij1, float *oi, float *lij0, int d) {
    const int y = threadIdx.y;

    for (int x = threadIdx.x; x < d; x += bc) {
        oi[y * d + x] /= lij0[y];
    }
    if (threadIdx.x == 0) {
        lij1[y] = logf(lij0[y]);
    }
}
};  // namespace flash_attention