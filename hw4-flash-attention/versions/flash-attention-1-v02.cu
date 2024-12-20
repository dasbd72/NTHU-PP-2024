#include <sys/time.h>

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
    int B, N, d;
    float *Q, *K, *V, *O;
};

void input(Data *data);
void output(Data *data);

template <typename T>
void pad_buffer(T *&buf, int B, int N, int d, int B_pad, int N_pad, int d_pad);
template <typename T>
void unpad_buffer(T *&buf, int B, int N, int d, int B_pad, int N_pad, int d_pad);

namespace flash_attention {
void flash_attention_switch(Data *data);
template <int bc, int br, int cr>
void flash_attention(Data *data);
__global__ void init_float_array(float *arr, int size, float val);
template <int bc, int br, int cr>
__global__ void flash_attention_kernel(float *O, float *Q, float *K, float *V, float *L, float *M, int N, int d);
template <int bc, int br>
__device__ void qk_dot_and_scalar(float *out, float *q, float *k, int d, float scalar);
template <int bc, int br>
__device__ void row_max(float *mij, float *sij, int n);
template <int bc, int br>
__device__ void minus_max_and_exp(float *pij, float *sij, float *mij);
template <int bc, int br>
__device__ void row_sum(float *lij, float *pij, int n);
template <int bc, int br>
__device__ void update_mlo(float *mi, float *li, float *oi, float *mij, float *lij, float *pij, float *vj, int n, int d);
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

    input(&data);

    flash_attention::flash_attention_switch(&data);

    output(&data);

    return 0;
}

void input(Data *data) {
    NVTX_RANGE_FUNC();
    FILE *file;
    int B, N, d;
    float *Q, *K, *V, *O;

    file = fopen(data->input_filename, "rb");
    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);

    Q = (float *)malloc(B * N * d * sizeof(float));
    K = (float *)malloc(B * N * d * sizeof(float));
    V = (float *)malloc(B * N * d * sizeof(float));
    O = (float *)malloc(B * N * d * sizeof(float));

    for (int i = 0; i < B; i++) {
        fread(Q + (i * N * d), sizeof(float), N * d, file);
        fread(K + (i * N * d), sizeof(float), N * d, file);
        fread(V + (i * N * d), sizeof(float), N * d, file);
    }
    memset(O, 0x00, B * N * d * sizeof(float));

    data->B = B;
    data->N = N;
    data->d = d;
    data->Q = Q;
    data->K = K;
    data->V = V;
    data->O = O;

    fclose(file);
}

void output(Data *data) {
    NVTX_RANGE_FUNC();
    FILE *file;
    int B = data->B;
    int N = data->N;
    int d = data->d;
    float *Q = data->Q;
    float *K = data->K;
    float *V = data->V;
    float *O = data->O;

    file = fopen(data->output_filename, "wb");
    fwrite(O, sizeof(float), B * N * d, file);

    fclose(file);
    free(Q);
    free(K);
    free(V);
    free(O);
}

/**
 * Pad buffer from (B, N, d) to (B_pad, N_pad, d_pad)
 **/
template <typename T>
void pad_buffer(T *&buf, int B, int N, int d, int B_pad, int N_pad, int d_pad) {
    NVTX_RANGE_FUNC();
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
    NVTX_RANGE_FUNC();
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

namespace flash_attention {
void flash_attention_switch(Data *data) {
    if (data->d <= 64) {
        flash_attention<32, 32, 1>(data);
    }
}

template <int bc, int br, int cr>
void flash_attention(Data *data) {
    NVTX_RANGE_FUNC();
    int B = data->B;
    int N = data->N;
    int d = data->d;

    // Pad buffer
    int b_lcm = std::lcm(bc, br * cr);
    int N_pad = (int)ceilf((float)N / b_lcm) * b_lcm;
    pad_buffer(data->Q, B, N, d, B, N_pad, d);
    pad_buffer(data->K, B, N, d, B, N_pad, d);
    pad_buffer(data->V, B, N, d, B, N_pad, d);
    pad_buffer(data->O, B, N, d, B, N_pad, d);

    float *Q = data->Q;
    float *K = data->K;
    float *V = data->V;
    float *O = data->O;

    float *d_Q, *d_K, *d_V, *d_O;
    float *d_L, *d_M;
    cudaMalloc(&d_Q, B * N_pad * d * sizeof(float));
    cudaMalloc(&d_K, B * N_pad * d * sizeof(float));
    cudaMalloc(&d_V, B * N_pad * d * sizeof(float));
    cudaMalloc(&d_O, B * N_pad * d * sizeof(float));
    cudaMalloc(&d_L, B * N_pad * sizeof(float));
    cudaMalloc(&d_M, B * N_pad * sizeof(float));

    // Create a CUDA stream for asynchronous operations
    cudaStream_t streams[B];
    for (int i = 0; i < B; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Register memory with cudaHostRegister
    cudaHostRegister(Q, B * N_pad * d * sizeof(float), cudaHostRegisterDefault);
    cudaHostRegister(K, B * N_pad * d * sizeof(float), cudaHostRegisterDefault);
    cudaHostRegister(V, B * N_pad * d * sizeof(float), cudaHostRegisterDefault);
    cudaHostRegister(O, B * N_pad * d * sizeof(float), cudaHostRegisterDefault);

    // Asynchronous memory copy
    for (int i = 0; i < B; i++) {
        cudaMemcpyAsync(d_Q + i * N_pad * d, Q + i * N_pad * d, N_pad * d * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_K + i * N_pad * d, K + i * N_pad * d, N_pad * d * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_V + i * N_pad * d, V + i * N_pad * d, N_pad * d * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        cudaMemsetAsync(d_O + i * N_pad * d, 0, N_pad * d * sizeof(float), streams[i]);
        cudaMemsetAsync(d_L + i * N_pad, 0, N_pad * sizeof(float), streams[i]);
        init_float_array<<<(int)ceil((float)N_pad / 256), 256, 0, streams[i]>>>(d_M + i * N_pad, N_pad, FLOAT_MIN);
    }

    // Kernel launch
    dim3 grid(1, (int)ceilf((float)N_pad / (br * cr)));
    dim3 block(bc, br);
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
    fprintf(stderr, "smem_size: %d bytes\n", smem_size);
    fprintf(stderr, "grid: (%d, %d), block: (%d, %d)\n", grid.x, grid.y, block.x, block.y);
    fprintf(stderr, "bc: %d, br: %d, cr: %d\n", bc, br, cr);
    fprintf(stderr, "B: %d, N: %d, d: %d, N_pad: %d\n", B, N, d, N_pad);
    for (int i = 0; i < B; i++) {
        flash_attention_kernel<bc, br, cr><<<grid, block, smem_size, streams[i]>>>(
            d_O + i * N_pad * d,
            d_Q + i * N_pad * d,
            d_K + i * N_pad * d,
            d_V + i * N_pad * d,
            d_L + i * N_pad,
            d_M + i * N_pad,
            N, d);
    }

    // Wait for the memory copy to finish before continuing
    for (int i = 0; i < B; i++) {
        cudaMemcpyAsync(O + i * N_pad * d, d_O + i * N_pad * d, N_pad * d * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    // Synchronize the stream to make sure all operations complete
    for (int i = 0; i < B; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    // Clean up
    for (int i = 0; i < B; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaHostUnregister(O);
    cudaHostUnregister(Q);
    cudaHostUnregister(K);
    cudaHostUnregister(V);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_L);
    cudaFree(d_M);

    // Unpad buffer
    unpad_buffer(data->O, B, N, d, B, N_pad, d);
}

__global__ void init_float_array(float *arr, int size, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        arr[idx] = val;
    }
}

template <int bc, int br, int cr>
__global__ void flash_attention_kernel(float *O, float *Q, float *K, float *V, float *L, float *M, int N, int d) {
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
    float *li = vj + bc * d;     // (br)
    float *mi = li + br;         // (br)
    float *lij = mi + br;        // (br)
    float *mij = lij + br;       // (br)
    float *sij = mij + br;       // (br, bc)
    float *pij = sij + br * bc;  // (br, bc)

    // Pointer to global memory
    float *o = O + blockIdx.y * cr * br * d;  // (cr, br, d)
    float *q = Q + blockIdx.y * cr * br * d;  // (cr, br, d)
    float *k = K;                             // (N, d)
    float *v = V;                             // (N, d)
    float *l = L + blockIdx.y * cr * br;      // (cr, br)
    float *m = M + blockIdx.y * cr * br;      // (cr, br)

    float scalar = 1.0 / sqrtf(d);

    for (int j = 0; j < tc; j++) {
        int n = min(N - j * bc, bc);
        // Load K and V to shared memory
        for (int y = ty; y < d; y += br) {
            kj[tx * d + y] = k[j * bc * d + tx * d + y];
            vj[tx * d + y] = v[j * bc * d + tx * d + y];
        }
        for (int i = 0; i < cr; i++) {
            // Load O, Q, l, m to shared memory
            for (int x = tx; x < d; x += bc) {
                oi[ty * d + x] = o[i * br * d + ty * d + x];
                qi[ty * d + x] = q[i * br * d + ty * d + x];
            }
            if (tx == 0) {
                li[ty] = l[i * br + ty];
                mi[ty] = m[i * br + ty];
            }
            __syncthreads();
            qk_dot_and_scalar<bc, br>(sij, qi, kj, d, scalar);
            __syncthreads();
            row_max<bc, br>(mij, sij, n);
            __syncthreads();
            minus_max_and_exp<bc, br>(pij, sij, mij);
            __syncthreads();
            row_sum<bc, br>(lij, pij, n);
            __syncthreads();
            update_mlo<bc, br>(mi, li, oi, mij, lij, pij, vj, n, d);
            __syncthreads();
            // Save O, l, m back to global memory
            for (int x = tx; x < d; x += bc) {
                o[i * br * d + ty * d + x] = oi[ty * d + x];
            }
            if (tx == 0) {
                l[i * br + ty] = li[ty];
                m[i * br + ty] = mi[ty];
            }
        }
    }
}

template <int bc, int br>
__device__ void qk_dot_and_scalar(float *out, float *q, float *k, int d, float scalar) {
    const int y = threadIdx.y;
    const int x = threadIdx.x;
    float sum = 0.0F;
    for (int t = 0; t < d; t++) {
        sum += q[y * d + t] * k[x * d + t];
    }
    out[y * bc + x] = sum * scalar;
}

template <int bc, int br>
__device__ void row_max(float *mij, float *sij, int n) {
    if (threadIdx.x == 0) {
        const int y = threadIdx.y;
        float mx = sij[y * bc + 0];
        for (int t = 1; t < n; t++) {
            mx = fmaxf(mx, sij[y * bc + t]);
        }
        mij[y] = mx;
    }
}

template <int bc, int br>
__device__ void minus_max_and_exp(float *pij, float *sij, float *mij) {
    const int y = threadIdx.y;
    const int x = threadIdx.x;
    pij[y * bc + x] = expf(sij[y * bc + x] - mij[y]);
}

template <int bc, int br>
__device__ void row_sum(float *lij, float *pij, int n) {
    if (threadIdx.x == 0) {
        const int y = threadIdx.y;
        float sum = pij[y * bc + 0];
        for (int t = 1; t < n; t++) {
            sum += pij[y * bc + t];
        }
        lij[y] = sum;
    }
}

template <int bc, int br>
__device__ void update_mlo(float *mi, float *li, float *oi, float *mij, float *lij, float *pij, float *vj, int n, int d) {
    const int y = threadIdx.y;

    float mi_new = fmaxf(mi[y], mij[y]);
    float val0 = expf(mi[y] - mi_new) * li[y];
    float val1 = expf(mij[y] - mi_new);
    float li_new = val0 + val1 * lij[y];

    for (int x = threadIdx.x; x < d; x += bc) {
        float sum = 0.0F;
        for (int t = 0; t < n; t++) {
            sum += pij[y * bc + t] * vj[t * d + x];
        }
        oi[y * d + x] = (val0 * oi[y * d + x] + val1 * sum) / li_new;
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        mi[y] = mi_new;
        li[y] = li_new;
    }
}
};  // namespace flash_attention