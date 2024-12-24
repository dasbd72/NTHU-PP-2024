#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

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
    int V, E;
};

constexpr int int_max = ((1 << 30) - 1);

double get_timestamp();

template <int nt, int ts, int bs>
void solve(Data *data);

template <int nt, int ts, int bs>
__device__ int blk_idx(int r, int c, int blk_pitch, int nblocks);

template <int nt, int ts, int bs>
__global__ void proc_1_glob(int *blk_dist, int k, int blk_pitch, int nblocks);
template <int nt, int ts, int bs>
__global__ void proc_2_glob(int *blk_dist, int s, int k, int blk_pitch, int nblocks);
template <int nt, int ts, int bs>
__global__ void proc_3_glob(int *blk_dist, int s_i, int s_j, int k, int blk_pitch, int nblocks);

template <int nt, int ts, int bs>
__global__ void init_dist(int *blk_dist, int blk_pitch, int nblocks);
template <int nt, int ts, int bs>
__global__ void build_dist(int *edge, int E, int *blk_dist, int blk_pitch, int nblocks);
template <int nt, int ts, int bs>
__global__ void copy_dist(int *blk_dist, int blk_pitch, int *dist, int pitch, int nblocks);

template <int nt, int ts, int bs>
__global__ void proc_1_blk_glob(int *blk_dist, int k, int pitch);
template <int nt, int ts, int bs>
__global__ void proc_2_blk_glob(int *blk_dist, int s, int k, int pitch);
template <int nt, int ts, int bs>
__global__ void proc_3_blk_glob(int *blk_dist, int s_i, int s_j, int k, int pitch);

__global__ void init_blk_dist(int *blk_dist, int pitch);
__global__ void build_blk_dist(int *edge, int E, int *blk_dist, int pitch);

int main(int argc, char **argv) {
    assert(argc == 3);

    Data data;

    data.input_filename = argv[1];
    data.output_filename = argv[2];
    data.input_file = fopen(data.input_filename, "rb");
    assert(data.input_file);
    data.output_file = fopen(data.output_filename, "wb");
    assert(data.output_file);

    fread(&data.V, sizeof(int), 1, data.input_file);
    fread(&data.E, sizeof(int), 1, data.input_file);
#ifdef PROFILING
    fprintf(stderr, "V: %d, E: %d\n", data.V, data.E);
#endif  // PROFILING

    solve<3, 26, 78>(&data);

    fclose(data.input_file);
    fclose(data.output_file);
    return 0;
}

double get_timestamp() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

template <int nt, int ts, int bs>
void solve(Data *data) {
#ifdef PROFILING
    fprintf(stderr, "nt: %d, ts: %d, bs: %d\n", nt, ts, bs);
#endif  // PROFILING

    FILE *input_file;
    FILE *output_file;
    int V, E;
    int *edge;
    int *dist;
    int VP;
    int nblocks;
    cudaDeviceProp deviceProp;
    double start_ts;

    input_file = data->input_file;
    output_file = data->output_file;
    V = data->V;
    E = data->E;
    start_ts = get_timestamp();

    cudaSetDevice(0);
    cudaGetDeviceProperties(&deviceProp, 0);
#ifdef PROFILING
    fprintf(stderr, "totalGlobalMem: %.3f GB\n", (float)deviceProp.totalGlobalMem / (1 << 30));
#endif  // PROFILING

    nblocks = (int)ceilf(float(V) / bs);
    VP = nblocks * bs;

    /* input */
    edge = (int *)malloc(sizeof(int) * 3 * E);
    fread(edge, sizeof(int), 3 * E, input_file);
    dist = (int *)malloc(sizeof(int) * V * V);

    /* calculate */
    int *edge_dev;
    int *blk_dist_dev;
    size_t blk_pitch;

    cudaHostRegister(edge, sizeof(int) * 3 * E, cudaHostRegisterReadOnly);
    cudaMalloc(&edge_dev, sizeof(int) * 3 * E);
    cudaHostRegister(dist, sizeof(int) * V * V, cudaHostRegisterDefault);
    cudaMalloc(&blk_dist_dev, sizeof(int) * bs * bs * nblocks * nblocks);
    blk_pitch = bs * bs;

    cudaMemcpy(edge_dev, edge, sizeof(int) * 3 * E, cudaMemcpyHostToDevice);

    init_dist<nt, ts, bs><<<dim3(VP / ts, VP / ts), dim3(ts, ts)>>>(blk_dist_dev, blk_pitch, nblocks);
    build_dist<nt, ts, bs><<<(int)ceilf((float)E / (ts * ts)), ts * ts>>>(edge_dev, E, blk_dist_dev, blk_pitch, nblocks);

    cudaHostUnregister(edge);
    cudaFree(edge_dev);

    dim3 blk(ts, ts);
    for (int k = 0; k < nblocks; k++) {
        /* Phase 1 */
        proc_1_glob<nt, ts, bs><<<1, blk>>>(blk_dist_dev, k, blk_pitch, nblocks);
        /* Phase 2 */
        proc_2_glob<nt, ts, bs><<<dim3(nblocks - 1, 2), blk>>>(blk_dist_dev, 0, k, blk_pitch, nblocks);
        /* Phase 3 */
        proc_3_glob<nt, ts, bs><<<dim3(nblocks - 1, nblocks - 1), blk>>>(blk_dist_dev, 0, 0, k, blk_pitch, nblocks);
    }

    cudaDeviceSynchronize();

    int num_dist_dev = min(
        (int)((deviceProp.totalGlobalMem - (size_t)bs * bs * nblocks * nblocks * sizeof(int)) / ((size_t)VP * bs * sizeof(int))),
        32);
    int *dist_dev[num_dist_dev];
    cudaStream_t stream[num_dist_dev];
    for (int i = 0; i < num_dist_dev; i++) {
        cudaStreamCreate(&stream[i]);
    }
    for (int i = 0; i < num_dist_dev; i++) {
        cudaMalloc(&dist_dev[i], sizeof(int) * VP * bs);
    }
    for (int chunk = 0; chunk < VP; chunk += bs) {
        int dev_height = min(bs, VP - chunk);
        int host_height = min(bs, V - chunk);
        int dev_idx = (chunk / bs) % num_dist_dev;
        copy_dist<nt, ts, bs><<<dim3(VP / ts, dev_height / ts), dim3(ts, ts), 0, stream[dev_idx]>>>(
            blk_dist_dev + chunk * bs * nblocks,
            blk_pitch,
            dist_dev[dev_idx],
            VP,
            nblocks);
        cudaMemcpy2DAsync(dist + chunk * V, sizeof(int) * V, dist_dev[dev_idx], sizeof(int) * VP, sizeof(int) * V, host_height, cudaMemcpyDeviceToHost, stream[dev_idx]);
    }
    for (int i = 0; i < num_dist_dev; i++) {
        cudaStreamSynchronize(stream[i]);
    }
    for (int i = 0; i < num_dist_dev; i++) {
        cudaStreamDestroy(stream[i]);
    }
    for (int i = 0; i < num_dist_dev; i++) {
        cudaFree(dist_dev[i]);
    }

    cudaHostUnregister(dist);
    cudaFree(blk_dist_dev);

    /* output */
    fwrite(dist, 1, sizeof(int) * V * V, output_file);

#ifdef PROFILING
    fprintf(stderr, "Took: %lf\n", get_timestamp() - start_ts);
#endif  // PROFILING

    free(edge);
    free(dist);
}

template <int nt, int ts, int bs>
__device__ int blk_idx(int r, int c, int blk_pitch, int nblocks) {
    return ((r / bs) * nblocks + (c / bs)) * blk_pitch + (r % bs) * bs + (c % bs);
}

#define _ref(i, j, r, c) blk_dist[(i * nblocks + j) * blk_pitch + (r) * bs + c]
template <int nt, int ts, int bs>
__global__ void proc_1_glob(int *blk_dist, int k, int blk_pitch, int nblocks) {
    __shared__ int k_k_sm[bs][bs];

    int r = threadIdx.y;
    int c = threadIdx.x;

#pragma unroll
    for (int rr = 0; rr < nt; rr++) {
#pragma unroll
        for (int cc = 0; cc < nt; cc++) {
            k_k_sm[r + rr * ts][c + cc * ts] = _ref(k, k, r + rr * ts, c + cc * ts);
        }
    }
    __syncthreads();

#pragma unroll
    for (int b = 0; b < bs; b++) {
#pragma unroll
        for (int rr = 0; rr < nt; rr++) {
#pragma unroll
            for (int cc = 0; cc < nt; cc++) {
                k_k_sm[r + rr * ts][c + cc * ts] = min(k_k_sm[r + rr * ts][c + cc * ts], k_k_sm[r + rr * ts][b] + k_k_sm[b][c + cc * ts]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int rr = 0; rr < nt; rr++) {
#pragma unroll
        for (int cc = 0; cc < nt; cc++) {
            _ref(k, k, r + rr * ts, c + cc * ts) = k_k_sm[r + rr * ts][c + cc * ts];
        }
    }
}
template <int nt, int ts, int bs>
__global__ void proc_2_glob(int *blk_dist, int s, int k, int blk_pitch, int nblocks) {
    __shared__ int k_k_sm[bs][bs];
    __shared__ int sm[bs][bs];

    int i = s + blockIdx.x;
    int r = threadIdx.y;
    int c = threadIdx.x;

    if (i >= k)
        i++;

#pragma unroll
    for (int rr = 0; rr < nt; rr++) {
#pragma unroll
        for (int cc = 0; cc < nt; cc++) {
            k_k_sm[r + rr * ts][c + cc * ts] = _ref(k, k, r + rr * ts, c + cc * ts);
        }
    }
    if (blockIdx.y == 0) {
        /* rows */
#pragma unroll
        for (int rr = 0; rr < nt; rr++) {
#pragma unroll
            for (int cc = 0; cc < nt; cc++) {
                sm[r + rr * ts][c + cc * ts] = _ref(i, k, r + rr * ts, c + cc * ts);
            }
        }
        __syncthreads();

#pragma unroll
        for (int b = 0; b < bs; b++) {
#pragma unroll
            for (int rr = 0; rr < nt; rr++) {
#pragma unroll
                for (int cc = 0; cc < nt; cc++) {
                    sm[r + rr * ts][c + cc * ts] = min(sm[r + rr * ts][c + cc * ts], sm[r + rr * ts][b] + k_k_sm[b][c + cc * ts]);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int rr = 0; rr < nt; rr++) {
#pragma unroll
            for (int cc = 0; cc < nt; cc++) {
                _ref(i, k, r + rr * ts, c + cc * ts) = sm[r + rr * ts][c + cc * ts];
            }
        }
    } else {
        /* cols */
#pragma unroll
        for (int rr = 0; rr < nt; rr++) {
#pragma unroll
            for (int cc = 0; cc < nt; cc++) {
                sm[r + rr * ts][c + cc * ts] = _ref(k, i, r + rr * ts, c + cc * ts);
            }
        }
        __syncthreads();

#pragma unroll
        for (int b = 0; b < bs; b++) {
#pragma unroll
            for (int rr = 0; rr < nt; rr++) {
#pragma unroll
                for (int cc = 0; cc < nt; cc++) {
                    sm[r + rr * ts][c + cc * ts] = min(sm[r + rr * ts][c + cc * ts], k_k_sm[r + rr * ts][b] + sm[b][c + cc * ts]);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int rr = 0; rr < nt; rr++) {
#pragma unroll
            for (int cc = 0; cc < nt; cc++) {
                _ref(k, i, r + rr * ts, c + cc * ts) = sm[r + rr * ts][c + cc * ts];
            }
        }
    }
}
template <int nt, int ts, int bs>
__global__ void proc_3_glob(int *blk_dist, int s_i, int s_j, int k, int blk_pitch, int nblocks) {
    __shared__ int i_k_sm[bs][bs];
    __shared__ int k_j_sm[bs][bs];

    int i = s_i + blockIdx.y;
    int j = s_j + blockIdx.x;
    int r = threadIdx.y;
    int c = threadIdx.x;
    int loc[nt][nt];

    if (i >= k)
        i++;
    if (j >= k)
        j++;

#pragma unroll
    for (int rr = 0; rr < nt; rr++) {
#pragma unroll
        for (int cc = 0; cc < nt; cc++) {
            i_k_sm[r + rr * ts][c + cc * ts] = _ref(i, k, r + rr * ts, c + cc * ts);
        }
    }
#pragma unroll
    for (int rr = 0; rr < nt; rr++) {
#pragma unroll
        for (int cc = 0; cc < nt; cc++) {
            k_j_sm[r + rr * ts][c + cc * ts] = _ref(k, j, r + rr * ts, c + cc * ts);
        }
    }
    __syncthreads();
#pragma unroll
    for (int rr = 0; rr < nt; rr++) {
#pragma unroll
        for (int cc = 0; cc < nt; cc++) {
            loc[rr][cc] = _ref(i, j, r + rr * ts, c + cc * ts);
        }
    }

#pragma unroll
    for (int b = 0; b < bs; b++) {
#pragma unroll
        for (int rr = 0; rr < nt; rr++) {
#pragma unroll
            for (int cc = 0; cc < nt; cc++) {
                loc[rr][cc] = min(loc[rr][cc], i_k_sm[r + rr * ts][b] + k_j_sm[b][c + cc * ts]);
            }
        }
    }
#pragma unroll
    for (int rr = 0; rr < nt; rr++) {
#pragma unroll
        for (int cc = 0; cc < nt; cc++) {
            _ref(i, j, r + rr * ts, c + cc * ts) = loc[rr][cc];
        }
    }
}
template <int nt, int ts, int bs>
__global__ void init_dist(int *blk_dist, int blk_pitch, int nblocks) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    blk_dist[blk_idx<nt, ts, bs>(r, c, blk_pitch, nblocks)] = (r != c) * int_max;
}
template <int nt, int ts, int bs>
__global__ void build_dist(int *edge, int E, int *blk_dist, int blk_pitch, int nblocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < E) {
        int src = *(edge + idx * 3);
        int dst = *(edge + idx * 3 + 1);
        int w = *(edge + idx * 3 + 2);
        blk_dist[blk_idx<nt, ts, bs>(src, dst, blk_pitch, nblocks)] = w;
    }
}
template <int nt, int ts, int bs>
__global__ void copy_dist(int *blk_dist, int blk_pitch, int *dist, int pitch, int nblocks) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    dist[r * pitch + c] = blk_dist[blk_idx<nt, ts, bs>(r, c, blk_pitch, nblocks)];
}

#define _ref_blk(i, j, r, c) blk_dist[i * bs * pitch + j * bs + (r) * pitch + c]
template <int nt, int ts, int bs>
__global__ void proc_1_blk_glob(int *blk_dist, int k, int pitch) {
    __shared__ int k_k_sm[bs][bs];

    int r = threadIdx.y;
    int c = threadIdx.x;

#pragma unroll
    for (int rr = 0; rr < nt; rr++) {
#pragma unroll
        for (int cc = 0; cc < nt; cc++) {
            k_k_sm[r + rr * ts][c + cc * ts] = _ref_blk(k, k, r + rr * ts, c + cc * ts);
        }
    }
    __syncthreads();

#pragma unroll
    for (int b = 0; b < bs; b++) {
#pragma unroll
        for (int rr = 0; rr < nt; rr++) {
#pragma unroll
            for (int cc = 0; cc < nt; cc++) {
                k_k_sm[r + rr * ts][c + cc * ts] = min(k_k_sm[r + rr * ts][c + cc * ts], k_k_sm[r + rr * ts][b] + k_k_sm[b][c + cc * ts]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int rr = 0; rr < nt; rr++) {
#pragma unroll
        for (int cc = 0; cc < nt; cc++) {
            _ref_blk(k, k, r + rr * ts, c + cc * ts) = k_k_sm[r + rr * ts][c + cc * ts];
        }
    }
}
template <int nt, int ts, int bs>
__global__ void proc_2_blk_glob(int *blk_dist, int s, int k, int pitch) {
    __shared__ int k_k_sm[bs][bs];
    __shared__ int sm[bs][bs];

    int i = s + blockIdx.x;
    int r = threadIdx.y;
    int c = threadIdx.x;

    if (i >= k)
        i++;

#pragma unroll
    for (int rr = 0; rr < nt; rr++) {
#pragma unroll
        for (int cc = 0; cc < nt; cc++) {
            k_k_sm[r + rr * ts][c + cc * ts] = _ref_blk(k, k, r + rr * ts, c + cc * ts);
        }
    }
    if (blockIdx.y == 0) {
        /* rows */
#pragma unroll
        for (int rr = 0; rr < nt; rr++) {
#pragma unroll
            for (int cc = 0; cc < nt; cc++) {
                sm[r + rr * ts][c + cc * ts] = _ref_blk(i, k, r + rr * ts, c + cc * ts);
            }
        }
        __syncthreads();

#pragma unroll
        for (int b = 0; b < bs; b++) {
#pragma unroll
            for (int rr = 0; rr < nt; rr++) {
#pragma unroll
                for (int cc = 0; cc < nt; cc++) {
                    sm[r + rr * ts][c + cc * ts] = min(sm[r + rr * ts][c + cc * ts], sm[r + rr * ts][b] + k_k_sm[b][c + cc * ts]);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int rr = 0; rr < nt; rr++) {
#pragma unroll
            for (int cc = 0; cc < nt; cc++) {
                _ref_blk(i, k, r + rr * ts, c + cc * ts) = sm[r + rr * ts][c + cc * ts];
            }
        }
    } else {
        /* cols */
#pragma unroll
        for (int rr = 0; rr < nt; rr++) {
#pragma unroll
            for (int cc = 0; cc < nt; cc++) {
                sm[r + rr * ts][c + cc * ts] = _ref_blk(k, i, r + rr * ts, c + cc * ts);
            }
        }
        __syncthreads();

#pragma unroll
        for (int b = 0; b < bs; b++) {
#pragma unroll
            for (int rr = 0; rr < nt; rr++) {
#pragma unroll
                for (int cc = 0; cc < nt; cc++) {
                    sm[r + rr * ts][c + cc * ts] = min(sm[r + rr * ts][c + cc * ts], k_k_sm[r + rr * ts][b] + sm[b][c + cc * ts]);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int rr = 0; rr < nt; rr++) {
#pragma unroll
            for (int cc = 0; cc < nt; cc++) {
                _ref_blk(k, i, r + rr * ts, c + cc * ts) = sm[r + rr * ts][c + cc * ts];
            }
        }
    }
}
template <int nt, int ts, int bs>
__global__ void proc_3_blk_glob(int *blk_dist, int s_i, int s_j, int k, int pitch) {
    __shared__ int i_k_sm[bs][bs];
    __shared__ int k_j_sm[bs][bs];

    int i = s_i + blockIdx.y;
    int j = s_j + blockIdx.x;
    int r = threadIdx.y;
    int c = threadIdx.x;
    int loc[nt][nt];

    if (i >= k)
        i++;
    if (j >= k)
        j++;

#pragma unroll
    for (int rr = 0; rr < nt; rr++) {
#pragma unroll
        for (int cc = 0; cc < nt; cc++) {
            i_k_sm[r + rr * ts][c + cc * ts] = _ref_blk(i, k, r + rr * ts, c + cc * ts);
        }
    }
#pragma unroll
    for (int rr = 0; rr < nt; rr++) {
#pragma unroll
        for (int cc = 0; cc < nt; cc++) {
            k_j_sm[r + rr * ts][c + cc * ts] = _ref_blk(k, j, r + rr * ts, c + cc * ts);
        }
    }
    __syncthreads();
#pragma unroll
    for (int rr = 0; rr < nt; rr++) {
#pragma unroll
        for (int cc = 0; cc < nt; cc++) {
            loc[rr][cc] = _ref_blk(i, j, r + rr * ts, c + cc * ts);
        }
    }

#pragma unroll
    for (int b = 0; b < bs; b++) {
#pragma unroll
        for (int rr = 0; rr < nt; rr++) {
#pragma unroll
            for (int cc = 0; cc < nt; cc++) {
                loc[rr][cc] = min(loc[rr][cc], i_k_sm[r + rr * ts][b] + k_j_sm[b][c + cc * ts]);
            }
        }
    }
#pragma unroll
    for (int rr = 0; rr < nt; rr++) {
#pragma unroll
        for (int cc = 0; cc < nt; cc++) {
            _ref_blk(i, j, r + rr * ts, c + cc * ts) = loc[rr][cc];
        }
    }
}
__global__ void init_blk_dist(int *blk_dist, int pitch) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    blk_dist[r * pitch + c] = (r != c) * int_max;
}
__global__ void build_blk_dist(int *edge, int E, int *blk_dist, int pitch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < E) {
        int src = *(edge + idx * 3);
        int dst = *(edge + idx * 3 + 1);
        int w = *(edge + idx * 3 + 2);
        blk_dist[src * pitch + dst] = w;
    }
}