
#include <hip/hip_runtime.h>
#include <immintrin.h>
#include <omp.h>
#include <pthread.h>

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
    if ((condition) != hipSuccess) {                                                                             \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", hipGetErrorString(hipGetLastError()), __FILE__, __LINE__); \
        exit(1);                                                                                                  \
    }

constexpr int int_max = ((1 << 30) - 1);

template <int nt, int ts, int bs>
int solve(int argc, char **argv);

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
    return solve<3, 26, 78>(argc, argv);
}

template <int nt, int ts, int bs>
int solve(int argc, char **argv) {
    assert(argc == 3);

    char *input_filename = argv[1];
    char *output_filename = argv[2];
    FILE *input_file;
    FILE *output_file;
    int ncpus = omp_get_max_threads();
    int device_cnt;
    int V, E;
    int *edge;
    int *dist;
    int VP;
    int nblocks;
    hipDeviceProp_t deviceProp;

    hipGetDeviceCount(&device_cnt);
    hipSetDevice(0);
    hipGetDeviceProperties(&deviceProp, 0);

    /* input */
    input_file = fopen(input_filename, "rb");
    assert(input_file);
    fread(&V, sizeof(int), 1, input_file);
    fread(&E, sizeof(int), 1, input_file);
    edge = (int *)malloc(sizeof(int) * 3 * E);
    fread(edge, sizeof(int), 3 * E, input_file);
    dist = (int *)malloc(sizeof(int) * V * V);
    fclose(input_file);

    nblocks = (int)ceilf(float(V) / bs);
    VP = nblocks * bs;

    /* calculate */
    if ((size_t)VP * VP * 2 + 2 * 3 * E + V * V <= deviceProp.totalGlobalMem / 4) {
        int *edge_dev;
        int *dist_dev;
        int *blk_dist_dev;
        size_t blk_pitch;

        hipHostRegister(edge, sizeof(int) * 3 * E, hipHostRegisterReadOnly);
        hipMalloc(&edge_dev, sizeof(int) * 3 * E);
        hipHostRegister(dist, sizeof(int) * V * V, hipHostRegisterDefault);
        hipMalloc(&blk_dist_dev, sizeof(int) * bs * bs * nblocks * nblocks);
        blk_pitch = bs * bs;

        hipMemcpy(edge_dev, edge, sizeof(int) * 3 * E, hipMemcpyDefault);

        init_dist<nt, ts, bs><<<dim3(VP / ts, VP / ts), dim3(ts, ts)>>>(blk_dist_dev, blk_pitch, nblocks);
        build_dist<nt, ts, bs><<<(int)ceilf((float)E / (ts * ts)), ts * ts>>>(edge_dev, E, blk_dist_dev, blk_pitch, nblocks);
        hipFree(edge_dev);

        dim3 blk(ts, ts);
        for (int k = 0, nk = nblocks - 1; k < nblocks; k++, nk--) {
            /* Phase 1 */
            proc_1_glob<nt, ts, bs><<<1, blk>>>(blk_dist_dev, k, blk_pitch, nblocks);
            /* Phase 2 */
            proc_2_glob<nt, ts, bs><<<dim3(nblocks - 1, 2), blk>>>(blk_dist_dev, 0, k, blk_pitch, nblocks);
            /* Phase 3 */
            proc_3_glob<nt, ts, bs><<<dim3(nblocks - 1, nblocks - 1), blk>>>(blk_dist_dev, 0, 0, k, blk_pitch, nblocks);
        }

        hipMalloc(&dist_dev, sizeof(int) * VP * VP);
        copy_dist<nt, ts, bs><<<dim3(VP / ts, VP / ts), dim3(ts, ts)>>>(blk_dist_dev, blk_pitch, dist_dev, VP, nblocks);
        hipMemcpy2D(dist, sizeof(int) * V, dist_dev, sizeof(int) * VP, sizeof(int) * V, V, hipMemcpyDefault);

        hipDeviceSynchronize();

        hipFree(blk_dist_dev);
        hipFree(dist_dev);
    } else {
        int *edge_dev;
        int *dist_dev;

        hipHostRegister(edge, sizeof(int) * 3 * E, hipHostRegisterReadOnly);
        hipMalloc(&edge_dev, sizeof(int) * 3 * E);
        hipHostRegister(dist, sizeof(int) * V * V, hipHostRegisterDefault);
        hipMalloc(&dist_dev, sizeof(int) * VP * VP);

        hipMemcpy(edge_dev, edge, sizeof(int) * 3 * E, hipMemcpyDefault);

        init_blk_dist<<<dim3(VP / ts, VP / ts), dim3(ts, ts)>>>(dist_dev, VP);
        build_blk_dist<<<(int)ceilf((float)E / (ts * ts)), ts * ts>>>(edge_dev, E, dist_dev, VP);
        hipFree(edge_dev);

        dim3 blk(ts, ts);
        for (int k = 0, nk = nblocks - 1; k < nblocks; k++, nk--) {
            /* Phase 1 */
            proc_1_blk_glob<nt, ts, bs><<<1, blk>>>(dist_dev, k, VP);
            /* Phase 2 */
            proc_2_blk_glob<nt, ts, bs><<<dim3(nblocks - 1, 2), blk>>>(dist_dev, 0, k, VP);
            /* Phase 3 */
            proc_3_blk_glob<nt, ts, bs><<<dim3(nblocks - 1, nblocks - 1), blk>>>(dist_dev, 0, 0, k, VP);
        }

        hipMemcpy2D(dist, sizeof(int) * V, dist_dev, sizeof(int) * VP, sizeof(int) * V, V, hipMemcpyDefault);

        hipDeviceSynchronize();
        hipFree(dist_dev);
    }
    /* output */
    output_file = fopen(output_filename, "w");
    assert(output_file);
    fwrite(dist, 1, sizeof(int) * V * V, output_file);
    fclose(output_file);

    free(edge);
    free(dist);
    return 0;
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