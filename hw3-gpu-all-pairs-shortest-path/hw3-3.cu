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
    if ((condition) != cudaSuccess) {                                                                             \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__); \
        exit(1);                                                                                                  \
    }

constexpr int int_max = ((1 << 30) - 1);

double get_timestamp();

template <int nt, int ts, int bs>
int solve(int argc, char **argv);

template <int nt, int ts, int bs>
__global__ void proc_1_glob(int *blk_dist, int k, int pitch);
template <int nt, int ts, int bs>
__global__ void proc_2_glob(int *blk_dist, int s, int k, int pitch);
template <int nt, int ts, int bs>
__global__ void proc_3_glob(int *blk_dist, int s_i, int s_j, int k, int pitch);

__global__ void init_dist(int *blk_dist, int pitch);
__global__ void build_dist(int *edge, int E, int *blk_dist, int pitch);

int main(int argc, char **argv) {
    return solve<3, 26, 78>(argc, argv);
}

double get_timestamp() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

template <int nt, int ts, int bs>
int solve(int argc, char **argv) {
    assert(argc == 3);

    char *input_filename = argv[1];
    char *output_filename = argv[2];
    FILE *input_file;
    FILE *output_file;
    int ncpus = omp_get_max_threads();
    int V, E;
    int *edge;
    int *edge_dev[2];
    int *dist;
    int *dist_dev[2];
    int VP;
    int nblocks;
    double start_ts;

    start_ts = get_timestamp();

    /* input */
    input_file = fopen(input_filename, "rb");
    assert(input_file);
    fread(&V, sizeof(int), 1, input_file);
    fread(&E, sizeof(int), 1, input_file);

    nblocks = (int)ceilf(float(V) / bs);
    VP = nblocks * bs;

    edge = (int *)malloc(sizeof(int) * 3 * E);
    fread(edge, sizeof(int), 3 * E, input_file);
    dist = (int *)malloc(sizeof(int) * V * V);
    fclose(input_file);

    /* calculate */
#pragma omp parallel num_threads(2) default(shared)
    {
        int tid = omp_get_thread_num();
        int peerid = !tid;
        int start, range;
        if (tid == 0) {
            start = 0;
            range = nblocks / 2;
        } else {
            start = nblocks / 2;
            range = nblocks - start;
        }

        cudaSetDevice(tid);

        // Enable peer access
        int can_access_peer = 0;
        cudaDeviceCanAccessPeer(&can_access_peer, tid, peerid);
        if (can_access_peer) {
            cudaDeviceEnablePeerAccess(peerid, 0);
        }

        cudaMalloc(&edge_dev[tid], sizeof(int) * 3 * E);
        cudaMalloc(&dist_dev[tid], sizeof(int) * VP * VP);

        cudaMemcpy(edge_dev[tid], edge, sizeof(int) * 3 * E, cudaMemcpyHostToDevice);
#pragma omp barrier

        init_dist<<<dim3(VP / ts, VP / ts), dim3(ts, ts)>>>(dist_dev[tid], VP);
        build_dist<<<(int)ceilf((float)E / (ts * ts)), ts * ts>>>(edge_dev[tid], E, dist_dev[tid], VP);
        cudaFree(edge_dev[tid]);

        dim3 blk(ts, ts);
        for (int k = 0; k < nblocks; k++) {
            /* Sync */
            if (range > 0 && k >= start && k < start + range)
                cudaMemcpy2D(
                    dist_dev[peerid] + VP * bs * k, sizeof(int) * VP,
                    dist_dev[tid] + VP * bs * k, sizeof(int) * VP,
                    sizeof(int) * VP, bs, cudaMemcpyDeviceToDevice);
#pragma omp barrier
            /* Phase 1 */
            proc_1_glob<nt, ts, bs><<<1, blk>>>(dist_dev[tid], k, VP);
            /* Phase 2 */
            if (nblocks - 1 > 0)
                proc_2_glob<nt, ts, bs><<<dim3(nblocks - 1, 2), blk>>>(dist_dev[tid], 0, k, VP);
            /* Phase 3 */
            if (nblocks - 1 > 0 && range > 0)
                proc_3_glob<nt, ts, bs><<<dim3(nblocks - 1, range), blk>>>(dist_dev[tid], start, 0, k, VP);
        }
        if (tid == 0) {
            cudaHostRegister(dist, sizeof(int) * V * V, cudaHostRegisterDefault);
#pragma omp barrier
            cudaMemcpy2D(dist, sizeof(int) * V, dist_dev[tid], sizeof(int) * VP, sizeof(int) * V, V, cudaMemcpyDeviceToHost);
            cudaHostUnregister(dist);
        } else {
            if (range > 0)
                cudaMemcpy2D(
                    dist_dev[peerid] + VP * bs * start, sizeof(int) * VP,
                    dist_dev[tid] + VP * bs * start, sizeof(int) * VP,
                    sizeof(int) * VP, bs * range, cudaMemcpyDeviceToDevice);
#pragma omp barrier
        }
        cudaFree(dist_dev[tid]);
    }

    /* output */
    output_file = fopen(output_filename, "w");
    assert(output_file);
    fwrite(dist, 1, sizeof(int) * V * V, output_file);
    fclose(output_file);

#ifdef PROFILING
    fprintf(stderr, "Took: %lf\n", get_timestamp() - start_ts);
#endif  // PROFILING

    /* finalize */
    free(edge);
    free(dist);
    return 0;
}

#define _ref(i, j, r, c) blk_dist[i * bs * pitch + j * bs + (r) * pitch + c]
template <int nt, int ts, int bs>
__global__ void proc_1_glob(int *blk_dist, int k, int pitch) {
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
__global__ void proc_2_glob(int *blk_dist, int s, int k, int pitch) {
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
__global__ void proc_3_glob(int *blk_dist, int s_i, int s_j, int k, int pitch) {
    __shared__ int i_k_sm[bs][bs];
    __shared__ int k_j_sm[bs][bs];

    int i = s_i + blockIdx.y;
    int j = s_j + blockIdx.x;
    int r = threadIdx.y;
    int c = threadIdx.x;
    int loc[nt][nt];

    if (i == k)
        return;
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
__global__ void init_dist(int *blk_dist, int pitch) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    blk_dist[r * pitch + c] = (r != c) * int_max;
}
__global__ void build_dist(int *edge, int E, int *blk_dist, int pitch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < E) {
        int src = *(edge + idx * 3);
        int dst = *(edge + idx * 3 + 1);
        int w = *(edge + idx * 3 + 2);
        blk_dist[src * pitch + dst] = w;
    }
}