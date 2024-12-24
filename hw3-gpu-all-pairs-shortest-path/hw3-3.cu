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

constexpr int TILE = 26;
constexpr int block_size = 78;
constexpr int div_block = 3;
constexpr int int_max = ((1 << 30) - 1);

__global__ void proc_1_glob(int *blk_dist, int k, int pitch);
__global__ void proc_2_glob(int *blk_dist, int s, int k, int pitch);
__global__ void proc_3_glob(int *blk_dist, int s_i, int s_j, int k, int pitch);

__global__ void init_dist(int *blk_dist, int pitch);
__global__ void build_dist(int *edge, int E, int *blk_dist, int pitch);

int main(int argc, char **argv) {
    assert(argc == 3);

    char *input_filename = argv[1];
    char *output_filename = argv[2];
    FILE *input_file;
    FILE *output_file;
    int ncpus = omp_get_max_threads();
    int device_cnt;
    int V, E;
    int *edge;
    int *edge_dev[2];
    int *dist;
    int *dist_dev[2];
    int VP;
    int nblocks;

    cudaGetDeviceCount(&device_cnt);

    /* input */
    input_file = fopen(input_filename, "rb");
    assert(input_file);
    fread(&V, sizeof(int), 1, input_file);
    fread(&E, sizeof(int), 1, input_file);
    edge = (int *)malloc(sizeof(int) * 3 * E);
    fread(edge, sizeof(int), 3 * E, input_file);
    dist = (int *)malloc(sizeof(int) * V * V);
    fclose(input_file);

    /* calculate */
    nblocks = (int)ceilf(float(V) / block_size);
    VP = nblocks * block_size;
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
        cudaMalloc(&edge_dev[tid], sizeof(int) * 3 * E);
        cudaMalloc(&dist_dev[tid], sizeof(int) * VP * VP);

        cudaMemcpy(edge_dev[tid], edge, sizeof(int) * 3 * E, cudaMemcpyDefault);
#pragma omp barrier

        init_dist<<<dim3(VP / TILE, VP / TILE), dim3(TILE, TILE)>>>(dist_dev[tid], VP);
        build_dist<<<(int)ceilf((float)E / (TILE * TILE)), TILE * TILE>>>(edge_dev[tid], E, dist_dev[tid], VP);
        cudaFree(edge_dev[tid]);

        dim3 blk(TILE, TILE);
        for (int k = 0, nk = nblocks - 1; k < nblocks; k++, nk--) {
            /* Sync */
            if (range > 0 && k >= start && k < start + range)
                cudaMemcpy2D(
                    dist_dev[peerid] + VP * block_size * k, sizeof(int) * VP,
                    dist_dev[tid] + VP * block_size * k, sizeof(int) * VP,
                    sizeof(int) * VP, block_size, cudaMemcpyDefault);
#pragma omp barrier
            /* Phase 1 */
            proc_1_glob<<<1, blk>>>(dist_dev[tid], k, VP);
            /* Phase 2 */
            if (nblocks - 1 > 0)
                proc_2_glob<<<dim3(nblocks - 1, 2), blk>>>(dist_dev[tid], 0, k, VP);
            /* Phase 3 */
            if (nblocks - 1 > 0 && range > 0)
                proc_3_glob<<<dim3(nblocks - 1, range), blk>>>(dist_dev[tid], start, 0, k, VP);
        }
        if (tid == 0) {
            cudaHostRegister(dist, sizeof(int) * V * V, cudaHostRegisterDefault);
#pragma omp barrier
            cudaMemcpy2D(dist, sizeof(int) * V, dist_dev[tid], sizeof(int) * VP, sizeof(int) * V, V, cudaMemcpyDefault);
            cudaHostUnregister(dist);
        } else {
            if (range > 0)
                cudaMemcpy2D(
                    dist_dev[peerid] + VP * block_size * start, sizeof(int) * VP,
                    dist_dev[tid] + VP * block_size * start, sizeof(int) * VP,
                    sizeof(int) * VP, block_size * range, cudaMemcpyDefault);
#pragma omp barrier
        }
        cudaFree(dist_dev[tid]);
    }

    /* output */
    output_file = fopen(output_filename, "w");
    assert(output_file);
    fwrite(dist, 1, sizeof(int) * V * V, output_file);
    fclose(output_file);

    /* finalize */
    free(edge);
    free(dist);
    return 0;
}

#define _ref(i, j, r, c) blk_dist[i * block_size * pitch + j * block_size + (r) * pitch + c]
__global__ void proc_1_glob(int *blk_dist, int k, int pitch) {
    __shared__ int k_k_sm[block_size][block_size];

    int r = threadIdx.y;
    int c = threadIdx.x;

#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            k_k_sm[r + rr * TILE][c + cc * TILE] = _ref(k, k, r + rr * TILE, c + cc * TILE);
        }
    }
    __syncthreads();

#pragma unroll
    for (int b = 0; b < block_size; b++) {
#pragma unroll
        for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
            for (int cc = 0; cc < div_block; cc++) {
                k_k_sm[r + rr * TILE][c + cc * TILE] = min(k_k_sm[r + rr * TILE][c + cc * TILE], k_k_sm[r + rr * TILE][b] + k_k_sm[b][c + cc * TILE]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            _ref(k, k, r + rr * TILE, c + cc * TILE) = k_k_sm[r + rr * TILE][c + cc * TILE];
        }
    }
}
__global__ void proc_2_glob(int *blk_dist, int s, int k, int pitch) {
    __shared__ int k_k_sm[block_size][block_size];
    __shared__ int sm[block_size][block_size];

    int i = s + blockIdx.x;
    int r = threadIdx.y;
    int c = threadIdx.x;

    if (i >= k)
        i++;

#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            k_k_sm[r + rr * TILE][c + cc * TILE] = _ref(k, k, r + rr * TILE, c + cc * TILE);
        }
    }
    if (blockIdx.y == 0) {
        /* rows */
#pragma unroll
        for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
            for (int cc = 0; cc < div_block; cc++) {
                sm[r + rr * TILE][c + cc * TILE] = _ref(i, k, r + rr * TILE, c + cc * TILE);
            }
        }
        __syncthreads();

#pragma unroll
        for (int b = 0; b < block_size; b++) {
#pragma unroll
            for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
                for (int cc = 0; cc < div_block; cc++) {
                    sm[r + rr * TILE][c + cc * TILE] = min(sm[r + rr * TILE][c + cc * TILE], sm[r + rr * TILE][b] + k_k_sm[b][c + cc * TILE]);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
            for (int cc = 0; cc < div_block; cc++) {
                _ref(i, k, r + rr * TILE, c + cc * TILE) = sm[r + rr * TILE][c + cc * TILE];
            }
        }
    } else {
        /* cols */
#pragma unroll
        for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
            for (int cc = 0; cc < div_block; cc++) {
                sm[r + rr * TILE][c + cc * TILE] = _ref(k, i, r + rr * TILE, c + cc * TILE);
            }
        }
        __syncthreads();

#pragma unroll
        for (int b = 0; b < block_size; b++) {
#pragma unroll
            for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
                for (int cc = 0; cc < div_block; cc++) {
                    sm[r + rr * TILE][c + cc * TILE] = min(sm[r + rr * TILE][c + cc * TILE], k_k_sm[r + rr * TILE][b] + sm[b][c + cc * TILE]);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
            for (int cc = 0; cc < div_block; cc++) {
                _ref(k, i, r + rr * TILE, c + cc * TILE) = sm[r + rr * TILE][c + cc * TILE];
            }
        }
    }
}
__global__ void proc_3_glob(int *blk_dist, int s_i, int s_j, int k, int pitch) {
    __shared__ int i_k_sm[block_size][block_size];
    __shared__ int k_j_sm[block_size][block_size];

    int i = s_i + blockIdx.y;
    int j = s_j + blockIdx.x;
    int r = threadIdx.y;
    int c = threadIdx.x;
    int loc[div_block][div_block];

    if (i == k)
        return;
    if (j >= k)
        j++;

#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            i_k_sm[r + rr * TILE][c + cc * TILE] = _ref(i, k, r + rr * TILE, c + cc * TILE);
        }
    }
#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            k_j_sm[r + rr * TILE][c + cc * TILE] = _ref(k, j, r + rr * TILE, c + cc * TILE);
        }
    }
    __syncthreads();
#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            loc[rr][cc] = _ref(i, j, r + rr * TILE, c + cc * TILE);
        }
    }

#pragma unroll
    for (int b = 0; b < block_size; b++) {
#pragma unroll
        for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
            for (int cc = 0; cc < div_block; cc++) {
                loc[rr][cc] = min(loc[rr][cc], i_k_sm[r + rr * TILE][b] + k_j_sm[b][c + cc * TILE]);
            }
        }
    }
#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            _ref(i, j, r + rr * TILE, c + cc * TILE) = loc[rr][cc];
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