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

constexpr int VEC_SIZE = 4;
constexpr int VEC_EXPAND = 4;
constexpr int N_STRIDE = 4;
constexpr int STRIDE = VEC_SIZE * VEC_EXPAND;
constexpr int BLOCK_SIZE = STRIDE * N_STRIDE;
constexpr float INV_BLOCK_SIZE = 1.0f / BLOCK_SIZE;
constexpr int INF = ((1 << 30) - 1);

struct edge_t {
    int src;
    int dst;
    int w;
};

__always_inline int calc_blk_idx(int r, int c, int nblocks);

__always_inline void proc(int *blk_dist, int s_i, int e_i, int s_j, int e_j, int k, int nblocks, int ncpus) __attribute((optimize("O3")));

int main(int argc, char **argv) {
    assert(argc == 3);

    char *input_filename = argv[1];
    char *output_filename = argv[2];
    FILE *input_file;
    FILE *output_file;
    int ncpus = omp_get_max_threads();
    int V, E;
    edge_t *edge;
    int *dist;
    int VP;
    int nblocks;
    int *blk_dist;

    // Read input
    input_file = fopen(input_filename, "rb");
    fread(&V, sizeof(int), 1, input_file);
    fread(&E, sizeof(int), 1, input_file);
    edge = (edge_t *)malloc(sizeof(edge_t) * E);
    fread(edge, sizeof(edge_t), E, input_file);
    dist = (int *)malloc(sizeof(int) * V * V);
#ifndef NO_FINALIZE
    fclose(input_file);
#endif

    // Initialize
    nblocks = (int)ceilf(float(V) * INV_BLOCK_SIZE);
    VP = nblocks * BLOCK_SIZE;
    blk_dist = (int *)malloc(sizeof(int) * VP * VP);
#pragma omp parallel for num_threads(ncpus) schedule(static) default(shared) collapse(2)
    for (int i = 0; i < VP; i++) {
        for (int j = 0; j < VP; j++) {
            if (i == j) {
                blk_dist[calc_blk_idx(i, j, nblocks)] = 0;
            } else {
                blk_dist[calc_blk_idx(i, j, nblocks)] = INF;
            }
        }
    }
#pragma omp parallel for num_threads(ncpus) schedule(static) default(shared)
    for (int i = 0; i < E; i++) {
        blk_dist[calc_blk_idx(edge[i].src, edge[i].dst, nblocks)] = edge[i].w;
    }

    // Blocked Floyd-Warshall
    for (int k = 0; k < nblocks; k++) {
        // Phase 1
        proc(blk_dist, k, k + 1, k, k + 1, k, nblocks, ncpus);
        // Phase 2
        proc(blk_dist, k, k + 1, 0, k, k, nblocks, ncpus);
        proc(blk_dist, k, k + 1, k + 1, nblocks, k, nblocks, ncpus);
        proc(blk_dist, 0, k, k, k + 1, k, nblocks, ncpus);
        proc(blk_dist, k + 1, nblocks, k, k + 1, k, nblocks, ncpus);
        // Phase 3
        proc(blk_dist, 0, k, 0, k, k, nblocks, ncpus);
        proc(blk_dist, 0, k, k + 1, nblocks, k, nblocks, ncpus);
        proc(blk_dist, k + 1, nblocks, 0, k, k, nblocks, ncpus);
        proc(blk_dist, k + 1, nblocks, k + 1, nblocks, k, nblocks, ncpus);
    }

    // Copy output to dist
#pragma omp parallel for num_threads(ncpus) schedule(static) default(shared) collapse(2)
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            int blk_idx = calc_blk_idx(i, j, nblocks);
            dist[i * V + j] = blk_dist[blk_idx] > INF ? INF : blk_dist[blk_idx];
        }
    }

    // Write output
    output_file = fopen(output_filename, "w");
    fwrite(dist, sizeof(int), V * V, output_file);
#ifndef NO_FINALIZE
    fclose(output_file);
#endif

    // Clean up
#ifndef NO_FINALIZE
    free(edge);
    free(dist);
    free(blk_dist);
#endif
    return 0;
}

__always_inline int calc_blk_idx(int r, int c, int nblocks) {
    return (int(r * INV_BLOCK_SIZE) * nblocks + int(c * INV_BLOCK_SIZE)) * (BLOCK_SIZE * BLOCK_SIZE) + (r % BLOCK_SIZE) * BLOCK_SIZE + (c % BLOCK_SIZE);
}

__always_inline void proc(int *blk_dist, int s_i, int e_i, int s_j, int e_j, int k, int nblocks, int ncpus) {
#pragma omp parallel for num_threads(ncpus) schedule(static) default(shared) collapse(2)
    for (int i = s_i; i < e_i; i++) {
        for (int j = s_j; j < e_j; j++) {
            int *ik_ptr = blk_dist + (i * nblocks + k) * (BLOCK_SIZE * BLOCK_SIZE);
            int *ij_ptr = blk_dist + (i * nblocks + j) * (BLOCK_SIZE * BLOCK_SIZE);
            int *kj_ptr = blk_dist + (k * nblocks + j) * (BLOCK_SIZE * BLOCK_SIZE);
            for (int b = 0; b < BLOCK_SIZE; b++) {
                for (int r = 0; r < BLOCK_SIZE; r++) {
#ifdef MANUAL_SIMD
                    __m128i vec_ik = _mm_set1_epi32(ik_ptr[r * BLOCK_SIZE + b]);
                    __m128i vec_kj[VEC_EXPAND];
                    __m128i vec_ij[VEC_EXPAND];
                    __m128i vec_sum[VEC_EXPAND];
                    __m128i vec_min[VEC_EXPAND];
                    for (int c = 0; c < BLOCK_SIZE; c += STRIDE) {
#pragma GCC unroll VEC_EXPAND
                        for (int v = 0; v < VEC_EXPAND; v++) {
                            vec_kj[v] = _mm_loadu_si128((__m128i *)(kj_ptr + b * BLOCK_SIZE + c + v * VEC_SIZE));
                        }
#pragma GCC unroll VEC_EXPAND
                        for (int v = 0; v < VEC_EXPAND; v++) {
                            vec_sum[v] = _mm_add_epi32(vec_ik, vec_kj[v]);
                        }
#pragma GCC unroll VEC_EXPAND
                        for (int v = 0; v < VEC_EXPAND; v++) {
                            vec_ij[v] = _mm_loadu_si128((__m128i *)(ij_ptr + r * BLOCK_SIZE + c + v * VEC_SIZE));
                        }
#pragma GCC unroll VEC_EXPAND
                        for (int v = 0; v < VEC_EXPAND; v++) {
                            vec_min[v] = _mm_min_epi32(vec_ij[v], vec_sum[v]);
                        }
#pragma GCC unroll VEC_EXPAND
                        for (int v = 0; v < VEC_EXPAND; v++) {
                            _mm_storeu_si128((__m128i *)(ij_ptr + r * BLOCK_SIZE + c + v * VEC_SIZE), vec_min[v]);
                        }
                    }
#else
#pragma GCC ivdep
                    for (int c = 0; c < BLOCK_SIZE; c++) {
                        ij_ptr[r * BLOCK_SIZE + c] = std::min(ij_ptr[r * BLOCK_SIZE + c], ik_ptr[r * BLOCK_SIZE + b] + kj_ptr[b * BLOCK_SIZE + c]);
                    }
#endif
                }
            }
        }
    }
}