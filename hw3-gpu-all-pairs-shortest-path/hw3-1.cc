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

constexpr int BLOCK_SIZE = 40;
constexpr float INV_BLOCK_SIZE = 1.0f / BLOCK_SIZE;
constexpr int INF = ((1 << 30) - 1);

struct edge_t {
    int src;
    int dst;
    int w;
};

inline int calc_blk_idx(int r, int c, int nblocks);

inline void proc(int *blk_dist, int s_i, int e_i, int s_j, int e_j, int k, int nblocks, int ncpus);

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
    fclose(input_file);

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
    fclose(output_file);

    // Clean up
    free(edge);
    free(dist);
    free(blk_dist);
    return 0;
}

inline int calc_blk_idx(int r, int c, int nblocks) {
    return (int(r * INV_BLOCK_SIZE) * nblocks + int(c * INV_BLOCK_SIZE)) * (BLOCK_SIZE * BLOCK_SIZE) + (r % BLOCK_SIZE) * BLOCK_SIZE + (c % BLOCK_SIZE);
}

inline void proc(int *blk_dist, int s_i, int e_i, int s_j, int e_j, int k, int nblocks, int ncpus) {
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
                    for (int c = 0; c < BLOCK_SIZE; c += 4) {
                        __m128i vec_kj = _mm_loadu_si128((__m128i *)(kj_ptr + b * BLOCK_SIZE + c));
                        __m128i vec_ij = _mm_loadu_si128((__m128i *)(ij_ptr + r * BLOCK_SIZE + c));
                        __m128i vec_sum = _mm_add_epi32(vec_ik, vec_kj);
                        __m128i vec_min = _mm_min_epi32(vec_ij, vec_sum);
                        _mm_storeu_si128((__m128i *)(ij_ptr + r * BLOCK_SIZE + c), vec_min);
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