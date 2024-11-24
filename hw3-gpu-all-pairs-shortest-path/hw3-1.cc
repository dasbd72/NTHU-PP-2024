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

constexpr int block_size = 40;
constexpr float inv_block_size = 1.0f / block_size;
constexpr int infinity = ((1 << 30) - 1);

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
    nblocks = (int)ceilf(float(V) * inv_block_size);
    VP = nblocks * block_size;
    blk_dist = (int *)malloc(sizeof(int) * VP * VP);
#pragma omp parallel for num_threads(ncpus) schedule(static) default(shared) collapse(2)
    for (int i = 0; i < VP; i++) {
        for (int j = 0; j < VP; j++) {
            if (i == j) {
                blk_dist[calc_blk_idx(i, j, nblocks)] = 0;
            } else {
                blk_dist[calc_blk_idx(i, j, nblocks)] = infinity;
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
            dist[i * V + j] = blk_dist[blk_idx] > infinity ? infinity : blk_dist[blk_idx];
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
    return (int(r * inv_block_size) * nblocks + int(c * inv_block_size)) * (block_size * block_size) + (r % block_size) * block_size + (c % block_size);
}

inline void proc(int *blk_dist, int s_i, int e_i, int s_j, int e_j, int k, int nblocks, int ncpus) {
#pragma omp parallel for num_threads(ncpus) schedule(static) default(shared) collapse(2)
    for (int i = s_i; i < e_i; i++) {
        for (int j = s_j; j < e_j; j++) {
            int *ik_ptr = blk_dist + (i * nblocks + k) * (block_size * block_size);
            int *ij_ptr = blk_dist + (i * nblocks + j) * (block_size * block_size);
            int *kj_ptr = blk_dist + (k * nblocks + j) * (block_size * block_size);
            for (int b = 0; b < block_size; b++) {
                for (int r = 0; r < block_size; r++) {
#pragma GCC ivdep
                    for (int c = 0; c < block_size; c++) {
                        ij_ptr[r * block_size + c] = std::min(ij_ptr[r * block_size + c], ik_ptr[r * block_size + b] + kj_ptr[b * block_size + c]);
                    }
                }
            }
        }
    }
}