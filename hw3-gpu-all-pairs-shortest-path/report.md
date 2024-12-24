# PP 2024 HW3 Report - All-Pairs Shortest Path

## Student information

- Student name: Sao-Hsuan Lin 林劭軒
- Student ID: 113062532

## 1. Implementation

### a. Which algorithm do you choose in hw3-1?

Blocked Floyd Warshall.

### b. How do you divide your data in hw3-2, hw3-3?

In hw3-2, I let each block of gpu process 64*64 entries of the graph. Each thread in the block process 4 entries of the blocks.

In hw3-3, first gpu process top half of the matrix, and second process bottom half of the matrix in phase 3.

### c. What’s your configuration in hw3-2, hw3-3? And why? (e.g. blocking factor, \#blocks, \#threads)

I set the blocking factor to block\_size, \#blocks to $\lceil V / block\_size \rceil * \lceil V / block\_size \rceil$, \#threads to TILE\*TILE, where block\_size is 78 and TILE is 26. Since we want to utilize memory access, we need to use as much shared memory as possible to reduce global memory accesses. Total amount of shared memory per block is 49152 Bytes, and we need two matrices with 4 bytes data for each blocks, $\sqrt{49152 / 2 / 4}=78$. So block size should maximum 78. Besides, maximum threads per block is $1024=32*32$, each side of thread is maximum 32. To let each thread process equal number of entries, we need to let block size to be multiple of thread. Hence I chose 26. Since, 26\*26 isn't a multiple of the warp size, (72,24) may be a better parameter. However, after testing, the result shows that (78,26) is better on hades, and (72,24) is better on NCHC.

### d. How do you implement the communication in hw3-3?

If the row k is processed by the thread in phase 3, then it sends row k to the peer device, otherwise it receives row k.

```c++
if (range > 0 && k >= start && k < start + range)
    cudaMemcpy2D(
        dist_dev[peerid] + int_pitch[peerid] * block_size * k, 
        pitch[peerid],
        dist_dev[tid] + int_pitch[tid] * block_size * k, 
        pitch[tid],
        sizeof(int) * VP, block_size, 
        cudaMemcpyDefault
    );
```

### e. Briefly describe your implementations in diagrams, figures or sentences

I implemeted two version of Floyd Warshall algorithm, blocked version and flatten version.

The blocked version is implemented similar to the sequential version. The difference is that since we need to launch onto gpu, we separate the three phases into three kernel. In each phases, we load shared data of each threads in a block into shared memory. Then we start block\_size rounds of processing. In phase 1 and phase 2, each round needs to sync threads, while in phase 3 we don't need. After computation, store the result to global memory. We total need nblocks rounds of phase 1 to phase 3.

The difference of flatten version to blocked version is the way I store data. $blk\_dist[(i * nblocks + j) * blk\_pitch + (r)*block\_size + c]$ versus $blk\_dist[i * block\_size * pitch + j * block\_size + (r)*pitch + c]$. $i, j$ are indices for blocks, $r, c$ are indices for threads in block. In flatten version, data in a block is all aligned together, which speeds up global loading throughput.

## 2. Profiling Results (hw3-2)

I profiled hw3-2 with p11k1 as input. This is the profiling result of my phase 3 kernel.

| Metrics                          | Min         | Max         | Ave         |
|----------------------------------|-------------|-------------|-------------|
| Achieved Occupancy               | 0.633448    | 0.634508    | 0.634090    |
| Multiprocessor Activity          | 99.68%      | 99.90%      | 99.87%      |
| Shared Memory Load Throughput    | 2721.1GB/s  | 2729.2GB/s  | 2725.9GB/s  |
| Shared Memory Store Throughput   | 272.75GB/s  | 273.56GB/s  | 273.22GB/s  |
| Global Load Throughput           | 279.09GB/s  | 279.92GB/s  | 279.58GB/s  |
| Global Store Throughput          | 118.29GB/s  | 118.64GB/s  | 118.49GB/s  |

## 3. Experiment & Analysis

### 3.1 System Spec

Using the `apollo-gpu` cluster.

### 3.2 Blocking Factor

Integer GPOS and global load throughput. Since block size is restricted by shared memory size, I
partitioned block size by threads, shown in {block size}-{thread size}.
    ![GPOS](https://i.imgur.com/MQh7NYO.png)
    ![GLT](https://i.imgur.com/rOJDwHp.png)

### 3.3 Optimizations

1. Page-locked memory
    - Use cudaHostRegister to pin host memory.
2. Padding
    - $VP=\lceil V/block\_size \rceil * block\_size$, VP is the padded matrix side.
3. 2D allocate
    - Tried using cudaMallocPitch to allocate 2D matrix and cudaMemcpy2D to copy matrix. cudaMallocPitch slows down the program while cudaMemcpy2D speeds up.
4. Shared memory
    - I used as much shared memory as possible to reduce memory bottleneck.
5. Handle bank conflict
    - Below are two different methods. I originally used the method at the top, which causes bank conflict. After profiling, I handled all bank conflicts by using the method at the bottom.
      ![bank conflict](https://i.imgur.com/zSKB2pe.png)
      ![no bank conflict](https://i.imgur.com/eRa5ErF.png)
6. Large blocking factor
    - Using more shared memory as possible also increases the blocking factor we can choose.
7. Reduce communication
    - Instead of copying a matrix to device, I copied the edges and initialize the matrix on device using kernel. One kernel clears the matrix to let entries $(i,j)$ where $i\neq j$ is zero and $i=j$ is INF, another set the weight of the edges.
8. Coalesced memory access
    - As shown in the above figure, the way I separated threads enables coalesced memory access.
9. Flatten blocks
    - I did some more effort to optimize memory usage. Instead of storing blocks in a 2D matrix, which exists a gap between rows in a block, I transfer the matrix into a $nblocks*nblocks*(78*78)$ matrix. $nblocks=\lceil V / block\_size \rceil$. This method is restricted by $VP * VP * 2 + 2 * 3 * E + V * V <= \texttt{deviceProp.totalGlobalMem} / 4$, hence I implemented both blocked method and flatten method in my program, and switches when needed global memory exceeds.
10. Occupancy optimization
    - As result, all my optimization and implementation leads to high occupancy.
11. Optimization Breakdown
    ![optimization breakdown](https://i.imgur.com/i8WT1eF.png)

### 3.4 Scalability

| Testcase | hw3-2 (s) | hw3-3 (s) |
|----------|-----------|-----------|
| p11k1    | 1.72      | 1.80      |
| p21k1    | 9.08      | 7.93      |
| p26k1    | 15.66     | 5.94      |
| p31k1    | 27.15     | 10.39     |

### 3.5 Time Breakdown

| Testcase | Computing (s) | Communication (s) | DtoH (s) | HtoD (s) | I/O (s) |
|----------|---------------|-------------------|----------|----------|---------|
| p30k1    | 22.688518     | 2.46867           | 0.57720  | 0.00728  | 6.016599|

### 3.6 Others

- \{block size\}-\{thread size\} on NCHC and hades
  - On NCHC, 78-26 gets 130.80s on hw3-2-judge, and 72-24 gets 118.04s. But on hades, 78-26 gets 194.89s on hw3-2-judge, and 72-24 gets 205.62s. This is probably because of the difference in memory throughput.

## 4. Experiment on AMD GPU

### Note

- My hw3-3 on cuda passed the judge but the hipified code failed to run on AMD GPU. Since there is no problem running hw3-2, I assume the problem due to the d2d communication.

### Comparison

- With `rocminfo` and some searching, I found that
  - MI210 has 64GB HBM2e with 1,638 GB/s bandwidth compared to GTX 1080 with 8GB GDDR5X with 320 GB/s bandwidth
  - MI210 is optimized with FP16, while GTX 1080 only supports FP16 with a very low throughput
  - GTX 1080 has 2560 CUDA cores operating at 1.6 GHz. MI210 has 6656 stream processors operating at 1.7 GHz.
  - MI210 has 22.63 TFLOPS peak FP32 performance, GTX 1080 has 8.9 TFLOPS peak FP32 performance, which is a 2.54x difference.
- Experiment results

    | Testcase | hw3-2 (s) | hw3-2-amd (s) | Speedup |
    |----------|-----------|---------------|---------|
    | p11k1    | 1.72      | 1.07          | 1.61    |
    | p21k1    | 9.08      | 4.31          | 2.11    |
    | p26k1    | 15.66     | 7.39          | 2.12    |
    | p31k1    | 27.15     | 12.46         | 2.18    |

- The speedup is not as high as expected given the TFLOPS. Though the bandwidth is also higher, the memory access pattern or parallelization is not optimized for AMD GPU. However, it still shows a significant speedup, showing the potential of AMD GPU.

## 5. Experience & conclusion

I learnt a lot in this homework. There are many details that we need to keep in mind when programming cuda code. Besides, the status of the machine running the program will also require some trade-off, such as on hades, we need to make more effort on memory utilization, since a better parallelized program may run slower on hades than a memory ultilized program.
