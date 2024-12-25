# PP 2024 HW4 Report - Flash Attention

## Student information

- Student name: Sao-Hsuan Lin 林劭軒
- Student ID: 113062532

## 1. Implementation

- I implemented flash attention 2
- Constants
  - `bc` same as the `bc` of the original paper
  - `br` same as the `br` of the original paper
  - `bc_pad` the padded size of `bc` to reduce bank conflict
  - `bd` the padded size of `d` or we can say the constant size of `d`
  - `num_warps` the number of threads responsible for x dimension
  - `threads_per_warp` the number of threads responsible for y dimension
- Variables
  - `B` the batch size
  - `N` the sequence length
  - `d` the dimension of the input
- Notes
  - For simplicity, this assignment does not require the implementation of the multi-head attention and masked attention.

### a. Key implementation details

- The version I implemented is flash attention 2.
- SRAM Requirement
  - A total of $2 * br * bd + 4 * bc\_pad * bd + 4 * br$ floats of SRAM is required.
    - $br * bd$ for $o_i$
    - $br * bd$ for $q_i$
    - $bc\_pad * bd$ for $k_j$
    - $bc\_pad * bd$ for $v_j$
    - $2 * br$ for $l_{i,j}$
      - Requires 2 buffers for values in previous loop
    - $2 * br$ for $m_{i,j}$
      - Requires 2 buffers for values in previous loop
    - $br * bc\_pad$ for $s_{i,j}$
    - $br * bc\_pad$ for $p_{i,j}$
- Blocked attention computation
  - Outer loop
    - Inner loop
      1. Compute $s_{i,j}$ with $q_i$ and $k_j$ in `qk_dot_and_scalar`.
      2. Compute $m_{i,j}$ with $s_{i,j}$ in `row_max`.
      3. Compute $p_{i,j}$ with $s_{i,j}$ and $m_{i,j}$ in `minus_max_and_exp`.
      4. Compute $l_{i,j}$ with $m_{i,j}$ and $p_{i,j}$ `row_sum`.
      5. Update $o_i$ with $p_{i,j}$, $m_{i,j}$, and $o_i$ in `inner_update_o`.
    - Update $o_i$ with $l_{i,j}$ and $l_{i,j}$ with $m_{i,j}$ and $l_{i,j}$ in `outer_update_lo`.
  
### b. Parallelization

- mini-batch parallelization
  - I parallelize mini-batches with streams, each mini-batch has size of $B/64$.
  - The constant decides the number of streams.
- sequence parallelization
  - Following the algorithm of flash attention 2, each thread block is responsible for sequence length of `bc`.
- thread parallelization
  - In different device functions, the threads have different jobs.
    - In `qk_dot_and_scalar`, `minus_max_and_exp`, and part of `inner_update_o`, each thread gets a position in x and y dimension. Then it starts computing the multiplication and sum of the position, and moves on to the next x or next y.
    - In `row_max`, `row_sum`, and part of `inner_update_o`, each thread gets a position in x dimension. Then it starts computing the max or sum of the row if it is in the range of `br`

### c. Constant choosing

- `bc` and `br`
  - I choose both `bc` and `br` to be 32
  - I have tried different values of `bc` and `br` and found that 32 is the best value for the performance.
  - Note that the `bc_pad` is set to 37, found after some experiments.

### d. Cuda Kernel Configuration

- For input `d` <= 32
  - `bc` = 32
  - `br` = 32
  - `bc_pad` = 37
  - `bd` = 37
  - `num_warps` = 8
  - `threads_per_warp` = 16
  - smem: $(2 * br * bd + 4 * bc\_pad * bd + 4 * br) * sizeof(float) = 15076$ bytes
  - grid: $(\lceil N/32 \rceil, \lceil B/64 \rceil)$
  - block: $(32, 32)$
- For input `d` <= 64
  - `bc` = 32
  - `br` = 32
  - `bc_pad` = 37
  - `bd` = 69
  - `num_warps` = 8
  - `threads_per_warp` = 32
  - smem: $(2 * br * bd + 4 * bc\_pad * bd + 4 * br) * sizeof(float) = 59024$ bytes
  - grid: $(\lceil N/32 \rceil, \lceil B/64 \rceil)$
  - block: $(32, 32)$

### e. Justification

## 2. Profilling Results

- Configurataion
  - Testcase: t02
  - Metrics and events
    - `--metrics achieved_occupancy,ipc,gld_throughput,gst_throughput,shared_load_throughput,shared_store_throughput --events shared_ld_bank_conflict,shared_st_bank_conflict`
- Results
  - Events of Kernel: `flash_attention::flash_attention_kernel`

    | Event                     | Min   | Max    | Avg    | Total  |
    |---------------------------|-------|--------|--------|--------|
    | `shared_ld_bank_conflict` | 2048  | 6144   | 6068   | 327680 |
    | `shared_st_bank_conflict` | 4864  | 14592  | 14411  | 778240 |

  - Metrics of Kernel: `flash_attention::flash_attention_kernel`

    | Metric                              | Description                       | Min         | Max         | Avg         |
    |-------------------------------------|-----------------------------------|-------------|-------------|-------------|
    | `achieved_occupancy`                | Achieved Occupancy                | 0.124858    | 0.124884    | 0.124874    |
    | `sm_efficiency`                     | Multiprocessor Activity           | 18.69%      | 56.87%      | 55.73%      |
    | `gld_throughput`                    | Global Load Throughput            | 11.647GB/s  | 35.586GB/s  | 34.579GB/s  |
    | `gst_throughput`                    | Global Store Throughput           | 1.3809GB/s  | 4.2193GB/s  | 4.0999GB/s  |
    | `shared_load_throughput`            | Shared Memory Load Throughput     | 363.93GB/s  | 1112.0GB/s  | 1080.5GB/s  |
    | `shared_store_throughput`           | Shared Memory Store Throughput    | 26.173GB/s  | 79.971GB/s  | 77.708GB/s  |

## 3. Experiment & Analysis

### a. System Spec

- Cluster: apollo-gpu
- GPU: GTX 1080

### b. Optimization

- Coalesced memory access
  - Let the threads in the same warp access the memory in a coalesced way.
  - Example
    ![coalesced](https://i.imgur.com/bXmk0fR.png)
- Memory alignment
  - I align the memory by adding padding to the data in HBM.
- Shared memory
  - Following the algorithm of the paper and described above, I use shared memory to store the intermediate results. Especially the `q`, `k`, and `v` matrices.
- Bank conflict handling
  - I pad the size of `bc` to 37 to reduce the bank conflict.
- Streaming
  - Use streams to parallelize the mini-batches.
- CPU and GPU overlap
  - With the streams, we can overlap the time reading QKV from the disk or writing O to the disk and the time computing the attention or copying between host and device.
- We can see the impact of each optimization in the next section.

### c. Others

- Experiment Method
  - Cast on testcase t02 since nvprof metrics is slow.
  - For each version, I run two commands to get the profiling results.
    - `nvprof` without extra options
    - `nvprof` with argument `--metrics achieved_occupancy,sm_efficiency,gld_throughput,gst_throughput,shared_load_throughput,shared_store_throughput --events shared_ld_bank_conflict,shared_st_bank_conflict`
  - The implementation version from 1 to 7 is
    - 1: best version
    - 2: no memory alignment
    - 3: no coalescing
    - 4: not handling bank conflict
    - 5: not overlapping cpu and gpu
    - 6: not streaming
    - 7: not using shared memory
- Result
  ![experiment](https://i.imgur.com/wfUmKTI.png)
- Explanation
  Based on the experiment results, the following observations and conclusions can be made:

  1. Kernel Execution Time
     - Version 1, the optimized implementation, exhibits the lowest average kernel execution time at approximately 35.993 ms.
     - Removing optimizations progressively increases the execution time, with the most significant jump observed in version 7 (189.67 ms), which lacks shared memory usage. This highlights the importance of shared memory for performance.  

  2. Shared Bank Conflicts
     - Shared memory bank conflicts are negligible in versions 1, 2, and 3, which employ techniques to minimize memory contention.
     - Starting from version 4, bank conflicts dramatically increase, as seen in the shared load bank conflict (691,010) and shared store bank conflict (189,629). As bank conflict increases, the execution time increases. This confirms that proper handling of bank conflicts is critical for performance.

  3. Achieved Occupancy and SM Efficiency
     - Achieved occupancy increases as optimizations are removed, though the occupancy is increased, total execution time is increased as well. Showing that the occupancy is not the only factor that affects the performance.
     - SM efficiency is highest in version 1 (56.85%) and slightly degrades across versions, with a noticeable dip in version 7 (53.24%). This also indicates that shared memory and optimized kernel usage contribute to maintaining high SM efficiency.  

  4. Global Memory Load and Store Throughput
     - GLD and GST throughput does not vary significantly across versions. Except the not optimized version, which does not use shared memory, has the highest GLD and GST throughput. This is because the shared memory is not used, and HBM is heavily used.

  5. Shared Memory Load and Store Throughput  
     - Shared memory throughput values are highest in version 4 (1793.1 GB/s for loads and 369.97 GB/s for stores), since we put the kernel launch together. However, this disables the overlap of CPU and GPU, which is not the best choice, shown in the increased execution time.
     - I set the values to `NaN` in version 7, as shared memory is entirely unused.

### Summary

The experiment demonstrates the critical role of optimizations, particularly shared memory utilization, memory coalescing, and bank conflict handling, in achieving high performance. Each optimization layer addresses specific bottlenecks, and their removal introduces measurable degradations in execution time, memory throughput, and SM efficiency. Version 1 showcases the effectiveness of combining all optimizations, while version 7 highlights the significant performance loss from neglecting these techniques.

## 4. Experience & Conclusion

I've learned a lot from this assignment. How to implement both flash attention 1 and 2, and their differences. I found that the critical part that effects performance in this assignment is the shared memory usage.

I suggest that the template can start from the flash attention 2, which is actually simpler, more widely used and more efficient.
