# NTHU PP 2024 HW1 - Odd Even Sort - Report

Name: Sao-Hsuan Lin (林劭軒)

Student ID: 113062532

## Implementation

### Task separation

First, I define $\texttt{MIN\_SIZE\_PER\_PROC}$ as 100000. If $\texttt{n}$ is less than $\texttt{MIN\_SIZE\_PER\_PROC}$, I will use the single process version. Otherwise, I will use the multi-process version.

If the multi-process version is used, I will first calculate the maximum size of buffer needed for each process.

$\texttt{max\_local\_size} = min(n, max(ceil(n / \texttt{size}), \texttt{MIN\_SIZE\_PER\_PROC}))$

Then, I will calculate the number of actual processes needed.

$\texttt{actual\_world\_size} = min(ceil(n / \texttt{max\_local\_size}), \texttt{size})$

Next, caculate start and end index for each process.

$\texttt{local\_start} = min(\texttt{rank} * \texttt{max\_local\_size}, n)$

$\texttt{local\_end} = min((\texttt{rank} + 1) * \texttt{max\_local\_size}, n)$

Then we can know the position to load of each process.

### Sorting

After reading the array using MPI IO, we have the array start at $\texttt{start}$ with $\texttt{count}$ elements. Before performing odd-even sort, we need to sort the local array. I used `boost::sort::spreadsort` as the sorting method. This is an sorting library that uses both radix and comparison, and performs 2 times faster than `std::sort`.

During the odd-even sort phase, we need at most $\texttt{actual\_world\_size} + 1$ phases, where each phase is either an odd phase or even phase. In odd phase, ranks that are odd exchange array with the ranks which is $\texttt{rank} + 1$. In even phase, ranks that are even exchange array with the ranks which is $\texttt{rank} + 1$. However, passing the large amount of data takes to much time, which isn't necessary in each phase. That is, when the rank at right is all larger than the left rank, there is no need to merge once again. So before we exchange all the elements, we exchange the first element of the right array and last element of the left array. If the element from right is larger than the element from left, we can skip the rest exchanging. Repeat the process until we finished the final phase completes the sorting.

### [Q1] How do you handle an arbitrary number of input items and processes?

After calculating local start and end index for each process, I will load the corresponding data for each process. For the last process, I will fill the remaining data with maximum value of `float`.

### [Q2] How do you sort in your program?

I follow the algorithm given in the spec. I use `boost::sort::spreadsort::float_sort` for local sorting. Then start `actual_world_size` rounds of odd-even sort.

## Optimizations

### Fused buffer

I use a fused buffer. The buffer has size of `max_local_size * 3`. Then I assign three pointers to the three parts of the buffer.

The three parts are `local`, `neighbor` and `temp`. The `local` part is used to store the local data. The `neighbor` part is used to store the data received from the neighbor process. The `temp` part is used to merge the `local` and `neighbor` part.

### Buffer swapping

Instead of copying the data from `temp` to `local`, I swap the pointers of `local` and `temp`. This saves the time of copying data.

### Minimum merging

Instead of using a single merge function to handle all merging cases, I use `merge_left` and `merge_right` to handle the left and right merging cases. This can save half of the merge time.

### Minumum communication

Exchange the border data before exchanging all data. This can save the time of communication.

### Flag controlled nvtx profiling

I use a flag to control the profiling. Simply set or unset compile flag `PROFILING` to enable or disable the profiling of nvtx.

### Reduce IO time

Use `MPI_COMM_SELF` instead of `MPI_COMM_WORLD` when opening the read or write file. Since `MPI_COMM_WORLD` will cause synchronization between all processes, it will significantly increase the IO time.

### Other efforts

I've tried to prune when all the ranks are sorted before they reached the final phase. However, we need to reduce all information from other ranks, which forced us to synchronize the processes, causes bottleneck. I've also tried to solve this problem using asynchronous collective routine, and only wait for the routine when the part of process is sorted, but it still doesn't perform better than the version no pruned.

I've also tried to cut each buffers into chunks, and only send the chunks that are needed. However, it takes one more communication, which turns out to be slower than the original version.

## Evaluation

### Setup

#### Testcases

- Orders
  - Random order
  - Reversed order
  - Skewed order
    - With total size n, $[0, n * 0.1)$ are random numbers from $[n*0.9, n)$
    - And the rest are ordered numbers from $[0, n*0.9)$
- Sizes
  - Small size  (  `1000000`)
    - Not used
  - Medium size ( `26843545`)
    - Not used
  - Large size  (`536870911`)
- Naming
  - `{order}_{size}`
  - Example: `rand_s`, `rev_m`, `rand_l`

#### System Spec

- Apollo Cluster
  - CPU: Intel(R) Xeon(R) CPU X5670 @ 2.93GHz
  - Socket: 2

#### Measurements

- I/O time
  - Time for reading and writing files
- Local sort time
- Odd-even sort time
  - Communication time
    - Pre-exchange time
    - Exchange time
    - Note that communication time includes the waiting time between the communicating processes
  - Merge time

#### Generation script

```bash
python -m scripts.generate -v random -n 536870911 rand_l
python -m scripts.generate -v reversed -n 536870911 rev_l
python -m scripts.generate -v skewed -n 536870911 skew_l
```

#### Profile script

```bash
# Scalability
testcase="skew_l" # rand_l, rev_l, skew_l
export PROFILING=1
python -m scripts.testcase --testcase-dir /share/judge_dir/.judge_exe.tc.$USER --profile nsys -N 1 -n 4  -w apollo[40]    $testcase
python -m scripts.testcase --testcase-dir /share/judge_dir/.judge_exe.tc.$USER --profile nsys -N 2 -n 8  -w apollo[40-41] $testcase
python -m scripts.testcase --testcase-dir /share/judge_dir/.judge_exe.tc.$USER --profile nsys -N 4 -n 16 -w apollo[40-43] $testcase
python -m scripts.testcase --testcase-dir /share/judge_dir/.judge_exe.tc.$USER --profile nsys -N 8 -n 32 -w apollo[40-47] $testcase
```

```bash
# Optimization Strategies
testcase="skew_l" # rand_l, skew_l
export PROFILING=1
python -m scripts.testcase --testcase-dir /share/judge_dir/.judge_exe.tc.$USER --report-name ${testcase}_opt_none      --profile nsys -N 4 -n 16 -w apollo[40-43] $testcase
python -m scripts.testcase --testcase-dir /share/judge_dir/.judge_exe.tc.$USER --report-name ${testcase}_opt_reduce_io --profile nsys -N 4 -n 16 -w apollo[40-43] $testcase
python -m scripts.testcase --testcase-dir /share/judge_dir/.judge_exe.tc.$USER --report-name ${testcase}_opt_min_comm  --profile nsys -N 4 -n 16 -w apollo[40-43] $testcase
python -m scripts.testcase --testcase-dir /share/judge_dir/.judge_exe.tc.$USER --report-name ${testcase}_opt_min_merge --profile nsys -N 4 -n 16 -w apollo[40-43] $testcase
python -m scripts.testcase --testcase-dir /share/judge_dir/.judge_exe.tc.$USER --report-name ${testcase}_opt_buff_swap --profile nsys -N 4 -n 16 -w apollo[40-43] $testcase
```

### Scalability

#### Scalability Experiment Setup

- Nodes: 1, 2, 4, 8
- Processes per node: 4
- Testcases: `rand_l`, `rev_l`, `skew_l`

#### Scalability with Random order

![rand_l scalability bar chart](https://i.imgur.com/RO7IaYq.png)
![rand_l speedup line chart](https://i.imgur.com/BoObw72.png)
![rand_l speedup line chart with ideal](https://i.imgur.com/8WXTZOb.png)

Nsys profile
![rand_l nsys](https://i.imgur.com/oq0HMsk.png)

#### Scalability with Reversed order

![rev_l scalability bar chart](https://i.imgur.com/x28ObZj.png)
![rev_l speedup line chart](https://i.imgur.com/RJRYTmj.png)
![rev_l speedup line chart with ideal](https://i.imgur.com/ABMfscH.png)

Nsys profile
![rev_l nsys](https://i.imgur.com/Evkg70c.png)

#### Scalability with Skewed order

![skew_l scalability bar chart](https://i.imgur.com/uhc0X5C.png)
![skew_l speedup line chart](https://i.imgur.com/n0JzWm6.png)
![skew_l speedup line chart with ideal](https://i.imgur.com/jhhnbQq.png)

Nsys profile
![skew_l nsys](https://i.imgur.com/zTAvEZV.png)

#### Explanation of Scalability

We can see in the random case, CPU and I/O time decreased as the number of processes increases. Though the communication time increased, we can see a significant decrease in total time, leading to a good scalability.
However, in the reversed case, although the CPU time and I/O time also decreased, the communication time increased significantly. This is due to the fact that the data is reversed, hence we must exchange the data in each phase without leveraging the `Minimum Communication` optimization. This leads to a bad scalability.

For the skewed case, if we look at the graph generated by nsys, we can see that if only one edge of processes takes more time local sorting, its neighbor must wait for it, and the neighbor's neighbor must wait for the neighbor, and so on. Causing a chain of waiting. However, without knowledge of the data distribution, I couldn't find an easy way to solve this problem. I observed this problem when profiling some of the testcases which has the same skewness characteristics.

### Optimization Strategies

#### Optimization Strategies Setup

- Nodes: 4
- Processes: 16
- Testcases: `rand_l`, `skew_l`

#### Optimization Strategies with Random ordered data

![rand_l optimization bar chart](https://imgur.com/0uASR03.png)
![rand_l optimization line chart](https://imgur.com/yperqRz.png)

#### Optimization Strategies with Skewed ordered data

![skew_l optimization bar chart](https://imgur.com/EVuVqAt.png)
![skew_l optimization line chart](https://imgur.com/l1x6awT.png)

#### Explanation of Optimization Strategies

I have shown the performance breakdown and improvement with each optimization strategies. Since given different data distribution, each optimization strategy has different impact on the performance. Hence I've conducted the experiment with both random and skewed data.

- Reduce IO
  - We can see that the I/O time decreased by half in **both** dataset. This is because we use `MPI_COMM_SELF` instead of `MPI_COMM_WORLD` when opening the read or write file. Since `MPI_COMM_WORLD` will cause synchronization between all processes, it will significantly increase the IO time. Besides, using `MPI_COMM_SELF` can let early done processes to write the file first without waiting for the other processes to finish.
- Minimum Communication
  - Minimum communication has no optimization on the random dataset, but gives a significant improvement on the skewed dataset. This is because in each phase of the skewed dataset, more than half of the processes do not need to exchange and merge the data, which is different from the random dataset. Hence, the minimum communication optimization can reduce both communication time and CPU merging time.
  - It is notable that instead of decreasing communication time, it actually reduces CPU time. I found out that most of the communication time are spent on waiting for the other processes of their previous phase. Hence reducing the size of data to be exchanged does not really reduces most of the measured communication time. And the CPU time is reduced because without transfering the whole data, we also do not need to merge the data.
  - In conclusion, with an imbalanced data distribution, the minimum communication optimization improves the performance a lot.
- Minimum Merge
  - Minimum merge has significant improvement on **both** dataset. This is because it reduces half of the merge time. Especially for the random dataset, where in each phase there are unavoidable merging.
- Buffer Swapping
  - Buffer swapping then improves the performance by reducing the time of copying data. The targets of optimization are the same as the minimum merge optimization. Hence, it also has significant improvement on **both** dataset, especially for the random dataset.
  
### Discussion

#### Discussion of Time Complexity

With some simple computation, we can know that the minimum time complexity of odd even sort is $O(n^2)$, which is worse than the $O(n \log n)$ of merge sort. However, with parallel programming, we can reduce the time complexity to $O(n^2 / p)$, where $p$ is the number of processes. This shows that the best sequential algorithm may not be the best parallel algorithm.

#### Discussion of Bottleneck

The main bottleneck differs with different data distribution.

- Random dataset
  - CPU time is the main bottleneck. Since each process has similar computation time, we can see the communication time less than other dataset, and CPU time dominates the total time.
- Reversed dataset
  - Both CPU time and communication time are bottlenecks.
- Skewed dataset
  - Communication time is the main bottleneck. As I've mentioned in the scalability section, the skewed dataset has a chain of waiting problem.

To optimize the CPU time, we can use the `Minimum Merge` and `Buffer Swapping` optimization. To optimize the communication time, we can use the `Minimum Communication` optimization.

#### Discussion of Scalability

If we only consider CPU time, the scalability is good. However, if we consider the total time, the scalability is not good. I have shown the speedup line chart with ideal speedup line. We can see that the actual speedup is far from the ideal speedup. This is because the communication time is not scalable, instead it increases as the number of processes increases.

I have improved the amount of data to be exchanged and merged, but the core problem with the skewed dataset is still there. I think the only way to solve this problem is to know the data distribution before the sorting, and we could assign the tasks to other processes to help the processes that are taking more time. However, this is not possible in the real world.

## Experience / Conclusion

I've learnt how odd-even sort works on MPI programming, it's very interesting. There are many factors that influences the performance during parallel programming. In this case, part of the bottleneck is IO. Also the problem minimum time complexity of sorting is also a bottleneck.

I've got more familiar with writing scripts. I've written multiple scripts for generating testcases, profiling, testing, and drawing the bar chart and line chart.

I've also learnt how to use NVTX to profile the program. Before this, I write a large section of macros defining the timing functions, which is very inconvenient and ugly. With NVTX, I can easily profile the program and get the information I need.

## Feedback

So interesting writing parallel program!
