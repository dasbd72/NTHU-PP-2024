# NTHU PP 2024 HW2 - Mandelbrot Set - Report

Name: Sao-Hsuan Lin (林劭軒)

Student ID: 113062532

## Implementation

### `pthreads` Implementation

Code hierarchy:

- `Solver::mandelbrot()`: The single process version of the Mandelbrot set solver.
  - `Solver::partial_mandelbrot()`: The function that start multiple threads to compute the Mandelbrot set.
    - `Solver::partial_mandelbrot_thread()`: Atomically fetches the next pixels to compute and computes them.
      - `Solver::partial_mandelbrot_single_thread()`: The function for the single-threaded version of partial_mandelbrot.
  - `Solver::write_png_controller_thread()`: The function that runs concurrently with partial_mandelbrot that detects completed rows and writes them to the image file.

Scheduling: I first create an array of `pixels` that stores the pixels to be computed. Then the threads will atomically fetch the next `batch_size` of pixels to compute. The `batch_size` is set to be `2048` if the number of tasks is greater than `10000000`, otherwise it is set to be `512`.

I/O: I use a separate thread to write the image to the file. The thread will check if the row is completed and write it to the file. The thread will incrementally check the next done pixel with an interval of `100us`.

Computation: I implemented the avx512 simd instructions to compute the Mandelbrot set. Since different pixels may have different number of iterations to escape, I make it able to early stop the computation, store the done entries, and get the next pixels to compute.

### `mpi + openmp` Implementation

Code hierarchy:

- `Solver::mandelbrot_mpi()`: The MPI version of the Mandelbrot set solver.
  - `Solver::partial_mandelbrot()`: The function that start multiple threads to compute the Mandelbrot set. Each processes computes different chunks of the image.
    - `Solver::partial_mandelbrot_thread()`: Atomically fetches the next pixels to compute and computes them.
      - `Solver::partial_mandelbrot_single_thread()`: The function for the single-threaded version of partial_mandelbrot.
  - `Solver::write_png()`: Writes the image to the file.

Scheduling: Different from the `pthreads` implementation, we have to distribute the tasks to different processes. After multiple different methods I tries, I found that the best method is to generate a random permutation of the pixels and distribute an even amount of pixels to each process. This avoids synchronization, reduces implementation effort, and has a good load balance.

### Combined Implementation

We can see that most of the functions are common between the two implementations, hence I combined them into a single file, and use the `#ifdef MPI_ENABLED` directive to switch between the two implementations.

## Optimizations

### Chunked Dynamic Scheduling

For the threads, instead of getting next tasks pixel by pixel, I fetch a batch of pixels to compute. This reduces the overhead of critical section and improves the performance.

### Avx512

I implemented the avx512 simd instructions to compute the Mandelbrot set. This improves the performance by a large margin. Besides, I found that the fused multiply-add / sub instruction can further improve the performance.

### Avx512 - Continuous Batching

Instead of binding each pixels together, which may have different number of iterations to escape. I make it able to early stop the computation, store the done entries, and get the next pixels to compute. This theoretically should improve the performance by a large margin, however, the overhead doing extra computation of the `x0` and `y0` and storing the done entries may offset the performance gain. Hence, instead of check in each iteration, I check every 200 iterations.

### MPI Chunked Randomized Static Scheduling

As above mentioned, I randomize the pixels that each process computes. This gives a good load balance. However, the sparse distribution of the pixels may cause cache miss and reduce the performance. Hence, instead of randomize each single pixels, I randomize the chunks of pixels. That is, I first divide the pixels into chunks, and then randomize the chunks. This not only gives a good load balance, but also keep more pixels close to each other.

## Evaluation

### Setup

#### Testcases

- Fast
  - Iterations: 10000
  - Region: (-0.5506211524210792 -0.5506132348972915 0.6273469513234796 0.6273427528329604)
  - Image size: 7680x4320
  - The pixels have smaller number of iterations to escape.
- Slow
  - Iterations: 10000
  - Region: (-0.5506164691618783 -0.5506164628264113 0.6273445437118131 0.6273445403522527)
  - Image size: 7680x4320
  - The pixels have larger number of iterations.

#### Environment

- QCT Cluster
  - CPU: INTEL(R) XEON(R) PLATINUM 8568Y+ @ 2.3GHz
  - Socket: 2

#### Measurements

- I/O time
  - Time for writing the image
- CPU time
  - Time for computing the Mandelbrot set
- Critical Section time
- Communication time

#### Profile script

`pthreads` version:

```bash
THREADS=1 # 1~12
TESTCASE=slow # fast / slow
python -m scripts.testcase --testcase-dir testcases.prof -c $THREADS --profile nsys hw2a $TESTCASE
```

`mpi` version:

```bash
THREADS=4
TESTCASE=slow # fast / slow
python -m scripts.testcase --testcase-dir testcases.prof -N 1 -n 1 -c $THREADS --profile nsys hw2b $TESTCASE
python -m scripts.testcase --testcase-dir testcases.prof -N 1 -n 2 -c $THREADS --profile nsys hw2b $TESTCASE
python -m scripts.testcase --testcase-dir testcases.prof -N 1 -n 4 -c $THREADS --profile nsys hw2b $TESTCASE
python -m scripts.testcase --testcase-dir testcases.prof -N 1 -n 8 -c $THREADS --profile nsys hw2b $TESTCASE
```

Optimization strategies:

```bash
THREADS=4
TESTCASE=fast # fast / slow
python -m scripts.testcase --testcase-dir testcases.prof --report-name baseline   -N 1 -n 4 -c $THREADS --profile nsys hw2b $TESTCASE
python -m scripts.testcase --testcase-dir testcases.prof --report-name avx        -N 1 -n 4 -c $THREADS --profile nsys hw2b $TESTCASE
python -m scripts.testcase --testcase-dir testcases.prof --report-name avx_cb     -N 1 -n 4 -c $THREADS --profile nsys hw2b $TESTCASE
python -m scripts.testcase --testcase-dir testcases.prof --report-name mpi_crs    -N 1 -n 4 -c $THREADS --profile nsys hw2b $TESTCASE
```

### `pthreads` Scalability

#### `pthreads` Scalability Experiment Setup

- Threads: 1~12
- Testcases: `fast` and `slow`

#### `pthreads` Scalability with Both Testcases

![`pthreads` `fast` scalability bar chart](https://i.imgur.com/IDj0Nog.png)
![`pthreads` `fast` speedup line chart with ideal](https://i.imgur.com/eWWjHlW.png)
![`pthreads` `slow` scalability bar chart](https://i.imgur.com/qe2PS5d.png)
![`pthreads` `slow` speedup line chart with ideal](https://i.imgur.com/rUq60gX.png)

#### Explanation of `pthreads` Scalability

We conducted scalability experiments on the `fast` and `slow` testcases. Though both testcases has same size of images and same maximum iterations, the `slow` testcases has an average of more iterations to complete before escaping. Hence, we can see that the `slow` testcase takes more time to complete under each number of threads. I chose the two testcases for the later section of optimization strategies.

From the scalability experiments, we can see a nearly perfect linear speedup for both testcases. This is because computation on each pixels are independent, and the computation is the bottleneck. Besides, the I/O time is negligible compared to the computation time as shown in the scalability bar chart.

### `mpi` Scalability & Load Balance

#### `mpi` Scalability & Load Balance Experiment Setup

- Nodes: 1
- Processes: 1, 2, 4, 8
- Threads: 4
- Testcases: `fast` and `slow`

#### `mpi` Scalability with Both Testcases

![`mpi` `fast` scalability bar chart](https://i.imgur.com/bFLuzI7.png)
![`mpi` `fast` speedup line chart with ideal](https://i.imgur.com/tShd4PP.png)
![`mpi` `slow` scalability bar chart](https://i.imgur.com/8JE4kKB.png)
![`mpi` `slow` speedup line chart with ideal](https://i.imgur.com/ukdAb2d.png)

#### `mpi` Load Balance with Both Testcases

![`mpi` `fast` n2 load balance](https://i.imgur.com/XInwrOu.png)
![`mpi` `fast` n4 load balance](https://i.imgur.com/fu5QLOC.png)
![`mpi` `fast` n8 load balance](https://i.imgur.com/stClTqW.png)

![`mpi` `slow` n2 load balance](https://i.imgur.com/doP0nzB.png)
![`mpi` `slow` n4 load balance](https://i.imgur.com/Bit440o.png)
![`mpi` `slow` n8 load balance](https://i.imgur.com/70WlOa0.png)

#### Explanation of `mpi` Scalability & Load Balance

We can see that the `mpi` scalability is not as good as the `pthreads` scalability. This is due to the imbalance of the iterations each process has to compute. This may be improved by implementing the master slave model, where the master process will distribute the tasks to the slave processes. However, this will introduce communication overhead and reduces performance, found after I implemented both versions.

To dive deeper, I found that in the fast testcase, the speedup curve left ideal from 4 processes, but in the slow testcase, the speedup curve left ideal from 8 processes. The reason is shown in the load balance figures. Despite from the increasing communication overhead, the load balance also matters. The fast testcase has a more non uniform distribution of the iterations, hence the load balance is worse. Hence we got a better speedup in the slow testcase.

### Optimization Strategies

![`pthreads` `fast` optimization strategies](https://i.imgur.com/I3AWkgq.png)
![`pthreads` `fast` optimization strategies speedup line chart](https://i.imgur.com/U2Y41Sm.png)

- First the `baseline` version is the version only with `chunked dynamic scheduling` and without `avx512`, `avx512 continuous batching`, and `mpi chunked randomized static scheduling`.
- The `avx` version is the `cds` version with `avx512`.
  - We can see that the `avx` version has a large performance gain in CPU time. Since communication time includes the time waiting other processes to complete, it also has a large performance gain in communication time.
- The `avx_cb` version is the `avx` version with `avx512 continuous batching`.
  - Although theoretically the `avx_cb` version should have a large performance gain, the overhead of doing extra computation of the `x0` and `y0` and storing the done entries offsets the performance gain. Hence, we can only see a slight performance gain.
- The `mpi_crs` version is the `avx_cb` version with `mpi chunked randomized static scheduling`.
  - We can see with the communication time has been reduced by a significant margin. This is because the `mpi chunked randomized static scheduling` gives a much more better load balance.
  - Notably, although we randomized the order of tasks, the CPU time has not been increased. This attributes to the chunked randomization, which keeps more pixels close to each other, hence reducing the cache miss.

## Experience / Conclusion / Feedback

Since we only need to parallelize the for loop to the processes and threads, it’s easy to reach an ideal scaling in the `pthreads` version.

SIMD optimization really helps to improve the performance. The ideal speedup is 8x, but we only got 5.33x due to the overhead of doing extra data movement and computation.

Load balance is a critical factor in the `mpi` version. It is important to find a way to make the load balance better.

Though I have done a lot of optimizations, I am still far from the 1st place, so I'm interested in how the 1st place did the optimization.
