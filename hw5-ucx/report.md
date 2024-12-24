# PP HW 5 Report
>
> - Please include both brief and detailed answers.
> - The report should be based on the UCX code.
> - Describe the code using the 'permalink' from [GitHub repository](https://github.com/NTHU-LSALAB/UCX-lsalab).

## Student Information

- Student name: Sao-Hsuan Lin 林劭軒
- Student ID: 113062532

## 1. Overview
>
> In conjunction with the UCP architecture mentioned in the lecture, please read [ucp_hello_world.c](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/pp2024/examples/ucp_hello_world.c)

1. Identify how UCP Objects (`ucp_context`, `ucp_worker`, `ucp_ep`) interact through the API, including at least the following functions:
    - `ucp_init`
    - `ucp_worker_create`
    - `ucp_ep_create`
2. UCX abstracts communication into three layers as below. Please provide a diagram illustrating the architectural design of UCX.
    - `ucp_context`
    - `ucp_worker`
    - `ucp_ep`

    > Please provide detailed example information in the diagram corresponding to the execution of the command `srun -N 2 ./send_recv.out` or `mpiucx --host HostA:1,HostB:1 ./send_recv.out`

3. Based on the description in HW5, where do you think the following information is loaded/created?
    - `UCX_TLS`
    - TLS selected by UCX

### Answers of section overview

1. Q: Identify how UCP Objects (`ucp_context`, `ucp_worker`, `ucp_ep`) interact through the API
   - In detail
     - `ucp_init` allocates and initializes a new `ucp_context` given the parameters configured in `ucp_params`.
     - `ucp_worker_create` creates a new `ucp_worker` associated with the `ucp_context`.
     - Before creating an endpoint, we need out of band communication to exchange the endpoint address.
        1. Through the specified socket in command line argument, client connects to server
        2. Server sends the endpoint address to the client.
     - Given the server endpoint address, client unpacks the address and creates an `ucp_ep` using `ucp_ep_create` which points to the server's endpoint given the client `ucp_worker`.
     - Through the created server endpoint, client sends its endpoint address to the server.
     - Same, server creates an `ucp_ep` using `ucp_ep_create` which points to the client's endpoint, given the server `ucp_worker`.
     - Then it remains the test message communication
       - Through the created client endpoint, server sends the test messages to the client.
       - After the communication is completed, we can clean up the resources by destroying the endpoints, worker, and context.
   - In short
     - `ucp_init` creates a new `ucp_context`.
     - `ucp_worker_create` creates a new `ucp_worker` associated with the `ucp_context`.
     - `ucp_ep_create` creates a new `ucp_ep` associated with the `ucp_worker` given the endpoint address.
   - After tracing the code in `ucp_hello_world.c`, I found that the interaction actually follows the pattern taught in the lecture.

2. Q: UCX abstracts communication into three layers as below. Please provide a diagram illustrating the architectural design of UCX.
    - `ucp_context`
    - `ucp_worker`
    - `ucp_ep`
    - Just as the diagram below, which is shown in the lecture slides, a `ucp_context` is created first, then one or more `ucp_worker` is created under the context, and finally, one or more `ucp_ep` is created under the worker when needed.
        ![three_layers_of_ucx](https://i.imgur.com/G4CxgT2.png)
    - In detail of how the command `srun -N 2 ./send_recv.out` is executed,
      - `./send_recv.out`
        - Application layer calls `MPI_Init`
          - OpenMPI layer calls `ucp_init`
            - UCP layer creates a `ucp_context`
          - OpenMPI layer calls `ucp_worker_create`
            - UCP layer creates a `ucp_worker`
            - UCT layer creates endpoints
              - Shared Memory (SysV, Posix)
              - InfiniBand (ibp3s0, rc_verbs, ud_verbs)
              - Loopback (self)
              - Other transports (TCP, CMA)
        - In `MPI_Init` or `MPI_Send`/`MPI_Recv`
          - OpenMPI layer calls `ucp_ep_create` to connect to the peer node
            - UCP layer creates a `ucp_ep`
          - OpenMPI layer sends and receives messages
            - UCT layer sends and receives messages

3. Q: Based on the description in HW5, where do you think the following information is loaded/created?
    - Guess
      - `UCX_TLS`
        - Since in the sample command provided instructs to add `UCX_TLS` in the environment variable, I guess it is set and loaded from the environment variable.
      - TLS selected by UCX
        - Without enough knowledge, I thought it might be selected when creating the context.
    - After tracing the code
      - `UCX_TLS`
        - `UCX_TLS` is set by the user in the environment variable `UCX_TLS`.
        - and is loaded by UCX by `ucp_config_read` function.
      - TLS selected by UCX
        - Created when creating worker and creating endpoints of the worker.
          - `ucp_worker_create`
          - `ucp_worker_mem_type_eps_create`
          - `ucp_ep_create_to_worker_addr`
          - `ucp_wireup_init_lanes`

## 2. Implementation
>
> Please complete the implementation according to the [spec](https://docs.google.com/document/d/1fmm0TFpLxbDP7neNcbLDn8nhZpqUBi9NGRzWjgxZaPE/edit?usp=sharing)
> Describe how you implemented the two special features of HW5.

1. Which files did you modify, and where did you choose to print Line 1 and Line 2?
   - Files and functions modified
     - `ucs/config/parser.c` function `ucs_config_parser_print_opts`
     - `ucs/config/types.h` declaration of `ucs_config_print_flags_t`
     - `ucp/core/ucp_worker.c` function `ucp_worker_print_used_tls`
   - Observed how the debug message is printed, I chose to print Line 1 and Line 2 in the function `ucp_worker_print_used_tls` in file `ucp/core/ucp_worker.c`
    ![print_of_line1_and_line2](https://i.imgur.com/LfcKYJd.png)
     - Line 1: printed by calling `ucp_config_print` with `UCS_CONFIG_PRINT_TLS` flag added in `types.h`
       - In `parser.c`, I added a function `ucs_config_parser_print_tls_opts` to print the TLS options
       - When `UCS_CONFIG_PRINT_TLS` is set, it calls `ucs_config_parser_print_tls_opts` to print the TLS options.
       - `ucs_config_parser_print_tls_opts` calls `ucs_config_parser_get_value` to get the value of `UCX_TLS` and prints it.
     - Line 2: printed the variable `strb` which contains the exact string required in the spec.
2. How do the functions in these files call each other? Why is it designed this way?
    - `ucp_worker_print_used_tls` and `ucp_worker_create`
      - Since the spec requires to print `UCX_TLS` and Line 2 together, I chose to put the two lines in the **same function**.
      - `ucp_worker_print_used_tls` is called when each endpoint is created, and it fills up the `strb` variable with the information of the TLS of the endpoint, which is Line 2. Originally, it calls `ucs_info` which prints the message if the log level is set lower or equal to `INFO`.
      - Hence I modified the function to print the message regardless of the log level and also printed `UCX_TLS` before this message.
    - `ucs_config_parser_print_opts`
      - To dive deeper, this function is used to print information from the `config` structure.
      - Given the `config` and `print_flags` as arguments, it prints the information accordingly.
3. Observe when Line 1 and 2 are printed during the call of which UCP API?
    - Both Line 1 and Line 2 are printed in
      - `ucp_worker_create`
        - `ucp_worker_mem_type_eps_create`
          - `ucp_ep_create_to_worker_addr`
            - `ucp_wireup_init_lanes`
              - `ucp_worker_get_ep_config`
                - `ucp_worker_print_used_tls`
4. Does it match your expectations for questions **1-3**? Why?
    - No, because in UCX it actually selects the TLS when creating the worker instead of creating the context.
5. In implementing the features, we see variables like lanes, tl_rsc, tl_name, tl_device, bitmap, iface, etc., used to store different Layer's protocol information. Please explain what information each of them stores.
    - `lanes`
      - Stores a list of active ways of communication between the two endpoints.
      - Each lane contains its corresponding resource index, destination information, device information, responsible operations and maximam fragment size.
    - `tl_rsc`
      - The descriptor of the enabled resources added to context.
      - Stores transport name, device name, and device index.
    - `tl_name`
      - The name of the transport layer.
    - `tl_device`
      - Queried device information of the transport layer.
    - `bitmap`
      - The `tl_bitmap` marks the available transports by seeing the `lanes`.
    - `iface`
      - The interface wrapping `uct iface`, adding extra attributes. Such as `overhead`, `latency`, `bandwidth`, etc for ucp to choose the best transport.

## 3. Optimize System

1. Below are the current configurations for OpenMPI and UCX in the system. Based on your learning, what methods can you use to optimize single-node performance by setting UCX environment variables?

```bash
-------------------------------------------------------------------
/opt/modulefiles/openmpi/ucx-pp:

module-whatis   {OpenMPI 4.1.6}
conflict        mpi
module          load ucx/1.15.0
prepend-path    PATH /opt/openmpi-4.1.6/bin
prepend-path    LD_LIBRARY_PATH /opt/openmpi-4.1.6/lib
prepend-path    MANPATH /opt/openmpi-4.1.6/share/man
prepend-path    CPATH /opt/openmpi-4.1.6/include
setenv          UCX_TLS ud_verbs
setenv          UCX_NET_DEVICES ibp3s0:1
-------------------------------------------------------------------
```

1. Please use the following commands to test different data sizes for latency and bandwidth, to verify your ideas:

    ```bash
    module load openmpi/ucx-pp
    mpiucx -n 2 $HOME/UCX-lsalab/test/mpi/osu/pt2pt/osu_latency
    mpiucx -n 2 $HOME/UCX-lsalab/test/mpi/osu/pt2pt/osu_bw
    ```

2. Please create a chart to illustrate the impact of different parameter options on various data sizes and the effects of different testsuite.
3. Based on the chart, explain the impact of different TLS implementations and hypothesize the possible reasons (references required).

### Answers of section optimize system

- Simply run `export UCX_TLS=all` improves the performance a lot.
  - It was set to `ud_verbs` in the original configuration, which uses IPC for communication, required cross node communication, but slower in single-node communication
  - By setting `UCX_TLS=all`, it enables all the available transports hence the sysv memory communication, which is faster in single-node communication.
  - For further optimization, we can set to `sysv` which is the fastest in single-node communication.
- Charts
    ![Latency](https://i.imgur.com/glL66ZP.png)
    ![Bandwidth](https://i.imgur.com/ZliiCCT.png)
  - Both latency and bandwidth are improved significantly by setting `UCX_TLS` to `all`, `shm`, `sysv` or `posix`.
  - Bandwidth of `ud_verbs` saturates at 2.4GB/s, while `shm`, `sysv` has about 8GB/s in average.
    - This is because `ud_verbs` uses IPC and requires extra management, which introduces many redundant overheads in single-node communication.
  - It is notable that bandwidth even goes up to 10GB/s at size of 8MB, 64MB, and 128MB in `posix` and `sysv`
  - References
    - [OpenUCX Frequently Asked Questions](https://openucx.readthedocs.io/en/master/faq.html)
    - PP 2024 Slides Chap10_UCX

### Advanced Challenge: Multi-Node Testing

This challenge involves testing the performance across multiple nodes. You can accomplish this by utilizing the sbatch script provided below. The task includes creating tables and providing explanations based on your findings. Notably, Writing a comprehensive report on this exercise can earn you up to 5 additional points.

- For information on sbatch, refer to the documentation at [Slurm's sbatch page](https://slurm.schedmd.com/sbatch.html).
- To conduct multi-node testing, use the following command:

```bash
cd ~/UCX-lsalab/test/
sbatch run.batch
```

#### Multi-Node Testing Report

- 6 times average of OSU latency and bandwidth benchmark
  ![Latency](https://i.imgur.com/gvZZprO.png)
  ![Bandwidth](https://i.imgur.com/LaGEpOQ.png)
- In multinode testing, we can see the benefit of ud_verbs or rc_verbs(all) in latency and bandwidth. Utilizing the InfiniBand network, the performance is significantly better than TCP
- Notably, choosing all gives both higher bandwidth and slightly lower latency than ud_verbs, showing that UCX can optimize the performance by selecting the best transport layer for us.

## 4. Experience & Conclusion

1. What have you learned from this homework?|
    - A lot. I learned how UCX chooses and uses the TLS between interaction of `ucp_context`, `ucp_worker`, and `ucp_ep`, and tried to use different TLS to observe the performance difference.
2. How long did you spend on the assignment?
    - One day, about 14 hours.
3. Feedback (optional)
    - Very good homework, learned a lot from it.
