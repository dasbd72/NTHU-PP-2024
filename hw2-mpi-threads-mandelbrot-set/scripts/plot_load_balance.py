import os
import pandas as pd
import matplotlib.pyplot as plt
import traceback


# Define a function to categorize time based on the function name
def categorize_function(func_name):
    func_name = func_name.strip(":")
    if func_name in ["write_png_rows"]:
        return "I/O"
    elif func_name in ["mpi_reduce"]:
        return "Communication"
    elif func_name in [
        "partial_mandelbrot_single_thread",
        "pixels_to_image_single_thread",
    ]:
        return "CPU"
    elif func_name in ["thread_critical"]:
        return "Critical"
    else:
        return None


# Load data for each node configuration and categorize the functions
def load_data_mpi(testcase, nodes, procs, num_threads):
    total_times = {}
    for rank in range(procs):
        total_times[rank] = 0

    report_dir = (
        f"./nsys-reports/hw2b/{testcase}-N{nodes}-n{procs}-c{num_threads}"
    )

    for rank in range(procs):
        if procs < 10:
            csv_path = f"{report_dir}/report_{rank}_nvtx_sum.csv"
        else:
            csv_path = f"{report_dir}/report_{rank:02d}_nvtx_sum.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                category = categorize_function(row["Range"])
                if category in ["CPU"]:
                    total_times[rank] += row["Total Time (ns)"]
        else:
            print(f"File not found: {csv_path}")

        # Average the total times across all processes and threads
        total_times[rank] /= num_threads

        # Convert ns to seconds for better readability
        total_times[rank] /= 1e9
    return total_times


def plot_one_mpi(testcase, nodes_list, procs_list, threads):
    os.makedirs("outputs", exist_ok=True)

    # Collect data for each node configuration
    scalability_data = []

    for nodes, procs in zip(nodes_list, procs_list):
        data = load_data_mpi(testcase, nodes, procs, threads)
        print(data)

        # Plot the load balance for each node configuration
        fig, ax = plt.subplots()
        ax.barh(range(procs), data.values(), align="center")
        ax.set_title(
            f"{testcase} Load Balance (N={nodes}, n={procs}, c={threads})"
        )
        ax.set_xlabel("Total Time (s)")
        ax.set_ylabel("MPI Rank")
        ax.set_yticks(range(procs))
        ax.grid(axis="x")
        plt.savefig(
            f"outputs/{testcase}-N{nodes}-n{procs}-c{threads}-load_balance.png"
        )
        plt.close()


# Plot for each testcase
for tc in ["slow", "fast"]:
    try:
        plot_one_mpi(tc, [1, 1, 1, 1], [1, 2, 4, 8], 4)
    except Exception as e:
        print(f"Error plotting {tc}: {e}")
        print(traceback.format_exc())
