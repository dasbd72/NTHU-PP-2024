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
def load_data(testcase, num_threads):
    total_times = {"I/O": 0, "CPU": 0, "Critical": 0}
    report_dir = f"./nsys-reports/hw2a/{testcase}-c{num_threads}"

    csv_path = f"{report_dir}/report_nvtx_sum.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            category = categorize_function(row["Range"])
            if category is not None:
                total_times[category] += row["Total Time (ns)"]
    else:
        print(f"File not found: {csv_path}")

    # Average the total times across all threads
    total_times["CPU"] /= num_threads

    # Convert ns to seconds for better readability
    for key in total_times:
        total_times[key] /= 1e9
    return total_times


# Load data for each node configuration and categorize the functions
def load_data_mpi(testcase, nodes, procs, num_threads):
    total_times = {"I/O": 0, "CPU": 0, "Communication": 0, "Critical": 0}
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
                if category in ["Communication", "MPI Init"]:
                    total_times[category] = max(
                        total_times[category], row["Total Time (ns)"]
                    )
                elif category is not None:
                    total_times[category] += row["Total Time (ns)"]
        else:
            print(f"File not found: {csv_path}")

    # Average the total times across all processes and threads
    total_times["CPU"] /= procs * num_threads

    # Convert ns to seconds for better readability
    for key in total_times:
        total_times[key] /= 1e9
    return total_times


def plot_one(testcase, threads_list):
    os.makedirs("outputs", exist_ok=True)

    # Collect data for each node configuration
    scalability_data = []
    baseline = None

    for threads in threads_list:
        data = load_data(testcase, threads)
        if baseline is None:
            baseline = sum(
                data.values()
            )  # Use the time for 1 node as the baseline
        data["Total Time"] = sum(data.values())
        data["Threads"] = threads
        data["Speedup"] = baseline / data["Total Time"]
        scalability_data.append(data)

    # Convert data to DataFrame for plotting
    scalability_df = pd.DataFrame(scalability_data)

    # Plotting scalability as a stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ["I/O", "CPU", "Critical"]
    scalability_df.set_index("Threads")[categories].plot(
        kind="bar", stacked=True, ax=ax, colormap="tab20b"
    )
    ax.set_title(f"Scalability of {testcase} with Different Number of Threads")
    ax.set_xlabel("# of Threads (1 node)")
    ax.set_ylabel("Time (s)")
    ax.legend(title="Category")
    plt.savefig(f"outputs/pthreads_{testcase}_stacked_bar_chart.png")

    # Plotting speedup as a line chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        scalability_df["Threads"],
        scalability_df["Speedup"],
        marker="o",
        linestyle="-",
    )
    ax.set_xticks(scalability_df["Threads"])
    ax.set_title(f"Speedup of {testcase} with Different Number of Threads")
    ax.set_xlabel("# of Threads (1 node)")
    ax.set_ylabel("Speedup")
    plt.grid(False)
    plt.savefig(f"outputs/pthreads_{testcase}_speedup.png")

    # Plotting speedup as a line chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        scalability_df["Threads"],
        scalability_df["Speedup"],
        marker="o",
        linestyle="-",
    )
    # Ideal
    ax.plot(
        scalability_df["Threads"],
        scalability_df["Threads"],
        linestyle="--",
        color="gray",
    )
    ax.legend(["Speedup", "Ideal"])
    ax.set_xticks(scalability_df["Threads"])
    ax.set_title(f"Speedup of {testcase} with Different Number of Threads")
    ax.set_xlabel("# of Threads (1 node)")
    ax.set_ylabel("Speedup")
    plt.grid(False)
    plt.savefig(f"outputs/pthreads_{testcase}_speedup_vs_ideal.png")


def plot_one_mpi(testcase, nodes_list, procs_list, threads):
    os.makedirs("outputs", exist_ok=True)

    # Collect data for each node configuration
    scalability_data = []
    baseline = None

    for nodes, procs in zip(nodes_list, procs_list):
        data = load_data_mpi(testcase, nodes, procs, threads)
        if baseline is None:
            baseline = sum(
                data.values()
            )  # Use the time for 1 node as the baseline
        data["Total Time"] = sum(data.values())
        data["Processes"] = procs
        data["Speedup"] = baseline / data["Total Time"]
        scalability_data.append(data)

    # Convert data to DataFrame for plotting
    scalability_df = pd.DataFrame(scalability_data)

    # Plotting scalability as a stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ["I/O", "CPU", "Communication", "Critical"]
    scalability_df.set_index("Processes")[categories].plot(
        kind="bar", stacked=True, ax=ax, colormap="tab20b"
    )
    ax.set_title(
        f"Scalability of {testcase} with Different Number of Processes"
    )
    ax.set_xlabel("# of Processes (1 node, 4 threads per process)")
    ax.set_ylabel("Time (s)")
    ax.legend(title="Category")
    plt.savefig(f"outputs/mpi_{testcase}_stacked_bar_chart.png")

    # Plotting speedup as a line chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        scalability_df["Processes"],
        scalability_df["Speedup"],
        marker="o",
        linestyle="-",
    )
    ax.set_xticks(scalability_df["Processes"])
    ax.set_title(f"Speedup of {testcase} with Different Number of Processes")
    ax.set_xlabel("# of Processes (1 node, 4 threads per process)")
    ax.set_ylabel("Speedup")
    plt.grid(False)
    plt.savefig(f"outputs/mpi_{testcase}_speedup.png")

    # Plotting speedup as a line chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        scalability_df["Processes"],
        scalability_df["Speedup"],
        marker="o",
        linestyle="-",
    )
    # Ideal
    ax.plot(
        scalability_df["Processes"],
        scalability_df["Processes"],
        linestyle="--",
        color="gray",
    )
    ax.legend(["Speedup", "Ideal"])
    ax.set_xticks(scalability_df["Processes"])
    ax.set_title(f"Speedup of {testcase} with Different Number of Processes")
    ax.set_xlabel("# of Processes (1 node, 4 threads per process)")
    ax.set_ylabel("Speedup")
    plt.grid(False)
    plt.savefig(f"outputs/mpi_{testcase}_speedup_vs_ideal.png")


# Plot for each testcase
for tc in ["slow", "fast"]:
    try:
        plot_one(tc, range(1, 13))
    except Exception as e:
        print(f"Error plotting {tc}: {e}")
        print(traceback.format_exc())

# Plot for each testcase
for tc in ["slow", "fast"]:
    try:
        plot_one_mpi(tc, [1, 1, 1, 1], [1, 2, 4, 8], 4)
    except Exception as e:
        print(f"Error plotting {tc}: {e}")
        print(traceback.format_exc())
