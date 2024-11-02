import os
import pandas as pd
import matplotlib.pyplot as plt
import traceback


# Define a function to categorize time based on the function name
def categorize_function(func_name):
    func_name = func_name.strip(":")
    if func_name in ["mpi_read", "mpi_write"]:
        return "I/O"
    elif func_name in [
        "mpi_exchange_left",
        "mpi_exchange_right",
        "mpi_pre_exchange_left",
        "mpi_pre_exchange_right",
    ]:
        return "Communication"
    elif func_name in ["local_sort", "merge_left", "merge_right"]:
        return "CPU"
    else:
        return None


# Load data for each node configuration and categorize the functions
def load_data(testcase, strategy, procs):
    total_times = {"I/O": 0, "CPU": 0, "Communication": 0}
    report_dir = "./nsys-reports/{}_{}".format(testcase, strategy)

    for rank in range(procs):
        csv_path = f"{report_dir}/report_{rank:02d}_nvtx_sum.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                category = categorize_function(row["Range"])
                if category is not None:
                    total_times[category] += row["Total Time (ns)"]
        else:
            print(f"File not found: {csv_path}")

    # Average the total times across all processes
    for key in total_times:
        total_times[key] /= procs

    # Convert ns to seconds for better readability
    for key in total_times:
        total_times[key] /= 1e9
    return total_times


procs = 16
strategy_list = [
    ("opt_none", "Baseline"),
    ("opt_reduce_io", "Reduce I/O"),
    ("opt_min_comm", "Minimum Communication"),
    ("opt_min_merge", "Minimum Merge"),
    ("opt_buff_swap", "Buffer Swapping"),
]


def plot_one(testcase):
    os.makedirs("outputs", exist_ok=True)

    # Collect data for each node configuration
    strategy_data = []
    baseline = None

    for strategy_id, strategy_name in strategy_list:
        data = load_data(testcase, strategy_id, procs)
        if baseline is None:
            baseline = sum(data.values())
        data["Total Time"] = sum(data.values())
        data["Strategy"] = strategy_name
        data["Speedup"] = baseline / data["Total Time"]
        strategy_data.append(data)

    # Convert data to DataFrame for plotting
    scalability_df = pd.DataFrame(strategy_data)

    # Plotting scalability as a stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ["I/O", "CPU", "Communication"]
    scalability_df.set_index("Strategy")[categories].plot(
        kind="bar", stacked=True, ax=ax, colormap="tab20b"
    )
    ax.set_title(
        f"Time Breakdown of {testcase} with Different Optimization Strategies"
    )
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Time (s)")
    ax.legend(title="Category")
    plt.xticks(rotation=0)
    plt.savefig(f"outputs/strategies_{testcase}_stacked_bar_chart.png")

    # Plotting speedup as a line chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        scalability_df["Strategy"],
        scalability_df["Speedup"],
        marker="o",
        linestyle="-",
    )
    ax.set_xticks(scalability_df["Strategy"])
    ax.set_title(
        f"Speedup of {testcase} with Different Optimization Strategies"
    )
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Speedup")
    plt.grid(False)
    plt.savefig(f"outputs/strategies_{testcase}_speedup.png")


# Plot for each testcase
for tc in ["rand_l", "skew_l"]:
    try:
        plot_one(tc)
    except Exception as e:
        print(f"Error plotting {tc}: {e}")
        print(traceback.format_exc())
