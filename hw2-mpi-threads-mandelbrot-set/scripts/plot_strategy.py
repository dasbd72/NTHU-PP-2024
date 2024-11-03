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
def load_data_mpi(strategy, nodes, procs, num_threads):
    total_times = {"I/O": 0, "CPU": 0, "Communication": 0, "Critical": 0}
    report_dir = f"./nsys-reports/hw2b/{strategy}"

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


def plot_one_mpi(strategies, nodes, procs, threads):
    os.makedirs("outputs", exist_ok=True)

    # Collect data for each strategy
    strategy_data = []
    baseline = None

    for strategy in strategies:
        data = load_data_mpi(strategy, nodes, procs, threads)
        if baseline is None:
            baseline = sum(
                data.values()
            )  # Use the time for 1 node as the baseline
        data["Total Time"] = sum(data.values())
        data["Strategy"] = strategy
        data["Speedup"] = baseline / data["Total Time"]
        strategy_data.append(data)

    # Convert data to DataFrame for plotting
    strategy_df = pd.DataFrame(strategy_data)

    # Plotting scalability as a stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ["I/O", "CPU", "Communication", "Critical"]
    strategy_df.set_index("Strategy")[categories].plot(
        kind="bar", stacked=True, ax=ax, colormap="tab20b"
    )
    ax.set_title(f"Breakdown of Time Spent with Different Strategies")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Time (s)")
    ax.legend(title="Category")
    plt.xticks(rotation=0)
    plt.savefig(f"outputs/mpi_strategies_breakdown.png")

    # Plotting speedup as a line chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        strategy_df["Strategy"],
        strategy_df["Speedup"],
        marker="o",
        linestyle="-",
    )
    ax.set_xticks(strategy_df["Strategy"])
    ax.set_title(f"Speedup of Different Strategies")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Speedup")
    plt.grid(False)
    plt.xticks(rotation=0)
    plt.savefig(f"outputs/mpi_strategies_speedup.png")


strategies = [
    "baseline",
    "avx",
    "avx_cb",
    "mpi_crs",
]
try:
    plot_one_mpi(strategies, 1, 4, 4)
except Exception as e:
    print(f"Error plotting {strategies}: {e}")
    print(traceback.format_exc())
