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
def load_data(testcase, nodes, procs):
    total_times = {"I/O": 0, "CPU": 0, "Communication": 0}
    report_dir = f"./nsys-reports/{testcase}-N{nodes}-n{procs}"

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


# Define node and process setups
nodes_list = [1, 2, 4, 8]
procs_per_node = 4


def plot_one(testcase):
    os.makedirs("outputs", exist_ok=True)

    # Collect data for each node configuration
    scalability_data = []
    baseline = None

    for nodes in nodes_list:
        procs = nodes * procs_per_node
        data = load_data(testcase, nodes, procs)
        if baseline is None:
            baseline = sum(
                data.values()
            )  # Use the time for 1 node as the baseline
        data["Total Time"] = sum(data.values())
        data["Nodes"] = nodes
        data["Speedup"] = baseline / data["Total Time"]
        scalability_data.append(data)

    # Convert data to DataFrame for plotting
    scalability_df = pd.DataFrame(scalability_data)

    # Plotting scalability as a stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ["I/O", "CPU", "Communication"]
    scalability_df.set_index("Nodes")[categories].plot(
        kind="bar", stacked=True, ax=ax, colormap="tab20b"
    )
    ax.set_title(
        f"Scalability of {testcase} with Different Node Configurations"
    )
    ax.set_xlabel("# of Nodes (4 processes per node)")
    ax.set_ylabel("Time (s)")
    ax.legend(title="Category")
    plt.savefig(f"outputs/{testcase}_stacked_bar_chart.png")

    # Plotting speedup as a line chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        scalability_df["Nodes"],
        scalability_df["Speedup"],
        marker="o",
        linestyle="-",
    )
    ax.set_xticks(scalability_df["Nodes"])
    ax.set_title(f"Speedup of {testcase} with Different Node Configurations")
    ax.set_xlabel("# of Nodes (4 processes per node)")
    ax.set_ylabel("Speedup")
    plt.grid(False)
    plt.savefig(f"outputs/{testcase}_speedup.png")

    # Plotting speedup as a line chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        scalability_df["Nodes"],
        scalability_df["Speedup"],
        marker="o",
        linestyle="-",
    )
    # Ideal
    ax.plot(
        scalability_df["Nodes"],
        scalability_df["Nodes"],
        linestyle="--",
        color="gray",
    )
    ax.legend(["Speedup", "Ideal"])
    ax.set_xticks(scalability_df["Nodes"])
    ax.set_title(f"Speedup of {testcase} with Different Node Configurations")
    ax.set_xlabel("# of Nodes (4 processes per node)")
    ax.set_ylabel("Speedup")
    plt.grid(False)
    plt.savefig(f"outputs/{testcase}_speedup_vs_ideal.png")


# Plot for each testcase
for tc in ["rand_l", "rev_l", "skew_l"]:
    try:
        plot_one(tc)
    except Exception as e:
        print(f"Error plotting {tc}: {e}")
        print(traceback.format_exc())
