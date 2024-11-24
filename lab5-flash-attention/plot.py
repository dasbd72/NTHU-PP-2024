import os
import json
import pandas as pd
import matplotlib.pyplot as plt


def process_benchmark_results(directory):
    results = []

    for file_name in os.listdir(directory):
        if not file_name.endswith(".json"):
            continue

        file_path = os.path.join(directory, file_name)
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
                # Extract parameters from the file name
                parts = file_name.split(".")[0].split("_")
                batch_size = int(parts[2])
                seq_len = int(parts[3])
                num_heads = int(parts[4])
                emb_dim = int(parts[5])
                impl = parts[6]
                causal = parts[7] == "true"

                results.append(
                    {
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "num_heads": num_heads,
                        "emb_dim": emb_dim,
                        "impl": impl,
                        "causal": causal,
                        "forward_time": data["forward"]["time(s)"],
                        "forward_flops": data["forward"]["FLOPS(TFLOPs/s)"],
                        "backward_time": data["backward"]["time(s)"],
                        "backward_flops": data["backward"]["FLOPS(TFLOPs/s)"],
                        "forward_backward_time": data["forward_backward"][
                            "time(s)"
                        ],
                        "forward_backward_flops": data["forward_backward"][
                            "FLOPS(TFLOPs/s)"
                        ],
                        "peak_memory": data["peak_memory_usage(MB)"],
                    }
                )
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Skipping file {file_name} due to error: {e}")

    return results


def plot_seq_lens(df, filters):
    for key, value in filters.items():
        df = df[df[key] == value]
    df = df.sort_values(by=["impl", "batch_size", "seq_len"])
    impls = df["impl"].unique()
    batch_sizes = df["batch_size"].unique()
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    for i, impl in enumerate(impls):
        df_impl = df[df["impl"] == impl]
        for batch_size in batch_sizes:
            df_batch = df_impl[df_impl["batch_size"] == batch_size]
            ax[i, 0].plot(
                df_batch["seq_len"],
                df_batch["forward_backward_flops"],
                label=batch_size,
            )
            ax[i, 1].plot(
                df_batch["seq_len"],
                df_batch["peak_memory"],
                label=batch_size,
            )
        ax[i, 0].set_title(f"{impl} FLOPS")
        ax[i, 0].set_xlabel("Sequence Length")
        ax[i, 0].set_xticks(df["seq_len"].unique())
        ax[i, 0].set_ylabel("TFLOPs/s")
        ax[i, 0].legend()
        ax[i, 1].set_title(f"{impl} Peak Memory Usage")
        ax[i, 1].set_xlabel("Sequence Length")
        ax[i, 1].set_xticks(df["seq_len"].unique())
        ax[i, 1].set_ylabel("Memory (MB)")
        ax[i, 1].legend()
    plt.savefig("seq_lens.png")


def plot_num_heads(df, filters):
    for key, value in filters.items():
        df = df[df[key] == value]
    df = df.sort_values(by=["impl", "batch_size", "num_heads"])
    impls = df["impl"].unique()
    batch_sizes = df["batch_size"].unique()
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    for i, impl in enumerate(impls):
        df_impl = df[df["impl"] == impl]
        for batch_size in batch_sizes:
            df_batch = df_impl[df_impl["batch_size"] == batch_size]
            ax[i, 0].plot(
                df_batch["num_heads"],
                df_batch["forward_backward_flops"],
                label=batch_size,
            )
            ax[i, 1].plot(
                df_batch["num_heads"],
                df_batch["peak_memory"],
                label=batch_size,
            )
        ax[i, 0].set_title(f"{impl} FLOPS")
        ax[i, 0].set_xlabel("Number of Heads")
        ax[i, 0].set_xticks(df["num_heads"].unique())
        ax[i, 0].set_ylabel("TFLOPs/s")
        ax[i, 0].legend()
        ax[i, 1].set_title(f"{impl} Peak Memory Usage")
        ax[i, 1].set_xlabel("Number of Heads")
        ax[i, 1].set_xticks(df["num_heads"].unique())
        ax[i, 1].set_ylabel("Memory (MB)")
        ax[i, 1].legend()
    plt.savefig("num_heads.png")


def plot_emb_dims(df, filters):
    for key, value in filters.items():
        df = df[df[key] == value]
    df = df.sort_values(by=["impl", "batch_size", "emb_dim"])
    impls = df["impl"].unique()
    batch_sizes = df["batch_size"].unique()
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    for i, impl in enumerate(impls):
        df_impl = df[df["impl"] == impl]
        for batch_size in batch_sizes:
            df_batch = df_impl[df_impl["batch_size"] == batch_size]
            ax[i, 0].plot(
                df_batch["emb_dim"],
                df_batch["forward_backward_flops"],
                label=batch_size,
            )
            ax[i, 1].plot(
                df_batch["emb_dim"],
                df_batch["peak_memory"],
                label=batch_size,
            )
        ax[i, 0].set_title(f"{impl} FLOPS")
        ax[i, 0].set_xlabel("Embedding Dimension")
        ax[i, 0].set_xticks(df["emb_dim"].unique())
        ax[i, 0].set_ylabel("TFLOPs/s")
        ax[i, 0].legend()
        ax[i, 1].set_title(f"{impl} Peak Memory Usage")
        ax[i, 1].set_xlabel("Embedding Dimension")
        ax[i, 1].set_xticks(df["emb_dim"].unique())
        ax[i, 1].set_ylabel("Memory (MB)")
        ax[i, 1].legend()
    plt.savefig("emb_dims.png")


if __name__ == "__main__":
    benchmark_dir = "benchmark_results"
    results = process_benchmark_results(benchmark_dir)
    df = pd.DataFrame(results)
    plot_seq_lens(
        df,
        {
            "num_heads": 16,
            "emb_dim": 2048,
            "causal": False,
        },
    )
    plot_num_heads(
        df,
        {
            "seq_len": 128,
            "emb_dim": 1024,
            "causal": False,
        },
    )
    plot_emb_dims(
        df,
        {
            "seq_len": 128,
            "num_heads": 16,
            "causal": False,
        },
    )
