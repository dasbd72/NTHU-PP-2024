import os
import signal
from typing import Literal


def sigint_handler(signum, frame):
    raise KeyboardInterrupt()


def benchmark(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    emb_dim: int,
    impl: Literal["Pytorch", "Flash2"],
    causal: bool,
    output: str,
):
    if causal:
        causal = "--causal"
    else:
        causal = ""
    return os.system(
        "python -m lab5 --batch_size {} --seq_len {} --num_heads {} --emb_dim {} --impl {} {} --output {}".format(
            batch_size, seq_len, num_heads, emb_dim, impl, causal, output
        )
    )


def main():
    os.makedirs("benchmark_results", exist_ok=True)
    for batch_size in [1, 16, 32, 64]:
        for seq_len in [128, 1024, 2048]:
            for num_heads in [4, 16, 64]:
                for emb_dim_mult in [4, 32, 128]:
                    emb_dim = num_heads * emb_dim_mult
                    for impl in ["Pytorch", "Flash2"]:
                        for causal in [False, True]:
                            output = "benchmark_results/benchmark_result_{:03d}_{:05d}_{:03d}_{:04d}_{}_{}.json".format(
                                batch_size,
                                seq_len,
                                num_heads,
                                emb_dim,
                                impl.lower(),
                                str(causal).lower(),
                            )
                            if os.path.exists(output):
                                print(f"Skipping {output}")
                                continue
                            try:
                                code = benchmark(
                                    batch_size,
                                    seq_len,
                                    num_heads,
                                    emb_dim,
                                    impl,
                                    causal,
                                    output,
                                )
                                if code != 0:
                                    print(
                                        "Benchmark failed with the following configuration:"
                                    )
                                    print(
                                        f"batch_size: {batch_size}, seq_len: {seq_len}, num_heads: {num_heads}, emb_dim: {emb_dim}, impl: {impl}, causal: {causal}, output: {output}"
                                    )
                                    continue
                            except Exception as e:
                                print(
                                    f"Exception occurred with the following configuration:"
                                )
                                print(
                                    f"batch_size: {batch_size}, seq_len: {seq_len}, num_heads: {num_heads}, emb_dim: {emb_dim}, impl: {impl}, causal: {causal}, output: {output}"
                                )
                                print(e)
                                continue


if __name__ == "__main__":
    signal.signal(signal.SIGINT, sigint_handler)
    main()
