import os
import struct
import tqdm
import json
import argparse
import random

username = os.getenv("USER")

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=536870911)
parser.add_argument(
    "-v",
    "--version",
    type=str,
    choices=["random", "reversed", "sorted", "skewed"],
    default="random",
)
parser.add_argument("testcase", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.n < 1:
        raise ValueError("Invalid value for n")
    if args.n > 536870911:
        raise ValueError("n is too large")
    if not args.testcase:
        raise ValueError("Testcase not provided")

    n = args.n
    testcase_dir = f"/share/judge_dir/.judge_exe.tc.{username}"
    testcase_in = f"{testcase_dir}/{args.testcase}.in"
    testcase_txt = f"{testcase_dir}/{args.testcase}.txt"
    batch_size = 100000

    if not os.path.exists(testcase_dir):
        os.makedirs(testcase_dir, exist_ok=True, mode=0o700)

    with open(testcase_txt, "w") as f:
        json.dump({"n": n, "nodes": 3, "procs": 12, "time": 60}, f)

    if not os.path.exists(testcase_in):
        with open(testcase_in, "wb") as f:
            if args.version == "random":
                random.seed(42)
                for i in tqdm.tqdm(range(0, n, batch_size)):
                    size = min(batch_size, n - i)
                    f.write(
                        struct.pack(
                            f"{size}f", *random.sample(range(1, n + 1), size)
                        )
                    )
            elif args.version == "sorted":
                for i in tqdm.tqdm(range(0, n, batch_size)):
                    size = min(batch_size, n - i)
                    f.write(
                        struct.pack(f"{size}f", *range(i + 1, i + 1 + size))
                    )
            elif args.version == "reversed":
                for i in tqdm.tqdm(range(0, n, batch_size)):
                    size = min(batch_size, n - i)
                    f.write(
                        struct.pack(
                            f"{size}f", *range(n - i, n - i - size, -1)
                        )
                    )
            elif args.version == "skewed":
                # random in [0, n)
                # ordered in [n, n + pivot)
                pivot = int(n * 0.1)
                for i in tqdm.tqdm(range(0, pivot, batch_size)):
                    size = min(batch_size, pivot - i)
                    f.write(
                        struct.pack(
                            f"{size}f",
                            *random.sample(range(n - pivot, n), size),
                        )
                    )
                for i in tqdm.tqdm(range(pivot, n, batch_size)):
                    size = min(batch_size, n - i)
                    f.write(
                        struct.pack(f"{size}f", *range(i + 1, i + 1 + size))
                    )
