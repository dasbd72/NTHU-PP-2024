import os
import shutil
import subprocess

versions = [
    "1-hw4.cu",
    "2-hw4.cu",
    "3-hw4.cu",
    "4-hw4.cu",
    "5-hw4.cu",
    "6-hw4.cu",
    "7-hw4.cu",
]

iters = 3
target_path = "hw4.cu"
source_dir = "versions/experiment"
results_dir = "versions/experiment/results"

for version in versions:
    file_path = os.path.join(source_dir, version)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")

for version in versions:
    file_path = os.path.join(source_dir, version)
    os.remove(target_path)
    shutil.copy(file_path, target_path)

    # python -m scripts.testcase t02 --verify --profile nvprof
    results_1 = []
    # python -m scripts.testcase t02 --verify --profile nvprof --nvprof-arg "--metrics achieved_occupancy,sm_efficiency,gld_throughput,gst_throughput,shared_load_throughput,shared_store_throughput --events shared_ld_bank_conflict,shared_st_bank_conflict"
    results_2 = []

    for i in range(iters):
        result = subprocess.run(
            [
                "python",
                "-m",
                "scripts.testcase",
                "t02",
                "--verify",
                "--profile",
                "nvprof",
                "-q",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        results_1.append(result)

    for i in range(iters):
        result = subprocess.run(
            [
                "python",
                "-m",
                "scripts.testcase",
                "t02",
                "--verify",
                "--profile",
                "nvprof",
                "--nvprof-arg",
                "--metrics achieved_occupancy,sm_efficiency,gld_throughput,gst_throughput,shared_load_throughput,shared_store_throughput --events shared_ld_bank_conflict,shared_st_bank_conflict",
                "-q",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        results_2.append(result)

    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/{version}.txt", "w") as f:
        f.write("Results 1\n")
        for i, result in enumerate(results_1):
            f.write(f"Result {i}\n")
            f.write(result.stdout)
            f.write(result.stderr)
        f.write("Results 2\n")
        for i, result in enumerate(results_2):
            f.write(f"Result {i}\n")
            f.write(result.stdout)
            f.write(result.stderr)
