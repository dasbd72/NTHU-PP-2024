import os
from .sync import sync
from .build import build
import argparse
from typing import Literal


class Args:
    local_dir = False
    version: Literal["pthread", "omp", "hybrid"]
    testcase = None


args = argparse.ArgumentParser()
args.add_argument("version", type=str, choices=["pthread", "omp", "hybrid"])
args.add_argument("testcase", type=str)


def read_testcase(path: str) -> dict:
    tc = {}
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            splitted = line.split("=")
            if len(splitted) != 2:
                continue
            key, value = splitted
            tc[key.strip()] = value.strip()
    return tc


if __name__ == "__main__":
    args: Args = args.parse_args()
    version = args.version
    testcase = args.testcase
    if not version:
        print("Version not provided")
        exit(1)
    if not testcase:
        print("Testcase not provided")
        exit(1)
    print(f"Running testcase {testcase} with version {version}")
    # testcase_dir =
    if version == "pthread":
        testcase_dir = "testcases_pthread"
    elif version == "omp":
        testcase_dir = "testcases_omp"
    elif version == "hybrid":
        testcase_dir = "testcases_hybrid"
    else:
        raise ValueError(f"Invalid version {version}")
    testcase_txt = f"{testcase_dir}/{testcase}"
    # Check if the testcase exists
    if not os.path.exists(testcase_txt):
        print("Testcase not found")
        exit(1)
    # Sync the files
    sync()
    # Build the program
    code = build()
    if code != 0:
        print("Build failed")
        exit(1)
    tc = read_testcase(testcase_txt)
    # Run the program
    if version == "pthread":
        cmd = f"srun -T {tc['ncpus']} ./lab2_pthread {tc['r']} {tc['k']}"
    elif version == "omp":
        cmd = f"srun -T {tc['ncpus']} ./lab2_omp {tc['r']} {tc['k']}"
    elif version == "hybrid":
        cmd = f"srun -n {tc['nproc']} -T {tc['ncpus']} ./lab2_hybrid {tc['r']} {tc['k']}"
    else:
        raise ValueError(f"Invalid version {version}")
    print(cmd)
    print("=====================================")
    code = os.system(cmd)
    print("=====================================")
    answer = tc["answer"]
    print(f"{cmd} finished with code {code}")
    print(f"Expected answer: {answer}")
    if code != 0:
        print("Execution failed")
        exit(1)
    exit(0)
