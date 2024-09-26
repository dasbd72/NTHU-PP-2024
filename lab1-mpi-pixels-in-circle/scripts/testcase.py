import os
import json
from .build import build
import argparse


class Args:
    local_dir = False
    testcase_dir = "testcases"
    testcase = None


args = argparse.ArgumentParser()
args.add_argument("--testcase-dir", type=str, default="testcases")
args.add_argument("testcase", type=str)


def read_testcase(path: str):
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
    testcase = args.testcase
    if not testcase:
        print("Testcase not provided")
        exit(1)
    print(f"Running testcase {testcase}")
    testcase_txt = f"{args.testcase_dir}/{testcase}.txt"
    # Check if the testcase exists
    if not os.path.exists(testcase_txt):
        print("Testcase not found")
        exit(1)
    # Build the program
    code = build()
    if code != 0:
        print("Build failed")
        exit(1)
    tc = read_testcase(testcase_txt)
    # Run the program
    cmd = f"srun -n {tc['nproc']} ./lab1 {tc['r']} {tc['k']}"
    print(cmd)
    code = os.system(cmd)
    answer = tc["answer"]
    print(f"{cmd} finished with code {code}")
    print(f"Expected answer: {answer}")
    if code != 0:
        print("Execution failed")
        exit(1)
    exit(0)
