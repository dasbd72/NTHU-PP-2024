import os
from .sync import sync
from .build import build
import argparse
from typing import Literal


class Args:
    profile: Literal["nsys", "vtune"]
    version: Literal["pthread", "omp", "hybrid"]
    testcase = None


args = argparse.ArgumentParser()
args.add_argument("--profile", type=str, choices=["nsys", "vtune"])
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
    if version == "pthread" or version == "omp":
        srun_cmd = f"srun -n 1 -c {tc['ncpus']}"
    elif version == "hybrid":
        srun_cmd = f"srun -n {tc['nproc']} -c {tc['ncpus']}"
    else:
        raise ValueError(f"Invalid version {version}")
    if args.profile == "nsys":
        os.makedirs("nsys_reports", exist_ok=True)
        if version == "pthread" or version == "omp":
            profiler_cmd = f"nsys profile -t nvtx,openmp -f true -o nsys_reports/{version}_{testcase}.nsys-rep --stats=true"
        elif version == "hybrid":
            profiler_cmd = f"nsys profile -t mpi,nvtx,openmp -f true -o nsys_reports/{version}_{testcase}.nsys-rep --mpi-impl openmpi --stats=true"
    elif args.profile == "vtune":
        os.makedirs("vtune_reports", exist_ok=True)
        target_path = f"vtune_reports/{version}_{testcase}"
        for file in os.listdir("vtune_reports"):
            if file.startswith(f"{version}_{testcase}"):
                os.system(f"rm -rf vtune_reports/{file}")
        if version == "pthread" or version == "omp":
            profiler_cmd = f"vtune -collect hotspots -r {target_path} --"
        elif version == "hybrid":
            profiler_cmd = f"vtune -collect hotspots -r {target_path} --"
    else:
        profiler_cmd = ""
    if version == "pthread":
        program_cmd = "./lab2_pthread"
    elif version == "omp":
        program_cmd = "./lab2_omp"
    elif version == "hybrid":
        program_cmd = "./lab2_hybrid"
    else:
        raise ValueError(f"Invalid version {version}")
    args_cmd = f"{tc['r']} {tc['k']}"
    cmd = f"{srun_cmd} {profiler_cmd} {program_cmd} {args_cmd}"
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
