import argparse
import os

from .build import build
from .sync import sync


class Args:
    verify = False
    local_dir = False
    testcase_dir = "testcases"
    nodes = 1
    procs = 1
    cpus = 1
    profile = "nsys"
    program = "hw2a"
    PROGRAMS = ["hw2a", "hw2b"]
    testcase = None


args = argparse.ArgumentParser()
args.add_argument("--verify", action="store_true")
args.add_argument("--local-dir", action="store_true")
args.add_argument("--testcase-dir", type=str, default="testcases")
args.add_argument("--nodes", "-N", type=int, default=1)
args.add_argument("--procs", "-n", type=int, default=1)
args.add_argument("--cpus", "-c", type=int, default=1)
args.add_argument(
    "--profile", type=str, choices=["nsys", "vtune", "none"], default="none"
)
args.add_argument("program", type=str, choices=Args.PROGRAMS)
args.add_argument("testcase", type=str)


if __name__ == "__main__":
    args: Args = args.parse_args()
    testcase = args.testcase
    if not testcase:
        print("Testcase not provided")
        exit(1)
    print(f"Running testcase {testcase}")
    testcase_txt = f"{args.testcase_dir}/{testcase}.txt"
    testcase_png = f"{args.testcase_dir}/{testcase}.png"
    if args.local_dir:
        outputs_dir = "outputs"
    else:
        outputs_dir = "/share/judge_dir/.judge_exe.pp24s105"
    outputs_png = f"{outputs_dir}/{testcase}.png"
    # Check if the testcase exists
    if not os.path.exists(testcase_txt):
        print("Testcase not found")
        exit(1)
    if not os.path.exists(testcase_png):
        print("Testcase image not found")
        exit(1)
    if os.path.exists(outputs_dir):
        # Check mode 700
        if os.stat(outputs_dir).st_mode & 0o777 != 0o700:
            os.chmod(outputs_dir, 0o700)
    os.makedirs(outputs_dir, exist_ok=True, mode=0o700)
    # Sync the program
    sync()
    # Build the program
    code = build()
    if code != 0:
        print("Build failed")
        exit(1)
    # Read the testcase
    with open(testcase_txt, "r") as f:
        tc = f.readline().strip()
    # Remove old output
    if os.path.exists(outputs_png):
        os.remove(outputs_png)
    # Run the program
    cmd_srun = f"srun -N {args.nodes} -n {args.procs} -c {args.cpus}"
    cmd_prog = f"./{args.program} {outputs_png} {tc}"
    if args.profile == "nsys":
        if args.program == "hw2a":
            outputs_report = f"nsys-reports/hw2a/{testcase}"
            os.makedirs("nsys-reports/hw2a", exist_ok=True)
            cmd = f"{cmd_srun} nsys profile -t openmp,nvtx,ucx,osrt --stats=true -f true -o {outputs_report} {cmd_prog}"
        elif args.program == "hw2b":
            outputs_report = f"nsys-reports/hw2b/{testcase}"
            os.makedirs("nsys-reports/hw2b", exist_ok=True)
            cmd = (
                f"{cmd_srun} ./scripts/wrapper.sh {outputs_report} {cmd_prog}"
            )
    elif args.profile == "vtune":
        if args.program == "hw2a":
            outputs_report = f"vtune-reports/hw2a/{testcase}"
            os.makedirs("vtune-reports/hw2a", exist_ok=True)
            cmd = f"{cmd_srun} vtune -collect hotspots -r {outputs_report} -- {cmd_prog}"
    else:
        cmd = f"{cmd_srun} {cmd_prog}"
    print(cmd)
    print("========== Program Output ==========")
    code = os.system(cmd)
    print("====================================")
    print(f"{cmd} finished with code {code}")
    if code != 0:
        print("Execution failed")
        exit(1)
    if not args.verify:
        exit(0)

    # Compare the output
    code = os.system(f"hw2-diff {testcase_png} {outputs_png}")
    if code != 0:
        print("Verification failed")
        exit(1)
    print("Test passed")
    exit(0)
