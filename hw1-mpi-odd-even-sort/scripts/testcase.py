import argparse
import json
import os
import struct

from .build import build


class Args:
    verify = False
    local_dir = False
    testcase_dir = "testcases"
    nodes = None
    procs = None
    profile = "nsys"
    testcase = None


args = argparse.ArgumentParser()
args.add_argument("--verify", action="store_true")
args.add_argument("--local-dir", action="store_true")
args.add_argument("--testcase-dir", type=str, default="testcases")
args.add_argument("--nodes", "-N", type=int)
args.add_argument("--procs", "-n", type=int)
args.add_argument(
    "--profile", type=str, choices=["nsys", "vtune", "none"], default="none"
)
args.add_argument("testcase", type=str)


def verify(n, outputs_out, testcase_out):
    with open(outputs_out, "rb") as output_f, open(
        testcase_out, "rb"
    ) as expected_f:
        batch_size = 100000
        for i in range(0, n, batch_size):
            b_output = output_f.read(batch_size * 4)
            b_expected = expected_f.read(batch_size * 4)
            size = min(batch_size, n - i)
            output = struct.unpack(f"{size}f", b_output)
            expected = struct.unpack(f"{size}f", b_expected)
            if output != expected:
                print("Test failed at index", i)
                if size < 10:
                    print(f"Expected: {expected}")
                    print(f"Got: {output}")
                else:
                    print(f"Expected: {expected[0:10]}...")
                    print(f"Got: {output[0:10]}...")
                return 1
    return 0


if __name__ == "__main__":
    args: Args = args.parse_args()
    testcase = args.testcase
    if not testcase:
        print("Testcase not provided")
        exit(1)
    print(f"Running testcase {testcase}")
    testcase_txt = f"{args.testcase_dir}/{testcase}.txt"
    testcase_in = f"{args.testcase_dir}/{testcase}.in"
    testcase_out = f"{args.testcase_dir}/{testcase}.out"
    if args.local_dir:
        outputs_dir = "outputs"
    else:
        outputs_dir = "/share/judge_dir/.judge_exe.pp24s105"
    outputs_out = f"{outputs_dir}/{testcase}.out"
    # Check if the testcase exists
    if not os.path.exists(testcase_txt):
        print("Testcase not found")
        exit(1)
    if not os.path.exists(testcase_in):
        print("Testcase input not found")
        exit(1)
    if args.verify and not os.path.exists(testcase_out):
        print("Testcase output not found")
        exit(1)
    if os.path.exists(outputs_dir):
        # Check mode 700
        if os.stat(outputs_dir).st_mode & 0o777 != 0o700:
            os.chmod(outputs_dir, 0o700)
    os.makedirs(outputs_dir, exist_ok=True, mode=0o700)
    # Build the program
    code = build()
    if code != 0:
        print("Build failed")
        exit(1)
    # Read the testcase
    tc = json.load(open(testcase_txt))
    if args.nodes is not None:
        tc["nodes"] = args.nodes
    if args.procs is not None:
        tc["procs"] = args.procs
    # Remove old output
    if os.path.exists(outputs_out):
        os.remove(outputs_out)
    # Run the program
    cmd_srun = f"srun -N {tc['nodes']} -n {tc['procs']}"
    cmd_prog = f"./hw1 {tc['n']} {testcase_in} {outputs_out}"
    if args.profile == "nsys":
        outputs_report = f"nsys-reports/{testcase}/report"
        os.makedirs(os.path.dirname(outputs_report), exist_ok=True)
        cmd = f"{cmd_srun} ./scripts/wrapper.sh {outputs_report} {cmd_prog}"
    elif args.profile == "vtune":
        outputs_report = f"vtune-reports/{testcase}"
        os.makedirs("vtune-reports", exist_ok=True)
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
    code = verify(tc["n"], outputs_out, testcase_out)
    if code != 0:
        print("Verification failed")
        exit(1)
    print("Test passed")
    exit(0)
