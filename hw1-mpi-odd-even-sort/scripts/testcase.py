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


def parse_arguments() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify output against expected results",
    )
    parser.add_argument(
        "--local-dir",
        action="store_true",
        help="Use local directory for output",
    )
    parser.add_argument(
        "--testcase-dir",
        type=str,
        default="testcases",
        help="Directory of testcases",
    )
    parser.add_argument("--nodes", "-N", type=int, help="Number of nodes")
    parser.add_argument("--procs", "-n", type=int, help="Number of processors")
    parser.add_argument(
        "--profile",
        type=str,
        choices=["nsys", "vtune", "none"],
        default="none",
        help="Profiling tool",
    )
    parser.add_argument("testcase", type=str, help="Testcase name")
    return parser.parse_args()


def verify_output(n, outputs_out, testcase_out):
    """Compares the program's output with the expected output."""
    batch_size = 100000
    with open(outputs_out, "rb") as output_f, open(
        testcase_out, "rb"
    ) as expected_f:
        for i in range(0, n, batch_size):
            output_chunk = output_f.read(batch_size * 4)
            expected_chunk = expected_f.read(batch_size * 4)
            size = min(batch_size, n - i)
            output = struct.unpack(f"{size}f", output_chunk)
            expected = struct.unpack(f"{size}f", expected_chunk)

            if output != expected:
                print(f"Test failed at index {i}")
                print_mismatch(output, expected, size)
                return False
    return True


def print_mismatch(output, expected, size):
    """Prints mismatched output for debugging."""
    if size < 10:
        print(f"Expected: {expected}")
        print(f"Got: {output}")
    else:
        print(f"Expected: {expected[:10]}...")
        print(f"Got: {output[:10]}...")


def validate_files_exist(files):
    """Ensures all required testcase files exist."""
    for file in files:
        if not os.path.exists(file):
            print(f"{file} not found")
            exit(1)


def ensure_output_directory_exists(directory):
    """Ensures the output directory exists with correct permissions."""
    if not os.path.exists(directory):
        os.makedirs(directory, mode=0o700)
    elif os.stat(directory).st_mode & 0o777 != 0o700:
        os.chmod(directory, 0o700)


def run_testcase(testcase, args: Args):
    """Main logic for running the testcase."""
    outputs_dir = (
        "outputs" if args.local_dir else "/share/judge_dir/.judge_exe.pp24s105"
    )
    outputs_out = os.path.join(outputs_dir, f"{testcase}.out")

    # Validate testcase files
    testcase_txt = os.path.join(args.testcase_dir, f"{testcase}.txt")
    testcase_in = os.path.join(args.testcase_dir, f"{testcase}.in")
    testcase_out = os.path.join(args.testcase_dir, f"{testcase}.out")
    validate_files_exist([testcase_txt, testcase_in])

    if args.verify:
        validate_files_exist([testcase_out])

    ensure_output_directory_exists(outputs_dir)

    # Build the program
    if build() != 0:
        print("Build failed")
        exit(1)

    # Run the program
    tc = load_testcase_config(testcase_txt, args)
    clean_old_output(outputs_out)
    execute_program(tc, testcase_in, outputs_out, args)

    # Verification if needed
    if args.verify:
        if not verify_output(tc["n"], outputs_out, testcase_out):
            print("Verification failed")
            exit(1)
        else:
            print("Test passed")
    exit(0)


def load_testcase_config(testcase_txt, args: Args):
    """Loads the testcase configuration and applies node/proc overrides."""
    with open(testcase_txt) as f:
        tc = json.load(f)
    if args.nodes is not None:
        tc["nodes"] = args.nodes
    if args.procs is not None:
        tc["procs"] = args.procs
    return tc


def clean_old_output(outputs_out):
    """Removes old output files if they exist."""
    if os.path.exists(outputs_out):
        os.remove(outputs_out)


def execute_program(tc, testcase_in, outputs_out, args: Args):
    """Builds and executes the command to run the program."""
    cmd_srun = f"srun -N {tc['nodes']} -n {tc['procs']}"
    cmd_prog = f"./hw1 {tc['n']} {testcase_in} {outputs_out}"

    if args.profile == "nsys":
        report_name = args.testcase
        if args.nodes is not None:
            report_name += f"-N{args.nodes}"
        if args.procs is not None:
            report_name += f"-n{args.procs}"
        outputs_report = f"nsys-reports/{report_name}/report"
        os.makedirs(os.path.dirname(outputs_report), exist_ok=True)
        cmd = f"{cmd_srun} ./scripts/wrapper.sh {outputs_report} {cmd_prog}"
    else:
        cmd = f"{cmd_srun} {cmd_prog}"

    print(cmd)
    print("========== Program Output ==========")
    code = os.system(cmd)
    print("====================================")

    if code != 0:
        print("Execution failed with code", code)
        exit(1)


if __name__ == "__main__":
    args = parse_arguments()
    run_testcase(args.testcase, args)
