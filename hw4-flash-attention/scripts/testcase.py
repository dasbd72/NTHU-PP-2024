import argparse
import os
import shutil

from .build import build


class Args:
    verify = False
    epsilon = 5e-3
    local_dir = False
    testcase_dir = "testcases"
    profiler = "nsys"
    nsys_args = ""
    nvprof_args = ""
    report_name = None
    quiet = False
    testcase = None


def parse_arguments() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify output against expected results",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=5e-3,
        help="Epsilon for floating point comparisons",
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
    parser.add_argument(
        "--profiler",
        type=str,
        choices=["nsys", "nvprof", "none"],
        default="none",
        help="Profiling tool",
    )
    parser.add_argument(
        "--nsys-args",
        type=str,
        default="",
        help="Additional arguments for nsys",
    )
    parser.add_argument(
        "--nvprof-args",
        type=str,
        default="",
        help="Additional arguments for nvprof",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        help="Name of the report (only used with --profiler=nsys)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress output",
    )
    parser.add_argument("testcase", type=str, help="Testcase name")
    args: Args = parser.parse_args()
    return args


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
    outputs_bin = f"{outputs_dir}/{testcase}.out"

    # Validate testcase files
    testcase_in = f"{args.testcase_dir}/{testcase}"
    testcase_bin = f"{args.testcase_dir}/{testcase}.out"
    validate_files_exist([testcase_in])

    if args.verify:
        validate_files_exist([testcase_bin])

    ensure_output_directory_exists(outputs_dir)

    # Build the program
    if build(args.quiet) != 0:
        print("Build failed")
        exit(1)

    # Run the program
    clean_old_output(outputs_bin)
    execute_program(testcase_in, outputs_bin, args)

    # Verification if needed
    if args.verify:
        if not verify_output(
            outputs_bin, testcase_bin, args.epsilon, args.quiet
        ):
            print("Verification failed")
            exit(1)
    exit(0)


def clean_old_output(outputs_bin):
    """Removes old output files."""
    if os.path.exists(outputs_bin):
        os.remove(outputs_bin)


def execute_program(testcase_in, outputs_bin, args: Args):
    """Executes the program with the given configuration."""
    cmd_srun = f"srun -N {1} -n {1} -c {1} --gres=gpu:1"
    cmd_prog = f"./hw4 {testcase_in} {outputs_bin}"

    if args.report_name is None:
        report_name = args.testcase
    else:
        report_name = args.report_name

    if args.profiler == "nsys":
        outputs_report = f"nsys-reports/{report_name}/report"
        os.makedirs(os.path.dirname(outputs_report), exist_ok=True)
        cmd = f"{cmd_srun} nsys profile -t nvtx,cuda --stats=true -f true -o {outputs_report} {args.nsys_args} {cmd_prog}"
    elif args.profiler == "nvprof":
        cmd = f"{cmd_srun} nvprof {args.nvprof_args} {cmd_prog}"
    else:
        cmd = f"{cmd_srun} {cmd_prog}"

    if not args.quiet:
        print(cmd)
        print("========== Program Output ==========")
    code = os.system(cmd)
    if not args.quiet:
        print("====================================")
        print(f"{cmd} finished with code {code}")

    if code != 0:
        print("Execution failed")
        exit(1)


def verify_output(outputs_bin, testcase_bin, epsilon, quiet=False):
    """Verifies the output of the program."""
    if not quiet:
        print("============ Verifying =============")
        print("Result: ", end="", flush=True)
    os.chdir("utils")
    if not quiet:
        os.system("make diff")
    else:
        os.system("make diff -q")
    os.chdir("..")
    code = (
        os.system(f"./utils/diff {testcase_bin} {outputs_bin} {epsilon}") == 0
    )
    if not quiet:
        print("====================================")
    return code


if __name__ == "__main__":
    args = parse_arguments()
    run_testcase(args.testcase, args)
