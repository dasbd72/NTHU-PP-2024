import argparse
import os
import shutil

from .build import build


class Args:
    verify = False
    local_dir = False
    testcase_dir = "testcases"
    profile = "nsys"
    report_name = None
    program = "sobel"
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
    parser.add_argument(
        "--profile",
        type=str,
        choices=["nsys", "none"],
        default="none",
        help="Profiling tool",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        help="Name of the report (only used with --profile=nsys)",
    )
    parser.add_argument(
        "program",
        type=str,
        choices=["sobel", "sobel-amd"],
        help="Program name",
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
    testcase_base = testcase.split(".")[0]
    outputs_dir = (
        "outputs" if args.local_dir else "/share/judge_dir/.judge_exe.pp24s105"
    )
    outputs_png = f"{outputs_dir}/{testcase_base}.out.png"

    # Validate testcase files
    testcase_txt = f"{args.testcase_dir}/{testcase}"
    testcase_png = f"{args.testcase_dir}/{testcase_base}.out.png"
    validate_files_exist([testcase_txt])

    if args.verify:
        validate_files_exist([testcase_png])

    ensure_output_directory_exists(outputs_dir)

    # Build the program
    if build() != 0:
        print("Build failed")
        exit(1)

    # Run the program
    tc = load_testcase_config(testcase_txt)
    clean_old_output(outputs_png)
    execute_program(tc, testcase_png, outputs_png, args)

    # Verification if needed
    if args.verify:
        if not verify_output(outputs_png, testcase_png):
            print("Verification failed")
            exit(1)
        else:
            print("Test passed")
    exit(0)


def load_testcase_config(testcase_txt):
    """Loads the configuration of the testcase."""
    tc = {}
    with open(testcase_txt, "r") as f:
        for line in f:
            key, value = line.strip().split("=")
            tc[key] = value
    return tc


def clean_old_output(outputs_png):
    """Removes old output files."""
    if os.path.exists(outputs_png):
        os.remove(outputs_png)


def execute_program(tc: dict, testcase_png, outputs_png, args: Args):
    """Executes the program with the given configuration."""
    cmd_srun = "srun -N 1 -n {} -c {} -p {} --gres=gpu:{}".format(
        tc.get("procs", 1),
        tc.get("threads", 1),
        tc.get("partition", "gpu"),
        tc.get("gpus", 1),
    )
    cmd_prog = f"./{args.program} {testcase_png} {outputs_png}"

    if args.report_name is None:
        report_name = args.testcase
    else:
        report_name = args.report_name

    if args.profile == "nsys":
        if args.program == "sobel":
            outputs_report = f"nsys-reports/sobel/{report_name}/report"
            os.makedirs(os.path.dirname(outputs_report), exist_ok=True)
            cmd = f"{cmd_srun} nsys profile -t cuda,nvtx --stats=true -f true -o {outputs_report} {cmd_prog}"
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


def verify_output(outputs_png, testcase_png):
    """Verifies the output of the program."""
    return os.system(f"png-diff {testcase_png} {outputs_png}") == 0


if __name__ == "__main__":
    args = parse_arguments()
    run_testcase(args.testcase, args)
