import argparse
import os
import shutil

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
    report_name = None
    program = "hw2a"
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
        "--nodes", "-N", type=int, help="Number of nodes", default=1
    )
    parser.add_argument(
        "--procs", "-n", type=int, help="Number of processors", default=1
    )
    parser.add_argument(
        "--cpus", "-c", type=int, help="Number of cpus", default=1
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["nsys", "vtune", "none"],
        default="none",
        help="Profiling tool",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        help="Name of the report (only used with --profile=nsys)",
    )
    parser.add_argument(
        "program", type=str, choices=["hw2a", "hw2b"], help="Program name"
    )
    parser.add_argument("testcase", type=str, help="Testcase name")
    args: Args = parser.parse_args()
    if (args.nodes > 1 or args.procs > 1) and args.program != "hw2b":
        raise ValueError("Only hw2b supports multiple nodes and processors")
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
    outputs_png = f"{outputs_dir}/{testcase}.png"

    # Validate testcase files
    testcase_txt = f"{args.testcase_dir}/{testcase}.txt"
    testcase_png = f"{args.testcase_dir}/{testcase}.png"
    validate_files_exist([testcase_txt])

    if args.verify:
        validate_files_exist([testcase_png])

    ensure_output_directory_exists(outputs_dir)

    # Sync the program
    sync()

    # Build the program
    if build() != 0:
        print("Build failed")
        exit(1)

    # Run the program
    tc = load_testcase_config(testcase_txt)
    clean_old_output(outputs_png)
    execute_program(tc, outputs_png, args)

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
    with open(testcase_txt, "r") as f:
        tc = f.readline().strip()
    return tc


def clean_old_output(outputs_png):
    """Removes old output files."""
    if os.path.exists(outputs_png):
        os.remove(outputs_png)


def execute_program(tc, outputs_png, args: Args):
    """Executes the program with the given configuration."""
    cmd_srun = f"srun -N {args.nodes} -n {args.procs} -c {args.cpus}"
    cmd_prog = f"./{args.program} {outputs_png} {tc}"

    if args.report_name is None:
        report_name = args.testcase
        if args.program == "hw2b":
            report_name += f"-N{args.nodes}"
            report_name += f"-n{args.procs}"
        report_name += f"-c{args.cpus}"
    else:
        report_name = args.report_name

    if args.profile == "nsys":
        if args.program == "hw2a":
            outputs_report = f"nsys-reports/hw2a/{report_name}/report"
            os.makedirs(os.path.dirname(outputs_report), exist_ok=True)
            cmd = f"{cmd_srun} nsys profile -t openmp,nvtx --stats=true -f true -o {outputs_report} {cmd_prog}"
        elif args.program == "hw2b":
            outputs_report = f"nsys-reports/hw2b/{report_name}/report"
            os.makedirs(os.path.dirname(outputs_report), exist_ok=True)
            cmd = (
                f"{cmd_srun} ./scripts/wrapper.sh {outputs_report} {cmd_prog}"
            )
    elif args.profile == "vtune":
        if args.program == "hw2a":
            outputs_report = f"vtune-reports/hw2a/{report_name}/report"
            if os.path.exists(os.path.dirname(outputs_report)):
                shutil.rmtree(os.path.dirname(outputs_report))
            os.makedirs(os.path.dirname(outputs_report), exist_ok=True)
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


def verify_output(outputs_png, testcase_png):
    """Verifies the output of the program."""
    return os.system(f"hw2-diff {testcase_png} {outputs_png}") == 0


if __name__ == "__main__":
    args = parse_arguments()
    run_testcase(args.testcase, args)
