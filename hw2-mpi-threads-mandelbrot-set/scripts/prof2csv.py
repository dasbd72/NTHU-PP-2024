import os
import subprocess


def one_report(report_base):
    report_path = "{}.sqlite".format(report_base)
    if not os.path.exists(report_path):
        raise FileNotFoundError(
            "Report file not found: {}".format(report_path)
        )

    if os.path.exists("{}_nvtx_sum.csv".format(report_base)):
        os.remove("{}_nvtx_sum.csv".format(report_base))

    cmd = "nsys stats -r nvtx_sum -q --format csv -o {} {}".format(
        report_base, report_path
    )
    subproc = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if subproc.returncode != 0:
        raise RuntimeError(
            "Failed to run command: {}\n{}".format(cmd, subproc.stderr)
        )


def one_report_mpi(report_base, procs):
    for rank in range(procs):
        if procs < 10:
            report_rank_path = "{}{:01d}".format(report_base, rank)
        else:
            report_rank_path = "{}{:02d}".format(report_base, rank)
        report_path = "{}.sqlite".format(report_rank_path)
        if not os.path.exists(report_path):
            raise FileNotFoundError(
                "Report file not found: {}".format(report_path)
            )

        if os.path.exists("{}_nvtx_sum.csv".format(report_rank_path)):
            os.remove("{}_nvtx_sum.csv".format(report_rank_path))

        cmd = "nsys stats -r nvtx_sum -q --format csv -o {} {}".format(
            report_rank_path, report_path
        )
        subproc = subprocess.run(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if subproc.returncode != 0:
            raise RuntimeError(
                "Failed to run command: {}\n{}".format(cmd, subproc.stderr)
            )


prog = "hw2a"
for tc in ["fast", "slow"]:
    for num_threads in range(1, 13):
        report_dir = "./nsys-reports/{}/{}-c{}".format(prog, tc, num_threads)
        report_base = "{}/report".format(report_dir)
        one_report(report_base)

prog = "hw2b"
for tc in ["fast", "slow"]:
    for num_procs in [1, 2, 4, 8]:
        report_dir = "./nsys-reports/{}/{}-N1-n{}-c{}".format(
            prog, tc, num_procs, 4
        )
        report_base = "{}/report_".format(report_dir)
        one_report_mpi(report_base, num_procs)

for strategy in [
    "baseline",
    "avx",
    "avx_cb",
    "mpi_crs",
]:
    report_dir = "./nsys-reports/hw2b/{}".format(strategy)
    report_base = "{}/report_".format(report_dir)
    one_report_mpi(report_base, 4)
