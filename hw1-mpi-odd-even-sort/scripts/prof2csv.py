import os
import subprocess
import shutil


def one_report(report_base, procs):
    for rank in range(procs):
        report_rank_path = "{}{:02d}".format(report_base, rank)
        report_path = "{}.sqlite".format(report_rank_path, rank)
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


def one_report_scalability(testcase, nodes, procs):
    report_dir = "./nsys-reports/{}-N{}-n{}".format(testcase, nodes, procs)
    report_base = "{}/report_".format(report_dir)
    one_report(report_base, procs)


def one_report_optimization(testcase, strategies, procs):
    for strategy in strategies:
        report_dir = "./nsys-reports/{}_{}".format(testcase, strategy)
        report_base = "{}/report_".format(report_dir)
        one_report(report_base, procs)


for tc in ["rand_l", "rev_l", "skew_l"]:
    for nodes in [1, 2, 4, 8]:
        one_report_scalability(tc, nodes, nodes * 4)


for tc in ["rand_l", "skew_l"]:
    one_report_optimization(
        tc,
        [
            "opt_none",
            "opt_reduce_io",
            "opt_min_comm",
            "opt_min_merge",
            "opt_buff_swap",
        ],
        16,
    )
