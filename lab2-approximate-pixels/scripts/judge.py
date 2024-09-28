import os
import signal
import shutil
import time

from .sync import sync
from .build import build

FILES = ["Makefile", "modules.list"]
JUDGE_FILES = {
    "lab2_omp-judge": ["lab2_omp.cc"],
    "lab2_pthread-judge": ["lab2_pthread.cc"],
    "lab2_hybrid-judge": ["lab2_hybrid.cc"],
}
JUDGES = []


def signal_int(signum, frame):
    print("Exiting...")
    exit(0)


def copy_move():
    # Copy files to a temporary directory and change directory
    judge_dir = os.path.expanduser("~/.tmp.judge")
    shutil.rmtree(judge_dir, ignore_errors=True)
    os.mkdir(judge_dir)
    for judge, files in JUDGE_FILES.items():
        for file in files:
            if not os.path.exists(file):
                continue
            shutil.copy(file, judge_dir)
            JUDGES.append(judge)
    for file in FILES:
        if not os.path.exists(file):
            continue
        shutil.copy(file, judge_dir)
    os.chdir(judge_dir)
    print(f"Changed directory to {judge_dir}")


def judge_loop():
    idx = 0
    while True:
        judge = JUDGES[idx]
        code = build()
        if code != 0:
            print("Build failed")
            time.sleep(1)
            continue
        os.system(judge)
        time.sleep(301)
        idx = (idx + 1) % len(JUDGES)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_int)
    sync()
    copy_move()
    judge_loop()
