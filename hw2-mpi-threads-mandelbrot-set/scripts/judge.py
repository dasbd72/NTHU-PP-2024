import os
import signal
import shutil
import time

from .build import build
from .sync import sync


judges = [
    "hw2a-judge",
    "hw2b-judge",
]


def signal_int(signum, frame):
    print("Exiting...")
    exit(0)


def copy_move():
    # Copy files to a temporary directory and change directory
    judge_dir = os.path.expanduser("~/.tmp.judge")
    shutil.rmtree(judge_dir, ignore_errors=True)
    os.mkdir(judge_dir)
    shutil.copy("Makefile", judge_dir)
    shutil.copy("modules.list", judge_dir)
    shutil.copy("hw2a.cc", judge_dir)
    shutil.copy("hw2b.cc", judge_dir)
    os.chdir(judge_dir)
    print(f"Changed directory to {judge_dir}")


def judge_loop():
    index = 0
    while True:
        os.system(judges[index])
        time.sleep(301)
        index = (index + 1) % len(judges)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_int)
    sync()
    copy_move()
    code = build()
    if code != 0:
        raise RuntimeError("Build failed")
    judge_loop()
