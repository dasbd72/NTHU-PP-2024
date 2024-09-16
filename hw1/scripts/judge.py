import os
import signal
import time

from .build import build


def signal_int(signum, frame):
    print("Exiting...")
    exit(0)


def judge_loop():
    while True:
        code = build()
        if code != 0:
            print("Build failed")
            time.sleep(1)
            continue
        os.system("hw1-judge")
        time.sleep(301)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_int)
    judge_loop()
