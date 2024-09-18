import os
import re
import argparse
import plotly.express as px
import pandas as pd
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("log", type=str)


def parse_timing_log_once(s):
    # TIMING_LOG_ONCE_START task mpi_read id 9 ts 0
    # TIMING_LOG_ONCE_END task mpi_read id 9 ts 0.158844
    m = re.match(
        r"TIMING_LOG_ONCE_(\w+) task (\w+) id (\d+) ts (\d+(\.\d+)?$)", s
    )
    if not m:
        return None
    log_type = "once"
    event = m.group(1).lower()
    task = m.group(2).lower()
    id = int(m.group(3))
    # ts add 20241001's timestamp
    ts = float(m.group(4))
    return {"type": log_type, "event": event, "task": task, "id": id, "ts": ts}


def parse_timing_log_multi(s):
    # TIMING_LOG_MULTI_COMM_START task mpi_exchange_1 id 10 iter 0 src 10 dst 11 ts 1.19635
    # TIMING_LOG_MULTI_COMM_END task mpi_exchange_1 id 10 iter 0 src 10 dst 11 ts 1.19645
    m = re.match(
        r"TIMING_LOG_MULTI_COMM_(\w+) task (\w+) id (\d+) iter (\d+) src (\d+) dst (\d+) ts (\d+(\.\d+)?$)",
        s,
    )
    if not m:
        return None
    log_type = "multi"
    event = m.group(1).lower()
    task = m.group(2).lower()
    id = int(m.group(3))
    iter = int(m.group(4))
    src = int(m.group(5))
    dst = int(m.group(6))
    ts = float(m.group(7))
    return {
        "type": log_type,
        "event": event,
        "task": task,
        "id": id,
        "iter": iter,
        "src": src,
        "dst": dst,
        "ts": ts,
    }


def parse_logs(log_file):
    once_logs = {}
    multi_logs = {}
    with open(log_file, "r") as f:
        for line in f:
            log = parse_timing_log_once(line)
            if log:
                key = (log["task"], log["id"])
                if key not in once_logs:
                    once_logs[key] = {}
                once_logs[key][log["event"]] = log["ts"]
            log = parse_timing_log_multi(line)
            if log:
                key = (
                    log["task"],
                    log["id"],
                    log["iter"],
                    log["src"],
                    log["dst"],
                )
                if key not in multi_logs:
                    multi_logs[key] = {}
                multi_logs[key][log["event"]] = log["ts"]
    logs: list[dict] = []
    for key, value in once_logs.items():
        logs.append(
            {
                "type": "once",
                "task": key[0],
                "id": key[1],
                "start": datetime.datetime.fromtimestamp(value["start"]),
                "end": datetime.datetime.fromtimestamp(value["end"]),
                "desc": f"task: {key[0]}",
            }
        )
    for key, value in multi_logs.items():
        logs.append(
            {
                "type": "multi",
                "task": f"{key[0]}-{key[2]:02d}",
                "id": key[1],
                "iter": key[2],
                "src": key[3],
                "dst": key[4],
                "start": datetime.datetime.fromtimestamp(value["start"]),
                "end": datetime.datetime.fromtimestamp(value["end"]),
                "desc": f"task: {key[0]}, iter: {key[2]}, src: {key[3]}, dst: {key[4]}",
            }
        )
    logs = sorted(logs, key=lambda x: x["task"])
    return logs


def plot_gantt_chart(logs: list[dict]):
    df = pd.DataFrame(logs)
    fig = px.timeline(
        df,
        x_start="start",
        x_end="end",
        y="id",
        color="task",
        hover_data="desc",
    )
    fig.write_image("outputs/gantt_chart.png")
    fig.write_html("outputs/gantt_chart.html")


if __name__ == "__main__":
    args = parser.parse_args()
    if not args.log:
        raise ValueError("Log file not provided")
    if not os.path.exists(args.log):
        raise ValueError("Log file does not exist")

    logs = parse_logs(args.log)
    plot_gantt_chart(logs)
