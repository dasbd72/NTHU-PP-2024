import re
import sys
from collections import defaultdict

# Sample log data
# log_data = """<Paste the log data here>"""
with open(sys.argv[1], "r") as f:
    log_data = f.read()

# Define regex patterns
kernel_avg_pattern = (
    r"void flash_attention::flash_attention_kernel.*?\bAvg\s+(\d+\.\d+\w*)"
)
metric_avg_pattern = (
    r"flash_attention::flash_attention_kernel.*?Metric Name.*?achieved_occupancy.*?Avg\s+(\d+\.\d+).*?"
    r"sm_efficiency.*?Avg\s+(\d+\.\d+).*?"
    r"gld_throughput.*?Avg\s+(\d+\.\d+).*?"
    r"gst_throughput.*?Avg\s+(\d+\.\d+).*?"
    r"shared_load_throughput.*?Avg\s+(\d+\.\d+).*?"
    r"shared_store_throughput.*?Avg\s+(\d+\.\d+)"
)
event_avg_pattern = (
    r"flash_attention::flash_attention_kernel.*?Event Name.*?shared_ld_bank_conflict.*?Avg\s+(\d+).*?"
    r"shared_st_bank_conflict.*?Avg\s+(\d+)"
)

# Extract values
kernel_avg = re.findall(kernel_avg_pattern, log_data, re.DOTALL)
metrics_avg = re.findall(metric_avg_pattern, log_data, re.DOTALL)
events_avg = re.findall(event_avg_pattern, log_data, re.DOTALL)

# Aggregate results
result = {
    "kernel_avg_time": kernel_avg,
    "metrics_avg": [
        {
            "achieved_occupancy": float(metric[0]),
            "sm_efficiency": float(metric[1]),
            "gld_throughput": float(metric[2]),
            "gst_throughput": float(metric[3]),
            "shared_load_throughput": float(metric[4]),
            "shared_store_throughput": float(metric[5]),
        }
        for metric in metrics_avg
    ],
    "events_avg": [
        {
            "shared_ld_bank_conflict": int(event[0]),
            "shared_st_bank_conflict": int(event[1]),
        }
        for event in events_avg
    ],
}

# Print results
print("Kernel Average Times:", result["kernel_avg_time"])
print("Metrics Averages:")
for metric in result["metrics_avg"]:
    print(metric)
print("Events Averages:")
for event in result["events_avg"]:
    print(event)
