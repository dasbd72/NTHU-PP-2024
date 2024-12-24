import re
import sys
from collections import defaultdict

# Input data (paste your data here as a string)
with open(sys.argv[1], 'r') as file:
    data = file.read()

def parse_and_average_bandwidth(data):
    # Regular expression to match size and bandwidth
    pattern = re.compile(r"^\s*(\d+)\s+([\d.]+)$")
    
    # Dictionary to store sizes and bandwidth values
    bandwidth_data = defaultdict(list)
    
    # Parse the data line by line
    for line in data.splitlines():
        match = pattern.match(line)
        if match:
            size = int(match.group(1))
            bandwidth = float(match.group(2))
            bandwidth_data[size].append(bandwidth)
    
    # Calculate averages
    averages = {size: sum(values) / len(values) for size, values in bandwidth_data.items()}
    
    # Convert to list of tuples sorted by size
    sorted_averages = sorted(averages.items())
    
    # Return as a list of averages
    return [avg for _, avg in sorted_averages]

# Run the parser and calculate averages
average_bandwidths = parse_and_average_bandwidth(data)

# Print results
print(average_bandwidths)