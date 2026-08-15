"""
Step 3: Parse raw HDFS logs into templates using Drain3.
This replaces manual regex parsing with automatic template mining.
"""
from drain3 import TemplateMiner

# Path to our log file
LOG_FILE = "data/HDFS_2k.log"

# Create the template miner — this is the Drain algorithm
template_miner = TemplateMiner()

# Read the log file line by line and feed each line to Drain
print("Parsing logs...\n")
with open(LOG_FILE, "r") as f:
    for line in f:
        line = line.strip()          # remove trailing newline/spaces
        if not line:                  # skip any blank lines
            continue
        template_miner.add_log_message(line)

# After processing all lines, show the templates Drain discovered
print("=== Discovered Log Templates ===\n")
clusters = template_miner.drain.clusters
for cluster in clusters:
    print(cluster)