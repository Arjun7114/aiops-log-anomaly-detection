"""
Step 4a: Parse raw HDFS logs into templates using Drain3,
and record which template ID each log line maps to.
Saves the result to a CSV so we can inspect the structured data.
"""
import csv
from drain3 import TemplateMiner

LOG_FILE = "data/HDFS_2k.log"
OUTPUT_CSV = "data/parsed_logs.csv"

template_miner = TemplateMiner()

# We'll collect one row per log line: the raw line + its template info
parsed_rows = []

print("Parsing logs...\n")
with open(LOG_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # add_log_message returns a dict describing the match
        result = template_miner.add_log_message(line)
        parsed_rows.append({
            "template_id": result["cluster_id"],        # which template this line matched
            "template": result["template_mined"],        # the template text
            "raw_log": line                              # the original line
        })

# Save every line's template mapping to a CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["template_id", "template", "raw_log"])
    writer.writeheader()
    writer.writerows(parsed_rows)

print(f"Done. Parsed {len(parsed_rows)} log lines.")
print(f"Structured output saved to: {OUTPUT_CSV}\n")

# Show a quick summary: how many distinct templates, and how often each appears
from collections import Counter
counts = Counter(row["template_id"] for row in parsed_rows)
print(f"Found {len(counts)} distinct templates.")
print("Top 5 most common templates by line count:")
for template_id, count in counts.most_common(5):
    print(f"  Template {template_id}: {count} lines")