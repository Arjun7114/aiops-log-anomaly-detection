"""
Step 7b: Build a labeled, block-based dataset from the full HDFS logs.

- Reads a slice of the large HDFS.log
- Parses each line into a template with Drain3
- Extracts the block ID (blk_...) from each line
- Groups lines by block ID into "sessions"
- Builds an event-count matrix (one row per block)
- Attaches each block's ground-truth label (Normal / Anomaly)
"""
import re
import pandas as pd
from collections import defaultdict, Counter
from drain3 import TemplateMiner

# --- Config ---
LOG_FILE = r"C:\Users\aadr5\Downloads\HDFS_full\HDFS.log"
LABELS_FILE = "data/anomaly_label.csv"
OUTPUT_CSV = "data/labeled_feature_matrix.csv"
MAX_LINES = 200_000   # how many lines of the big log to read (start small)

# Regex to pull the block id out of a line, e.g. blk_-6952295868487656571
BLOCK_RE = re.compile(r"(blk_-?\d+)")

# --- 1. Load the labels into a dict: block_id -> "Normal"/"Anomaly" ---
print("Loading labels...")
labels_df = pd.read_csv(LABELS_FILE)
label_map = dict(zip(labels_df["BlockId"], labels_df["Label"]))
print(f"Loaded {len(label_map)} block labels.\n")

# --- 2. Parse a slice of the log, grouping template IDs by block ---
template_miner = TemplateMiner()
# block_id -> list of template ids seen for that block
block_events = defaultdict(list)

print(f"Reading and parsing up to {MAX_LINES:,} lines...")
with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f):
        if i >= MAX_LINES:
            break
        line = line.strip()
        if not line:
            continue

        # find the block id(s) in this line
        match = BLOCK_RE.search(line)
        if not match:
            continue  # skip lines with no block id
        block_id = match.group(1)

        # parse the line into a template
        result = template_miner.add_log_message(line)
        template_id = result["cluster_id"]

        # record this event under its block
        block_events[block_id].append(template_id)

print(f"Parsed slice. Found {len(block_events)} distinct blocks.\n")

# --- 3. Build one feature row per block (event counts), attach label ---
rows = []
skipped_no_label = 0
for block_id, template_ids in block_events.items():
    if block_id not in label_map:
        skipped_no_label += 1
        continue
    counts = Counter(template_ids)           # {template_id: count}
    row = dict(counts)
    row["block_id"] = block_id
    row["label"] = label_map[block_id]
    rows.append(row)

print(f"Blocks with labels: {len(rows)} (skipped {skipped_no_label} without labels)")

# --- 4. Turn into a tidy DataFrame ---
df = pd.DataFrame(rows).fillna(0)

# Separate the meta columns from the numeric template-count columns
meta_cols = ["block_id", "label"]
template_cols = [c for c in df.columns if c not in meta_cols]
template_cols = sorted(template_cols)                       # tidy order
df = df[template_cols + meta_cols]                          # reorder
df[template_cols] = df[template_cols].astype(int)
df.columns = [f"template_{c}" if c not in meta_cols else c for c in df.columns]

# --- 5. Report label balance and save ---
label_counts = df["label"].value_counts()
print("\nLabel distribution in this slice:")
print(label_counts.to_string())

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved labeled feature matrix to: {OUTPUT_CSV}")
print(f"Shape: {df.shape[0]} blocks x {len(template_cols)} templates (+ block_id, label)")