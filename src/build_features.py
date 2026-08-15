"""
Step 5: Build a feature matrix from parsed logs.

Each row  = a window of consecutive log lines.
Each column = a template ID.
Each cell = how many times that template appeared in that window.

This numeric matrix is what the Isolation Forest model will train on.
"""
import pandas as pd
from collections import Counter

PARSED_CSV = "data/parsed_logs.csv"
FEATURES_CSV = "data/feature_matrix.csv"
WINDOW_SIZE = 20   # how many log lines per window

# 1. Load the parsed logs we created in Step 4
df = pd.read_csv(PARSED_CSV)
print(f"Loaded {len(df)} parsed log lines.")

# 2. Grab the sequence of template IDs, in order
template_ids = df["template_id"].tolist()

# 3. Slice the sequence into fixed-size windows and count templates in each
rows = []
for start in range(0, len(template_ids), WINDOW_SIZE):
    window = template_ids[start : start + WINDOW_SIZE]
    counts = Counter(window)          # e.g. {1: 5, 2: 3, 7: 2, ...}
    rows.append(counts)

print(f"Created {len(rows)} windows of {WINDOW_SIZE} lines each.")

# 4. Turn the list of count-dicts into a table.
#    Missing template = 0 occurrences in that window.
feature_matrix = pd.DataFrame(rows).fillna(0).astype(int)

# 5. Sort columns by template ID so the table is tidy, and name them clearly
feature_matrix = feature_matrix.reindex(sorted(feature_matrix.columns), axis=1)
feature_matrix.columns = [f"template_{c}" for c in feature_matrix.columns]

# 6. Save it and show a preview
feature_matrix.to_csv(FEATURES_CSV, index=False)
print(f"\nFeature matrix saved to: {FEATURES_CSV}")
print(f"Shape: {feature_matrix.shape[0]} windows x {feature_matrix.shape[1]} templates\n")
print("Preview of first 5 windows:")
print(feature_matrix.head())