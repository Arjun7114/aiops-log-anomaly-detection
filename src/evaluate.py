"""
Step 7c: Evaluate the Isolation Forest against ground-truth labels.

Trains on the block-based feature matrix, predicts anomalies,
then computes precision, recall, and F1 by comparing to true labels.
"""
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

DATA_CSV = "data/labeled_feature_matrix.csv"
CONTAMINATION = 0.02

# --- 1. Load the labeled data ---
df = pd.read_csv(DATA_CSV)
print(f"Loaded {len(df)} labeled blocks.\n")

# Separate features (template counts) from the labels/meta
meta_cols = ["block_id", "label"]
feature_cols = [c for c in df.columns if c not in meta_cols]
X = df[feature_cols]

# Convert true labels to 1 = anomaly, 0 = normal (our "ground truth")
y_true = (df["label"] == "Anomaly").astype(int)
print(f"True anomalies: {y_true.sum()} / {len(y_true)} blocks\n")

# --- 2-5. Sweep contamination values and compare ---
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

contamination_values = [0.02, 0.05, 0.08, 0.10, 0.15]

print("=== Contamination Sweep ===")
print(f"{'contam':>8} | {'flagged':>7} | {'precision':>9} | {'recall':>6} | {'F1':>5}")
print("-" * 50)

best_f1 = 0
best_contam = None
for contam in contamination_values:
    model = IsolationForest(
        n_estimators=200,
        contamination=contam,
        random_state=42
    )
    model.fit(X)
    y_pred = (model.predict(X) == -1).astype(int)

    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f = f1_score(y_true, y_pred)

    print(f"{contam:>8.2f} | {y_pred.sum():>7} | {p:>9.3f} | {r:>6.3f} | {f:>5.3f}")

    if f > best_f1:
        best_f1 = f
        best_contam = contam

print("-" * 50)
print(f"\nBest F1: {best_f1:.3f} at contamination = {best_contam}")