"""
Step 9: Visualize anomaly detection results.

Plot 1: Distribution of anomaly scores, split by true label.
        Shows whether the model separates anomalies from normal blocks.
Plot 2: Precision & recall vs contamination.
        Shows the tuning trade-off.

Saves both plots as PNGs in results/ for the README.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # render to file without needing a display window
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score

DATA_CSV = "data/labeled_feature_matrix.csv"
os.makedirs("results", exist_ok=True)

# --- Load data ---
df = pd.read_csv(DATA_CSV)
meta_cols = ["block_id", "label"]
feature_cols = [c for c in df.columns if c not in meta_cols]
X = df[feature_cols]
y_true = (df["label"] == "Anomaly").astype(int)

# ============================================================
# PLOT 1: Anomaly score distribution by true label
# ============================================================
model = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
model.fit(X)
scores = model.decision_function(X)   # higher = more normal, lower = more anomalous

df_plot = pd.DataFrame({"score": scores, "label": df["label"]})
normal_scores = df_plot[df_plot["label"] == "Normal"]["score"]
anomaly_scores = df_plot[df_plot["label"] == "Anomaly"]["score"]

plt.figure(figsize=(9, 5))
plt.hist(normal_scores, bins=50, alpha=0.6, label="Normal blocks", color="#4C9F70")
plt.hist(anomaly_scores, bins=50, alpha=0.6, label="Anomaly blocks", color="#D64550")
plt.axvline(x=0, color="black", linestyle="--", linewidth=1, label="Decision boundary")
plt.xlabel("Anomaly score  (lower = more anomalous)")
plt.ylabel("Number of blocks")
plt.yscale("log")
plt.title("Anomaly Score Distribution by True Label (log scale)")
plt.legend()
plt.tight_layout()
plt.savefig("results/anomaly_score_distribution.png", dpi=120)
plt.close()
print("Saved: results/anomaly_score_distribution.png")

# ============================================================
# PLOT 2: Precision & recall vs contamination
# ============================================================
contamination_values = [0.02, 0.05, 0.08, 0.10, 0.15]
precisions, recalls = [], []

for contam in contamination_values:
    m = IsolationForest(n_estimators=200, contamination=contam, random_state=42)
    m.fit(X)
    y_pred = (m.predict(X) == -1).astype(int)
    precisions.append(precision_score(y_true, y_pred))
    recalls.append(recall_score(y_true, y_pred))

plt.figure(figsize=(9, 5))
plt.plot(contamination_values, precisions, marker="o", label="Precision", color="#2E86AB")
plt.plot(contamination_values, recalls, marker="s", label="Recall", color="#E8871E")
plt.xlabel("Contamination (expected anomaly fraction)")
plt.ylabel("Score")
plt.title("Precision vs Recall Trade-off Across Contamination")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/precision_recall_tradeoff.png", dpi=120)
plt.close()
print("Saved: results/precision_recall_tradeoff.png")

print("\nBoth plots saved to results/")