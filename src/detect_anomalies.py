"""
Step 6: Train an Isolation Forest to detect anomalous log windows.

Isolation Forest scores each window by how easily it can be "isolated"
from the rest. Anomalies isolate in few splits; normal points take many.
We flag the most anomalous windows based on the contamination setting.
"""
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURES_CSV = "data/feature_matrix.csv"
RESULTS_CSV = "results/anomaly_results.csv"
CONTAMINATION = 0.05   # expect ~5% of windows to be anomalies

# 1. Load the feature matrix we built in Step 5
X = pd.read_csv(FEATURES_CSV)
print(f"Loaded feature matrix: {X.shape[0]} windows x {X.shape[1]} templates\n")

# 2. Create and train the Isolation Forest
#    - n_estimators: how many trees in the forest (more = more stable)
#    - contamination: expected fraction of anomalies (sets the cutoff)
#    - random_state: fixes randomness so results are reproducible
model = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,
    random_state=42
)
model.fit(X)

# 3. Score and predict each window
#    predict(): -1 = anomaly, 1 = normal
#    decision_function(): raw score (lower = more anomalous)
predictions = model.predict(X)
scores = model.decision_function(X)

# 4. Assemble results into a readable table
results = X.copy()
results["window_index"] = range(len(X))
results["anomaly_score"] = scores
results["is_anomaly"] = (predictions == -1)

# 5. Save full results
import os
os.makedirs("results", exist_ok=True)
results.to_csv(RESULTS_CSV, index=False)

# 6. Report what was flagged
num_anomalies = results["is_anomaly"].sum()
print(f"Flagged {num_anomalies} anomalous windows out of {len(X)}.\n")
print("=== Anomalous windows (most anomalous first) ===")
anomalies = results[results["is_anomaly"]].sort_values("anomaly_score")
# show template columns + the score, for each flagged window
cols = list(X.columns) + ["window_index", "anomaly_score"]
print(anomalies[cols].to_string(index=False))

print(f"\nFull results saved to: {RESULTS_CSV}")