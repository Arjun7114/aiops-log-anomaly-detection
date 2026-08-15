"""
Step 8: Improved anomaly detection using TF-IDF weighted features.

Instead of raw template counts, we weight each template by how rare it
is across all blocks (TF-IDF). Rare, informative events get emphasized;
common background events get downweighted. This helps the Isolation
Forest isolate genuine anomalies.

We sweep contamination and compare against the raw-count baseline.
"""
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_score, recall_score, f1_score

DATA_CSV = "data/labeled_feature_matrix.csv"

# --- 1. Load labeled data ---
df = pd.read_csv(DATA_CSV)
print(f"Loaded {len(df)} labeled blocks.\n")

meta_cols = ["block_id", "label"]
feature_cols = [c for c in df.columns if c not in meta_cols]
X_counts = df[feature_cols]
y_true = (df["label"] == "Anomaly").astype(int)
print(f"True anomalies: {y_true.sum()} / {len(y_true)} blocks\n")

# --- 2. Transform raw counts into TF-IDF weighted features ---
# TfidfTransformer takes a count matrix and reweights it.
tfidf = TfidfTransformer()
X_tfidf = tfidf.fit_transform(X_counts)   # returns a sparse matrix
print("Applied TF-IDF weighting to features.\n")

# --- 3. Sweep contamination, same as before, but on TF-IDF features ---
contamination_values = [0.02, 0.05, 0.08, 0.10, 0.15]

print("=== TF-IDF Contamination Sweep ===")
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
    model.fit(X_tfidf)
    y_pred = (model.predict(X_tfidf) == -1).astype(int)

    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f = f1_score(y_true, y_pred)
    print(f"{contam:>8.2f} | {y_pred.sum():>7} | {p:>9.3f} | {r:>6.3f} | {f:>5.3f}")

    if f > best_f1:
        best_f1 = f
        best_contam = contam

print("-" * 50)
print(f"\nBest F1 (TF-IDF): {best_f1:.3f} at contamination = {best_contam}")
print("Compare this to the raw-count best F1 of 0.564 from Step 7.")