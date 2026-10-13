"""MLBC project - credit-card fraud detection (ML step-by-step notebook as a script).

Steps (mirroring the class pipeline):
 1. Import data
 2. Explore / basic checks
 3. Preprocess (features, split, imbalance handling)
 4. Choose a model (small comparison)
 5. Train
 6. Evaluate (metrics, confusion matrix, ROC)
 7. Visualize (plots saved to artifacts/)
 8. Save the trained model (+ metadata the frontend needs)
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # headless backend (no display needed)
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             RocCurveDisplay)
from sklearn.preprocessing import StandardScaler

os.makedirs("artifacts", exist_ok=True)
os.makedirs("artifacts/plots", exist_ok=True)

# --------------------------------------------------------------------------- #
# Step 1 - Import data
# --------------------------------------------------------------------------- #
df = pd.read_csv("data/transactions.csv")
schema = json.load(open("data/schema.json"))
features = schema["feature_order"]
print("shape:", df.shape)
print(df["fraud"].value_counts().to_dict())

# --------------------------------------------------------------------------- #
# Step 2 - Quick exploration
# --------------------------------------------------------------------------- #
# (a) class imbalance
fig, ax = plt.subplots(figsize=(5, 4))
df["fraud"].value_counts().sort_index().plot.bar(ax=ax, color=["#4ade80", "#f87171"])
ax.set_title("Class distribution")
ax.set_xticklabels(["Legit (0)", "Fraud (1)"], rotation=0)
fig.tight_layout()
fig.savefig("artifacts/plots/class_distribution.png")
plt.close(fig)

# --------------------------------------------------------------------------- #
# Step 3 - Preprocess: feature matrix X, target y, train/test split
# --------------------------------------------------------------------------- #
X = df[features]
y = df["fraud"]

# stratified split keeps the fraud ratio in both sets
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# scale only for models that need it (LogisticRegression). Trees don't need it.
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xtr)
Xte_s = scaler.transform(Xte)

# --------------------------------------------------------------------------- #
# Step 4 - Choose a model: quick comparison of two candidates
# --------------------------------------------------------------------------- #
print("\n--- model comparison (validation on 20% split) ---")

# candidate A: LogisticRegression (linear baseline, needs scaled data, class_weight)
lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
lr.fit(Xtr_s, ytr)
lr_proba = lr.predict_proba(Xte_s)[:, 1]
lr_pred = (lr_proba >= 0.5).astype(int)
print(f"LogisticRegression  acc={accuracy_score(yte,lr_pred):.4f} "
      f"prec={precision_score(yte,lr_pred,zero_division=0):.4f} "
      f"rec={recall_score(yte,lr_pred,zero_division=0):.4f} "
      f"f1={f1_score(yte,lr_pred,zero_division=0):.4f} "
      f"auc={roc_auc_score(yte,lr_proba):.4f}")

# candidate B: RandomForest (handles non-linearity, no scaling needed)
# chosen as final model -> higher recall/auc on this tabular data
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                           random_state=42, n_jobs=-1, min_samples_leaf=2)
rf.fit(Xtr, ytr)
rf_proba = rf.predict_proba(Xte)[:, 1]
rf_pred = (rf_proba >= 0.5).astype(int)
print(f"RandomForest        acc={accuracy_score(yte,rf_pred):.4f} "
      f"prec={precision_score(yte,rf_pred,zero_division=0):.4f} "
      f"rec={recall_score(yte,rf_pred,zero_division=0):.4f} "
      f"f1={f1_score(yte,rf_pred,zero_division=0):.4f} "
      f"auc={roc_auc_score(yte,rf_proba):.4f}")

model = rf   # final choice

# --------------------------------------------------------------------------- #
# Step 5 - Train (already done above for RF; expose fitted model)
# --------------------------------------------------------------------------- #
print("\nfinal model trained:", model)

# --------------------------------------------------------------------------- #
# Step 6 - Evaluate
# --------------------------------------------------------------------------- #
proba = model.predict_proba(Xte)[:, 1]
pred = (proba >= 0.3).astype(int)
metrics = {
    "n_train": len(Xtr), "n_test": len(Xte),
    "fraud_rate": float(y.mean()),
    "accuracy": float(accuracy_score(yte, pred)),
    "precision": float(precision_score(yte, pred, zero_division=0)),
    "recall": float(recall_score(yte, pred, zero_division=0)),
    "f1": float(f1_score(yte, pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(yte, proba)),
    "confusion_matrix": confusion_matrix(yte, pred).tolist(),
    "threshold": 0.3,
}
print(json.dumps(metrics, indent=2))

# confusion matrix plot
cm = confusion_matrix(yte, pred)
fig, ax = plt.subplots(figsize=(4.5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred legit", "Pred fraud"])
ax.set_yticklabels(["Actual legit", "Actual fraud"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
ax.set_title("Confusion matrix")
fig.tight_layout()
fig.savefig("artifacts/plots/confusion_matrix.png")
plt.close(fig)

# ROC curve
fig, ax = plt.subplots(figsize=(5, 4))
RocCurveDisplay.from_predictions(yte, proba, ax=ax, name="RandomForest")
ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
ax.set_title("ROC curve")
fig.tight_layout()
fig.savefig("artifacts/plots/roc_curve.png")
plt.close(fig)

# --------------------------------------------------------------------------- #
# Step 7 - Visualize feature importance
# --------------------------------------------------------------------------- #
fi = sorted(
    [{"feature": f, "importance": float(v)}
     for f, v in zip(features, model.feature_importances_)],
    key=lambda d: d["importance"], reverse=True)

fig, ax = plt.subplots(figsize=(7, 5))
names = [d["feature"] for d in fi][::-1]
vals = [d["importance"] for d in fi][::-1]
ax.barh(names, vals, color="#60a5fa")
ax.set_title("Feature importance")
fig.tight_layout()
fig.savefig("artifacts/plots/feature_importance.png")
plt.close(fig)

# --------------------------------------------------------------------------- #
# Step 8 - Save the trained model + metadata the HTML frontend needs
# --------------------------------------------------------------------------- #
joblib.dump(model, "artifacts/model.pkl")

means = {}
for cls in (0, 1):
    sub = df[df["fraud"] == cls]
    means[str(cls)] = {f: float(sub[f].mean()) for f in features}
stats = {
    "n_total": len(df),
    "fraud_count": int(y.sum()),
    "legit_count": int((y == 0).sum()),
    "fraud_rate": float(y.mean()),
    "avg_amount_fraud": float(df[df.fraud == 1].amount.mean()),
    "avg_amount_legit": float(df[df.fraud == 0].amount.mean()),
    "class_means": means,
}

samples = []
for label in (0, 1):
    sub = df[df.fraud == label].sample(5, random_state=42)
    for _, r in sub.iterrows():
        samples.append({
            "label": "fraud" if label == 1 else "legit",
            "values": {f: (r[f].item() if hasattr(r[f], "item") else r[f]) for f in features},
        })

meta_features = [
    {"name": "amount", "label": "Amount (USD)", "type": "number", "min": 0, "step": 0.01},
    {"name": "hour", "label": "Hour of day (0-23)", "type": "number", "min": 0, "max": 23, "step": 1},
    {"name": "day_of_week", "label": "Day of week (0=Mon)", "type": "number", "min": 0, "max": 6, "step": 1},
    {"name": "merchant_category", "label": "Merchant category", "type": "select",
     "options": [{"value": i, "label": m} for i, m in enumerate(schema["merchants"])]},
    {"name": "distance_from_home", "label": "Distance from home (km)", "type": "number", "min": 0, "step": 0.1},
    {"name": "distance_from_last_tx", "label": "Distance from last tx (km)", "type": "number", "min": 0, "step": 0.1},
    {"name": "ratio_to_median", "label": "Ratio to median purchase", "type": "number", "min": 0, "step": 0.01},
    {"name": "is_chip", "label": "Chip used?", "type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
    {"name": "is_pin_used", "label": "PIN used?", "type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
    {"name": "velocity_24h", "label": "Tx count last 24h", "type": "number", "min": 0, "step": 1},
    {"name": "foreign_transaction", "label": "Foreign transaction?", "type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
    {"name": "card_present", "label": "Card present?", "type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
    {"name": "account_age_months", "label": "Account age (months)", "type": "number", "min": 1, "step": 1},
]

schema_out = dict(schema)
schema_out["feature_meta"] = meta_features
schema_out["feature_order"] = features

with open("artifacts/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
with open("artifacts/feature_importance.json", "w") as f:
    json.dump(fi, f, indent=2)
with open("artifacts/stats.json", "w") as f:
    json.dump(stats, f, indent=2)
with open("artifacts/samples.json", "w") as f:
    json.dump(samples, f, indent=2)
with open("artifacts/schema.json", "w") as f:
    json.dump(schema_out, f, indent=2)

print("\nsaved artifacts/model.pkl + json + plots. done.")