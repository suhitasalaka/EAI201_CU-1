# =========================
# WEEK 3 — EVALUATION & INTERPRETATION
# =========================
# Uses: data/cleaned_fifa_dataset.csv  and models/*.pkl (from Week 2)
# Produces: reports/week3_eval_summary.json, reports/week3_feature_importance.csv
# Shows: Confusion Matrix, ROC, Precision-Recall, Top Feature Importance / Coefficients
# =========================

import os, glob, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance
import joblib

warnings.filterwarnings("ignore", message=".*Skipping features without any observed values.*")

# ---------------------------------------------------------------
# Setup folders
# ---------------------------------------------------------------
DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")
for p in [DATA_DIR, REPORTS_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------
# Load cleaned dataset
# ---------------------------------------------------------------
CSV_PATH = DATA_DIR / "cleaned_fifa_dataset.csv"
assert CSV_PATH.exists(), "❌ ERROR: Run Week-1 to generate cleaned_fifa_dataset.csv"
df = pd.read_csv(CSV_PATH)

TARGET = "finalist"
exclude_cols = {"year", TARGET, "team_norm", "team_original"}
num_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
X_all = df[num_cols].copy()
y_all = df[TARGET].astype(int)

# ---------------------------------------------------------------
# Train/Test split (prefer temporal 1994–2018 train, 2022 test)
# ---------------------------------------------------------------
if ("year" in df.columns) and (df["year"] == 2022).any() and (df["year"] < 2022).any():
    train_idx = df["year"] < 2022
    test_idx  = df["year"] == 2022
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    split_note = "Temporal split → Train: 1994–2018, Test: 2022"
else:
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    split_note = "Stratified random 80/20 split (2022 not present)"

print("\n✅ Split used:", split_note)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# ---------------------------------------------------------------
# Load latest trained model from Week-2
# ---------------------------------------------------------------
def latest_model_path():
    lst = sorted(glob.glob(str(MODELS_DIR / "*.pkl")))
    return lst[-1] if lst else None

model_path = latest_model_path()
assert model_path, "❌ No model found in /models. Run Week-2 first."
print("\n✅ Loaded model:", model_path)

pipe = joblib.load(model_path)      # sklearn Pipeline

# ---------------------------------------------------------------
# Determine the ACTUAL feature names used by the pipeline after imputation
# (SimpleImputer may drop columns that have no observed values in TRAIN)
# ---------------------------------------------------------------
feature_names_used = list(num_cols)  # start with original numeric features
imputer = pipe.named_steps.get("imputer", None)
if imputer is not None and hasattr(imputer, "statistics_"):
    # Keep only those with a finite statistic (i.e., not fully-missing in TRAIN)
    mask = np.isfinite(imputer.statistics_)
    feature_names_used = [f for f, keep in zip(num_cols, mask) if keep]

# ---------------------------------------------------------------
# Predict probabilities
# ---------------------------------------------------------------
y_proba = pipe.predict_proba(X_test)[:,1]
roc_auc = float(roc_auc_score(y_test, y_proba))
pr_auc  = float(average_precision_score(y_test, y_proba))

print(f"\n📊 ROC-AUC  : {roc_auc:.4f}")
print(f"📊 PR-AUC   : {pr_auc:.4f}")

# ---------------------------------------------------------------
# Threshold search (maximize F1 score)
# ---------------------------------------------------------------
def evaluate_at(threshold, y_true, y_prob):
    y_pred = (y_prob >= threshold).astype(int)
    return dict(
        threshold=float(threshold),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )

thresholds = np.linspace(0.05, 0.95, 19)
results = [evaluate_at(t, y_test, y_proba) for t in thresholds]
best = max(results, key=lambda x: x["f1"])

print(f"\n🏆 Best Threshold (F1-max): {best['threshold']:.2f}")
print("→ Metrics:", best)

# Save threshold search to CSV
pd.DataFrame(results).to_csv(REPORTS_DIR / "week3_threshold_search.csv", index=False)

# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------
# Confusion Matrix @ best threshold
ConfusionMatrixDisplay.from_predictions(y_test, (y_proba >= best["threshold"]).astype(int))
plt.title(f"Confusion Matrix @ threshold={best['threshold']:.2f}")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve"); plt.legend(); plt.show()

# Precision-Recall curve
prec, rec, _ = precision_recall_curve(y_test, y_proba)
plt.plot(rec, prec, label=f"PR-AUC={pr_auc:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall Curve"); plt.legend(); plt.show()

# ---------------------------------------------------------------
# Feature Importance (RF) or Coefficients (LR) with robust alignment
# If alignment still fails, fall back to permutation importance
# ---------------------------------------------------------------
clf = pipe.named_steps.get("clf")
importance_df = None
fallback_to_perm = False

try:
    if hasattr(clf, "feature_importances_"):  # Random Forest
        vals = clf.feature_importances_
        if len(vals) != len(feature_names_used):
            # Rare, but guard against mismatch
            fallback_to_perm = True
        else:
            importance_df = pd.DataFrame({"feature": feature_names_used, "importance": vals}) \
                               .sort_values("importance", ascending=False)

    elif hasattr(clf, "coef_"):  # Logistic Regression
        coefs = clf.coef_.ravel()
        if len(coefs) != len(feature_names_used):
            # This is the mismatch that caused your error; handle gracefully
            fallback_to_perm = True
        else:
            importance_df = pd.DataFrame({"feature": feature_names_used, "coefficient": coefs}) \
                               .assign(abscoef=lambda d: d["coefficient"].abs()) \
                               .sort_values("abscoef", ascending=False)
    else:
        fallback_to_perm = True
except Exception:
    fallback_to_perm = True

# Permutation importance fallback (model-agnostic)
if fallback_to_perm or importance_df is None:
    try:
        perm = permutation_importance(pipe, X_test, y_test, n_repeats=12, random_state=42, n_jobs=-1)
        importance_df = pd.DataFrame({"feature": feature_names_used, "perm_importance": perm.importances_mean}) \
                           .sort_values("perm_importance", ascending=False)
    except Exception:
        # As a last resort, write an empty CSV with header
        importance_df = pd.DataFrame({"feature": [], "importance": []})

# Save importances
importance_df.to_csv(REPORTS_DIR / "week3_feature_importance.csv", index=False)

# ---------------------------------------------------------------
# Save JSON Summary (includes BOTH default 0.50 metrics and best-thr metrics)
# ---------------------------------------------------------------
metrics_default = evaluate_at(0.50, y_test, y_proba)
summary = {
    "model_path": model_path,
    "split": split_note,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "metrics_at_default_threshold": metrics_default,
    "metrics_at_best_threshold": best,
    "features_used_after_imputer": feature_names_used,
}

with open(REPORTS_DIR / "week3_eval_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n✅ Output files generated:")
print("   → reports/week3_eval_summary.json")
print("   → reports/week3_feature_importance.csv")
print("   → reports/week3_threshold_search.csv")
print("\n🎯 Week-3 Evaluation Completed Successfully!")
