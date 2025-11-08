
# =========================
# WEEK 2 — MODEL BUILDING & TRAINING
# =========================
!pip install scikit-learn joblib matplotlib pandas numpy

import warnings
warnings.filterwarnings("ignore", message=".*Skipping features without any observed values.*")

import os, json, textwrap, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import joblib

# -------------------------
# Paths & setup
# -------------------------
DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")
for p in [DATA_DIR, REPORTS_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "cleaned_fifa_dataset.csv"
if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Missing {CSV_PATH}. Run Week-1 pipeline first to generate the cleaned dataset."
    )

# -------------------------
# Load & inspect
# -------------------------
df = pd.read_csv(CSV_PATH)

# Target and feature candidates
TARGET_COL = "finalist"
if TARGET_COL not in df.columns:
    raise ValueError(f"Column '{TARGET_COL}' not found in dataset.")

# Use numeric features only; exclude obvious identifiers / leakage columns
exclude_cols = {"year", "finalist", "team_norm", "team_original"}
num_cols = [c for c in df.columns
            if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]

if not num_cols:
    raise ValueError("No numeric features found to model on.")

X_all = df[num_cols].copy()
y_all = df[TARGET_COL].astype(int)

# -------------------------
# Train/Test split strategy
# -------------------------
# Preferred: train on 1994–2018, test on 2022
if "year" in df.columns and (df["year"] == 2022).any() and (df["year"] < 2022).any():
    train_idx = df["year"] < 2022
    test_idx  = df["year"] == 2022
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    split_note = "Temporal split: Train=1994–2018, Test=2022"
else:
    # Fallback: stratified random split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    split_note = "Random stratified split (2022 not found)"

print(f"Features used ({len(num_cols)}): {num_cols}")
print(f"Split: {split_note}")
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# -------------------------
# Pipelines
# -------------------------
# Logistic Regression — needs scaling
lr_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, n_jobs=None))
])

# Random Forest — scaling not required, but safe to impute
rf_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", RandomForestClassifier(random_state=42))
])

# -------------------------
# Hyperparameter grids
# -------------------------
lr_param_grid = {
    "clf__C": [0.05, 0.1, 0.5, 1, 5],
    "clf__solver": ["lbfgs", "liblinear"],
    "clf__penalty": ["l2"]
}
rf_param_grid = {
    "clf__n_estimators": [150, 300, 500],
    "clf__max_depth": [5, 10, 20, None],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__class_weight": [None, "balanced"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# -------------------------
# Grid search: Logistic Regression
# -------------------------
print("\n=== GridSearch: Logistic Regression ===")
lr_grid = GridSearchCV(
    estimator=lr_pipe,
    param_grid=lr_param_grid,
    cv=cv,
    scoring="f1",
    n_jobs=-1,
    verbose=0
)
lr_grid.fit(X_train, y_train)
print("Best params (LR):", lr_grid.best_params_)
print("Best CV F1 (LR):", round(lr_grid.best_score_, 4))

# -------------------------
# Grid search: Random Forest
# -------------------------
print("\n=== GridSearch: Random Forest ===")
rf_grid = GridSearchCV(
    estimator=rf_pipe,
    param_grid=rf_param_grid,
    cv=cv,
    scoring="f1",
    n_jobs=-1,
    verbose=0
)
rf_grid.fit(X_train, y_train)
print("Best params (RF):", rf_grid.best_params_)
print("Best CV F1 (RF):", round(rf_grid.best_score_, 4))

# -------------------------
# Evaluate on test set
# -------------------------
def evaluate(model, name, X_test, y_test, feature_names):
    y_pred = model.predict(X_test)
    # Some models may not have predict_proba; guard it
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        y_proba, auc = None, np.nan

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n{name} — Test Metrics")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}" if not np.isnan(auc) else "ROC-AUC  : N/A")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"{name} — Confusion Matrix")
    plt.show()

    # ROC
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
        plt.plot([0,1], [0,1], "--")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"{name} — ROC Curve"); plt.legend(); plt.show()

    # Feature importance / coefficients (top 12)
    try:
        # If it's the RF pipeline, last step is clf; get feature importances
        if hasattr(model.named_steps["clf"], "feature_importances_"):
            importances = model.named_steps["clf"].feature_importances_
            idx = np.argsort(importances)[::-1][:12]
            plt.barh(np.array(feature_names)[idx][::-1], importances[idx][::-1])
            plt.title(f"{name} — Top 12 Feature Importances")
            plt.xlabel("Importance"); plt.tight_layout(); plt.show()
        elif hasattr(model.named_steps["clf"], "coef_"):
            coefs = model.named_steps["clf"].coef_.ravel()
            idx = np.argsort(np.abs(coefs))[::-1][:12]
            plt.barh(np.array(feature_names)[idx][::-1], coefs[idx][::-1])
            plt.title(f"{name} — Top 12 Coefficients (LR)")
            plt.xlabel("Coefficient"); plt.tight_layout(); plt.show()
    except Exception:
        pass

    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "roc_auc": None if np.isnan(auc) else auc
    }

lr_best = lr_grid.best_estimator_
rf_best = rf_grid.best_estimator_

lr_metrics = evaluate(lr_best, "Logistic Regression", X_test, y_test, num_cols)
rf_metrics = evaluate(rf_best, "Random Forest", X_test, y_test, num_cols)

# -------------------------
# Pick champion model by F1 (or ROC-AUC if you prefer)
# -------------------------
def pick_best(m1, m2, key="f1"):
    return m1 if (m1[key] or 0) >= (m2[key] or 0) else m2

champion_name, champion_model, champion_metrics = None, None, None
if pick_best(lr_metrics, rf_metrics, key="f1") is lr_metrics:
    champion_name, champion_model, champion_metrics = "Logistic Regression", lr_best, lr_metrics
else:
    champion_name, champion_model, champion_metrics = "Random Forest", rf_best, rf_metrics

print(f"\n🏆 Champion model by F1: {champion_name} → {champion_metrics}")

# -------------------------
# Save models & short report
# -------------------------
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = MODELS_DIR / f"week2_{champion_name.replace(' ','_').lower()}_{ts}.pkl"
joblib.dump(champion_model, model_path)

all_results = {
    "split_note": split_note,
    "features_used": num_cols,
    "lr_best_params": lr_grid.best_params_,
    "rf_best_params": rf_grid.best_params_,
    "lr_metrics": lr_metrics,
    "rf_metrics": rf_metrics,
    "champion": {"name": champion_name, "metrics": champion_metrics, "model_path": str(model_path)}
}

report_md = REPORTS_DIR / "week2_model_report.md"
with open(report_md, "w", encoding="utf-8") as f:
    f.write("# Week 2 — Model Training Report\n\n")
    f.write(f"- Split: {split_note}\n")
    f.write(f"- Features used ({len(num_cols)}): {', '.join(num_cols)}\n\n")
    f.write("## Logistic Regression\n")
    f.write(f"- Best params: {lr_grid.best_params_}\n")
    f.write(f"- Test metrics: {json.dumps(lr_metrics, indent=2)}\n\n")
    f.write("## Random Forest\n")
    f.write(f"- Best params: {rf_grid.best_params_}\n")
    f.write(f"- Test metrics: {json.dumps(rf_metrics, indent=2)}\n\n")
    f.write(f"## Champion Model: {champion_name}\n")
    f.write(f"- Saved to: {model_path}\n")

print(f"\n✅ Saved champion model to: {model_path}")
print(f"📝 Wrote report: {report_md}")
