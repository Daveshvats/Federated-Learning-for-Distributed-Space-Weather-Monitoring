"""
centralized_baseline.py
────────────────────────
Trains an XGBoost classifier on the entire training set (all clients' data
pooled) as a CENTRALISED UPPER BOUND.

This answers the key question in federated learning papers:
"How much accuracy do we sacrifice for privacy?"

If federated F1 ≈ centralised F1 → privacy comes nearly for free.
If federated F1 << centralised F1 → further tuning or more rounds needed.

Also runs a Logistic Regression baseline so the paper has a 3-model
comparison table that reviewers expect.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
from xgboost import XGBClassifier
import shap

from config import RANDOM_STATE, THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_centralized(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> Dict:
    """
    Train LR and XGBoost on the full pooled training set.

    NOTE: sklearn models (LogisticRegression) are CPU-only Python objects.
    They do NOT have a .to(device) method — that is PyTorch syntax only.
    XGBoost can use GPU via device='cuda' (XGBoost >= 2.0) or
    tree_method='gpu_hist' (XGBoost < 2.0).
    """
    models = {}

    # ── Logistic Regression (sklearn — CPU only, no .to() needed) ──────────
    print("[Centralised] Training Logistic Regression ...")
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=-1          # uses all CPU cores — that's the correct speedup for LR
    )
    lr.fit(X_train, y_train)
    models["logistic_regression"] = lr
    print("[Centralised] Logistic Regression done.\n")

    # ── XGBoost (GPU-accelerated if CUDA available) ─────────────────────────
    print("[Centralised] Training XGBoost ...")
    cuda_available = torch.cuda.is_available()

    # XGBoost >= 2.0 uses device='cuda'; older versions use tree_method='gpu_hist'
    # We try the modern syntax first and fall back gracefully
    xgb_params = dict(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    )

    if cuda_available:
        print("[Centralised] CUDA detected — attempting XGBoost GPU acceleration ...")
        try:
            # XGBoost >= 2.0 syntax
            xgb = XGBClassifier(**xgb_params, device="cuda")
            xgb.fit(X_train, y_train)
            print("[Centralised] XGBoost running on GPU (device='cuda').")
        except TypeError:
            try:
                # XGBoost 1.x syntax
                xgb = XGBClassifier(**xgb_params, tree_method="gpu_hist", gpu_id=0)
                xgb.fit(X_train, y_train)
                print("[Centralised] XGBoost running on GPU (tree_method='gpu_hist').")
            except Exception as e:
                print(f"[Centralised] GPU XGBoost failed ({e}), falling back to CPU.")
                xgb = XGBClassifier(**xgb_params)
                xgb.fit(X_train, y_train)
    else:
        print("[Centralised] No CUDA — XGBoost on CPU.")
        xgb = XGBClassifier(**xgb_params)
        xgb.fit(X_train, y_train)

    models["xgboost"] = xgb
    print("[Centralised] XGBoost done.\n")

    return models


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_centralized(
    models:    Dict,
    X_test:    np.ndarray,
    y_test:    np.ndarray,
    threshold: float = THRESHOLD
) -> Dict[str, Dict]:
    """Evaluate each centralised model; return a results dict."""
    results = {}

    for name, model in models.items():
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= threshold).astype(int)

        results[name] = {
            "accuracy":  accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall":    recall_score(y_test, preds, zero_division=0),
            "f1":        f1_score(y_test, preds, zero_division=0),
            "roc_auc":   roc_auc_score(y_test, probs),
            "probs":     probs,
            "preds":     preds,
        }
        print(f"[Centralised] {name:<25} | "
              f"F1: {results[name]['f1']:.3f} | "
              f"Recall: {results[name]['recall']:.3f} | "
              f"ROC-AUC: {results[name]['roc_auc']:.3f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SHAP FEATURE IMPORTANCE (XGBoost only)
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap(
    xgb_model:     XGBClassifier,
    X_test:        np.ndarray,
    feature_names: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SHAP values for XGBoost.
    Returns (shap_values, mean_abs_shap) for plotting.
    SHAP shows PER-PREDICTION importance — far more powerful than
    default XGBoost feature_importances_ for a reviewer audience.
    """
    print("[SHAP] Computing SHAP values (may take ~30s) ...")
    # Use a subset for speed; 500 samples is sufficient for importance ranking
    sample = X_test[:500]
    explainer = shap.TreeExplainer(xgb_model)
    shap_vals  = explainer.shap_values(sample)
    mean_shap  = np.abs(shap_vals).mean(axis=0)
    print("[SHAP] Done.\n")
    return shap_vals, mean_shap