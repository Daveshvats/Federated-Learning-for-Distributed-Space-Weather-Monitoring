"""
centralized_baseline.py
─────────────────
Trains XGBoost and Logistic Regression baselines.
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


def train_centralized(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> Dict:
    """
    Train LR and XGBoost on pooled training set.
    """
    models = {}

    # ── Logistic Regression ──────────────────────────────
    print("[Centralised] Training Logistic Regression ...")
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
        # n_jobs removed — deprecated in sklearn 1.8+, has no effect since 1.8
    )
    lr.fit(X_train, y_train)
    models["logistic_regression"] = lr
    print("[Centralised] Logistic Regression done.\n")

    # ── XGBoost (GPU-accelerated) ────────────────────────
    print("[Centralised] Training XGBoost ...")
    cuda_available = torch.cuda.is_available()

    # ✅ FIXED: Use only compatible parameters for all XGBoost versions
    xgb_params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=10.0,  # ✅ Fixed: Handle test imbalance
        random_state=RANDOM_STATE,
        verbosity=0,
    )

    if cuda_available:
        print("[Centralised] CUDA detected — using GPU acceleration ...")
        try:
            # Try modern XGBoost 2.0+ syntax first
            xgb = XGBClassifier(**xgb_params, device="cuda", tree_method="hist")
            xgb.fit(X_train, y_train)
            print("[Centralised] XGBoost running on GPU.")
        except Exception as e1:
            try:
                # Fallback to older syntax
                xgb = XGBClassifier(**xgb_params, tree_method="gpu_hist", gpu_id=0)
                xgb.fit(X_train, y_train)
                print("[Centralised] XGBoost running on GPU (legacy mode).")
            except Exception as e2:
                print(f"[Centralised] GPU failed ({e2}), using CPU.")
                xgb = XGBClassifier(**xgb_params)
                xgb.fit(X_train, y_train)
    else:
        print("[Centralised] No CUDA — XGBoost on CPU.")
        xgb = XGBClassifier(**xgb_params)
        xgb.fit(X_train, y_train)

    models["xgboost"] = xgb
    print("[Centralised] XGBoost done.\n")

    return models


def evaluate_centralized(
    models:    Dict,
    X_test:    np.ndarray,
    y_test:    np.ndarray,
    threshold: float = THRESHOLD
) -> Dict[str, Dict]:
    """Evaluate each centralised model; return results dict."""
    results = {}

    for name, model in models.items():
        try:
            probs = model.predict_proba(X_test)[:, 1]
        except:
            # Fallback for models without predict_proba
            probs = model.predict(X_test).astype(float)
        
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


def compute_shap(
    xgb_model:     XGBClassifier,
    X_test:        np.ndarray,
    feature_names: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SHAP values for XGBoost.
    Returns (shap_values, mean_abs_shap) for plotting.
    """
    print("[SHAP] Computing SHAP values (may take ~30s) ...")
    
    try:
        # Use subset for speed
        sample_size = min(500, len(X_test))
        sample = X_test[:sample_size]
        
        explainer = shap.TreeExplainer(xgb_model)
        shap_vals = explainer.shap_values(sample)
        mean_shap = np.abs(shap_vals).mean(axis=0)
        print("[SHAP] Done.\n")
        return shap_vals, mean_shap
        
    except Exception as e:
        print(f"[SHAP] Failed: {e}\n")
        return None, None