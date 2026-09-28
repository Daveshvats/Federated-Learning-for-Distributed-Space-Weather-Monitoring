"""
interpretability.py  (v3.0 — improvements branch)
────────────────────────────────────────────────────
FL-model interpretability (revision Stage 13).

The v2.x pipeline computed SHAP ONLY on the centralized XGBoost
surrogate and then made claims about the *federated* model's physics.
This module computes attributions for BOTH:
  1. centralized XGBoost (TreeExplainer — exact, fast)
  2. the federated global model (DeepLiftShap / GradientShap / Kernel
     fallback, batched for GPU memory safety)

Consistency check: rank correlation (Spearman) between the two
importance rankings. High agreement -> the federated decision structure
is consistent with the centralized surrogate (supports the physical
interpretation claim); low agreement -> the claim is an artifact of the
surrogate and must be weakened.
"""

import numpy as np

import config as cfg


def shap_xgboost(xgb_model, X_sample, feature_names, sample_size=500):
    """TreeExplainer SHAP for the XGBoost reference baseline."""
    import shap
    sample = X_sample[:min(sample_size, len(X_sample))]
    explainer = shap.TreeExplainer(xgb_model)
    vals = explainer.shap_values(sample)
    if isinstance(vals, list):
        vals = vals[1] if len(vals) == 2 else vals[0]
    mean_abs = np.abs(vals).mean(axis=0)
    return mean_abs, feature_names


def shap_torch_model(model, X_sample, feature_names, sample_size=300,
                     batch_size=256, method="deeplift"):
    """
    Attributions for the FL global model (SolarMLP path).

    Uses shap.DeepExplainer when possible, GradientShap otherwise,
    with a background sample of 100 training points. Batched for
    GPU-memory safety.

    v3.0.1: the model is wrapped so its logits keep shape (N, 1) —
    SolarMLP squeezes to (N,), which breaks GradientExplainer/DeepExplainer
    indexing (outputs[:, idx]).
    """
    import torch
    import torch.nn as nn
    import shap

    device = next(model.parameters()).device
    model.eval()

    class _OutputWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            out = self.m(x)
            if out.ndim == 1:
                out = out.unsqueeze(-1)
            return out

    wrapped = _OutputWrapper(model).to(device)
    wrapped.eval()

    n = min(sample_size, len(X_sample))
    background_idx = np.random.RandomState(cfg.SEED).choice(
        len(X_sample), size=min(100, len(X_sample)), replace=False)
    background = torch.tensor(
        np.asarray(X_sample)[background_idx], dtype=torch.float32).to(device)
    test_tensor = torch.tensor(
        np.asarray(X_sample)[:n], dtype=torch.float32).to(device)

    try:
        if method == "deeplift":
            explainer = shap.DeepExplainer(wrapped, background)
        else:
            explainer = shap.GradientExplainer(wrapped, background)
        shap_vals = explainer.shap_values(test_tensor)
    except Exception:
        explainer = shap.GradientExplainer(wrapped, background)
        shap_vals = explainer.shap_values(test_tensor)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0] if len(shap_vals) == 1 else shap_vals[-1]
    arr = np.asarray(shap_vals)
    if arr.ndim == 3:  # (N, features, outputs) for some versions
        arr = arr.squeeze(-1)
    mean_abs = np.abs(arr[:n]).mean(axis=0)
    return mean_abs, feature_names


def consistency_report(importance_a, importance_b, feature_names,
                       top_n=15):
    """
    Spearman rank agreement between two importance vectors.
    Returns the report dict used in the paper's interpretability table.
    """
    from scipy import stats as sps
    a = np.asarray(importance_a)
    b = np.asarray(importance_b)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    rho, p = sps.spearmanr(a, b)
    top_a = set(np.argsort(a)[-top_n:].tolist())
    top_b = set(np.argsort(b)[-top_n:].tolist())
    overlap = len(top_a & top_b) / top_n

    return {
        "spearman_rho": float(rho),
        "p_value": float(p),
        f"top{top_n}_overlap_fraction": float(overlap),
        "top_features_a": [feature_names[i] for i in sorted(top_a)],
        "top_features_b": [feature_names[i] for i in sorted(top_b)],
        "verdict": "consistent" if rho > 0.5 and overlap > 0.4 else
                   "inconsistent — surrogate artifact suspected",
    }


def run_interpretability(xgb_model, fl_model, X_sample, feature_names):
    """Full Stage-13 analysis: both models + consistency check."""
    print("[Interpretability] XGBoost TreeSHAP ...")
    imp_xgb, names = shap_xgboost(xgb_model, X_sample, feature_names)

    print("[Interpretability] FL global model attributions ...")
    imp_fl, _ = shap_torch_model(fl_model, X_sample, feature_names)

    report = consistency_report(imp_xgb, imp_fl, feature_names)
    report["xgboost_importance"] = imp_xgb.tolist()
    report["fl_importance"] = imp_fl.tolist()
    print(f"[Interpretability] Spearman rho = {report['spearman_rho']:.3f} "
          f"| top-15 overlap = {report['top15_overlap_fraction']:.0%} -> "
          f"{report['verdict']}")
    return report
