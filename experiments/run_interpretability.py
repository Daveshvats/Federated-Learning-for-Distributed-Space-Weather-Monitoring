"""
experiments/run_interpretability.py  (v3.0)
────────────────────────────────────────────
Interpretability validation (revision Stage 13).

Extends SHAP beyond the centralized XGBoost surrogate:
  1. XGBoost reference (TreeSHAP)
  2. Federated global model (FedProx) attributions
  3. Cross-model consistency: Spearman rank correlation + top-15 overlap

Compares against physically expected drivers (R-value, current helicity,
Lorentz force, magnetic flux) — consistency across models indicates the
federated decision structure is physically interpretable rather than an
artifact of one centralized surrogate.

Usage:
    python experiments/run_interpretability.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from interpretability import run_interpretability


def main():
    print("=" * 70)
    print("  SF-9 INTERPRETABILITY VALIDATION (Stage 13)")
    print("=" * 70)

    # phase-cache aware loading
    data_npz = os.path.join("data", "cache", "phase_data.npz")
    assert os.path.exists(data_npz), \
        "run main.py first (phase cache with splits required)"
    z = np.load(data_npz, allow_pickle=False)
    X_test = z["X_test"]
    feature_names = [str(s) for s in z["feature_names"]]
    print(f"[data] phase cache: test={X_test.shape}")

    # cached models from the completed main pipeline run
    baselines = os.path.join("data", "cache", "phase_baselines.pt")
    fedprox_pt = os.path.join("data", "cache", "phase_fedprox.pt")
    assert os.path.exists(baselines), "phase_baselines.pt missing (run main.py)"
    assert os.path.exists(fedprox_pt), "phase_fedprox.pt missing (run main.py)"

    import torch
    c_models = torch.load(baselines, weights_only=False)
    xgb_model = c_models["xgboost"]
    fl_state = torch.load(fedprox_pt, weights_only=False)
    assert fl_state.get("done"), "FedProx phase incomplete — run main.py"
    fl_model = fl_state["model"]
    print("[models] XGBoost + FedProx global model loaded from phase cache")

    # attribution sample: first 500 test rows (deterministic, no selection)
    sample_size = 500
    X_sample = X_test[:sample_size]

    report = run_interpretability(xgb_model, fl_model, X_sample,
                                  feature_names)

    # physically meaningful feature groups (SWAN-SF base parameters)
    groups = {
        "R_value (Schrijver flux emergence)": ["R_VALUE"],
        "current helicity (TOTUSJH/ABSNJZH/MEANJZH)": [
            "TOTUSJH", "ABSNJZH", "MEANJZH", "TOTUSJZ", "MEANJZD"],
        "Lorentz force (TOTFZ/TOTFY/TOTFX)": ["TOTFZ", "TOTFY", "TOTFX"],
        "magnetic flux (USFLUX/TOTBSQ)": ["USFLUX", "TOTBSQ"],
        "free energy / potential (TOTPOT/MEANPOT)": ["TOTPOT", "MEANPOT"],
    }
    # aggregate importance per physical group (sum of member stat-features)
    names = feature_names if len(feature_names) == len(
        report["xgboost_importance"]) else [
        f"f{i}" for i in range(len(report["xgboost_importance"]))]

    def group_importance(importance):
        out = {}
        for g, members in groups.items():
            total = 0.0
            for n, v in zip(names, importance):
                base = n.split("_", 1)[1] if "_" in n else n
                if base in members:
                    total += float(v)
            out[g] = total
        return out

    report["physical_groups"] = {
        "xgboost": group_importance(report["xgboost_importance"]),
        "fedprox_fl": group_importance(report["fl_importance"]),
    }
    report["sample_size"] = sample_size
    report["note"] = ("attributions computed on the first 500 TEST samples "
                      "(deterministic slice, no cherry-picking)")

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(cfg.OUTPUT_DIR, "interpretability.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[Done] -> {path}")


if __name__ == "__main__":
    main()
