"""
experiments/analyze_smote_validity.py  (v3.0)
──────────────────────────────────────────────
SMOTE physical-plausibility analysis (revision Stage 12).

SMOTE interpolates between minority-class neighbours in feature space.
For SWAN-SF magnetic parameters, naive interpolation can produce
physically implausible combinations (e.g. huge free energy with tiny
flux, or negative concentrations after scaling) that a model can
exploit to inflate minority-class separability.

Checks:
  1. Marginal distribution shift per feature (Wasserstein-1 distance,
     real minority vs synthetic samples)
  2. Correlation-structure preservation (|Δcorr| matrix, mean + max)
  3. Physical-range violations: synthetic samples outside the
     [min, max] envelope of REAL minority observations per feature
  4. Classification separability: can a 1-NN/RF distinguish synthetic
     from real minority samples? (train-test AUC near 0.5 = plausible)

Usage:
    python experiments/analyze_smote_validity.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from data_preparation import load_or_generate_data, preprocess, apply_smote


def wasserstein_1(a, b):
    """1-D Wasserstein distance (quantile approximation, robust for
    plausibility screening)."""
    qs = np.linspace(0.0, 1.0, 101)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.abs(np.quantile(a, qs) - np.quantile(b, qs)).mean())


def analyse(X_real_pos, X_synth, feature_names):
    """Compare real minority samples with synthetic SMOTE samples."""
    n_feat = X_real_pos.shape[1]
    report = {}

    # 1. per-feature Wasserstein distance
    wd = [wasserstein_1(X_real_pos[:, j], X_synth[:, j]) for j in range(n_feat)]
    report["wasserstein_mean"] = float(np.mean(wd))
    report["wasserstein_max"] = float(np.max(wd))
    worst = np.argsort(wd)[-5:]
    report["worst_features"] = [
        {"feature": feature_names[j] if j < len(feature_names) else f"f{j}",
         "wasserstein": float(wd[j])} for j in worst]

    # 2. correlation preservation
    if X_real_pos.shape[0] > 3 and X_synth.shape[0] > 3:
        corr_real = np.corrcoef(X_real_pos.T)
        corr_syn = np.corrcoef(X_synth.T)
        iu = np.triu_indices(n_feat, k=1)
        d = np.abs(corr_real[iu] - corr_syn[iu])
        report["mean_abs_delta_corr"] = float(d.mean())
        report["max_abs_delta_corr"] = float(d.max())
    else:
        report["mean_abs_delta_corr"] = None

    # 3. physical-range violations (outside real minority envelope)
    lo, hi = X_real_pos.min(axis=0), X_real_pos.max(axis=0)
    below = (X_synth < lo).any(axis=1).sum()
    above = (X_synth > hi).any(axis=1).sum()
    report["range_violation_fraction"] = float(
        (below + above) / len(X_synth)) if len(X_synth) else 0.0

    # 4. separability of synthetic vs real (logistic regression AUC)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        X_all = np.vstack([X_real_pos, X_synth])
        y_all = np.concatenate([np.zeros(len(X_real_pos)),
                                np.ones(len(X_synth))])
        if len(y_all) > 20 and 0 < y_all.mean() < 1:
            clf = LogisticRegression(max_iter=500)
            aucs = cross_val_score(clf, X_all, y_all, cv=3, scoring="roc_auc")
            report["real_vs_synth_auc"] = float(aucs.mean())
    except Exception as e:
        report["real_vs_synth_auc"] = f"failed: {e}"

    return report


def main():
    print("=" * 70)
    print("  SMOTE PHYSICAL-VALIDITY ANALYSIS (Stage 12)")
    print("=" * 70)

    df = load_or_generate_data()
    splits = preprocess(df, val_fraction=cfg.VAL_SPLIT)
    X_train, y_train = splits.X_train, splits.y_train
    feature_names = splits.features

    pos = np.where(y_train == 1)[0]
    X_pos = X_train[pos]

    print(f"\n[SMOTE] real minority samples: {len(X_pos)}")
    X_res, y_res = apply_smote(X_train, y_train, seed=cfg.SEED)
    synth_mask = y_res == 1
    n_synth = int(synth_mask.sum() - len(pos))
    if n_synth <= 0:
        print("[SMOTE] no synthetic samples generated (ratio/prevalence). "
              "Increase SMOTE_RATIO or check data.")
        return
    # recover synthetic-only samples (SMOTE appends them)
    X_synth = X_res[y_res == 1][len(pos):]

    report = analyse(X_pos, X_synth, feature_names)
    print("\n[SMOTE validity]")
    print(f"  synthetic samples: {n_synth}")
    print(f"  Wasserstein-1  mean/max: {report['wasserstein_mean']:.4f} / "
          f"{report['wasserstein_max']:.4f}")
    if report.get("mean_abs_delta_corr") is not None:
        print(f"  |Δcorr| mean/max: {report['mean_abs_delta_corr']:.4f} / "
              f"{report['max_abs_delta_corr']:.4f}")
    print(f"  range violations: {report['range_violation_fraction']:.1%}")
    if isinstance(report.get("real_vs_synth_auc"), float):
        auc = report["real_vs_synth_auc"]
        verdict = "PLAUSIBLE" if 0.5 <= auc < 0.8 else \
                  "SEPARABLE (physically suspect)"
        print(f"  real-vs-synthetic AUC: {auc:.3f} -> {verdict}")

    report["n_real_minority"] = int(len(X_pos))
    report["n_synthetic"] = int(n_synth)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(cfg.OUTPUT_DIR, "smote_validity.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[Done] -> {path}")


if __name__ == "__main__":
    main()
