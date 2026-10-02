"""
experiments/run_raw_substrate.py  (review item 9: raw-substrate retraining;
also executes the SCAFFOLD half of item 8)
────────────────────────────────────────────────────────
Retrain the FROZEN protocol on the raw, natural-prevalence SWAN-SF
benchmark — no RUS-Tomek-TimeGAN rebalancing, no synthetic positives —
using the FPCKNN/LSBZM reproduction of experiments/raw_substrate.py
(parameters fitted on the train pool only, verified against the released
cleaned export), and evaluate ONCE on raw partition 5.

Substrate (leakage-free, temporally-preceding):
    train P1-4 raw (natural ~2.05% prevalence, stratified 84/16 val
    carve, seed 42) -> single-pass test P5 raw (natural 1.31%).

Protocol (unchanged / frozen): Dirichlet alpha=1.0, 6 clients, 50
rounds, 10 local epochs, Fed-Focal, seed 42, prior-shift calibration,
val-frozen F-beta thresholds + frozen-FPR operating points.

Arms: logistic regression, XGBoost, centralised MLP, FedAvg, FedProx,
SCAFFOLD (item 8's remaining arm; identical compute budget).

Round-resumable via data/cache/rawsubstrate/ checkpoints.  CPU-only.

Usage:
    python experiments/run_raw_substrate.py --raw-dir /tmp/swansf_raw
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg

FPR_TARGETS = (0.005, 0.01, 0.02, 0.05)

# reference numbers for the comparison columns
# v4.0 (review M1): both reference dicts are READ from the committed
# machine-readable artefacts at run time — never hand-typed.
def _load_reference(path, key):
    """Read {model: {roc_auc, pr_auc}} from a committed eval JSON."""
    import json as _json
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        d = _json.load(f)
    out = {}
    for name, entry in d.get("results", {}).items():
        m = entry.get(key) or entry.get("test") or {}
        if isinstance(m, dict) and "roc_auc" in m and "pr_auc" in m:
            out[name] = {"roc_auc": float(m["roc_auc"]),
                         "pr_auc": float(m["pr_auc"]),
                         "source": os.path.basename(path)}
    return out


def load_shipped_reference():
    """In-partition (shipped-protocol) reference metrics from
    outputs/results.json (test_metrics block)."""
    import json as _json
    path = cfg.RESULTS_JSON
    with open(path) as f:
        res = _json.load(f)
    out = {}
    for name, m in res.get("test_metrics", {}).items():
        if isinstance(m, dict) and "roc_auc" in m and "pr_auc" in m:
            out[name] = {"roc_auc": float(m["roc_auc"]),
                         "pr_auc": float(m["pr_auc"]),
                         "source": "outputs/results.json"}
    return out


SHIPPED = load_shipped_reference()          # in-partition (memorisation-tainted)
FOLD = _load_reference(                     # leakage-free, cleaned substrate (v3.4)
    os.path.join("outputs", "partition_disjoint_eval.json"),
    "test")


def _atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def _atomic_torch(obj, path):
    import torch
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="/tmp/swansf_raw")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    t0 = time.time()
    from federated_learning import run_fedavg, run_fedprox, run_scaffold, \
        get_model_probs
    from partition_clients import partition_data_dirichlet
    from centralized_baseline import train_centralized, model_probs
    from evaluation import (make_calibrator, find_optimal_threshold_fbeta,
                            compute_all_metrics,
                            select_fpr_thresholds_on_validation,
                            frozen_operating_point_metrics)
    from experiments.raw_substrate import build, CACHE

    cfg.USE_LSTM = False
    cfg.USE_SCAFFOLD = True          # item 8 arm, same budget as FedProx
    out_path = args.output or "outputs/raw_substrate_eval.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # ── 1. data (built+cached by experiments/raw_substrate.py) ────────
    d = build(args.raw_dir, CACHE)
    X_train, y_train = d["X_train"], d["y_train"]
    X_val, y_val = d["X_val"], d["y_val"]
    X_test, y_test = d["X_test"], d["y_test"]
    print(f"[RawSub] train {len(y_train):,} (pos {y_train.mean():.2%}) | "
          f"val {len(y_val):,} (pos {y_val.mean():.2%}) | "
          f"test {len(y_test):,} (pos {y_test.mean():.2%})", flush=True)

    # ── 2. shards (identical recipe: Dirichlet alpha=1.0, seed 42) ────
    shards = partition_data_dirichlet(
        X_train, y_train, alpha=1.0, n_clients=cfg.N_CLIENTS,
        seed=cfg.SEED, min_samples=cfg.MIN_SAMPLES_PER_CLIENT)

    # ── 3. pooled baselines (cached) ────────────────────────────────────
    import torch
    bcache = os.path.join(CACHE, "baselines.pt")
    if os.path.exists(bcache):
        c_models = torch.load(bcache, weights_only=False)
        print("[RawSub] baselines resumed from cache", flush=True)
    else:
        print("\n[RawSub] training pooled baselines on raw P1-4 ...",
              flush=True)
        c_models = train_centralized(X_train, y_train, X_val, y_val,
                                     seed=cfg.SEED)
        _atomic_torch(c_models, bcache)

    # ── 4. federated arms (round-resumable) ─────────────────────────────
    print("\n[RawSub] FedAvg (50 rounds, frozen protocol) ...", flush=True)
    fedavg_model, _ = run_fedavg(
        shards, X_val, y_val, n_rounds=cfg.N_ROUNDS, use_lstm=False,
        eval_batch_size=cfg.EVAL_BATCH_SIZE, seed=cfg.SEED,
        resume_path=os.path.join(CACHE, "fedavg.pt"))
    print("\n[RawSub] FedProx (50 rounds, frozen protocol) ...", flush=True)
    fedprox_model, _ = run_fedprox(
        shards, X_val, y_val, n_rounds=cfg.N_ROUNDS, mu=cfg.MU,
        use_lstm=False, eval_batch_size=cfg.EVAL_BATCH_SIZE,
        seed=cfg.SEED, resume_path=os.path.join(CACHE, "fedprox.pt"))
    print("\n[RawSub] SCAFFOLD (50 rounds, frozen protocol) ...",
          flush=True)
    scaffold_model, _ = run_scaffold(
        shards, X_val, y_val, n_rounds=cfg.N_ROUNDS, use_lstm=False,
        eval_batch_size=cfg.EVAL_BATCH_SIZE, seed=cfg.SEED,
        resume_path=os.path.join(CACHE, "scaffold.pt"))

    # ── 5. evaluation: identical frozen protocol ────────────────────────
    print("\n[RawSub] evaluation (prior_shift, val-frozen thresholds)",
          flush=True)
    device = next(fedprox_model.parameters()).device

    def mprobs(model, X):
        return np.asarray(model_probs(model, X), dtype=float)

    arms = {
        "logistic_regression": lambda X: mprobs(c_models["logistic_regression"], X),
        "xgboost": lambda X: mprobs(c_models["xgboost"], X),
        "centralized_mlp": lambda X: mprobs(c_models["centralized_mlp"], X),
        "fedavg_mlp": lambda X: get_model_probs(fedavg_model, X, device,
                                                cfg.EVAL_BATCH_SIZE),
        "fedprox_mlp": lambda X: get_model_probs(fedprox_model, X, device,
                                                 cfg.EVAL_BATCH_SIZE),
        "scaffold_mlp": lambda X: get_model_probs(scaffold_model, X, device,
                                                  cfg.EVAL_BATCH_SIZE),
    }

    results, probs_dump = {}, {}
    for name, fn in arms.items():
        p_val = fn(X_val)
        p_test = fn(X_test)
        cal = make_calibrator(cfg.CALIBRATION_METHOD).fit(y_val, p_val)
        if cfg.CALIBRATION_METHOD == "prior_shift":
            cal.set_prevalences(train_rate=float(y_train.mean()),
                                test_rate=float(y_val.mean()))
        p_val_c = cal.transform(p_val)
        p_test_c = cal.transform(p_test)
        t, fb = find_optimal_threshold_fbeta(
            y_val, p_val_c, beta=cfg.FBETA_BETA, grid=cfg.THRESHOLD_GRID)
        metrics = compute_all_metrics(y_test, p_test_c, t,
                                      beta=cfg.FBETA_BETA)
        frozen = select_fpr_thresholds_on_validation(y_val, p_val_c,
                                                     FPR_TARGETS)
        ops = frozen_operating_point_metrics(y_test, p_test_c, frozen)
        results[name] = {
            "threshold": float(t), "val_fbeta": float(fb),
            "test": metrics, "frozen_operating_points": ops,
            "shipped_in_partition": SHIPPED.get(name),
            "fold_leakage_free_cleaned": FOLD.get(name),
        }
        probs_dump[name] = np.asarray(p_test_c, dtype=np.float32)
        print(f"  {name:<20s} ROC-AUC {metrics['roc_auc']:.3f} "
              f"(fold {FOLD.get(name, {}).get('roc_auc')}) "
              f"PR-AUC {metrics['pr_auc']:.3f} "
              f"(fold {FOLD.get(name, {}).get('pr_auc')})", flush=True)

    # ── 6. report ──────────────────────────────────────────────────────
    substrate_stats, verification = {}, {}
    sp = os.path.join(CACHE, "substrate_stats.json")
    if os.path.exists(sp):
        substrate_stats = json.load(open(sp))
    vp = "outputs/raw_substrate_verification.json"
    if os.path.exists(vp):
        v = json.load(open(vp))
        verification = {k: v[k] for k in (
            "n_matched_windows", "match_fraction_p5",
            "median_spearman_all", "median_spearman_observed_nonzero",
            "min_spearman_observed_nonzero", "median_spearman_imputed")
            if k in v}

    report = {
        "purpose": ("review item 9: frozen protocol retrained on the RAW "
                    "unbalanced SWAN-SF benchmark (natural prevalence, no "
                    "RUS-Tomek-TimeGAN, no synthetic positives) with "
                    "FPCKNN/LSBZM reproduced in-pipeline (train-only "
                    "parameter fit); single-pass test on raw P5"),
        "protocol": {"rounds": cfg.N_ROUNDS, "mu": cfg.MU, "alpha": 1.0,
                     "seed": cfg.SEED, "clients": cfg.N_CLIENTS,
                     "val_split": cfg.VAL_SPLIT,
                     "flatten": cfg.FLATTEN_METHOD,
                     "calibration": cfg.CALIBRATION_METHOD,
                     "loss": cfg.LOSS_VARIANT},
        "substrate": {
            "train": int(len(y_train)), "val": int(len(y_val)),
            "test": int(len(y_test)),
            "train_pos": float(y_train.mean()),
            "val_pos": float(y_val.mean()),
            "test_pos": float(y_test.mean()),
            "imputation": "FPCKNN-style (partial gaps: within-window "
                          "linear interpolation; fully-missing series: "
                          "Pearson-correlation-based kNN transfer, k=5, "
                          "train-only references)",
            "normalization": "LSBZM-style chain (robust clip -> shift -> "
                             "log/sqrt/Box-Cox by fitted lambda -> z-score "
                             "-> min-max), train-only parameters",
            "stats": substrate_stats, "verification": verification,
        },
        "results": results,
        "elapsed_s": time.time() - t0,
    }
    _atomic_json(report, out_path)
    np.savez_compressed(os.path.join(CACHE, "test_probs.npz"),
                        y_test=y_test.astype(np.int8), **probs_dump)
    print(f"\n[RawSub] report -> {out_path} ({time.time()-t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
