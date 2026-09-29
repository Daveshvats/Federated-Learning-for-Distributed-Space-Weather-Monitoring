"""
experiments/run_alpha_promotion.py  (review R3 / queued item 7)
────────────────────────────────────────────────────────────────
Sweep-winner promotion run at the full headline protocol length.

The mu/alpha sensitivity sweep (9 configurations, 15 rounds) selected
alpha=5.0 as the validation-optimal heterogeneity setting. Promoting
it to the headline requires demonstrating that it transfers to the
full 50-round frozen protocol. This runner executes exactly that:
repartition at the requested alpha (default 5.0), train FedAvg and
FedProx under the headline protocol, and evaluate with the identical
frozen pipeline (prior_shift calibration + validation-frozen F-beta
threshold, single test pass).

Round-resumable via resume_path checkpoints; the main.py phase cache
is reused for the data matrices only (the partition is recomputed at
the requested alpha, so no contamination of the headline caches is
possible).

Usage:
    python experiments/run_alpha_promotion.py
    python experiments/run_alpha_promotion.py --alpha 5.0 --rounds 50
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


def _atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--rounds", type=int, default=cfg.N_ROUNDS)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    t0 = time.time()
    from federated_learning import run_fedavg, run_fedprox, get_model_probs
    from partition_clients import partition_data_dirichlet
    from evaluation import (make_calibrator, find_optimal_threshold_fbeta,
                            compute_all_metrics,
                            select_fpr_thresholds_on_validation,
                            frozen_operating_point_metrics)

    cfg.USE_LSTM = False
    cache = os.path.join("data", "cache", "alpha5")
    os.makedirs(cache, exist_ok=True)

    # ── data: reuse main.py phase cache; repartition at the target alpha ──
    data_cache = os.path.join("data", "cache", "phase_data.npz")
    if os.path.exists(data_cache):
        z = np.load(data_cache, allow_pickle=False)
        X_train, y_train = z["X_train"], z["y_train"]
        X_val, y_val = z["X_val"], z["y_val"]
        X_test, y_test = z["X_test"], z["y_test"]
        print("[AlphaPromotion] data reused from main.py phase cache")
    else:
        from data_preparation import load_or_generate_data, preprocess
        df = load_or_generate_data()
        splits = preprocess(df, val_fraction=cfg.VAL_SPLIT)
        X_train, y_train = splits.X_train, splits.y_train
        X_val, y_val = splits.X_val, splits.y_val
        X_test, y_test = splits.X_test, splits.y_test

    shards = partition_data_dirichlet(
        X_train, y_train, alpha=args.alpha, n_clients=cfg.N_CLIENTS,
        seed=cfg.SEED, min_samples=cfg.MIN_SAMPLES_PER_CLIENT)

    # ── federated arms at the target alpha (round-resumable) ─────────────
    print(f"\n=== FedAvg alpha={args.alpha} ===")
    fedavg_model, _ = run_fedavg(
        shards, X_val, y_val, n_rounds=args.rounds,
        use_lstm=False, eval_batch_size=cfg.EVAL_BATCH_SIZE,
        seed=cfg.SEED, resume_path=os.path.join(cache, "fedavg.pt"))

    print(f"\n=== FedProx alpha={args.alpha} ===")
    fedprox_model, _ = run_fedprox(
        shards, X_val, y_val, n_rounds=args.rounds, mu=cfg.MU,
        use_lstm=False, eval_batch_size=cfg.EVAL_BATCH_SIZE,
        seed=cfg.SEED, resume_path=os.path.join(cache, "fedprox.pt"))

    # ── identical evaluation protocol to the headline run ────────────────
    print("\n=== Evaluation (prior_shift, val-frozen thresholds) ===")
    device = next(fedprox_model.parameters()).device

    def evaluate(model):
        p_val = get_model_probs(model, X_val, device, cfg.EVAL_BATCH_SIZE)
        p_test = get_model_probs(model, X_test, device, cfg.EVAL_BATCH_SIZE)
        cal = make_calibrator(cfg.CALIBRATION_METHOD).fit(y_val, p_val)
        if cfg.CALIBRATION_METHOD == "prior_shift":
            cal.set_prevalences(train_rate=float(y_train.mean()),
                                test_rate=float(y_val.mean()))
        p_val_c = cal.transform(p_val)
        p_test_c = cal.transform(p_test)
        t, fb = find_optimal_threshold_fbeta(
            y_val, p_val_c, beta=cfg.FBETA_BETA, grid=cfg.THRESHOLD_GRID)
        metrics = compute_all_metrics(y_test, p_test_c, t, beta=cfg.FBETA_BETA)
        frozen = select_fpr_thresholds_on_validation(y_val, p_val_c,
                                                     FPR_TARGETS)
        ops = frozen_operating_point_metrics(y_test, p_test_c, frozen)
        return {"threshold": float(t), "val_fbeta": float(fb),
                "test": metrics, "frozen_operating_points": ops}

    report = {
        "alpha": args.alpha, "seed": cfg.SEED, "rounds": args.rounds,
        "mu": cfg.MU, "calibration": cfg.CALIBRATION_METHOD,
        "n_clients": cfg.N_CLIENTS,
        "models": {
            "fedavg_mlp": evaluate(fedavg_model),
            "fedprox_mlp": evaluate(fedprox_model),
        },
        "alpha_1.0_reference": {
            "fedprox_roc_auc": 0.954, "fedprox_pr_auc": 0.307,
            "source": "outputs/results.json (frozen protocol, reproduced)"},
        "elapsed_s": round(time.time() - t0, 1),
    }

    path = args.output or os.path.join(
        cfg.OUTPUT_DIR,
        "alpha5_promotion.json" if args.alpha == 5.0
        else f"alpha{args.alpha:g}_promotion.json")
    _atomic_json(report, path)
    print(f"\n[AlphaPromotion] report -> {path}")
    for name, m in report["models"].items():
        t = m["test"]
        op = m["frozen_operating_points"].get("2%FPR", {})
        print(f"  {name:<12} ROC-AUC {t['roc_auc']:.3f} | PR-AUC "
              f"{t['pr_auc']:.3f} | F1 {t['f1']:.3f} | R@2%FPR "
              f"{op.get('recall', float('nan')):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
