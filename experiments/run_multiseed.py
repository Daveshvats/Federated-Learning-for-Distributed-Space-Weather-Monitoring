"""
experiments/run_multiseed.py  (v3.0)
─────────────────────────────────────
Multi-seed statistical validation (revision Stage 9, audit B11).

Runs the frozen final configuration across N seeds and reports
mean / std / 95% confidence intervals for every headline metric.
Also runs paired Wilcoxon tests (FedProx vs FedAvg) when scipy is
available.

Usage:
    python experiments/run_multiseed.py [--seeds 5] [--rounds 20]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from data_preparation import load_or_generate_data, preprocess
from partition_clients import partition_data_dirichlet
from federated_learning import run_fedavg, run_fedprox, get_model_probs
from evaluation import (find_optimal_threshold_fbeta, compute_all_metrics,
                        make_calibrator)


def _summary(values):
    """mean / std / 95% CI (t-distribution) for a list of scalars."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    out = {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if n > 1 else 0.0,
           "n": n}
    if n > 1:
        try:
            from scipy import stats
            ci = stats.t.ppf(1 - (1 - cfg.CONFIDENCE) / 2, df=n - 1)
            out["ci95_half"] = float(ci * arr.std(ddof=1) / np.sqrt(n))
        except ImportError:
            out["ci95_half"] = float(1.96 * arr.std(ddof=1) / np.sqrt(n))
    else:
        out["ci95_half"] = 0.0
    return out


def _atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def _load_splits():
    """Use the shared phase cache when valid; else load from source."""
    cache = os.path.join("data", "cache")
    data_npz = os.path.join(cache, "phase_data.npz")
    try:
        if os.path.exists(data_npz):
            z = np.load(data_npz, allow_pickle=False)
            return (z["X_train"], z["y_train"], z["X_val"], z["y_val"],
                    z["X_test"], z["y_test"], True)
    except Exception as e:
        print(f"[cache] unusable ({e}) — loading from source")
    df = load_or_generate_data()
    splits = preprocess(df, val_fraction=cfg.VAL_SPLIT)
    return (splits.X_train, splits.y_train, splits.X_val, splits.y_val,
            splits.X_test, splits.y_test, False)


def run_seed(seed, n_rounds, splits_cache=None, state_dir=None):
    """One full frozen-protocol run at a given seed."""
    if splits_cache is not None:
        (X_train, y_train, X_val, y_val, X_test, y_test) = splits_cache
    else:
        df = load_or_generate_data()
        splits = preprocess(df, val_fraction=cfg.VAL_SPLIT)
        X_train, y_train = splits.X_train, splits.y_train
        X_val, y_val = splits.X_val, splits.y_val
        X_test, y_test = splits.X_test, splits.y_test
    cfg.USE_LSTM = False

    shards = partition_data_dirichlet(
        X_train, y_train, alpha=cfg.DIRICHLET_ALPHA, n_clients=cfg.N_CLIENTS,
        seed=seed, min_samples=cfg.MIN_SAMPLES_PER_CLIENT)

    out = {"seed": seed}
    import torch

    for algo, runner in (("fedavg", run_fedavg), ("fedprox", run_fedprox)):
        resume_path = None
        if state_dir:
            resume_path = os.path.join(state_dir,
                                       f"{algo}_s{seed}_r{n_rounds}.pt")
        model, hist = runner(shards, X_val, y_val, n_rounds=n_rounds,
                             seed=seed,
                             resume_path=resume_path) if algo == "fedavg" else \
            run_fedprox(shards, X_val, y_val, n_rounds=n_rounds,
                        mu=cfg.MU, seed=seed, resume_path=resume_path)
        device = next(model.parameters()).device
        p_val = get_model_probs(model, X_val, device)
        cal = make_calibrator(cfg.CALIBRATION_METHOD).fit(y_val, p_val)
        if cfg.CALIBRATION_METHOD == "prior_shift":
            cal.set_prevalences(
                train_rate=float(np.concatenate([y for _, y in shards]).mean()),
                test_rate=float(y_val.mean()))
        t, _ = find_optimal_threshold_fbeta(y_val, cal.transform(p_val),
                                            beta=cfg.FBETA_BETA,
                                            grid=cfg.THRESHOLD_GRID)
        p_test = get_model_probs(model, X_test, device)
        m = compute_all_metrics(y_test, cal.transform(p_test), t,
                                beta=cfg.FBETA_BETA)
        m.pop("preds", None)
        out[algo] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=cfg.N_SEEDS)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--base-seed", type=int, default=cfg.SEED)
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 70)
    print(f"  SF-9 MULTI-SEED VALIDATION ({args.seeds} seeds, "
          f"95% CI, frozen protocol)")
    print("=" * 70)

    X_train, y_train, X_val, y_val, X_test, y_test, cached = _load_splits()
    print(f"[data] train={len(y_train):,} val={len(y_val):,} "
          f"test={len(y_test):,} ({'phase cache' if cached else 'source'})")
    splits_cache = (X_train, y_train, X_val, y_val, X_test, y_test)
    cfg.USE_LSTM = False

    RESULTS_PATH = os.path.join(cfg.OUTPUT_DIR, "multiseed_results.json")
    STATE_DIR = os.path.join(cfg.OUTPUT_DIR, "multiseed_state")
    os.makedirs(STATE_DIR, exist_ok=True)

    # incremental: skip seeds already completed in earlier sessions
    payload = {"rounds": args.rounds, "base_seed": args.base_seed, "runs": []}
    if os.path.exists(RESULTS_PATH):
        try:
            prev = json.load(open(RESULTS_PATH))
            if prev.get("rounds") == args.rounds and \
               prev.get("base_seed") == args.base_seed:
                payload = prev
                done = {r.get("seed") for r in prev.get("runs", [])
                        if "fedavg" in r and "fedprox" in r}
                print(f"[resume] {len(done)} completed seeds skipped")
        except Exception:
            pass

    runs = payload["runs"]
    for k in range(args.seeds):
        seed = args.base_seed + k
        if seed in {r.get("seed") for r in runs
                    if "fedavg" in r and "fedprox" in r}:
            continue  # already completed
        print(f"\n──── seed {seed} ({k+1}/{args.seeds}) ────")
        try:
            runs.append(run_seed(seed, args.rounds,
                                 splits_cache=splits_cache,
                                 state_dir=STATE_DIR))
            # incremental atomic save after every seed
            payload["runs"] = runs
            payload["elapsed_s"] = round(time.time() - t0, 1)
            _atomic_json(payload, RESULTS_PATH)
            print(f"[save] incremental -> {RESULTS_PATH} ({len(runs)} seeds)")
        except Exception as e:
            print(f"[multiseed] seed {seed} FAILED: {e}")

    metrics = ["accuracy", "precision", "recall", "f1", "f2",
               "roc_auc", "pr_auc", "brier"]
    stats = {}
    for algo in ("fedavg", "fedprox"):
        algo_runs = [r[algo] for r in runs if algo in r]
        if not algo_runs:
            continue
        stats[algo] = {}
        for m in metrics:
            vals = [r[m] for r in algo_runs if m in r]
            if vals:
                stats[algo][m] = _summary(vals)

    # paired test FedProx > FedAvg
    tests = {}
    paired = [r for r in runs if "fedavg" in r and "fedprox" in r]
    if len(paired) >= 3:
        try:
            from scipy import stats as sps
            for m in ("pr_auc", "f1", "recall"):
                a = [r["fedavg"][m] for r in paired]
                b = [r["fedprox"][m] for r in paired]
                w, p = sps.wilcoxon(b, a, alternative="greater")
                tests[m] = {"wilcoxon_stat": float(w), "p_value": float(p),
                            "fedprox_better_in": int(sum(
                                1 for x, y in zip(a, b) if y > x))}
        except ImportError:
            tests["note"] = "scipy unavailable — paired tests skipped"

    out = {"n_seeds": len(runs), "seeds": [r["seed"] for r in runs],
           "rounds": args.rounds, "statistics": stats,
           "paired_tests": tests, "runs": runs,
           "elapsed_s": round(time.time() - t0, 1)}

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(cfg.OUTPUT_DIR, "multiseed_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[Done] multi-seed statistics -> {path}")

    for algo, mstats in stats.items():
        print(f"\n  {algo}:")
        for m in ("pr_auc", "f1", "recall"):
            if m in mstats:
                s = mstats[m]
                print(f"    {m:<10} {s['mean']:.3f} ± {s['std']:.3f} "
                      f"(CI95 ±{s['ci95_half']:.3f})")


if __name__ == "__main__":
    main()
