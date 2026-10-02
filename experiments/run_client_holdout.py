"""
experiments/run_client_holdout.py  (review R14)
────────────────────────────────────────────────
Untouched-holdout client-level evaluation.

Protocol (R14):
    1. Split every client's shard into a FEDERATION-TRAIN portion (80%)
       and a completely UNTOUCHED holdout (20%) BEFORE any training.
    2. Train FedAvg + FedProx on the federation-train portions only —
       the holdouts never influence any weight update.
    3. Train per-client local-only models on the same federation-train
       portions (same local budget) as the comparison baseline.
    4. Evaluate local vs global models on each client's untouched
       holdout — clean per-client generalisation estimates, free of the
       optimistic bias of the legacy within-shard-slice protocol.

The main.py phase cache is reused for SPLITS/SHARDS ONLY (identical
frozen-protocol identity); global models are deliberately retrained on
the 80% portions (round-resumable checkpoints) because the phase-cache
models saw the full shards and would contaminate the holdouts.

Usage:
    python experiments/run_client_holdout.py
    python experiments/run_client_holdout.py --rounds 50 --holdout-fraction 0.2
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg


def _atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=cfg.N_ROUNDS)
    ap.add_argument("--holdout-fraction", type=float, default=0.2)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    t0 = time.time()
    from partition_clients import partition_data_dirichlet
    from evaluate_clients import (split_shards_for_holdout,
                                  evaluate_client_level)
    from federated_learning import run_fedavg, run_fedprox
    from data_preparation import load_or_generate_data, preprocess

    cfg.USE_LSTM = False

    # ── 1. splits/shards: reuse main.py phase cache when it exists ──────
    cache_dir = os.path.join("data", "cache")
    data_cache = os.path.join(cache_dir, "phase_data.npz")
    assign_cache = os.path.join(cache_dir, "phase_assignment.npz")

    if os.path.exists(data_cache) and os.path.exists(assign_cache):
        z = np.load(data_cache, allow_pickle=False)
        X_train, y_train = z["X_train"], z["y_train"]
        X_val, y_val = z["X_val"], z["y_val"]
        assignment = np.load(assign_cache)["assignment"]
        X_arr, y_arr = np.asarray(X_train), np.asarray(y_train)
        shards = []
        for k in range(cfg.N_CLIENTS):
            idx = np.where(assignment == k)[0]
            rng = np.random.RandomState(cfg.SEED * 1000 + k)
            idx = idx[rng.permutation(len(idx))]
            shards.append((X_arr[idx], y_arr[idx]))
        splits_source = "main.py phase cache (frozen protocol)"
        print("[Holdout] splits/shards reused from main.py phase cache")
    else:
        df = load_or_generate_data()
        splits = preprocess(df, val_fraction=cfg.VAL_SPLIT)
        X_train, y_train = splits.X_train, splits.y_train
        X_val, y_val = splits.X_val, splits.y_val
        shards = partition_data_dirichlet(
            X_train, y_train, alpha=cfg.DIRICHLET_ALPHA,
            n_clients=cfg.N_CLIENTS, seed=cfg.SEED,
            min_samples=cfg.MIN_SAMPLES_PER_CLIENT)
        splits_source = "recomputed"

    # ── 2. untouched holdouts carved BEFORE training ─────────────────────
    fed_shards, holdouts = split_shards_for_holdout(
        shards, holdout_fraction=args.holdout_fraction, seed=cfg.SEED)
    print(f"[Holdout] carved {args.holdout_fraction:.0%} untouched holdouts "
          "from every client shard (seed 42)")

    # ── 3. global models trained on federation-train portions ONLY ──────
    # (round-resumable; deliberately NOT reused from main.py's phase cache
    #  because those models saw the full shards)
    fedavg, _ = run_fedavg(
        fed_shards, X_val, y_val, n_rounds=args.rounds,
        seed=cfg.SEED,
        resume_path=os.path.join(cfg.OUTPUT_DIR, "holdout_fedavg_state.pt"))
    fedprox, _ = run_fedprox(
        fed_shards, X_val, y_val, n_rounds=args.rounds, mu=cfg.MU,
        seed=cfg.SEED,
        resume_path=os.path.join(cfg.OUTPUT_DIR, "holdout_fedprox_state.pt"))

    # ── 4. local vs global on the untouched holdouts ─────────────────────
    results = evaluate_client_level(
        fed_shards, (fedavg, fedprox), "MLP",
        batch_size=cfg.EVAL_BATCH_SIZE, holdouts=holdouts)

    results["protocol"] = {
        "rounds": args.rounds,
        "seed": cfg.SEED,
        "holdout_fraction": args.holdout_fraction,
        "mu": cfg.MU,
        "alpha": cfg.DIRICHLET_ALPHA,
        "splits_source": splits_source,
        "global_models_trained_on": "federation-train portions only",
        "note": ("holdouts carved before training; global models "
                 "deliberately retrained on the 80% portions (phase-cache "
                 "models saw the full shards and would contaminate)"),
    }
    results["elapsed_s"] = time.time() - t0

    path = args.output or os.path.join(cfg.OUTPUT_DIR,
                                       "client_holdout_eval.json")
    _atomic_json(results, path)
    print(f"\n[Holdout] report -> {path}")
    s = results["summary"]
    print(f"[Holdout] protocol: {s['evaluation_protocol']} | "
          f"clients evaluated: {s['n_clients_evaluated']}")
    print(f"[Holdout] mean local PR-AUC {s['mean_local_pr_auc']:.3f} | "
          f"mean FedAvg PR-AUC {s['mean_fedavg_pr_auc']:.3f} | "
          f"mean FedProx PR-AUC {s['mean_fedprox_pr_auc']:.3f}")
    print(f"[Holdout] clients helped by FedProx: "
          f"{s['clients_helped_by_federation']}/{s['n_clients_evaluated']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
