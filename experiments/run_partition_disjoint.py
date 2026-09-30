"""
experiments/run_partition_disjoint.py  (raw-metadata review items 1/2/9)
────────────────────────────────────────────────────────
Leakage-free evaluation fold following the SWAN-SF benchmark's
intended temporally-preceding protocol (and the Cleaned-SWANSF
release's own usage): train on the train pkls of partitions 1..4,
evaluate ONCE on the test pkl of partition 5.

MOTIVATION (raw-metadata audit finding, 2026-09-30)
    The shipped headline protocol pooled the train pkls of ALL five
    partitions and evaluated on the test pkls of the SAME five
    partitions. Aligning every cleaned window to the raw benchmark
    instances (Harvard Dataverse doi:10.7910/DVN/EBCFKM, argmax/argmin
    position-invariant value matching, 99.6-100% verification, 100%
    label agreement) shows the two sets share instances: the TEST pkl
    of partition p contains ALL raw instances of p, while the TRAIN pkl
    of p is a RUS-Tomek-TimeGAN rebalanced SUBSET of the same
    instances. 100% of verified train windows (56,005/56,006) are also
    test windows, including every M/X-flaring test instance (6,234).
    The published in-partition numbers therefore partially measure
    memorised data. This runner produces the leakage-free comparison:
    no instance, active region, or time span is shared between
    training and test (verified: 0 ARs span partitions).

Round-resumable via data/cache/pdisjoint/ checkpoints; the main.py
phase caches are NOT reused (the data substrate differs).

Usage:
    python experiments/run_partition_disjoint.py [--test-part 5]
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

# reference (shipped, in-partition) numbers from outputs/results.json
SHIPPED = {
    "xgboost": {"roc_auc": 0.977, "pr_auc": 0.493},
    "fedprox_mlp": {"roc_auc": 0.954, "pr_auc": 0.307},
    "fedavg_mlp": {"roc_auc": 0.575, "pr_auc": 0.199},
    "centralized_mlp": {"roc_auc": 0.888, "pr_auc": 0.153},
    "logistic_regression": {"roc_auc": 0.818, "pr_auc": 0.114},
}


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


def load_fold(test_part):
    """Train pkls of all partitions except test_part; test pkl of
    test_part. Flatten 3D->144 features with the frozen method."""
    import pickle
    from load_cleaned_data import flatten_3d_to_2d

    parts = [p for p in (1, 2, 3, 4, 5) if p != test_part]
    Xs, ys = [], []
    for p in parts:
        tag = "RUS-Tomek-TimeGAN_LSBZM-Norm_WithoutC_FPCKNN-impute"
        X = pickle.load(open(f"data/cleaned/train/Partition{p}_{tag}.pkl", "rb"))
        y = pickle.load(open(
            f"data/cleaned/train/Partition{p}_Labels_{tag}.pkl", "rb"))
        X = flatten_3d_to_2d(np.asarray(X), method=cfg.FLATTEN_METHOD)
        Xs.append(X.astype(np.float32))
        ys.append(np.asarray(y).astype(np.float32))
        print(f"  [fold] train Partition{p}: {X.shape}")
    X_tr = np.vstack(Xs)
    y_tr = np.concatenate(ys)

    tag = "LSBZM-Norm_FPCKNN-impute"
    X_te = pickle.load(open(
        f"data/cleaned/test/Partition{test_part}_{tag}.pkl", "rb"))
    y_te = pickle.load(open(
        f"data/cleaned/test/Partition{test_part}_Labels_{tag}.pkl", "rb"))
    X_te = flatten_3d_to_2d(np.asarray(X_te),
                            method=cfg.FLATTEN_METHOD).astype(np.float32)
    y_te = np.asarray(y_te).astype(np.float32)
    print(f"  [fold] test Partition{test_part}: {X_te.shape} "
          f"(prevalence {y_te.mean():.4%})")
    return X_tr, y_tr, X_te, y_te, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-part", type=int, default=5)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    t0 = time.time()
    from federated_learning import run_fedavg, run_fedprox, get_model_probs
    from partition_clients import partition_data_dirichlet
    from centralized_baseline import train_centralized, model_probs
    from evaluation import (make_calibrator, find_optimal_threshold_fbeta,
                            compute_all_metrics,
                            select_fpr_thresholds_on_validation,
                            frozen_operating_point_metrics)

    cfg.USE_LSTM = False
    cache = os.path.join("data", "cache", "pdisjoint")
    os.makedirs(cache, exist_ok=True)
    out_path = args.output or "outputs/partition_disjoint_eval.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # ── 1. data (cached) ────────────────────────────────────────────────
    dcache = os.path.join(cache, "data.npz")
    if os.path.exists(dcache):
        z = np.load(dcache)
        X_train, y_train = z["X_train"], z["y_train"]
        X_val, y_val = z["X_val"], z["y_val"]
        X_test, y_test = z["X_test"], z["y_test"]
        print("[PDisjoint] data resumed from cache")
    else:
        from sklearn.model_selection import train_test_split
        print(f"\n[PDisjoint] building fold: train P1-4 -> test "
              f"P{args.test_part}")
        X_pool, y_pool, X_test, y_test, parts = load_fold(args.test_part)
        # validation carve: EXACT preprocess() recipe (stratified, seed 42)
        val_idx = train_test_split(
            np.arange(len(y_pool)), test_size=cfg.VAL_SPLIT,
            random_state=cfg.SEED, stratify=y_pool)[1]
        val_set = set(val_idx.tolist())
        train_idx = np.array([i for i in range(len(y_pool))
                              if i not in val_set])
        X_train, y_train = X_pool[train_idx], y_pool[train_idx]
        X_val, y_val = X_pool[val_idx], y_pool[val_idx]
        np.savez_compressed(dcache, X_train=X_train, y_train=y_train,
                            X_val=X_val, y_val=y_val, X_test=X_test,
                            y_test=y_test)
        print(f"[PDisjoint] pool {len(y_pool):,} -> train {len(y_train):,} "
              f"(pos {y_train.mean():.2%}) | val {len(y_val):,} "
              f"(pos {y_val.mean():.2%}) | test {len(y_test):,} "
              f"(pos {y_test.mean():.2%}); cached")

    # ── 2. shards (identical recipe: Dirichlet alpha=1.0, seed 42) ──────
    shards = partition_data_dirichlet(
        X_train, y_train, alpha=1.0, n_clients=cfg.N_CLIENTS,
        seed=cfg.SEED, min_samples=cfg.MIN_SAMPLES_PER_CLIENT)

    # ── 3. pooled baselines (cached) ────────────────────────────────────
    import torch
    bcache = os.path.join(cache, "baselines.pt")
    if os.path.exists(bcache):
        c_models = torch.load(bcache, weights_only=False)
        print("[PDisjoint] baselines resumed from cache")
    else:
        print("\n[PDisjoint] training pooled baselines on P1-4 ...")
        c_models = train_centralized(X_train, y_train, X_val, y_val,
                                     seed=cfg.SEED)
        _atomic_torch(c_models, bcache)

    # ── 4. federated arms (round-resumable) ─────────────────────────────
    print("\n[PDisjoint] FedAvg (50 rounds, frozen protocol) ...")
    fedavg_model, _ = run_fedavg(
        shards, X_val, y_val, n_rounds=cfg.N_ROUNDS, use_lstm=False,
        eval_batch_size=cfg.EVAL_BATCH_SIZE, seed=cfg.SEED,
        resume_path=os.path.join(cache, "fedavg.pt"))
    print("\n[PDisjoint] FedProx (50 rounds, frozen protocol) ...")
    fedprox_model, _ = run_fedprox(
        shards, X_val, y_val, n_rounds=cfg.N_ROUNDS, mu=cfg.MU,
        use_lstm=False, eval_batch_size=cfg.EVAL_BATCH_SIZE,
        seed=cfg.SEED, resume_path=os.path.join(cache, "fedprox.pt"))

    # ── 5. evaluation: identical frozen protocol ────────────────────────
    print("\n[PDisjoint] evaluation (prior_shift, val-frozen thresholds)")
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
        results[name] = {"threshold": float(t), "val_fbeta": float(fb),
                         "test": metrics, "frozen_operating_points": ops,
                         "shipped_in_partition": SHIPPED.get(name)}
        probs_dump[name] = np.asarray(p_test_c, dtype=np.float32)
        print(f"  {name:<20s} ROC-AUC {metrics['roc_auc']:.3f} "
              f"(shipped {SHIPPED.get(name, {}).get('roc_auc')}) "
              f"PR-AUC {metrics['pr_auc']:.3f} "
              f"(shipped {SHIPPED.get(name, {}).get('pr_auc')})")

    report = {
        "purpose": ("leakage-free fold: train P1-4 -> test P5 "
                    "(SWAN-SF temporally-preceding protocol); raw-metadata "
                    "audit found the shipped all-5/all-5 pairing shares "
                    "instances between train and test"),
        "test_partition": args.test_part,
        "protocol": {"rounds": cfg.N_ROUNDS, "mu": cfg.MU, "alpha": 1.0,
                     "seed": cfg.SEED, "clients": cfg.N_CLIENTS,
                     "val_split": cfg.VAL_SPLIT,
                     "flatten": cfg.FLATTEN_METHOD,
                     "calibration": cfg.CALIBRATION_METHOD},
        "sizes": {"train": int(len(y_train)), "val": int(len(y_val)),
                  "test": int(len(y_test)),
                  "train_pos": float(y_train.mean()),
                  "val_pos": float(y_val.mean()),
                  "test_pos": float(y_test.mean())},
        "results": results,
        "elapsed_s": time.time() - t0,
    }
    _atomic_json(report, out_path)
    np.savez_compressed(os.path.join(cache, "test_probs.npz"),
                        y_test=y_test.astype(np.int8), **probs_dump)
    print(f"\n[PDisjoint] report -> {out_path} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
