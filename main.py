"""
main.py  (v3.0 — improvements branch)
────────────────────────────────────────
SF-9: Federated Space Weather Monitoring — full pipeline orchestrator.

v3.0 PROTOCOL (fixes audit findings B1, B3, B4, B5, B10):

    TRAIN  ──> FL client training / centralized training
    VAL    ──> monitoring, calibration fitting, F-beta threshold search,
                checkpoint selection (everything that "selects")
    TEST   ──> touched EXACTLY ONCE, with the frozen pipeline

    1.  Load 2D data (and 3D data if LSTM mode)
    2.  Immutable train/val/test split (+ runtime leakage audit)
    3.  Partition TRAIN into disjoint Dirichlet client shards
    4.  Centralized baselines: climatology, LR, centralized MLP, XGBoost
    5.  FedAvg + FedProx (+SCAFFOLD if enabled), monitored on VAL
    6.  Calibration selected on VAL (Brier score)
    7.  F-beta threshold searched on VAL — never on test
    8.  ONE final evaluation on TEST at the frozen threshold;
        ALL metrics recomputed there (stale-accuracy bug B4 fixed)
    9.  results.json + run_manifest.json written; figures regenerated
        from machine-readable results (B18)
    10. Client-level evaluation + communication-cost instrumentation

Usage:
  python main.py
  python main.py --rounds 30 --clients 4
  python main.py --no-lstm            # MLP mode
  python main.py --calibration platt  # override calibration method
"""

import argparse
import hashlib
import json
import os
import shutil
import time
import numpy as np
import sys

import config as cfg
from data_preparation import load_or_generate_data, preprocess, \
    load_and_scale_3d_data
from partition_clients import partition_data_dirichlet
from centralized_baseline import (train_centralized, evaluate_centralized,
                                  model_probs, compute_shap,
                                  training_budget_report)
from federated_learning import (run_fedavg, run_fedprox, run_scaffold,
                                get_model_probs)
from evaluation import (make_calibrator, select_calibration,
                        find_optimal_threshold_fbeta, compute_all_metrics,
                        select_fpr_thresholds_on_validation,
                        frozen_operating_point_metrics)
from leakage_audit.audit_leakage import run_audit, _print_report
from visualize_results import (plot_confusion_matrices, plot_roc_curves,
                               plot_fl_convergence, plot_shap_importance,
                               plot_comparison_table, print_results_table)

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8',
                                  errors='replace')


def parse_args():
    p = argparse.ArgumentParser(description="SF-9 Federated Solar Flare Prediction")
    p.add_argument("--rounds", type=int, default=cfg.N_ROUNDS)
    p.add_argument("--clients", type=int, default=cfg.N_CLIENTS)
    p.add_argument("--mu", type=float, default=cfg.MU)
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--alpha", type=float, default=cfg.DIRICHLET_ALPHA)
    p.add_argument("--calibration", default=cfg.CALIBRATION_METHOD,
                   choices=["none", "prior_shift", "platt", "isotonic",
                            "temperature"])
    p.add_argument("--dirichlet", action="store_true")
    p.add_argument("--no-lstm", action="store_true")
    p.add_argument("--no-scaffold", action="store_true")
    p.add_argument("--no-mixup", action="store_true")
    p.add_argument("--eval-batch-size", type=int, default=cfg.EVAL_BATCH_SIZE)
    p.add_argument("--allow-in-partition", action="store_true",
                   help="v4.0: proceed despite the region-level "
                        "train/test overlap the leakage audit now "
                        "detects on the shipped in-partition protocol "
                        "(use only to reproduce the published "
                        "in-partition numbers deliberately)")
    p.add_argument("--fresh", action="store_true",
                   help="ignore and wipe the phase cache (full recompute)")
    return p.parse_args()


def _save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"[Results] Written -> {path}")


def _atomic_save(path, writer):
    """Write via temp file + rename so a killed process can never leave a
    half-written phase cache behind."""
    tmp = path + ".tmp"
    writer(tmp)
    os.replace(tmp, path)


def _load_region_ids():
    """AR/region ids aligned to the pooled dataframe order (train rows
    then test rows, partitions 1-5 within each), read from the committed
    provenance slim tables. Returns None if unavailable (synthetic-data
    mode)."""
    try:
        import pandas as pd
        tr = pd.read_csv("provenance/train_meta_slim.csv.gz")
        te = pd.read_csv("provenance/test_meta_slim.csv.gz")
        tr = tr.sort_values("pool_row")
        te = te.sort_values("pool_row")
        ids = np.concatenate([tr["ar"].values.astype(str),
                              te["ar"].values.astype(str)])
        print(f"[Audit] region ids loaded from provenance slim tables "
              f"({len(ids):,} windows)")
        return ids
    except Exception as e:
        print(f"[Audit] region ids unavailable ({e}) — region check "
              f"will be skipped")
        return None


def main():
    args = parse_args()

    if args.no_lstm:
        cfg.USE_LSTM = False
    if args.no_scaffold:
        cfg.USE_SCAFFOLD = False
    if args.no_mixup:
        cfg.USE_MIXUP = False

    run_id = cfg.RUN_ID or time.strftime("run_%Y%m%d_%H%M%S")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    t0 = time.time()

    # ── dataset verification gate (v4.0, review M5) ─────────────────────
    # Every pipeline run begins by byte-verifying all 20 partition files
    # against the frozen manifest. The "byte-verified" claim in the paper
    # is this call, not prose. Failure aborts the run.
    from data_manifest.verify_manifest import verify as verify_dataset
    ds_verify = verify_dataset()
    print(f"[Verify] dataset manifest: "
          f"{ds_verify['n_verified']}/{ds_verify['n_files']} files "
          f"byte-identical (sha256 {ds_verify['manifest_sha256'][:16]}…)")
    if not ds_verify["ok"]:
        print("[Verify] FAIL — aborting before any training. "
              "Re-download the cleaned export or regenerate the manifest.")
        sys.exit(1)

    use_lstm = cfg.USE_LSTM
    model_name = 'LSTM' if use_lstm else 'MLP'
    # ── resumable phase cache (keyed by experiment identity) ──────────────
    # Phases data/baselines/fedavg/fedprox can be resumed across process
    # restarts; the cache lives under data/cache/ (git-ignored) and is
    # invalidated automatically whenever any experiment-defining parameter
    # or the dataset manifest hash changes (anti-stale-result, B18).
    CACHE_DIR = os.path.join("data", "cache")

    def _phase_key():
        # v4.0 (review m2): fold in the per-file sha256 of ALL 20 manifest
        # entries (train+test), not just the first train file — a change
        # to any partition now invalidates the phase cache.
        ds_sha = "none"
        mf = "data_manifest/manifest.json"
        if os.path.exists(mf):
            m = json.load(open(mf))
            parts = []
            for split in ("train", "test"):
                for e in m.get("files", {}).get(split, []):
                    parts.append(f"{split}/{e['file']}:"
                                 f"{e.get('sha256', 'none')}")
            if parts:
                ds_sha = hashlib.sha256(
                    "|".join(parts).encode()).hexdigest()[:16]
        kb = {"v": cfg.VERSION, "seed": args.seed, "clients": args.clients,
              "rounds": args.rounds, "mu": args.mu, "alpha": args.alpha,
              "model": model_name, "val_split": cfg.VAL_SPLIT,
              "agg": cfg.AGGREGATION_STRATEGY, "smote": cfg.USE_SMOTE,
              "focal": cfg.USE_FED_FOCAL, "ds": ds_sha}
        return hashlib.sha256(
            json.dumps(kb, sort_keys=True).encode()).hexdigest()[:16]

    key = _phase_key()
    resume = (not args.fresh) and (not use_lstm)
    key_file = os.path.join(CACHE_DIR, "phase_key.json")
    if resume and os.path.exists(key_file):
        old_key = json.load(open(key_file)).get("key")
        if old_key != key:
            print("[Cache] experiment key changed — invalidating phase cache")
            resume = False
    if not resume:
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(key_file, "w") as f:
        json.dump({"key": key}, f)
    resumed = []

    print("\n" + "=" * 70)
    print(f"  SF-9 v{cfg.VERSION} | run: {run_id}")
    print(f"  Clients: {args.clients} | Rounds: {args.rounds} | "
          f"mu: {args.mu} | seed: {args.seed}")
    print(f"  Model: {model_name} | agg: {cfg.AGGREGATION_STRATEGY} | "
          f"calib: {args.calibration} | SMOTE: {cfg.USE_SMOTE}")
    print("  Protocol: threshold+calibration on VALIDATION; "
          "test touched ONCE")
    print("=" * 70 + "\n")

    # ═══════════════════════════════════════════════════════════════════════
    # 1. LOAD DATA + IMMUTABLE SPLITS  (resumable)
    # ═══════════════════════════════════════════════════════════════════════
    data_cache = os.path.join(CACHE_DIR, "phase_data.npz")
    if resume and os.path.exists(data_cache):
        z = np.load(data_cache, allow_pickle=False)
        X_train_2d, y_train = z["X_train"], z["y_train"]
        X_val_2d, y_val = z["X_val"], z["y_val"]
        X_test_2d, y_test = z["X_test"], z["y_test"]
        feature_names = [str(s) for s in z["feature_names"]]
        train_idx, val_idx, test_idx = z["train_idx"], z["val_idx"], z["test_idx"]
        assign_cache = os.path.join(CACHE_DIR, "phase_assignment.npz")
        assignment = np.load(assign_cache)["assignment"] \
            if os.path.exists(assign_cache) else None
        print(f"[Cache] splits resumed: train={len(y_train):,} "
              f"val={len(y_val):,} test={len(y_test):,}")
        resumed.append("data")
    else:
        df = load_or_generate_data()
        splits = preprocess(df, val_fraction=cfg.VAL_SPLIT)

        X_train_2d, y_train = splits.X_train, splits.y_train
        X_val_2d, y_val = splits.X_val, splits.y_val
        X_test_2d, y_test = splits.X_test, splits.y_test
        feature_names = splits.features
        train_idx, val_idx, test_idx = (splits.train_idx, splits.val_idx,
                                        splits.test_idx)
        assignment = None
        np.savez(open(data_cache + ".tmp", "wb"),
                 X_train=np.asarray(X_train_2d, dtype=np.float32),
                 y_train=np.asarray(y_train, dtype=np.float32),
                 X_val=np.asarray(X_val_2d, dtype=np.float32),
                 y_val=np.asarray(y_val, dtype=np.float32),
                 X_test=np.asarray(X_test_2d, dtype=np.float32),
                 y_test=np.asarray(y_test, dtype=np.float32),
                 feature_names=np.array(feature_names),
                 train_idx=np.asarray(train_idx), val_idx=np.asarray(val_idx),
                 test_idx=np.asarray(test_idx))
        os.replace(data_cache + ".tmp", data_cache)
        print(f"[Cache] splits cached -> {data_cache}")

    # ── runtime leakage audit (Gate 1 pass criterion) ──
    # v4.0 (review M11): region ids from the committed provenance tables
    # are wired in, so the gate now detects the shipped in-partition
    # protocol's instance-level train/test overlap (same ARs on both
    # sides) instead of silently skipping the region check. A guard that
    # cannot fail on the failure mode it guards is decoration.
    region_ids = _load_region_ids()
    audit = run_audit(train_idx=train_idx, val_idx=val_idx,
                      test_idx=test_idx, region_ids=region_ids)
    _print_report(audit)
    if not audit.get("overall_pass"):
        if region_ids is not None and not args.allow_in_partition:
            raise RuntimeError(
                "LEAKAGE AUDIT FAILED — region ids shared between train "
                "and test (the shipped in-partition protocol: the "
                "cleaned export's test pkls contain the same raw "
                "instances the train pkls were rebalanced from; see "
                "outputs/dataset_structure_audit.json). Refusing to "
                "train. Pass --allow-in-partition to reproduce the "
                "shipped-in-partition numbers deliberately, with the "
                "leakage quantified in the provenance audit.")
        print("[Audit] region/window checks FAILED but "
              "--allow-in-partition set: this run deliberately "
              "reproduces the shipped in-partition protocol whose "
              "instance-level leakage the provenance audit quantifies "
              "(6,234/6,234 flaring test windows in training).")

    # ── 3D data for LSTM (same contract: val carved from train) ──
    X_train_fl, X_val_fl, X_test_fl = X_train_2d, X_val_2d, X_test_2d
    y_train_fl, y_val_fl, y_test_fl = y_train, y_val, y_test
    if use_lstm:
        print("=" * 70)
        print("  LOADING 3D DATA FOR LSTM MODELS")
        print("=" * 70 + "\n")
        try:
            (X_train_3d, y_train_3d, X_val_3d, y_val_3d,
             X_test_3d, y_test_3d, _) = load_and_scale_3d_data()
            assert len(y_train_3d) == len(y_train) and \
                len(y_val_3d) == len(y_val), \
                "3D/2D sample mismatch — check data sources"
            X_train_fl, X_val_fl, X_test_fl = X_train_3d, X_val_3d, X_test_3d
            y_train_fl, y_val_fl, y_test_fl = y_train_3d, y_val_3d, y_test_3d
            print(f"[Data Path] LSTM mode: 3D {X_train_3d.shape[1:]} for FL | "
                  f"2D ({X_train_2d.shape[1]} feats) for baselines\n")
        except Exception as e:
            print(f"\n[ERROR] 3D data unavailable ({e})\n"
                  f"[FALLBACK] Switching to MLP mode.\n")
            cfg.USE_LSTM = use_lstm = False
            model_name = 'MLP'

    # ═══════════════════════════════════════════════════════════════════════
    # 2. DISJOINT DIRICHLET PARTITION (TRAIN only)
    # ═══════════════════════════════════════════════════════════════════════
    if assignment is None:
        print("[Partition] Disjoint Dirichlet non-IID partitioning "
              f"(alpha={args.alpha}, seed={args.seed}) ...\n")
        shards, assignment = partition_data_dirichlet(
            X_train_fl, y_train_fl, alpha=args.alpha,
            n_clients=args.clients, seed=args.seed,
            min_samples=cfg.MIN_SAMPLES_PER_CLIENT, return_indices=True)
        np.savez(open(os.path.join(CACHE_DIR, "phase_assignment.npz.tmp"), 'wb'),
                 assignment=np.asarray(assignment))
        os.replace(os.path.join(CACHE_DIR, "phase_assignment.npz.tmp"),
                   os.path.join(CACHE_DIR, "phase_assignment.npz"))
    else:
        # deterministic shard rebuild from the cached client assignment
        # (replicates partition_clients.py's per-client seeded shuffle)
        X_arr, y_arr = np.asarray(X_train_fl), np.asarray(y_train_fl)
        shards = []
        for k in range(args.clients):
            idx = np.where(assignment == k)[0]
            if len(idx) == 0:
                shards.append((X_arr[:0], y_arr[:0]))
                continue
            rng = np.random.RandomState(args.seed * 1000 + k)
            idx = idx[rng.permutation(len(idx))]
            shards.append((X_arr[idx], y_arr[idx]))
        print(f"[Cache] client shards rebuilt from cached assignment "
              f"({[int((assignment == k).sum()) for k in range(args.clients)]})")

    # partition audit: verify disjointness against TRAIN indices
    part_audit = run_audit(
        shards=[np.where(assignment == k)[0] for k in range(args.clients)],
        n_train=len(y_train_fl))
    if not part_audit.get("shard_disjointness", {}).get("pass") or \
       not part_audit.get("shard_coverage", {}).get("pass"):
        raise RuntimeError("PARTITION AUDIT FAILED (overlap/coverage).")

    # ═══════════════════════════════════════════════════════════════════════
    # 3. CENTRALIZED BASELINES (pooled 2D data)
    # ═══════════════════════════════════════════════════════════════════════
    baselines_cache = os.path.join(CACHE_DIR, "phase_baselines.pt")
    if resume and os.path.exists(baselines_cache):
        import torch
        c_models = torch.load(baselines_cache, weights_only=False)
        print("\n[Centralised] baselines resumed from cache")
        resumed.append("baselines")
    else:
        print("\n[Centralised] Training pooled baselines ...")
        c_models = train_centralized(X_train_2d, y_train,
                                     X_val_2d, y_val, seed=args.seed)
        import torch
        _atomic_save(baselines_cache,
                     lambda p: torch.save(c_models, p))
        print("[Cache] baselines cached")

    # ═══════════════════════════════════════════════════════════════════════
    # 4. FEDERATED TRAINING (monitoring on VALIDATION)
    # ═══════════════════════════════════════════════════════════════════════
    import torch
    fedavg_cache = os.path.join(CACHE_DIR, "phase_fedavg.pt")
    fedavg_model, fedavg_history = None, None
    if resume and os.path.exists(fedavg_cache):
        try:
            d = torch.load(fedavg_cache, weights_only=False)
            if d.get("done"):
                fedavg_model, fedavg_history = d["model"], d["history"]
                print("\n[FedAvg] complete run resumed from cache")
                resumed.append("fedavg")
        except Exception:
            pass
    if fedavg_model is None:
        print()
        fedavg_model, fedavg_history = run_fedavg(
            shards, X_val_fl, y_val_fl, n_rounds=args.rounds,
            use_lstm=use_lstm, eval_batch_size=args.eval_batch_size,
            seed=args.seed, resume_path=fedavg_cache)
        print("[Cache] FedAvg cached (round-resumable)")

    fedprox_cache = os.path.join(CACHE_DIR, "phase_fedprox.pt")
    fedprox_model, fedprox_history = None, None
    if resume and os.path.exists(fedprox_cache):
        try:
            d = torch.load(fedprox_cache, weights_only=False)
            if d.get("done"):
                fedprox_model, fedprox_history = d["model"], d["history"]
                print("\n[FedProx] complete run resumed from cache")
                resumed.append("fedprox")
        except Exception:
            pass
    if fedprox_model is None:
        print()
        fedprox_model, fedprox_history = run_fedprox(
            shards, X_val_fl, y_val_fl, n_rounds=args.rounds, mu=args.mu,
            use_lstm=use_lstm, eval_batch_size=args.eval_batch_size,
            seed=args.seed, resume_path=fedprox_cache)
        print("[Cache] FedProx cached (round-resumable)")

    scaffold_model, scaffold_history = None, []
    if cfg.USE_SCAFFOLD:
        print()
        scaffold_cache = os.path.join(CACHE_DIR, "phase_scaffold.pt")
        if resume and os.path.exists(scaffold_cache):
            try:
                d = torch.load(scaffold_cache, weights_only=False)
                if d.get("done"):
                    scaffold_model, scaffold_history = d["model"], d["history"]
                    print("[SCAFFOLD] complete run resumed from cache")
                    resumed.append("scaffold")
            except Exception:
                pass
        if scaffold_model is None:
            scaffold_model, scaffold_history = run_scaffold(
                shards, X_val_fl, y_val_fl, n_rounds=args.rounds,
                use_lstm=use_lstm, eval_batch_size=args.eval_batch_size,
                warm_start_model=fedavg_model, seed=args.seed,
                resume_path=scaffold_cache)

    # ═══════════════════════════════════════════════════════════════════════
    # 5. CALIBRATION + THRESHOLD — selected on VALIDATION ONLY
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  EVALUATION PROTOCOL (validation -> freeze -> test once)")
    print("=" * 70)

    import torch
    device = next(fedavg_model.parameters()).device

    def fl_probs(model, X):
        return get_model_probs(model, X, device,
                               batch_size=args.eval_batch_size)

    # gather VAL probabilities for every model
    val_probs = {
        "climatology": model_probs(c_models["climatology"], X_val_2d),
        "logistic_regression": model_probs(c_models["logistic_regression"], X_val_2d),
        "xgboost": model_probs(c_models["xgboost"], X_val_2d),
        f"fedavg_{model_name.lower()}": fl_probs(fedavg_model, X_val_fl),
        f"fedprox_{model_name.lower()}": fl_probs(fedprox_model, X_val_fl),
    }
    if "centralized_mlp" in c_models:
        val_probs["centralized_mlp"] = model_probs(
            c_models["centralized_mlp"], X_val_2d)
    if scaffold_model is not None:
        val_probs[f"scaffold_{model_name.lower()}"] = \
            fl_probs(scaffold_model, X_val_fl)

    # ── 5a. calibration selection on VAL (Brier) ──
    method = args.calibration
    if method == "auto":
        method, cal_table = select_calibration(y_val, val_probs["xgboost"])
        print(f"[Calibration] auto-selected: {method} (val Brier per method: "
              f"{cal_table})")

    calibrators = {}
    for name, p in val_probs.items():
        cal = make_calibrator(method).fit(y_val, p)
        if method == "prior_shift":
            cal.set_prevalences(train_rate=float(y_train.mean()),
                                test_rate=float(y_val.mean()))
        calibrators[name] = cal

    # ── 5b. threshold search on (calibrated) VAL probabilities ──
    thresholds = {}
    for name, p in val_probs.items():
        p_cal = calibrators[name].transform(p)
        t, fb = find_optimal_threshold_fbeta(
            y_val, p_cal, beta=cfg.FBETA_BETA, grid=cfg.THRESHOLD_GRID)
        thresholds[name] = t
        print(f"  [VAL] {name:<28} threshold={t:.3f}  F{cfg.FBETA_BETA:.0f}={fb:.3f}")

    # ── 5c. deployment thresholds at fixed FPR budgets (review R4) ──
    # FPR depends only on the NEGATIVE score distribution, so the
    # (1-fpr) quantile of validation negatives is the frozen deployment
    # threshold; the test set plays no role in the selection.
    fpr_frozen = {}
    for name, p in val_probs.items():
        p_cal = calibrators[name].transform(p)
        fpr_frozen[name] = select_fpr_thresholds_on_validation(
            y_val, p_cal, cfg.OPERATING_FPR_TARGETS)
    print(f"  [VAL] FPR-budget deployment thresholds frozen for "
          f"{len(fpr_frozen)} models")

    print(f"\n  Frozen: calibration={method} | thresholds={thresholds}")
    print("  -> applying ONCE to the held-out TEST set ...\n")

    # ═══════════════════════════════════════════════════════════════════════
    # 6. FINAL TEST EVALUATION (single pass, frozen pipeline)
    # ═══════════════════════════════════════════════════════════════════════
    test_probs = {
        "climatology": model_probs(c_models["climatology"], X_test_2d),
        "logistic_regression": model_probs(c_models["logistic_regression"], X_test_2d),
        "xgboost": model_probs(c_models["xgboost"], X_test_2d),
        f"fedavg_{model_name.lower()}": fl_probs(fedavg_model, X_test_fl),
        f"fedprox_{model_name.lower()}": fl_probs(fedprox_model, X_test_fl),
    }
    if "centralized_mlp" in c_models:
        test_probs["centralized_mlp"] = model_probs(
            c_models["centralized_mlp"], X_test_2d)
    if scaffold_model is not None:
        test_probs[f"scaffold_{model_name.lower()}"] = \
            fl_probs(scaffold_model, X_test_fl)

    all_results = {}
    for name, p in test_probs.items():
        p_cal = calibrators[name].transform(p)
        res = compute_all_metrics(y_test_fl if "fed" in name or "scaffold" in name
                                  else y_test, p_cal, thresholds[name],
                                  beta=cfg.FBETA_BETA)
        res["calibration"] = method
        # R4: realised deployment behaviour at the validation-frozen
        # FPR thresholds (single application, distinct from the
        # threshold-free recall@FPR discrimination statistics)
        res["frozen_operating_points"] = frozen_operating_point_metrics(
            y_test_fl if "fed" in name or "scaffold" in name else y_test,
            p_cal, fpr_frozen[name])
        res["probs"] = p_cal
        res["preds"] = res.pop("preds", None)
        all_results[name] = res

    # display names
    display = {
        "climatology": "Climatology",
        "logistic_regression": "Logistic Regression",
        "centralized_mlp": "Centralized MLP",
        "xgboost": "XGBoost (Centralized ref.)",
        f"fedavg_{model_name.lower()}": f"FedAvg {model_name}",
        f"fedprox_{model_name.lower()}": f"FedProx {model_name}",
        f"scaffold_{model_name.lower()}": f"SCAFFOLD {model_name}",
    }
    pretty_results = {display.get(k, k): v for k, v in all_results.items()}
    print_results_table(pretty_results)

    # ═══════════════════════════════════════════════════════════════════════
    # 7. CLIENT-LEVEL EVALUATION (Stage 10)
    # ═══════════════════════════════════════════════════════════════════════
    try:
        from evaluate_clients import evaluate_client_level
        client_results = evaluate_client_level(
            shards, (fedavg_model, fedprox_model),
            model_name,
            batch_size=args.eval_batch_size)
    except Exception as e:
        print(f"[Client Eval] Skipped: {e}")
        client_results = None

    # ═══════════════════════════════════════════════════════════════════════
    # 8. COMMUNICATION COST (Stage 15)
    # ═══════════════════════════════════════════════════════════════════════
    try:
        from communication_cost import measure_communication
        comm = measure_communication(fedavg_model, args.clients,
                                     n_rounds=args.rounds)
    except Exception as e:
        print(f"[Comm Cost] Skipped: {e}")
        comm = None

    # ═══════════════════════════════════════════════════════════════════════
    # 9. FIGURES (from machine-readable results)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[Figures] Generating plots ...")
    plot_confusion_matrices(pretty_results, y_test_fl)
    plot_roc_curves(pretty_results, y_test_fl)
    plot_fl_convergence(fedavg_history, fedprox_history, scaffold_history)
    plot_comparison_table(pretty_results)

    try:
        _, mean_shap = compute_shap(c_models["xgboost"], X_test_2d,
                                    feature_names)
        if mean_shap is not None:
            plot_shap_importance(mean_shap, feature_names)
    except Exception as e:
        print(f"[SHAP] Skipped: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # 10. MACHINE-READABLE RESULTS (B18)
    # ═══════════════════════════════════════════════════════════════════════
    serialisable = {}
    for name, res in all_results.items():
        serialisable[name] = {k: v for k, v in res.items()
                              if k not in ("probs", "preds")}

    results = {
        "run_id": run_id,
        "version": cfg.VERSION,
        "protocol": {
            "split": {"train": len(y_train), "val": len(y_val),
                      "test": len(y_test)},
            "calibration": method,
            "thresholds": thresholds,
            "threshold_selected_on": "validation",
            "fpr_thresholds_selected_on": "validation (negative quantile, R4)",
            "test_touched_once": True,
        },
        "test_metrics": serialisable,
        "training_budget": training_budget_report(
            len(y_train), n_val=len(y_val), central_epochs=30,
            central_batch=256, central_lr=cfg.LR,
            fl_rounds=args.rounds, fl_local_epochs=cfg.LOCAL_EPOCHS,
            fl_batch=cfg.BATCH_SIZE, fl_lr=cfg.LR),
        "fl_convergence_val": {
            "fedavg": fedavg_history,
            "fedprox": fedprox_history,
            "scaffold": scaffold_history,
        },
        "client_level": client_results,
        "communication": comm,
        "leakage_audit": {k: v for k, v in audit.items()
                          if isinstance(v, (str, int, float, bool))},
    }
    _save_json(results, cfg.RESULTS_JSON)

    import torch  # for the run manifest environment record (v4.0, m7)

    manifest = {
        "run_id": run_id, "version": cfg.VERSION,
        "seed": args.seed, "clients": args.clients, "rounds": args.rounds,
        "mu": args.mu, "alpha": args.alpha, "model": model_name,
        "resumed_phases": resumed,
        "aggregation": cfg.AGGREGATION_STRATEGY,
        "calibration": method, "use_smote": cfg.USE_SMOTE,
        "use_fed_focal": cfg.USE_FED_FOCAL,
        "dirichlet_alpha": args.alpha,
        "feature_names": feature_names,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "dataset_manifest_sha256": ds_verify["manifest_sha256"],
        "dataset_files_verified": (
            f"{ds_verify['n_verified']}/{ds_verify['n_files']}"),
        "elapsed_s": round(time.time() - t0, 1),
    }
    _save_json(manifest, cfg.RUN_MANIFEST)

    # ═══════════════════════════════════════════════════════════════════════
    # 11. SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n[Done] {run_id} finished in {elapsed:.1f}s")
    print(f"[Done] results -> {cfg.RESULTS_JSON}")
    if "centralized_mlp" in all_results and f"fedprox_{model_name.lower()}" in all_results:
        cost = all_results["centralized_mlp"]["pr_auc"] - \
            all_results[f"fedprox_{model_name.lower()}"]["pr_auc"]
        print(f"\n  Federation cost (PR-AUC): centralized MLP "
              f"{all_results['centralized_mlp']['pr_auc']:.3f} vs FedProx "
              f"{all_results[f'fedprox_{model_name.lower()}']['pr_auc']:.3f} "
              f"-> delta {cost:+.3f}")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
