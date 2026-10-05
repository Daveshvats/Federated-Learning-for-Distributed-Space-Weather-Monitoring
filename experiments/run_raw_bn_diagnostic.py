#!/usr/bin/env python3
"""
experiments/run_raw_bn_diagnostic.py  (v4.8 / owner-side content review,
point 1 — the raw-substrate BN-transport diagnostic; Dossier R-FS9-R8
era, post-closure errata programme E1)
────────────────────────────────────────────────────────
Question: the raw-substrate federated MLP arms (published test ROC-AUC
0.875 / 0.930 / 0.768 for FedAvg / FedProx / SCAFFOLD) were evaluated
through the SAME untransported-BatchNorm evaluation path (init running
buffers, model.eval()) that the in-partition diagnostic
(outputs/bn_diagnostic.json, Section 6.9 of the paper) showed
manufactures apparent federated collapse.  Is the raw-substrate MLP
degradation therefore BN-evaluation-conditional, or does it survive
correct normalisation statistics?

Method: NO retraining.  The raw-substrate runner
(experiments/run_raw_substrate.py) is round-resumable and leaves its
three checkpoint files in data/cache/rawsubstrate/ (fedavg.pt,
fedprox.pt, scaffold.pt; done=True, carrying the returned model = the
best-VALIDATION-checkpoint weights that produced the published
numbers).  This runner rebuilds the seed-42 substrate and the seed-42
Dirichlet shards from the same cache, loads those checkpoints, and
re-evaluates the IDENTICAL weights under three BN treatments:

  A  init buffers (status quo) — the published evaluation path.
     ARM A IS A REPRODUCTION GATE: its ROC/PR-AUC must reproduce the
     committed outputs/raw_substrate_eval.json within --tolerance
     (default 1e-4), else the runner exits non-zero — the checkpoint
     or substrate is not what produced the published numbers, and a
     silent mismatch would poison the counterfactual framing.
  B  recalibrated buffers: exact per-client BN input statistics pooled
     by the law of total variance (computed on the TRAINING shards
     only — no validation/test leakage).  An optimistic oracle for
     what buffer transport could recover.
  C  eval-time batch statistics (BN in train mode, dropout in eval).
     An adaptive-normalisation diagnostic of information content, not
     a deployable protocol.

Arm D (the element-wise weighted average of the running statistics
local training actually produced — what a buffer-transporting
implementation would evaluate) is NOT reconstructible from the
checkpoints: only parameters were ever aggregated and persisted, the
buffers each round's local trainings produced were discarded at
aggregation.  Arm D for the raw substrate requires the faithful
re-run with per-round capture (the in-partition diagnostic's phase-C
method) and stays queued owner-side compute; arms B and C bracket it.

Evaluation mirrors run_raw_substrate.py's section 5 exactly, per
treatment: prior-shift calibration fit on that treatment's VALIDATION
probabilities, val-frozen F-beta threshold, frozen-FPR operating
points, single-pass test evaluation.  An event-level layer (per
treatment) runs when the raw audit metadata is present under
--raw-dir (match_test_p5.csv, test_meta_pooled.csv, the integrated
GOES csv — the same inputs run_event_level_raw.py consumes);
otherwise that layer is skipped with a declared note.

Output: outputs/raw_bn_diagnostic.json.  CPU-capable, minutes (pure
evaluation; no training).  Deterministic given the frozen seed-42
protocol and the cached substrate.

Usage:
    python experiments/run_raw_bn_diagnostic.py [--raw-dir /tmp/swansf_raw]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

try:
    import torch
except ImportError:          # evaluated lazily in main() with a
    torch = None               # loud, actionable message

CLONE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CLONE)
os.chdir(CLONE)

import config as cfg                                        # noqa: E402
from partition_clients import partition_data_dirichlet      # noqa: E402
from evaluation import (                                    # noqa: E402
    make_calibrator, find_optimal_threshold_fbeta,
    compute_all_metrics, select_fpr_thresholds_on_validation,
    frozen_operating_point_metrics)
from experiments.raw_substrate import build, CACHE          # noqa: E402

FPR_TARGETS = (0.005, 0.01, 0.02, 0.05)
ARMS = ("fedavg", "fedprox", "scaffold")          # checkpoint basenames
TREATMENTS = ("A_init_buffers", "B_recalibrated", "C_batch_stats")
PUBLISHED_EVAL = os.path.join("outputs", "raw_substrate_eval.json")
OUT_DEFAULT = os.path.join("outputs", "raw_bn_diagnostic.json")
RAW_DEFAULT = "/tmp/swansf_raw"


def _atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str, allow_nan=False)
    os.replace(tmp, path)


# ─────────────────────────────────────────────────────────────────────────────
# BN helpers (self-contained copies of the run_bn_diagnostic.py contracts;
# that module is an executable script and must not be imported)
# ─────────────────────────────────────────────────────────────────────────────

def bn_modules(model):
    import torch.nn as nn
    return [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]


def bn_buffer_state(model):
    """Readable summary of every BN buffer in the model (phase-A style)."""
    out = []
    for name, buf in model.named_buffers():
        if buf.dtype in (torch.float32, torch.float64):
            out.append({"layer": name,
                        "mean_abs": float(buf.mean().abs()),
                        "var_mean": float(buf.var())
                        if buf.numel() > 1 else float(buf[0])})
        else:
            out.append({"layer": name, "value": int(buf.reshape(-1)[0])})
    return out


def probs_fixed_mode(model, X, device, batch_size=cfg.EVAL_BATCH_SIZE,
                     bn_train=False):
    """Batched probabilities WITHOUT touching module train/eval state
    (get_model_probs calls model.eval(), which would undo a BN-train
    arm).  NaN/clip semantics identical to get_model_probs so that
    arm A reproduces the published evaluation path exactly.

    bn_train=True → BN layers normalise with per-batch statistics
    (adaptive-normalisation oracle); dropout stays in eval.
    """
    import torch.nn as nn
    was_training = model.training
    model.eval()                       # dropout off everywhere
    if bn_train:
        for m in bn_modules(model):
            m.train()                  # BN uses batch stats
    n = len(X)
    all_probs = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            Xb = torch.tensor(np.ascontiguousarray(X[s:e]),
                              dtype=torch.float32).to(device)
            all_probs[s:e] = torch.sigmoid(model(Xb)) \
                .cpu().numpy().flatten()
            del Xb
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if bn_train:
        for m in bn_modules(model):
            m.eval()
    if was_training:
        model.train()
    nan_c = int(np.isnan(all_probs).sum())
    inf_c = int(np.isinf(all_probs).sum())
    if nan_c > 0 or inf_c > 0:
        print(f"  [Warning] {nan_c} NaN + {inf_c} Inf probabilities "
              f"-> replaced with 0.5")
        all_probs = np.nan_to_num(all_probs, nan=0.5, posinf=1.0,
                                  neginf=0.0)
    return np.clip(all_probs, 1e-7, 1 - 1e-7)


def exact_bn_stats(model, X, device, batch_size=cfg.EVAL_BATCH_SIZE):
    """Exact mean/var of every BN layer's INPUT over the full array X,
    via forward pre-hooks.  Returns list of (mean, var_unbiased) CPU
    tensors."""
    bns = bn_modules(model)
    acc = [{"n": 0, "sum": None, "sumsq": None} for _ in bns]
    handles = []

    def make_hook(i):
        def hook(mod, inp):
            x = inp[0].detach()
            acc[i]["n"] += x.shape[0]
            s = x.sum(dim=0)
            ss = (x * x).sum(dim=0)
            acc[i]["sum"] = s if acc[i]["sum"] is None \
                else acc[i]["sum"] + s
            acc[i]["sumsq"] = ss if acc[i]["sumsq"] is None \
                else acc[i]["sumsq"] + ss
        return hook

    for i, m in enumerate(bns):
        handles.append(m.register_forward_pre_hook(make_hook(i)))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            e = min(s + batch_size, len(X))
            Xb = torch.tensor(np.ascontiguousarray(X[s:e]),
                              dtype=torch.float32).to(device)
            model(Xb)
            del Xb
    for h in handles:
        h.remove()
    if was_training:
        model.train()
    out = []
    for a in acc:
        n = a["n"]
        mean = a["sum"] / n
        var_b = a["sumsq"] / n - mean * mean
        var_b = torch.clamp(var_b, min=0.0)
        var_u = var_b * n / max(n - 1, 1)   # PyTorch running_var semantics
        out.append((mean.cpu(), var_u.cpu()))
    return out


def set_bn_stats(model, stats):
    """stats: list of (mean, var) per BN layer, in module order."""
    for m, (mean, var) in zip(bn_modules(model), stats):
        with torch.no_grad():
            m.running_mean.copy_(mean.to(m.running_mean.dtype)
                                  .to(m.running_mean.device))
            m.running_var.copy_(var.to(m.running_var.dtype)
                                .to(m.running_var.device))
            m.num_batches_tracked.fill_(1)   # >0: eval uses running stats


def pool_client_stats(per_client, sizes):
    """Law-of-total-variance pooling of per-client (mean, var) lists."""
    total = float(sum(sizes))
    pooled = []
    n_layers = len(per_client[0])
    for li in range(n_layers):
        w_mean = None
        e_x2 = None
        for (mean, var), sz in zip([c[li] for c in per_client], sizes):
            w = sz / total
            m = mean * w
            e_x2w = (var + mean * mean) * w
            w_mean = m if w_mean is None else w_mean + m
            e_x2 = e_x2w if e_x2 is None else e_x2 + e_x2w
        pooled.append((w_mean, torch.clamp(e_x2 - w_mean * w_mean,
                                           min=0.0)))
    return pooled


# ─────────────────────────────────────────────────────────────────────────────
# evaluation (mirrors run_raw_substrate.py section 5, per treatment)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_treatment(model, device, X_val, y_val, X_test, y_test,
                       y_train_rate, tag):
    """Full frozen-protocol evaluation of one model under one BN
    treatment.  Returns the results block + the calibrated test
    probabilities (for the optional event layer)."""
    p_val = probs_fixed_mode(model, X_val, device,
                             bn_train=(tag == "C_batch_stats"))
    p_test = probs_fixed_mode(model, X_test, device,
                              bn_train=(tag == "C_batch_stats"))
    cal = make_calibrator(cfg.CALIBRATION_METHOD).fit(y_val, p_val)
    if cfg.CALIBRATION_METHOD == "prior_shift":
        cal.set_prevalences(train_rate=float(y_train_rate),
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
    block = {"threshold": float(t), "val_fbeta": float(fb),
             "test": metrics, "frozen_operating_points": ops,
             "treatment": tag}
    return block, np.asarray(p_test_c, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# optional event-level layer (same metadata contract as
# run_event_level_raw.py; runs only when the audit files are present)
# ─────────────────────────────────────────────────────────────────────────────

def event_layer(raw_dir, y_test, probs_by_arm, thresholds):
    import pandas as pd
    from experiments.run_event_level import event_level_metrics
    need = [os.path.join(raw_dir, "match_test_p5.csv"),
            os.path.join(raw_dir, "test_meta_pooled.csv"),
            os.path.join(raw_dir, "integrated_flare_data",
                         "goes_flares_integrated.csv")]
    if not all(os.path.exists(p) for p in need):
        print(f"[event] metadata incomplete under {raw_dir} "
              f"(match/meta/GOES) — event-level layer SKIPPED "
              f"(declared; window-level results stand alone)")
        return None, None
    COOLDOWN = 5   # 5 windows = 1 h at 12-min cadence (frozen)

    match = pd.read_csv(need[0], low_memory=False)
    match = match[match["verified"] == True]                 # noqa: E712
    match = match.drop_duplicates("raw_idx")
    r2p = dict(zip(match["raw_idx"], match["pkl_row"]))
    pooled = pd.read_csv(need[1], low_memory=False)
    pooled = pooled[pooled["partition"] == 5]
    p2meta = {r.pkl_row: r for r in pooled.itertuples()}
    rows = []
    for i in range(len(y_test)):
        p = r2p.get(i)
        if p is None or p not in p2meta:
            continue
        m = p2meta[p]
        rows.append((i, p, m.event_id, m.ts_end_min, str(m.flare_class),
                     int(m.label)))
    meta = pd.DataFrame(rows, columns=["raw_idx", "pkl_row", "event_id",
                                       "ts_end_min", "flare_class",
                                       "label"])
    assert (meta["label"].values ==
            y_test[meta["raw_idx"].values]).all()
    idx = meta["raw_idx"].values
    ev = meta["event_id"].values
    ts = meta["ts_end_min"].values.astype(float)
    yy = y_test[idx]
    true_groups = meta.loc[meta["label"] == 1, "event_id"].unique()

    fl = pd.read_csv(need[2], low_memory=False)
    fl["peak_dt"] = pd.to_datetime(fl["peak_time"])
    fl["cls"] = fl["goes_class"].astype(str)
    peak_map, matched, ambiguous = {}, 0, 0
    for g in true_groups:
        sub = meta[meta["event_id"] == g]
        last_end = sub["ts_end_min"].max()
        cls = str(sub["flare_class"].iloc[0])
        lo = pd.Timestamp(last_end, unit="m")
        cand = fl[(fl["cls"] == cls) & (fl["peak_dt"] >= lo)
                  & (fl["peak_dt"] <= lo + pd.Timedelta(hours=48))]
        if len(cand):
            peak_map[g] = (cand["peak_dt"].iloc[0]
                           - pd.Timestamp(0)).total_seconds() / 60.0
            matched += 1
            ambiguous += int(len(cand) > 1)
    print(f"[event] {len(yy):,} windows matched "
          f"({len(meta)/len(y_test):.2%}) | {meta['event_id'].nunique()} "
          f"groups | {len(true_groups)} true events | peaks matched "
          f"{matched}/{len(true_groups)} (ambiguous {ambiguous})")

    out = {}
    for key, p_full in probs_by_arm.items():
        arm, treat = key
        p = p_full[idx]
        thr = thresholds[key]
        r = event_level_metrics(yy, p, ev, threshold=thr,
                                timestamps=ts, cooldown_windows=COOLDOWN)
        alerts = (p >= thr).astype(int)
        leads = []
        for g, peak_min in peak_map.items():
            gi = np.where(ev == g)[0]
            g_alerts = alerts[gi]
            if g_alerts.sum() > 0:
                first_alert = ts[gi][g_alerts == 1].min()
                leads.append(peak_min - first_alert)
        r["lead_to_flare_peak_minutes"] = {
            "median": float(np.median(leads)) if leads else None,
            "p10": float(np.percentile(leads, 10)) if leads else None,
            "p90": float(np.percentile(leads, 90)) if leads else None,
            "n": len(leads),
            "negative_means_alert_after_peak": True,
        }
        out[f"{arm}:{treat}"] = r
        lt = r["lead_to_flare_peak_minutes"]
        lead_h = (lt["median"] / 60 if lt["median"] is not None
                  else float("nan"))
        print(f"[event] {arm}:{treat:18s} events "
              f"{r['n_detected_events']:>2d}/{r['n_true_events']} "
              f"| FA/day {r['false_alarm_windows_per_day']:.1f} "
              f"| lead {lead_h:.1f} h", flush=True)
    summary = {"cooldown_windows": COOLDOWN,
               "n_windows": int(len(yy)),
               "n_matched_by_audit": int(len(meta)),
               "n_event_groups": int(meta["event_id"].nunique()),
               "n_true_events": int(len(true_groups)),
               "flare_peak_match_rate": matched / max(len(true_groups), 1)}
    return out, summary


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(algo):
    """Load one raw-substrate FL checkpoint and verify its protocol
    fingerprint against the frozen configuration (never silently
    evaluate a wrong-protocol checkpoint)."""
    path = os.path.join(CACHE, f"{algo}.pt")
    if not os.path.exists(path):
        sys.exit(f"[rawbn] checkpoint missing: {path}\n"
                 f"       The raw-substrate checkpoints exist only "
                 f"owner-side (gitignored data/). Run "
                 f"experiments/run_raw_substrate.py --raw-dir "
                 f"<raw benchmark dir> first (round-resumable; the "
                 f"completed checkpoints then carry done=True).")
    st = torch.load(path, weights_only=False, map_location="cpu")
    if not st.get("done") or st.get("model") is None:
        sys.exit(f"[rawbn] {path} is not a completed checkpoint "
                 f"(done={st.get('done')}) — re-run "
                 f"experiments/run_raw_substrate.py to completion "
                 f"first.")
    expected = {
        "algorithm": algo,
        "mu": cfg.MU if algo == "fedprox" else 0.0,
        "seed": cfg.SEED,
        "n_rounds": cfg.N_ROUNDS,
        "agg": cfg.AGGREGATION_STRATEGY,
        "smote": cfg.USE_SMOTE,
        "focal": cfg.USE_FED_FOCAL,
        "loss_variant": getattr(cfg, "LOSS_VARIANT", "fed_focal"),
        "use_lstm": False,
    }
    bad = {k: (st.get(k), v) for k, v in expected.items()
           if st.get(k) != v}
    if bad:
        sys.exit(f"[rawbn] {path} protocol fingerprint mismatch "
                 f"{bad} — this checkpoint did not run the frozen "
                 f"protocol; refusing to evaluate it.")
    m = st["model"]
    m.eval()
    return m, {"best_val_f1": float(st["best_val_f1"]),
               "n_history": len(st.get("history", [])),
               "bn_buffers": bn_buffer_state(m)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=RAW_DEFAULT,
                    help="raw benchmark dir (substrate build cache miss "
                         "and the optional event-level metadata)")
    ap.add_argument("--output", default=OUT_DEFAULT)
    ap.add_argument("--tolerance", type=float, default=1e-4,
                    help="reproduction-gate tolerance on ROC/PR-AUC")
    args = ap.parse_args()

    t0 = time.time()
    if torch is None:
        sys.exit("[rawbn] torch is required (evaluation of torch "
                 "checkpoints); the torch-less battery guards cover "
                 "this runner statically via tests/"
                 "test_raw_bn_diagnostic.py")

    cfg.USE_LSTM = False
    cfg.USE_SCAFFOLD = True

    # ── 1. substrate + shards (identical recipe, from cache) ───────────
    d = build(args.raw_dir, CACHE)
    X_train, y_train = d["X_train"], d["y_train"]
    X_val, y_val = d["X_val"], d["y_val"]
    X_test, y_test = d["X_test"], d["y_test"]
    print(f"[rawbn] substrate: train {len(y_train):,} "
          f"(pos {y_train.mean():.2%}) | val {len(y_val):,} "
          f"(pos {y_val.mean():.2%}) | test {len(y_test):,} "
          f"(pos {y_test.mean():.2%})", flush=True)

    shards = partition_data_dirichlet(
        X_train, y_train, alpha=1.0, n_clients=cfg.N_CLIENTS,
        seed=cfg.SEED, min_samples=cfg.MIN_SAMPLES_PER_CLIENT,
        verbose=False)
    print(f"[rawbn] shards rebuilt (Dirichlet alpha=1.0, seed "
          f"{cfg.SEED}): sizes {[len(s[1]) for s in shards]}",
          flush=True)

    # ── 2. published reference for the reproduction gate ──────────────
    if not os.path.exists(PUBLISHED_EVAL):
        sys.exit(f"[rawbn] published reference missing: "
                 f"{PUBLISHED_EVAL} — the reproduction gate cannot "
                 f"run; restore the committed artefact first.")
    published = json.load(open(PUBLISHED_EVAL))["results"]

    # ── 3. per-arm BN treatments on the identical weights ────────────
    results = {}
    probs_by_arm = {}
    thresholds = {}
    reproduction = {}
    device = torch.device("cpu")
    for algo in ARMS:
        model, meta = load_checkpoint(algo)
        nbt = [b.get("value") for b in meta["bn_buffers"]
               if "value" in b]
        print(f"\n[rawbn] {algo}: best_val_f1 "
              f"{meta['best_val_f1']:.4f}, BN num_batches_tracked="
              f"{nbt}", flush=True)
        arm_results = {"checkpoint": meta}

        # Arm A — status quo (init buffers) on a pristine clone.
        from model import clone_model
        mA = clone_model(model).to(device)
        blk, pA = evaluate_treatment(mA, device, X_val, y_val,
                                     X_test, y_test, y_train.mean(),
                                     "A_init_buffers")
        arm_results["A_init_buffers"] = blk
        probs_by_arm[(algo, "A_init_buffers")] = pA
        thresholds[(algo, "A_init_buffers")] = blk["threshold"]

        # Arm B — recalibrated from client training shards (pooled).
        per_client, sizes = [], []
        for X_c, y_c in shards:
            stats = exact_bn_stats(mA, X_c, device)
            per_client.append(stats)
            sizes.append(len(y_c))
        pooled = pool_client_stats(per_client, sizes)
        mB = clone_model(model).to(device)
        set_bn_stats(mB, pooled)
        blk, pB = evaluate_treatment(mB, device, X_val, y_val,
                                     X_test, y_test, y_train.mean(),
                                     "B_recalibrated")
        arm_results["B_recalibrated"] = blk
        probs_by_arm[(algo, "B_recalibrated")] = pB
        thresholds[(algo, "B_recalibrated")] = blk["threshold"]

        # Arm C — eval-time batch statistics on a pristine clone.
        mC = clone_model(model).to(device)
        blk, pC = evaluate_treatment(mC, device, X_val, y_val,
                                     X_test, y_test, y_train.mean(),
                                     "C_batch_stats")
        arm_results["C_batch_stats"] = blk
        probs_by_arm[(algo, "C_batch_stats")] = pC
        thresholds[(algo, "C_batch_stats")] = blk["threshold"]

        results[f"{algo}_mlp"] = arm_results

        # reproduction gate (arm A vs the published numbers)
        pub = published[f"{algo}_mlp"]["test"]
        got = arm_results["A_init_buffers"]["test"]
        d_roc = abs(float(got["roc_auc"]) - float(pub["roc_auc"]))
        d_pr = abs(float(got["pr_auc"]) - float(pub["pr_auc"]))
        verdict = ("exact" if max(d_roc, d_pr) < 1e-9 else
                   "match" if max(d_roc, d_pr) <= args.tolerance
                   else "MISMATCH")
        reproduction[f"{algo}_mlp"] = {
            "published_roc_auc": float(pub["roc_auc"]),
            "reproduced_roc_auc": float(got["roc_auc"]),
            "delta_roc_auc": d_roc,
            "published_pr_auc": float(pub["pr_auc"]),
            "reproduced_pr_auc": float(got["pr_auc"]),
            "delta_pr_auc": d_pr,
            "tolerance": args.tolerance,
            "verdict": verdict,
        }
        print(f"[rawbn] {algo}: A roc {got['roc_auc']:.5f} "
              f"(published {pub['roc_auc']:.5f}, d={d_roc:.2e}) | "
              f"B {arm_results['B_recalibrated']['test']['roc_auc']:.5f} "
              f"| C {arm_results['C_batch_stats']['test']['roc_auc']:.5f} "
              f"| gate: {verdict}", flush=True)

    gate_failed = any(v["verdict"] == "MISMATCH"
                      for v in reproduction.values())

    # ── 4. optional event-level layer ─────────────────────────────────
    event_results, event_summary = event_layer(
        args.raw_dir, y_test, probs_by_arm, thresholds)

    # ── 5. report ──────────────────────────────────────────────────────
    report = {
        "runner": "experiments/run_raw_bn_diagnostic.py",
        "purpose": ("v4.8 / E1: BN-treatment re-evaluation of the "
                    "raw-substrate federated MLP checkpoints — is the "
                    "raw-substrate MLP degradation BN-evaluation-"
                    "conditional?  Arms A (init buffers, the published "
                    "path), B (pooled per-client recalibration, "
                    "train-shards-only), C (eval-time batch stats). "
                    "Arm D (transported round buffers) is not "
                    "reconstructible from the checkpoints (parameters "
                    "only were ever aggregated) and stays queued "
                    "owner compute."),
        "protocol": {
            "seed": cfg.SEED, "rounds": cfg.N_ROUNDS,
            "alpha": 1.0, "clients": cfg.N_CLIENTS,
            "aggregation": cfg.AGGREGATION_STRATEGY,
            "loss_variant": getattr(cfg, "LOSS_VARIANT", "fed_focal"),
            "calibration": cfg.CALIBRATION_METHOD,
            "evaluation_device": "cpu",
            "note": ("identical weights per treatment; no retraining; "
                     "frozen seed-42 substrate/shards from the "
                     "rawsubstrate cache"),
        },
        "substrate": {
            "train": int(len(y_train)), "val": int(len(y_val)),
            "test": int(len(y_test)),
            "train_pos": float(y_train.mean()),
            "val_pos": float(y_val.mean()),
            "test_pos": float(y_test.mean()),
        },
        "reproduction_gate": reproduction,
        "reproduction_gate_verdict": (
            "MISMATCH" if gate_failed else
            "reproduced (arm A within tolerance on every arm)"),
        "results": results,
        "published_event_level_reference": _published_event_ref(),
        "elapsed_s": time.time() - t0,
    }
    if event_results is not None:
        report["event_level"] = event_results
        report["event_level_summary"] = event_summary
    else:
        report["event_level"] = None
        report["event_level_note"] = (
            "skipped: raw audit metadata not found under "
            f"{args.raw_dir} (window-level results stand alone)")

    _atomic_json(report, args.output)
    print(f"\n[rawbn] report -> {args.output} "
          f"({time.time() - t0:.0f}s)")
    print(f"[rawbn] reproduction gate: "
          f"{report['reproduction_gate_verdict']}")
    if gate_failed:
        print("[rawbn] GATE FAILED — the checkpoint/substrate did not "
              "reproduce the published arm-A numbers; do not use these "
              "results as a same-weights counterfactual (investigate "
              "before reporting)")
        sys.exit(1)
    print("[rawbn] done.")


def _published_event_ref():
    """Published event-level reference (fed MLP arms) for side-by-side
    comparison, read from the committed artefact — never hand-typed."""
    p = os.path.join("outputs", "event_level_raw_p5.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return {k: {"n_detected_events": v["n_detected_events"],
                "n_true_events": v["n_true_events"],
                "event_detection_rate": v["event_detection_rate"],
                "false_alarm_windows_per_day":
                    v["false_alarm_windows_per_day"],
                "threshold": v["threshold"]}
            for k, v in d.get("models", {}).items()
            if k in ("fedavg_mlp", "fedprox_mlp", "scaffold_mlp")}


if __name__ == "__main__":
    main()
