#!/usr/bin/env python3
"""
experiments/run_arm_b_central_sanity.py  (v4.9.2 / Dossier R-FS9-R10,
master change register item A1 — the centralised arm-B sanity check)
────────────────────────────────────────────────────────
Question: is arm B (pooled per-client BN recalibration) a working
instrument or a broken implementation?  The Round-13 adjudication
agreed with the third-party reviewer that Table 8's B column —
"pooled-statistics recalibration collapses every federated arm to
chance" — is an implementation-level claim wearing a scientific
conclusion's clothes until the one decisive check runs: apply the
SAME arm-B machinery to the CENTRALISED raw MLP.  The centralised
model was trained on the pooled training data (the union of the
seed-42 shards), so pooling the per-client BN input statistics by
the law of total variance should approximately RECONSTRUCT the
model's own running statistics — a working arm B therefore
preserves the central model's ranking (~0.971 ROC-AUC / ~0.445
PR-AUC, the published centralised numbers).  A chance-level
collapse there (the federated arms' 0.500-0.501 signature) means
arm B is BROKEN: the B column withdraws and the "actively
destructive" sentence goes.  This check decides which world we
are in, and it costs seconds.

Method: NO retraining.  Load data/cache/rawsubstrate/baselines.pt
(the centralised MLP the owner's ask-#4 retrain left behind; the
v3.x-era central checkpoint, like the v3.x FL checkpoints, was
never persisted — the same disclosed premise amendment RUNLOG ask
#4 carries).  Rebuild the substrate and the seed-42 Dirichlet
shards from the cache (the identical recipe
run_raw_bn_diagnostic.py uses), then evaluate the identical
weights under two BN treatments:

  A  own running buffers (as trained, the published path) — TWO
     gates, both v4.9.3 (the first ask-#8 execution, 2026-10-06,
     exposed the v4.9.2 single-tolerance gate's miscalibration;
     the amendment is evidence-based and disclosed in the RUNLOG
     v4.9.3 row):
       1. IDENTITY gate — the wrong-checkpoint tripwire: arm A vs
          the ask-#4 RETRAIN RECORD committed at v4.9
          (outputs/raw_substrate_rerun.json, centralized_mlp) at
          --identity-tolerance (default 1e-3).  The tightest
          reference that exists for THIS checkpoint — same
          weights, same frozen evaluation path; the first ask-#8
          execution reproduced it to <1e-5.  Ten times tighter
          than the v4.9.2 published bracket.
       2. PUBLISHED retrain bracket, PER METRIC: arm A vs the
          PUBLISHED outputs/raw_substrate_eval.json
          centralized_mlp numbers at --gate-tolerance (ROC-AUC,
          default 1e-2, unchanged from v4.9.2) AND
          --gate-tolerance-pr (PR-AUC, default 2.5e-2, NEW).
          Basis: the frozen ask-#4 retrain drift itself —
          ROC deltas LR +1.7e-4 / XGB +1.5e-3 / MLP -2.32e-3
          (bracket 1e-2 = 4x headroom); PR deltas LR +8.3e-4 /
          XGB -1.9e-3 / MLP -1.93e-2 (bracket 2.5e-2).  The
          v4.9.2 single 1e-2 bracket cited the ROC-only
          "<= 2.3e-3 pooled-baseline drift" measurement — the
          paper's own disclosure is metric-scoped ("...to
          <= 2.3e-3 ROC-AUC") — and generalised it to PR-AUC,
          which the record's own MLP PR drift (1.93e-2) made
          unsatisfiable by construction: the first ask-#8 run
          failed the gate on the PR side alone (ROC 2.32e-3 was
          inside) and the verdict was correctly NOT read off it.
     A MISMATCH on EITHER gate exits 1 — never read the arm-B
     verdict off a wrong checkpoint.  (The 1e-4 same-weights
     gate vs the PUBLISHED numbers is impossible by construction
     for retrained weights — the ask-#4 lesson, unchanged.)
  B  pooled per-client recalibration: exact per-client BN input
     statistics over the TRAINING shards (forward pre-hooks,
     law-of-total-variance pooling, non-negativity clamp — the
     byte-identical helper contracts of the diagnostic runner),
     set into a pristine clone.

Also recorded: the per-layer distance between the pooled
statistics and the model's own running buffers (mean relative
L2, var relative delta) — supporting diagnostics that make the
reconstruction visible in the artefact; the EVALUATION under
treatment B is the decisive instrument.

Verdict bands (declared before the run, R-FS9-R10 §6 item A1):
  arm_b_validated : ROC_B >= 0.95 and PR_B >= 0.35
                    (ranking preserved at the central model's level)
  arm_b_broken    : ROC_B <= 0.60 or PR_B <= 0.10
                    (chance-level collapse — the federated signature)
  inconclusive    : anything between (needs adjudication before
                    Table 8's reading changes)

Exit codes: 0 for a DECISIVE verdict (validated or broken — both
are findings, the check did its job); 1 for execution defects
(missing checkpoint, identity-gate MISMATCH — baselines.pt is
not the checkpoint the ask-#4 record describes — published-
bracket MISMATCH on arm A, or an INCONCLUSIVE verdict (undecided
means Table 8's status cannot be settled this run; adjudicate
before proceeding).  The artefact is written BEFORE any exit.

Output: outputs/arm_b_central_sanity.json.  CPU-capable, seconds
(pure evaluation; no training).  Deterministic given the frozen
seed-42 protocol and the cached substrate.

Usage:
    python experiments/run_arm_b_central_sanity.py \
        [--raw-dir /tmp/swansf_raw] [--output ...] [--gate-tolerance 1e-2]
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
    torch = None             # loud, actionable message

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
PUBLISHED_EVAL = os.path.join("outputs", "raw_substrate_eval.json")
RERUN_RECORD = os.path.join("outputs", "raw_substrate_rerun.json")
BASELINES = os.path.join(CACHE, "baselines.pt")
OUT_DEFAULT = os.path.join("outputs", "arm_b_central_sanity.json")
RAW_DEFAULT = "/tmp/swansf_raw"

# verdict bands (declared in the docstring; pinned here so the
# battery's static layer can freeze them — see
# tests/test_raw_bn_diagnostic.py)
VERDICT_PASS_ROC, VERDICT_PASS_PR = 0.95, 0.35
VERDICT_FAIL_ROC, VERDICT_FAIL_PR = 0.60, 0.10


def _atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str, allow_nan=False)
    os.replace(tmp, path)


# ─────────────────────────────────────────────────────────────────────────────
# BN helpers (self-contained copies of the run_raw_bn_diagnostic.py
# contracts — that module is an executable script and must not be
# imported; the byte-identical helpers are what makes THIS check a
# validation of THAT runner's arm B)
# ─────────────────────────────────────────────────────────────────────────────

def bn_modules(model):
    import torch.nn as nn
    return [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]


def probs_fixed_mode(model, X, device, batch_size=cfg.EVAL_BATCH_SIZE):
    """Batched probabilities WITHOUT touching module train/eval state.
    NaN/clip semantics identical to get_model_probs so that the own-
    buffers arm A reproduces the published evaluation path exactly.
    (No bn_train mode here: both central arms evaluate through
    running statistics — that is the entire point of the check.)"""
    was_training = model.training
    model.eval()
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
    tensors.  (Byte-identical contract to the diagnostic runner's.)"""
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
# evaluation (mirrors run_raw_substrate.py section 5 / the diagnostic's
# evaluate_treatment, per treatment — prior-shift calibration on that
# treatment's VALIDATION probabilities, val-frozen F-beta threshold,
# frozen-FPR operating points, single-pass test evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_treatment(model, device, X_val, y_val, X_test, y_test,
                       y_train_rate, tag):
    p_val = probs_fixed_mode(model, X_val, device)
    p_test = probs_fixed_mode(model, X_test, device)
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
    return {"threshold": float(t), "val_fbeta": float(fb),
            "test": metrics, "frozen_operating_points": ops,
            "treatment": tag}


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def load_central_model():
    """Load the centralised raw-substrate MLP from the baselines cache.

    baselines.pt carries the models dict train_centralized() returned
    (logistic_regression / xgboost / centralized_mlp) — it has NO
    protocol-fingerprint fields (the centralised path never had the
    FL runner's fingerprint discipline), so the arm-A reproduction
    gate below is the fingerprint surrogate: a wrong or stale
    checkpoint fails the gate loudly instead of silently passing a
    fingerprint check that cannot exist."""
    if not os.path.exists(BASELINES):
        sys.exit(f"[a1] baselines cache missing: {BASELINES}\n"
                 f"       The centralised raw MLP exists only owner-side "
                 f"(gitignored data/). Run "
                 f"experiments/run_raw_substrate.py (or its ask-#4 "
                 f"repair-sequence form) first — baselines.pt is the "
                 f"by-product it leaves in the cache.")
    c_models = torch.load(BASELINES, weights_only=False,
                          map_location="cpu")
    if not isinstance(c_models, dict) or \
            "centralized_mlp" not in c_models:
        sys.exit(f"[a1] {BASELINES} does not carry a "
                 f"centralized_mlp entry — unexpected cache format; "
                 f"re-run experiments/run_raw_substrate.py.")
    m = c_models["centralized_mlp"]
    from model import SolarMLP
    if not isinstance(m, SolarMLP):
        sys.exit(f"[a1] {BASELINES}: centralized_mlp is not a "
                 f"SolarMLP instance — wrong cache.")
    if not bn_modules(m):
        sys.exit(f"[a1] the centralised MLP carries no BatchNorm1d "
                 f"layers — the arm-B sanity check is meaningless "
                 f"for a BN-free architecture.")
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=RAW_DEFAULT,
                    help="raw benchmark dir (substrate build cache miss)")
    ap.add_argument("--output", default=OUT_DEFAULT)
    ap.add_argument("--gate-tolerance", type=float, default=1e-2,
                    help="arm-A reproduction-gate tolerance, ROC-AUC "
                         "side of the retrain bracket (NOT the 1e-4 "
                         "same-weights gate)")
    ap.add_argument("--gate-tolerance-pr", type=float, default=2.5e-2,
                    help="arm-A gate tolerance, PR-AUC side of the "
                         "retrain bracket (v4.9.3: the frozen ask-#4 "
                         "MLP PR drift is 1.93e-2 — the v4.9.2 single "
                         "1e-2 bracket was unsatisfiable by "
                         "construction)")
    ap.add_argument("--identity-tolerance", type=float, default=1e-3,
                    help="arm-A identity-gate tolerance vs the ask-#4 "
                         "retrain record (raw_substrate_rerun.json — "
                         "the wrong-checkpoint tripwire, the tightest "
                         "reference that exists for this checkpoint)")
    args = ap.parse_args()

    t0 = time.time()
    if torch is None:
        sys.exit("[a1] torch is required (evaluation of torch "
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
    print(f"[a1] substrate: train {len(y_train):,} "
          f"(pos {y_train.mean():.2%}) | val {len(y_val):,} "
          f"(pos {y_val.mean():.2%}) | test {len(y_test):,} "
          f"(pos {y_test.mean():.2%})", flush=True)

    shards = partition_data_dirichlet(
        X_train, y_train, alpha=1.0, n_clients=cfg.N_CLIENTS,
        seed=cfg.SEED, min_samples=cfg.MIN_SAMPLES_PER_CLIENT,
        verbose=False)
    print(f"[a1] shards rebuilt (Dirichlet alpha=1.0, seed "
          f"{cfg.SEED}): sizes {[len(s[1]) for s in shards]}", flush=True)

    # ── 2. published reference for the arm-A mini gate ─────────────────
    if not os.path.exists(PUBLISHED_EVAL):
        sys.exit(f"[a1] published reference missing: {PUBLISHED_EVAL} "
                 f"— the centralised gate cannot run; restore the "
                 f"committed artefact first.")
    published = json.load(open(PUBLISHED_EVAL))["results"]
    pub = published["centralized_mlp"]["test"]
    # v4.9.3: the identity gate's reference — the ask-#4 retrain
    # record, committed at v4.9. The tightest reference that exists
    # for THIS checkpoint (same weights, same frozen eval path).
    if not os.path.exists(RERUN_RECORD):
        sys.exit(f"[a1] ask-#4 retrain record missing: "
                 f"{RERUN_RECORD} — the identity gate cannot run; "
                 f"restore the committed v4.9 artefact first.")
    rerun_ref = json.load(open(RERUN_RECORD))["results"]
    rer = rerun_ref["centralized_mlp"]["test"]

    # ── 3. the centralised model, own buffers vs pooled ────────────────
    model = load_central_model()
    device = torch.device("cpu")
    from model import clone_model

    # arm A — own running buffers (the published path), pristine clone
    mA = clone_model(model).to(device)
    blkA = evaluate_treatment(mA, device, X_val, y_val, X_test, y_test,
                              y_train.mean(), "A_own_buffers")
    d_roc = abs(float(blkA["test"]["roc_auc"]) - float(pub["roc_auc"]))
    d_pr = abs(float(blkA["test"]["pr_auc"]) - float(pub["pr_auc"]))
    # gate 1 — the v4.9.3 identity gate (the wrong-checkpoint tripwire)
    id_d_roc = abs(float(blkA["test"]["roc_auc"]) -
                   float(rer["roc_auc"]))
    id_d_pr = abs(float(blkA["test"]["pr_auc"]) -
                  float(rer["pr_auc"]))
    identity_verdict = ("match" if max(id_d_roc, id_d_pr)
                        <= args.identity_tolerance else "MISMATCH")
    print(f"[a1] identity gate (vs the ask-#4 retrain record): ROC "
          f"{blkA['test']['roc_auc']:.5f} (record "
          f"{rer['roc_auc']:.5f}, d={id_d_roc:.2e}) | PR "
          f"{blkA['test']['pr_auc']:.5f} (record {rer['pr_auc']:.5f}, "
          f"d={id_d_pr:.2e}) | tol {args.identity_tolerance:g} | "
          f"gate: {identity_verdict}", flush=True)
    # gate 2 — the published retrain bracket, PER METRIC (v4.9.3)
    gate_verdict = ("match" if (d_roc <= args.gate_tolerance and
                                d_pr <= args.gate_tolerance_pr)
                    else "MISMATCH")
    print(f"[a1] central A (own buffers): ROC "
          f"{blkA['test']['roc_auc']:.5f} (published "
          f"{pub['roc_auc']:.5f}, d={d_roc:.2e}, tol "
          f"{args.gate_tolerance:g}) | PR "
          f"{blkA['test']['pr_auc']:.5f} (published "
          f"{pub['pr_auc']:.5f}, d={d_pr:.2e}, tol "
          f"{args.gate_tolerance_pr:g}) | gate: {gate_verdict}",
          flush=True)

    # per-client exact stats over the TRAINING shards, pooled
    per_client, sizes = [], []
    for X_c, y_c in shards:
        stats = exact_bn_stats(mA, X_c, device)
        per_client.append(stats)
        sizes.append(len(y_c))
    pooled = pool_client_stats(per_client, sizes)

    # supporting diagnostics: pooled stats vs the model's OWN buffers
    own = [(m.running_mean.detach().cpu().clone(),
            m.running_var.detach().cpu().clone())
           for m in bn_modules(mA)]
    buffer_deltas = []
    for li, ((pm, pv), (om, ov)) in enumerate(zip(pooled, own)):
        mean_scale = float(torch.linalg.norm(om)) or 1e-12
        rel_mean = float(torch.linalg.norm(pm - om)) / mean_scale
        rel_var = float(torch.abs(pv - ov).div(
            torch.clamp(ov, min=1e-12)).mean())
        buffer_deltas.append(
            {"layer": li,
             "pooled_vs_own_mean_rel_l2": rel_mean,
             "pooled_vs_own_var_mean_rel_delta": rel_var})
        print(f"[a1] BN layer {li}: pooled-vs-own mean rel L2 "
              f"{rel_mean:.4f} | var rel delta {rel_var:.4f}",
              flush=True)

    # arm B — the pooled statistics set into a pristine clone
    mB = clone_model(model).to(device)
    set_bn_stats(mB, pooled)
    blkB = evaluate_treatment(mB, device, X_val, y_val, X_test, y_test,
                              y_train.mean(), "B_pooled_recalibrated")
    roc_b = float(blkB["test"]["roc_auc"])
    pr_b = float(blkB["test"]["pr_auc"])
    print(f"[a1] central B (pooled recalibration): ROC {roc_b:.5f} | "
          f"PR {pr_b:.5f}", flush=True)

    # ── 4. verdict bands (declared in the docstring, pinned above) ────
    if roc_b >= VERDICT_PASS_ROC and pr_b >= VERDICT_PASS_PR:
        verdict = "arm_b_validated"
    elif roc_b <= VERDICT_FAIL_ROC or pr_b <= VERDICT_FAIL_PR:
        verdict = "arm_b_broken"
    else:
        verdict = "inconclusive"
    verdict_basis = {
        "pass_bands": {"roc_ge": VERDICT_PASS_ROC,
                       "pr_ge": VERDICT_PASS_PR},
        "fail_bands": {"roc_le": VERDICT_FAIL_ROC,
                       "pr_le": VERDICT_FAIL_PR},
        "observed": {"roc_auc": roc_b, "pr_auc": pr_b},
    }
    print(f"[a1] verdict: {verdict}", flush=True)

    # ── 5. report ──────────────────────────────────────────────────────
    report = {
        "runner": "experiments/run_arm_b_central_sanity.py",
        "purpose": ("R-FS9-R10 master change register item A1: the "
                    "centralised arm-B sanity check. The third-party "
                    "review's decisive experiment — apply arm B (pooled "
                    "per-client BN recalibration over the seed-42 train "
                    "shards) to the CENTRALISED raw MLP, where pooled "
                    "statistics should reconstruct the model's own "
                    "running statistics. Validated: Table 8's B column "
                    "stands and becomes the validated proxy for the "
                    "transport fix. Broken: the column withdraws and "
                    "the 'actively destructive' sentence goes. Costs "
                    "seconds; gates the reading of everything else in "
                    "Table 8."),
        "protocol": {
            "seed": cfg.SEED, "alpha": 1.0, "clients": cfg.N_CLIENTS,
            "calibration": cfg.CALIBRATION_METHOD,
            "evaluation_device": "cpu",
            "note": ("identical weights per treatment; no retraining; "
                     "frozen seed-42 substrate/shards from the "
                     "rawsubstrate cache; the centralised checkpoint "
                     "is the ask-#4 retrain's baselines.pt (the "
                     "v3.x-era central checkpoint was never "
                     "persisted — same premise amendment as RUNLOG "
                     "ask #4)"),
        },
        "substrate": {
            "train": int(len(y_train)), "val": int(len(y_val)),
            "test": int(len(y_test)),
            "train_pos": float(y_train.mean()),
            "val_pos": float(y_val.mean()),
            "test_pos": float(y_test.mean()),
        },
        "reproduction_gate": {
            "published_roc_auc": float(pub["roc_auc"]),
            "reproduced_roc_auc": float(blkA["test"]["roc_auc"]),
            "delta_roc_auc": d_roc,
            "published_pr_auc": float(pub["pr_auc"]),
            "reproduced_pr_auc": float(blkA["test"]["pr_auc"]),
            "delta_pr_auc": d_pr,
            "tolerance": {"roc_auc": args.gate_tolerance,
                          "pr_auc": args.gate_tolerance_pr},
            "tolerance_basis": (
                "per-metric retrain bracket (v4.9.3): the frozen "
                "ask-#4 pooled-baseline drift itself — ROC deltas "
                "LR +1.7e-4 / XGBoost +1.5e-3 / MLP -2.32e-3 vs "
                "bracket 1e-2; PR deltas LR +8.3e-4 / XGBoost "
                "-1.9e-3 / MLP -1.93e-2 vs bracket 2.5e-2. The "
                "v4.9.2 single 1e-2 bracket cited the ROC-only "
                "'<= 2.3e-3 pooled-baseline drift' measurement "
                "(the paper's own disclosure is metric-scoped) "
                "and generalised it to PR-AUC, which the record's "
                "own MLP PR drift (1.93e-2) made unsatisfiable "
                "by construction; the first ask-#8 execution "
                "(2026-10-06) failed it on the PR side alone and "
                "that MISMATCH is frozen in the RUNLOG; NOT the "
                "1e-4 same-weights gate"),
            "verdict": gate_verdict,
        },
        "reproduction_gate_verdict": gate_verdict,
        "identity_gate": {
            "reference": ("outputs/raw_substrate_rerun.json (the "
                          "ask-#4 retrain record, committed at "
                          "v4.9 — the tightest reference that "
                          "exists for THIS checkpoint: same "
                          "weights, same frozen evaluation path)"),
            "retrain_roc_auc": float(rer["roc_auc"]),
            "retrain_pr_auc": float(rer["pr_auc"]),
            "reproduced_roc_auc": float(blkA["test"]["roc_auc"]),
            "reproduced_pr_auc": float(blkA["test"]["pr_auc"]),
            "delta_roc_auc": id_d_roc,
            "delta_pr_auc": id_d_pr,
            "tolerance": args.identity_tolerance,
            "tolerance_basis": (
                "same-weights reproduction bracket (v4.9.3): the "
                "first ask-#8 execution reproduced the record to "
                "<1e-5, so 1e-3 leaves 100x margin over that "
                "while being 10x tighter than the v4.9.2 "
                "published bracket — the wrong-checkpoint "
                "tripwire the v4.9.2 design reached for"),
            "verdict": identity_verdict,
        },
        "identity_gate_verdict": identity_verdict,
        "results": {
            "centralized_mlp": {
                "A_own_buffers": blkA,
                "B_pooled_recalibrated": blkB,
            },
        },
        "pooled_vs_own_buffer_deltas": buffer_deltas,
        "verdict": verdict,
        "verdict_basis": verdict_basis,
        "elapsed_s": time.time() - t0,
    }
    _atomic_json(report, args.output)
    print(f"\n[a1] report -> {args.output} "
          f"({time.time() - t0:.0f}s)")

    if identity_verdict == "MISMATCH":
        print("[a1] IDENTITY GATE FAILED — baselines.pt did not "
              "reproduce the ask-#4 retrain record within the "
              "same-weights bracket: it is not the checkpoint the "
              "frozen record describes (re-run the ask-#4 repair "
              "sequence, or widen --identity-tolerance with a "
              "RUNLOG disclosure); do not read the arm-B verdict "
              "off a wrong checkpoint")
        sys.exit(1)
    if gate_verdict == "MISMATCH":
        print("[a1] GATE FAILED — the centralised checkpoint did not "
              "reproduce the published numbers within the per-metric "
              "retrain bracket; do not read the arm-B verdict off a "
              "wrong checkpoint (investigate before reporting)")
        sys.exit(1)
    if verdict == "inconclusive":
        print("[a1] INCONCLUSIVE — the observed arm-B numbers fall "
              "between the declared bands; adjudicate before Table "
              "8's reading changes")
        sys.exit(1)
    if verdict == "arm_b_validated":
        print("[a1] done — arm B stands: the pooled recalibration "
              "preserves the centralised ranking, so Table 8's B "
              "column is a validated instrument (and the validated "
              "proxy for the transport fix, register item A2).")
    else:
        print("[a1] done — arm B is BROKEN on the centralised model: "
              "the pooled recalibration collapses a model whose own "
              "buffers score 0.971; Table 8's B column withdraws and "
              "the 'actively destructive' sentence goes (register "
              "item A1, the withdrawal branch).")
    sys.exit(0)


if __name__ == "__main__":
    main()
