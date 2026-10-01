#!/usr/bin/env python3
"""BN checkpoint diagnostic (v3.9 revision programme, step 2).

Question (external review): is the FedAvg instability — and the returned
artefact's weak test scores — substantially an artefact of the BatchNorm
implementation, in which BN running statistics are never transported
(get_weights/set_weights move parameters only, model.py)?

Phases (select with argv[1], e.g. "A,B" or "C"):

  A  checkpoint-state audit
     - BN buffers of the saved global FedAvg / FedProx models
       (expect num_batches_tracked == 0, running_var == 1 → never
       updated server-side)
     - which round the degenerate val-F1 "best checkpoint" selector
       actually restored

  B  BN-treatment arms on the saved (returned) artefacts
     For the exact weights that produced the published numbers:
       A  init buffers (status quo — reproduces published metrics)
       B  recalibrated buffers: exact per-client BN input statistics
          pooled by law of total variance (what a buffer-transporting
          implementation would evaluate; recalibration uses TRAINING
          shards only — no test/val leakage)
       C  eval-time batch statistics (BN in train mode, dropout off)
     → val + test ROC/PR under each treatment.

  C  faithful 50-round FedAvg re-run with per-round capture
     Replicates _run_fl_loop call order exactly (same seeds, same
     shard order, same aggregation), capturing at every logged round:
     global parameters + per-client BN buffers. r5 (pre-collapse),
     r20 (the restored "best" round) and r50 (diverged) are then
     evaluated under arms A/B/C — decomposing the published 0.575
     test ROC into (i) generalisation gap, (ii) stale BN statistics,
     (iii) checkpoint-selection failure. Round-level resumable.

Output: outputs/bn_diagnostic.json (+ data/cache/bn_diag_state.pt).
Deterministic given the frozen seed-42 protocol; CPU-friendly.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

CLONE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CLONE)
os.chdir(CLONE)

PHASES = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else {"A", "B", "C"}

import config as cfg                                    # noqa: E402
from model import make_fresh_model, clone_model         # noqa: E402
from federated_learning import (                        # noqa: E402
    local_train_fedavg, aggregate, evaluate_model)
from evaluation import compute_all_metrics              # noqa: E402

T0 = time.time()
DEVICE = torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def bn_modules(model):
    return [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]


def bn_buffer_state(model):
    """Readable summary of every BN buffer in the model."""
    out = []
    for name, buf in model.named_buffers():
        if buf.dtype in (torch.float32, torch.float64):
            out.append({"layer": name, "mean_abs": float(buf.mean().abs()),
                        "var_mean": float(buf.var()) if buf.numel() > 1 else float(buf[0])})
        else:
            out.append({"layer": name, "value": int(buf.reshape(-1)[0])})
    return out


def probs_fixed_mode(model, X, batch_size=cfg.EVAL_BATCH_SIZE, bn_train=False):
    """Batched probabilities WITHOUT touching module train/eval state
    (get_model_probs calls model.eval(), which would undo a BN-train arm).

    bn_train=True → BN layers normalise with per-batch statistics
    (adaptive-normalisation oracle); dropout stays in eval."""
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
            Xb = torch.tensor(np.ascontiguousarray(X[s:e]), dtype=torch.float32)
            all_probs[s:e] = torch.sigmoid(model(Xb)).numpy()
    if bn_train:
        for m in bn_modules(model):
            m.eval()
    if was_training:
        model.train()
    all_probs = np.nan_to_num(all_probs, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(all_probs, 1e-7, 1 - 1e-7)


def exact_bn_stats(model, X, batch_size=cfg.EVAL_BATCH_SIZE):
    """Exact mean/var of every BN layer's INPUT over the full array X,
    via forward pre-hooks. Returns list of (mean, var_unbiased) tensors."""
    bns = bn_modules(model)
    acc = [{"n": 0, "sum": None, "sumsq": None} for _ in bns]
    handles = []

    def make_hook(i):
        def hook(mod, inp):
            x = inp[0].detach()
            acc[i]["n"] += x.shape[0]
            s = x.sum(dim=0)
            ss = (x * x).sum(dim=0)
            acc[i]["sum"] = s if acc[i]["sum"] is None else acc[i]["sum"] + s
            acc[i]["sumsq"] = ss if acc[i]["sumsq"] is None else acc[i]["sumsq"] + ss
        return hook

    for i, m in enumerate(bns):
        handles.append(m.register_forward_pre_hook(make_hook(i)))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            e = min(s + batch_size, len(X))
            Xb = torch.tensor(np.ascontiguousarray(X[s:e]), dtype=torch.float32)
            model(Xb)
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
        var_u = var_b * n / max(n - 1, 1)            # PyTorch running_var semantics
        out.append((mean, var_u))
    return out


def set_bn_stats(model, stats):
    """stats: list of (mean, var) per BN layer, in module order."""
    for m, (mean, var) in zip(bn_modules(model), stats):
        with torch.no_grad():
            m.running_mean.copy_(mean.to(m.running_mean.dtype))
            m.running_var.copy_(var.to(m.running_var.dtype))
            m.num_batches_tracked.fill_(1)            # >0: eval uses running stats


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
        pooled.append((w_mean, torch.clamp(e_x2 - w_mean * w_mean, min=0.0)))
    return pooled


def metric_block(y, p, tag):
    m = compute_all_metrics(y, p, cfg.DEFAULT_THRESHOLD, beta=2.0)
    keep = {k: round(float(v), 4) for k, v in m.items()
            if k in ("roc_auc", "pr_auc", "f1", "recall", "precision", "brier")}
    keep["prob_min"] = round(float(np.min(p)), 6)
    keep["prob_median"] = round(float(np.median(p)), 6)
    keep["prob_max"] = round(float(np.max(p)), 6)
    keep["treatment"] = tag
    return keep


def eval_arms(model, X_val, y_val, X_test, y_test, shards, label):
    """Arms A (init buffers), B (recalibrated), C (batch stats)."""
    res = {"label": label, "val": {}, "test": {}}

    # Arm A — status quo (init buffers). Do not mutate the incoming model.
    mA = clone_model(model)
    p_val = probs_fixed_mode(mA, X_val)
    p_test = probs_fixed_mode(mA, X_test)
    res["val"]["A_init_buffers"] = metric_block(y_val, p_val, "A")
    res["test"]["A_init_buffers"] = metric_block(y_test, p_test, "A")

    # Arm B — recalibrated from client training shards (pooled exact stats).
    per_client, sizes = [], []
    for X_c, y_c in shards:
        stats = exact_bn_stats(mA, X_c)
        per_client.append(stats)
        sizes.append(len(y_c))
    pooled = pool_client_stats(per_client, sizes)
    mB = clone_model(model)
    set_bn_stats(mB, pooled)
    p_val = probs_fixed_mode(mB, X_val)
    p_test = probs_fixed_mode(mB, X_test)
    res["val"]["B_recalibrated"] = metric_block(y_val, p_val, "B")
    res["test"]["B_recalibrated"] = metric_block(y_test, p_test, "B")

    # Arm C — eval-time batch statistics.
    p_val = probs_fixed_mode(mA, X_val, bn_train=True)
    p_test = probs_fixed_mode(mA, X_test, bn_train=True)
    res["val"]["C_batch_stats"] = metric_block(y_val, p_val, "C")
    res["test"]["C_batch_stats"] = metric_block(y_test, p_test, "C")

    return res


def load_weights_into(model, weights):
    with torch.no_grad():
        for p, w in zip(model.parameters(), weights):
            p.data.copy_(torch.tensor(np.asarray(w), dtype=torch.float32))
    return model


# ─────────────────────────────────────────────────────────────────────────────
# data (identical to the main-run cache path)
# ─────────────────────────────────────────────────────────────────────────────

z = np.load("data/cache/phase_data.npz", allow_pickle=False)
X_train, y_train = z["X_train"], z["y_train"]
X_val, y_val = z["X_val"], z["y_val"]
X_test, y_test = z["X_test"], z["y_test"]
assignment = np.load("data/cache/phase_assignment.npz")["assignment"]

# deterministic shard rebuild (replicates main.py's cached-assignment branch)
X_arr, y_arr = np.asarray(X_train), np.asarray(y_train)
shards = []
for k in range(cfg.N_CLIENTS):
    idx = np.where(assignment == k)[0]
    if len(idx) == 0:
        shards.append((X_arr[:0], y_arr[:0]))
        continue
    rng = np.random.RandomState(cfg.SEED * 1000 + k)
    idx = idx[rng.permutation(len(idx))]
    shards.append((X_arr[idx], y_arr[idx]))

print(f"[data] train={len(y_train):,} val={len(y_val):,} test={len(y_test):,} "
      f"| shard sizes {[len(s[1]) for s in shards]}", flush=True)

report = {"runner": "experiments/run_bn_diagnostic.py",
          "protocol": {"seed": cfg.SEED, "rounds": cfg.N_ROUNDS,
                       "aggregation": cfg.AGGREGATION_STRATEGY,
                       "loss_variant": getattr(cfg, "LOSS_VARIANT", "fed_focal"),
                       "note": "frozen seed-42 protocol, phase-cache data"},
          "elapsed_s": None}

REPORT_PATH = os.path.join(cfg.OUTPUT_DIR, "bn_diagnostic.json")
if os.path.exists(REPORT_PATH):
    with open(REPORT_PATH) as f:
        prev = json.load(f)
    for k in ("phaseA_checkpoint_state", "phaseB_returned_artefacts",
              "phaseC_rerun", "phaseC_round_arms"):
        if k in prev:
            report[k] = prev[k]
    report["elapsed_s"] = prev.get("elapsed_s")

# ─────────────────────────────────────────────────────────────────────────────
# Phase A — checkpoint-state audit
# ─────────────────────────────────────────────────────────────────────────────

phaseA = report.get("phaseA_checkpoint_state", {})
if "A" in PHASES:
    print("\n=== Phase A: checkpoint-state audit ===", flush=True)
    for algo in ("fedavg", "fedprox"):
        d = torch.load(f"data/cache/phase_{algo}.pt", weights_only=False)
        assert d.get("done")
        m = d["model"]
        hist = d["history"]
        best_round = min(hist, key=lambda h: abs(h["f1"] - d["best_val_f1"]))["round"] \
            if hist else None
        phaseA[algo] = {
            "best_val_f1": round(float(d["best_val_f1"]), 6),
            "best_round_restored": best_round,
            "n_history": len(hist),
            "bn_buffers": bn_buffer_state(m),
            "history_rounds": {h["round"]: {"f1": round(h["f1"], 4),
                                            "roc": round(h["roc_auc"], 4)}
                               for h in hist},
        }
        nbt = [int(b.reshape(-1)[0]) for n, b in m.named_buffers()
               if "num_batches" in n]
        print(f"[{algo}] best_val_f1={d['best_val_f1']:.4f} "
              f"(restored round ~{best_round}) | num_batches_tracked={nbt}",
              flush=True)
    report["phaseA_checkpoint_state"] = phaseA

# ─────────────────────────────────────────────────────────────────────────────
# Phase B — arms on the returned artefacts
# ─────────────────────────────────────────────────────────────────────────────

if "B" in PHASES:
    print("\n=== Phase B: BN treatments on returned artefacts ===", flush=True)
    phaseB = {}
    for algo in ("fedavg", "fedprox"):
        d = torch.load(f"data/cache/phase_{algo}.pt", weights_only=False)
        m = d["model"]
        r = eval_arms(m, X_val, y_val, X_test, y_test, shards,
                      f"{algo}_returned_artefact")
        phaseB[algo] = r
        print(f"[{algo}] test ROC A={r['test']['A_init_buffers']['roc_auc']:.3f} "
              f"B={r['test']['B_recalibrated']['roc_auc']:.3f} "
              f"C={r['test']['C_batch_stats']['roc_auc']:.3f} | "
              f"test PR A={r['test']['A_init_buffers']['pr_auc']:.3f} "
              f"B={r['test']['B_recalibrated']['pr_auc']:.3f} "
              f"C={r['test']['C_batch_stats']['pr_auc']:.3f}", flush=True)
    report["phaseB_returned_artefacts"] = phaseB

# ─────────────────────────────────────────────────────────────────────────────
# Phase C — faithful FedAvg re-run with per-round capture (resumable)
# ─────────────────────────────────────────────────────────────────────────────

STATE_PATH = "data/cache/bn_diag_state.pt"

if "C" in PHASES:
    print("\n=== Phase C: FedAvg 50-round re-run with capture ===", flush=True)

    # replicate _run_fl_loop's RNG discipline exactly
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    rng = np.random.RandomState(cfg.SEED)

    sample_X = shards[0][0]
    input_dim = sample_X.shape[1]
    global_model, device = make_fresh_model(input_dim, use_lstm=False)
    global_pos_rate = float(np.concatenate(
        [np.asarray(y) for _, y in shards]).mean())

    history = []
    best_val_f1, best_round = 0.0, None
    capture = {}
    start_round = 1

    if os.path.exists(STATE_PATH):
        try:
            st = torch.load(STATE_PATH, weights_only=False)
            global_model.load_state_dict(st["model_state"])
            history = st["history"]
            best_val_f1 = st["best_val_f1"]
            best_round = st["best_round"]
            capture = st["capture"]
            rng.set_state(st["rng_state"])
            np.random.set_state(st["np_rng_state"])
            torch.set_rng_state(st["torch_rng_state"])
            start_round = st["next_round"]
            print(f"[capture] resuming from round {start_round}", flush=True)
        except Exception as e:
            print(f"[capture] state unreadable ({e}) — restarting", flush=True)

    def _save_state(next_round):
        tmp = STATE_PATH + ".tmp"
        torch.save({
            "model_state": global_model.state_dict(),
            "history": history, "best_val_f1": best_val_f1,
            "best_round": best_round, "capture": capture,
            "rng_state": rng.get_state(),
            "np_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "next_round": next_round,
        }, tmp)
        os.replace(tmp, STATE_PATH)

    for rnd in range(start_round, cfg.N_ROUNDS + 1):
        selected = rng.choice(cfg.N_CLIENTS,
                              max(1, int(cfg.FRACTION_FIT * cfg.N_CLIENTS)),
                              replace=False)
        client_weights, client_sizes, client_labels = [], [], []
        round_client_bn = []    # list of (cid, [(mean, var, nbt) per BN layer])

        for cid in selected:
            X_c, y_c = shards[cid]
            if len(X_c) == 0:
                continue
            local = clone_model(global_model)
            local = local_train_fedavg(local, X_c, y_c,
                                       current_round=rnd,
                                       total_rounds=cfg.N_ROUNDS, seed=cfg.SEED,
                                       global_pos_rate=global_pos_rate)
            client_weights.append([p.data.cpu().numpy().copy()
                                   for p in local.parameters()])
            round_client_bn.append(
                (cid, [(m.running_mean.detach().clone(),
                        m.running_var.detach().clone(),
                        int(m.num_batches_tracked.item()))
                       for m in bn_modules(local)]))
            client_sizes.append(len(X_c))
            client_labels.append(y_c)
            del local

        new_weights = aggregate(client_weights, client_sizes, client_labels,
                                global_pos_rate, round_num=rnd)
        if any(np.isnan(w).any() for w in new_weights):
            print(f"  [fedavg-cap] NaN at round {rnd} — reverting round",
                  flush=True)
            continue

        with torch.no_grad():
            for p, w in zip(global_model.parameters(), new_weights):
                p.data.copy_(torch.tensor(w, dtype=torch.float32))

        if rnd % 5 == 0 or rnd == cfg.N_ROUNDS:
            metrics = evaluate_model(global_model, X_val, y_val,
                                     batch_size=cfg.EVAL_BATCH_SIZE)
            history.append({"round": rnd,
                            **{k: v for k, v in metrics.items() if k != "preds"}})
            print(f"  Round {rnd:>3} | val F1: {metrics['f1']:.4f} | "
                  f"val ROC: {metrics['roc_auc']:.3f}", flush=True)
            if metrics["f1"] > best_val_f1:
                best_val_f1 = metrics["f1"]
                best_round = rnd
            capture[rnd] = {
                "params": [w.copy() for w in new_weights],
                "client_bn": round_client_bn,
                "val": {k: round(float(v), 4) for k, v in metrics.items()
                        if k in ("f1", "recall", "precision", "roc_auc", "pr_auc")},
            }

        _save_state(rnd + 1)

    print(f"[capture] best_val_f1={best_val_f1:.4f} "
          f"restored at round {best_round}", flush=True)

    # faithfulness check against the original run
    orig = phaseA.get("fedavg", {}).get("history_rounds", {})
    same = all(abs(orig[r["round"]]["roc"] - r["roc_auc"]) < 0.01
               for r in history if r["round"] in orig) if orig else None
    report["phaseC_rerun"] = {
        "history": history,
        "best_val_f1": round(float(best_val_f1), 6),
        "best_round_restored": best_round,
        "faithful_replication": same,
    }
    print(f"[capture] faithful vs frozen history (±0.01 ROC): {same}",
          flush=True)

    # evaluate captured rounds under the three arms + round-transport arm
    phaseC_arms = {}
    for r in (5, 20, 50):
        if r not in capture:
            continue
        cap = capture[r]
        m = make_fresh_model(input_dim, use_lstm=False)[0]
        load_weights_into(m, cap["params"])
        arms = eval_arms(m, X_val, y_val, X_test, y_test, shards,
                         f"fedavg_round{r}_weights")
        # Arm D: buffers exactly as a buffer-transporting implementation
        # would have averaged them THIS round (element-wise weighted mean
        # of the per-client running stats local training produced)
        sizes_all = [len(shards[c][1]) for c in range(cfg.N_CLIENTS)]
        n_layers = len(cap["client_bn"][0][1])
        naive = []
        for li in range(n_layers):
            tot = float(sum(sizes_all[cid] for cid, _ in cap["client_bn"]))
            wm, wv = None, None
            for cid, cl in cap["client_bn"]:
                w = sizes_all[cid] / tot
                mm = cl[li][0] * w
                vv = cl[li][1] * w
                wm = mm if wm is None else wm + mm
                wv = vv if wv is None else wv + vv
            naive.append((wm, wv))
        mN = make_fresh_model(input_dim, use_lstm=False)[0]
        load_weights_into(mN, cap["params"])
        set_bn_stats(mN, naive)
        p_val = probs_fixed_mode(mN, X_val)
        p_test = probs_fixed_mode(mN, X_test)
        arms["val"]["D_round_client_buffers"] = metric_block(y_val, p_val, "D")
        arms["test"]["D_round_client_buffers"] = metric_block(y_test, p_test, "D")
        phaseC_arms[f"round{r}"] = arms
        t = arms["test"]
        print(f"[round {r}] test ROC A={t['A_init_buffers']['roc_auc']:.3f} "
              f"B={t['B_recalibrated']['roc_auc']:.3f} "
              f"C={t['C_batch_stats']['roc_auc']:.3f} "
              f"D={t['D_round_client_buffers']['roc_auc']:.3f}", flush=True)
    report["phaseC_round_arms"] = phaseC_arms

report["elapsed_s"] = round(time.time() - T0, 1)
_atomic_json(report, REPORT_PATH)
print(f"\n[BNdiag] report -> {REPORT_PATH} ({report['elapsed_s']}s)", flush=True)
