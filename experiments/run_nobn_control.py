#!/usr/bin/env python3
"""Controlled no-BatchNorm experiment (v3.9 revision programme, step 3).

Motivated by the BN diagnostic (outputs/bn_diagnostic.json): the FedAvg
weights never lose discriminative content (test ROC ~0.95 under correct
normalisation at every round), but the shipped server-side evaluation
initialises BN running statistics to (0, 1) and never updates them, so
the evaluated behaviour drifts to 0.575/0.098 as rounds progress. The
open question is therefore the TRAINING-dynamics side: does plain
FedAvg remain stable when the architecture has no BN layers at all
(no buffers → the shipped evaluation is exact)?

Design: identical frozen seed-42 protocol (data, partition, rounds,
local epochs, loss, aggregation, monitoring, best-checkpoint rule),
with SolarMLP's three BatchNorm1d layers replaced by Identity. Run as:

    python experiments/run_nobn_control.py fedavg
    python experiments/run_nobn_control.py fedprox   (attribution control)

Output: outputs/nobn_control.json (merged across invocations),
round-level resumable state in data/cache/nobn_<algo>_state.pt.
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

ALGO = sys.argv[1] if len(sys.argv) > 1 else "fedavg"
assert ALGO in ("fedavg", "fedprox"), "usage: run_nobn_control.py [fedavg|fedprox]"

import config as cfg                                    # noqa: E402
import model as model_mod                               # noqa: E402
from model import SolarMLP, make_fresh_model, clone_model  # noqa: E402
from federated_learning import (                        # noqa: E402
    local_train_fedavg, local_train_fedprox, aggregate, evaluate_model)
from evaluation import compute_all_metrics              # noqa: E402

T0 = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# No-BN architecture: identical SolarMLP with BatchNorm1d → Identity
# ─────────────────────────────────────────────────────────────────────────────

class NoBNSolarMLP(SolarMLP):
    """SolarMLP with all BatchNorm1d layers replaced by Identity.

    Same Linear/ReLU/Dropout stack, same parameter families minus
    (gamma, beta); no running statistics exist, so weight transport is
    exact — there is no buffer state to drift."""
    def __init__(self, input_dim=None):
        super().__init__(input_dim=input_dim if input_dim is not None
                         else cfg.INPUT_DIM)
        for i, m in enumerate(self.net):
            if isinstance(m, nn.BatchNorm1d):
                self.net[i] = nn.Identity()


# patch the module globals used by make_fresh_model / clone_model so the
# entire FL machinery constructs no-BN models transparently
model_mod.SolarMLP = NoBNSolarMLP
SolarMLP = NoBNSolarMLP


def _atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def metric_block(y, p):
    m = compute_all_metrics(y, p, cfg.DEFAULT_THRESHOLD, beta=2.0)
    keep = {k: round(float(v), 4) for k, v in m.items()
            if k in ("roc_auc", "pr_auc", "f1", "recall", "precision", "brier")}
    keep["prob_min"] = round(float(np.min(p)), 6)
    keep["prob_median"] = round(float(np.median(p)), 6)
    keep["prob_max"] = round(float(np.max(p)), 6)
    return keep


def probs(model, X, batch_size=cfg.EVAL_BATCH_SIZE):
    model.eval()
    n = len(X)
    out = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            Xb = torch.tensor(np.ascontiguousarray(X[s:e]), dtype=torch.float32)
            out[s:e] = torch.sigmoid(model(Xb)).numpy()
    out = np.nan_to_num(out, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(out, 1e-7, 1 - 1e-7)


# ─────────────────────────────────────────────────────────────────────────────
# data (identical to the main-run cache path)
# ─────────────────────────────────────────────────────────────────────────────

z = np.load("data/cache/phase_data.npz", allow_pickle=False)
X_train, y_train = z["X_train"], z["y_train"]
X_val, y_val = z["X_val"], z["y_val"]
X_test, y_test = z["X_test"], z["y_test"]
assignment = np.load("data/cache/phase_assignment.npz")["assignment"]

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

print(f"[data] train={len(y_train):,} val={len(y_val):,} "
      f"test={len(y_test):,} | shards {[len(s[1]) for s in shards]}",
      flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# faithful loop replication (mirrors _run_fl_loop / bn-diag Phase C)
# ─────────────────────────────────────────────────────────────────────────────

STATE_PATH = f"data/cache/nobn_{ALGO}_state.pt"

torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
rng = np.random.RandomState(cfg.SEED)

sample_X = shards[0][0]
input_dim = sample_X.shape[1]
global_model, device = make_fresh_model(input_dim, use_lstm=False)
assert isinstance(global_model, NoBNSolarMLP) and not any(
    isinstance(m, nn.BatchNorm1d) for m in global_model.modules()), \
    "no-BN patch failed"
n_params = sum(p.numel() for p in global_model.parameters())
print(f"[arch] NoBNSolarMLP | params={n_params:,} "
      f"(BN baseline 29,377)", flush=True)

global_pos_rate = float(np.concatenate(
    [np.asarray(y) for _, y in shards]).mean())

history = []
best_val_f1, best_round = 0.0, None
best_weights = None
round_params = {}
start_round = 1

if os.path.exists(STATE_PATH):
    try:
        st = torch.load(STATE_PATH, weights_only=False)
        global_model.load_state_dict(st["model_state"])
        history = st["history"]
        best_val_f1 = st["best_val_f1"]
        best_round = st["best_round"]
        best_weights = st["best_weights"]
        round_params = st["round_params"]
        rng.set_state(st["rng_state"])
        np.random.set_state(st["np_rng_state"])
        torch.set_rng_state(st["torch_rng_state"])
        start_round = st["next_round"]
        print(f"[{ALGO}-noBN] resuming from round {start_round}", flush=True)
    except Exception as e:
        print(f"[{ALGO}-noBN] state unreadable ({e}) — restarting", flush=True)


def _save_state(next_round):
    tmp = STATE_PATH + ".tmp"
    torch.save({
        "model_state": global_model.state_dict(),
        "history": history, "best_val_f1": best_val_f1,
        "best_round": best_round, "best_weights": best_weights,
        "round_params": round_params,
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

    for cid in selected:
        X_c, y_c = shards[cid]
        if len(X_c) == 0:
            continue
        local = clone_model(global_model)
        if ALGO == "fedavg":
            local = local_train_fedavg(local, X_c, y_c,
                                       current_round=rnd,
                                       total_rounds=cfg.N_ROUNDS,
                                       seed=cfg.SEED,
                                       global_pos_rate=global_pos_rate)
        else:
            local = local_train_fedprox(local, global_model, X_c, y_c,
                                        mu=cfg.MU, current_round=rnd,
                                        total_rounds=cfg.N_ROUNDS,
                                        seed=cfg.SEED,
                                        global_pos_rate=global_pos_rate)
        client_weights.append([p.data.cpu().numpy().copy()
                               for p in local.parameters()])
        client_sizes.append(len(X_c))
        client_labels.append(y_c)
        del local

    new_weights = aggregate(client_weights, client_sizes, client_labels,
                            global_pos_rate, round_num=rnd)
    if any(np.isnan(w).any() for w in new_weights):
        print(f"  [{ALGO}-noBN] NaN at round {rnd} — reverting round",
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
            best_weights = [w.copy() for w in new_weights]
        round_params[rnd] = [w.copy() for w in new_weights]

    _save_state(rnd + 1)

print(f"[{ALGO}-noBN] best_val_f1={best_val_f1:.4f} at round {best_round}",
      flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# evaluation: shipped selection rule (best-val checkpoint) + per-round test
# (disclosed as a controlled diagnostic; no test-based selection happens)
# ─────────────────────────────────────────────────────────────────────────────

report = {"runner": "experiments/run_nobn_control.py",
          "algorithm": ALGO,
          "protocol": {"seed": cfg.SEED, "rounds": cfg.N_ROUNDS,
                       "arch": "SolarMLP with BatchNorm1d→Identity",
                       "n_params": int(n_params),
                       "mu": cfg.MU if ALGO == "fedprox" else None},
          "history": history,
          "best_val_f1": round(float(best_val_f1), 6),
          "best_round": best_round}

# shipped selection rule: restore best-val weights, evaluate test
m = make_fresh_model(input_dim, use_lstm=False)[0]
with torch.no_grad():
    for p, w in zip(m.parameters(), best_weights):
        p.data.copy_(torch.tensor(np.asarray(w), dtype=torch.float32))
report["selected_checkpoint_test"] = {
    "round": best_round,
    **metric_block(y_test, probs(m, X_test)),
}
print(f"[{ALGO}-noBN] selected checkpoint (r{best_round}) test: "
      f"ROC={report['selected_checkpoint_test']['roc_auc']:.3f} "
      f"PR={report['selected_checkpoint_test']['pr_auc']:.3f}", flush=True)

# per-round test trajectory (diagnostic disclosure)
traj = {}
for r in sorted(round_params):
    mm = make_fresh_model(input_dim, use_lstm=False)[0]
    with torch.no_grad():
        for p, w in zip(mm.parameters(), round_params[r]):
            p.data.copy_(torch.tensor(np.asarray(w), dtype=torch.float32))
    traj[r] = metric_block(y_test, probs(mm, X_test))
report["test_trajectory"] = traj
for r, v in traj.items():
    print(f"  [{ALGO}-noBN] r{r}: test ROC={v['roc_auc']:.3f} "
          f"PR={v['pr_auc']:.3f}", flush=True)

report["elapsed_s"] = round(time.time() - T0, 1)

# merge with any sibling arm already written
OUT = os.path.join(cfg.OUTPUT_DIR, "nobn_control.json")
merged = {"runner": "experiments/run_nobn_control.py"}
if os.path.exists(OUT):
    with open(OUT) as f:
        prev = json.load(f)
    merged.update(prev)
merged[ALGO] = report
_atomic_json(merged, OUT)
print(f"\n[noBN] report -> {OUT} ({report['elapsed_s']}s)", flush=True)
