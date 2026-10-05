#!/usr/bin/env python3
"""
experiments/run_raw_nobn.py  (v4.9.2 / Dossier R-FS9-R10, master
change register item A3 — no-BatchNorm MLP federation on the raw
substrate)
────────────────────────────────────────────────────────
Question: is the raw-substrate federated-MLP degradation
ENCODER-conditional or BatchNorm-under-federation-conditional?
The paper's only no-BN control is the in-partition one
(experiments/run_nobn_control.py, Section 6.11: identity in place
of the three BatchNorm1d layers, both algorithms stable at
0.871-0.873 test ROC-AUC). Until MLP-without-BN is federated on
the RAW substrate, "encoder-conditional" cannot be separated
from "BatchNorm-under-federation-conditional" — exactly the
Round-13 / third-party-review point 4, and the FedBN line the
paper itself cites keeps the alternative attribution alive. If
no-BN FedAvg and FedProx reach LSTM-level quality on raw
(~0.958 / ~0.970 ROC), contribution 5's attribution inverts and
must be rewritten; if they collapse like their BN counterparts,
the encoder-conditional claim finally earns its name.

Method: the frozen seed-42 raw protocol, UNCHANGED except for
the architecture — SolarMLP with its three BatchNorm1d layers
replaced by Identity (the run_nobn_control.py construction:
same Linear/ReLU/Dropout stack, same parameter families minus
gamma/beta; no running statistics exist, so weight transport is
exact and the shipped evaluation path IS the exact evaluation).
FedAvg and FedProx both (register item A3 names both; SCAFFOLD
is not part of the register's demand). Round-resumable state in
data/cache/rawsubstrate/nobn_<algo>_state.pt — separate
filenames from the BN arms' checkpoints, so the two programme
arms never collide.

Evaluation mirrors run_raw_substrate.py section 5 exactly (the
comparability requirement): prior-shift calibration fit on
validation, val-frozen F-beta threshold, frozen-FPR operating
points, single-pass test evaluation.

Checkpoint-selection hygiene (register item B1, folded in here
so the NEW arms carry it from birth): the shipped best-val-F1
selection is evaluated as-is AND (a) the selected round is
disclosed, (b) a validation-ROC-selected sensitivity row is
recorded beside it (disclosed as a diagnostic; no test-based
selection happens), and (c) a per-round test trajectory is
recorded at the monitoring cadence. The degenerate-threshold
pathology the Round-13 review caught on the BN arms cannot hide
an operating point here.

Output: outputs/raw_nobn_eval.json (merged across arms; re-run
either arm with --only).  GPU preferred (the owner's box), CPU
works (round-resumable — the same budget the ask-#4 retrain
already ran).  Deterministic given the frozen seed-42 protocol
and the cached substrate.

Usage:
    python experiments/run_raw_nobn.py            (both arms)
    python experiments/run_raw_nobn.py --only fedavg
"""
import argparse
import json
import os
import sys
import time

import numpy as np

try:
    import torch
except ImportError:          # the battery's torch-less convention:
    torch = None             # fail loudly at the gate in main(),
                               # never with a raw import traceback

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
ALGOS = ("fedavg", "fedprox")
OUT_DEFAULT = os.path.join("outputs", "raw_nobn_eval.json")
RAW_DEFAULT = "/tmp/swansf_raw"

# torch-dependent machinery, injected into module globals by
# _load_torch_stack() from main() INSIDE the torch gate (the names
# below are bound at call time, so run_nobn_arm resolves them only
# after the gate has passed)
nn = None                    # torch.nn
model_mod = None            # the model module (patched)
NoBNSolarMLP = None         # the no-BN architecture class
make_fresh_model = None
clone_model = None
get_weights = None
local_train_fedavg = None
local_train_fedprox = None
aggregate = None
evaluate_model = None


def _load_torch_stack():
    """Import the torch-dependent machinery INSIDE the gate, define
    the NoBN architecture, and patch model.SolarMLP so the entire FL
    machinery constructs no-BN models transparently (the
    run_nobn_control.py mechanism, verbatim: same Linear/ReLU/Dropout
    stack, same parameter families minus gamma/beta, no running
    statistics, so weight transport is exact)."""
    global nn, model_mod, NoBNSolarMLP, make_fresh_model, clone_model
    global get_weights, local_train_fedavg, local_train_fedprox
    global aggregate, evaluate_model
    import torch.nn as _nn
    import model as _model_mod
    from model import (SolarMLP as _SolarMLP,
                       make_fresh_model as _mfm,
                       clone_model as _cm, get_weights as _gw)
    from federated_learning import (
        local_train_fedavg as _lta, local_train_fedprox as _ltp,
        aggregate as _agg, evaluate_model as _em)

    class _NoBNSolarMLP(_SolarMLP):
        """SolarMLP with all BatchNorm1d layers replaced by Identity.

        Same Linear/ReLU/Dropout stack, same parameter families minus
        (gamma, beta); no running statistics exist, so weight
        transport is exact — there is no buffer state to drift."""
        def __init__(self, input_dim=None):
            super().__init__(input_dim=input_dim if input_dim is not None
                             else cfg.INPUT_DIM)
            for i, m in enumerate(self.net):
                if isinstance(m, _nn.BatchNorm1d):
                    self.net[i] = _nn.Identity()

    _model_mod.SolarMLP = _NoBNSolarMLP
    nn = _nn
    model_mod = _model_mod
    NoBNSolarMLP = _NoBNSolarMLP
    make_fresh_model = _mfm
    clone_model = _cm
    get_weights = _gw
    local_train_fedavg = _lta
    local_train_fedprox = _ltp
    aggregate = _agg
    evaluate_model = _em


def _atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str, allow_nan=False)
    os.replace(tmp, path)


def probs(model, X, device, batch_size=cfg.EVAL_BATCH_SIZE):
    """Batched probabilities, NaN/clip semantics identical to
    get_model_probs (device-safe; no BN layers exist, so eval mode is
    the exact path by construction)."""
    model.eval()
    n = len(X)
    out = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            Xb = torch.tensor(np.ascontiguousarray(X[s:e]),
                              dtype=torch.float32).to(device)
            out[s:e] = torch.sigmoid(model(Xb)).cpu().numpy().flatten()
            del Xb
            if device.type == "cuda":
                torch.cuda.empty_cache()
    out = np.nan_to_num(out, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(out, 1e-7, 1 - 1e-7)


def frozen_protocol_eval(model, device, X_val, y_val, X_test, y_test,
                         y_train_rate):
    """run_raw_substrate.py section 5, verbatim — the comparability
    contract with the BN raw arms."""
    p_val = probs(model, X_val, device)
    p_test = probs(model, X_test, device)
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
            "test": metrics, "frozen_operating_points": ops}


# ─────────────────────────────────────────────────────────────────────────────
# the faithful loop (mirrors run_nobn_control.py: the _run_fl_loop
# primitives — local_train_* / aggregate / evaluate_model — with
# per-round checkpoint capture and the resume discipline)
# ─────────────────────────────────────────────────────────────────────────────

def run_nobn_arm(algo, shards, X_val, y_val, X_test, y_test, device,
                 input_dim):
    STATE_PATH = os.path.join(CACHE, f"nobn_{algo}_state.pt")

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    rng = np.random.RandomState(cfg.SEED)

    global_model = make_fresh_model(input_dim, use_lstm=False)[0]
    assert isinstance(global_model, NoBNSolarMLP) and not any(
        isinstance(m, nn.BatchNorm1d) for m in global_model.modules()), \
        "no-BN patch failed"
    n_params = sum(p.numel() for p in global_model.parameters())
    print(f"[{algo}-noBN] NoBNSolarMLP | params={n_params:,} "
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
            if st.get("arch") != "nobn":
                sys.exit(f"[{algo}-noBN] {STATE_PATH} is not a no-BN "
                         f"state file (arch={st.get('arch')!r}) — "
                         f"refusing to resume a foreign checkpoint.")
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
            print(f"[{algo}-noBN] resuming from round {start_round}",
                  flush=True)
        except SystemExit:
            raise
        except Exception as e:
            print(f"[{algo}-noBN] state unreadable ({e}) — "
                  f"restarting", flush=True)

    def _save_state(next_round):
        tmp = STATE_PATH + ".tmp"
        torch.save({
            "arch": "nobn",
            "algorithm": algo,
            "mu": cfg.MU if algo == "fedprox" else 0.0,
            "seed": cfg.SEED,
            "n_rounds": cfg.N_ROUNDS,
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
                              max(1, int(cfg.FRACTION_FIT *
                                         cfg.N_CLIENTS)),
                              replace=False)
        client_weights, client_sizes, client_labels = [], [], []

        for cid in selected:
            X_c, y_c = shards[cid]
            if len(X_c) == 0:
                continue
            local = clone_model(global_model)
            if algo == "fedavg":
                local = local_train_fedavg(local, X_c, y_c,
                                           current_round=rnd,
                                           total_rounds=cfg.N_ROUNDS,
                                           seed=cfg.SEED,
                                           global_pos_rate=global_pos_rate)
            else:
                local = local_train_fedprox(local, global_model, X_c,
                                            y_c, mu=cfg.MU,
                                            current_round=rnd,
                                            total_rounds=cfg.N_ROUNDS,
                                            seed=cfg.SEED,
                                            global_pos_rate=global_pos_rate)
            client_weights.append([w.copy() for w in
                                   get_weights(local)])
            client_sizes.append(len(X_c))
            client_labels.append(y_c)
            del local

        new_weights = aggregate(client_weights, client_sizes,
                                client_labels, global_pos_rate,
                                round_num=rnd)
        if any(np.isnan(w).any() for w in new_weights):
            print(f"  [{algo}-noBN] NaN at round {rnd} — reverting "
                  f"round", flush=True)
            continue

        with torch.no_grad():
            for p, w in zip(global_model.parameters(), new_weights):
                p.data.copy_(torch.tensor(w, dtype=torch.float32))

        if rnd % 5 == 0 or rnd == cfg.N_ROUNDS:
            metrics = evaluate_model(global_model, X_val, y_val,
                                     batch_size=cfg.EVAL_BATCH_SIZE)
            history.append({"round": rnd,
                            **{k: v for k, v in metrics.items()
                               if k != "preds"}})
            print(f"  Round {rnd:>3} | val F1: {metrics['f1']:.4f} | "
                  f"val ROC: {metrics['roc_auc']:.3f}", flush=True)
            if metrics["f1"] > best_val_f1:
                best_val_f1 = metrics["f1"]
                best_round = rnd
                best_weights = [w.copy() for w in new_weights]
            round_params[rnd] = [w.copy() for w in new_weights]

        _save_state(rnd + 1)

    print(f"[{algo}-noBN] best_val_f1={best_val_f1:.4f} at round "
          f"{best_round}", flush=True)
    if best_weights is None:
        # the degenerate selection rule (every monitored round scored
        # val F1 = 0.0 — the exact pathology the Round-13 review
        # caught on the BN raw arms, where the 0.05-threshold F-beta
        # monitor cannot see an operating point at 1.31% prevalence):
        # the shipped rule's fallback is the FINAL round's weights,
        # disclosed here loudly (the _run_fl_loop precedent — it
        # restores nothing and ships the final model). The B1
        # sensitivity row and the per-round test trajectory below
        # carry the full picture regardless.
        best_round = int(max(round_params))
        best_weights = [w.copy() for w in round_params[best_round]]
        print(f"[{algo}-noBN] DEGENERATE selection rule disclosed: "
              f"val F1 was 0.0 at every monitored round; the shipped "
              f"fallback is the FINAL round {best_round}", flush=True)

    # ── selection + the B1 hygiene block ───────────────────────────────
    # shipped rule: best-val-F1 checkpoint; sensitivity row: the
    # validation-ROC-selected round (both disclosed; NO test-based
    # selection happens anywhere in this runner)
    y_train_pos = global_pos_rate
    val_roc_by_round = {h["round"]: float(h["roc_auc"])
                        for h in history}
    roc_round = max(val_roc_by_round, key=val_roc_by_round)

    def _eval_checkpoint(weights, tag):
        m = make_fresh_model(input_dim, use_lstm=False)[0]
        with torch.no_grad():
            for p, w in zip(m.parameters(), weights):
                p.data.copy_(torch.tensor(np.asarray(w),
                                          dtype=torch.float32))
        blk = frozen_protocol_eval(m, device, X_val, y_val, X_test,
                                   y_test, float(y_train_pos))
        blk["selection_rule"] = tag
        return blk

    selected_blk = _eval_checkpoint(best_weights,
                                    "best_val_f1 (shipped rule)")
    roc_blk = _eval_checkpoint(round_params[roc_round],
                               f"best_val_roc (sensitivity row, "
                               f"B1 hygiene)")

    # per-round test trajectory (disclosed diagnostic)
    traj = {}
    for r in sorted(round_params):
        m = make_fresh_model(input_dim, use_lstm=False)[0]
        with torch.no_grad():
            for p, w in zip(m.parameters(), round_params[r]):
                p.data.copy_(torch.tensor(np.asarray(w),
                                          dtype=torch.float32))
        p_test = probs(m, X_test, device)
        mm = compute_all_metrics(y_test, p_test, cfg.DEFAULT_THRESHOLD,
                                 beta=cfg.FBETA_BETA)
        traj[str(r)] = {"roc_auc": round(float(mm["roc_auc"]), 6),
                        "pr_auc": round(float(mm["pr_auc"]), 6)}
        print(f"  [{algo}-noBN] r{r}: test ROC={mm['roc_auc']:.3f} "
              f"PR={mm['pr_auc']:.3f}", flush=True)

    return {
        "runner": "experiments/run_raw_nobn.py",
        "algorithm": algo,
        "protocol": {"seed": cfg.SEED, "rounds": cfg.N_ROUNDS,
                     "arch": "SolarMLP with BatchNorm1d->Identity",
                     "n_params": int(n_params),
                     "mu": cfg.MU if algo == "fedprox" else 0.0,
                     "alpha": 1.0, "clients": cfg.N_CLIENTS,
                     "calibration": cfg.CALIBRATION_METHOD,
                     "note": ("frozen raw protocol, architecture-only "
                              "intervention; state file "
                              f"nobn_{algo}_state.pt carries "
                              "arch='nobn'")},
        "history": history,
        "best_val_f1": round(float(best_val_f1), 6),
        "best_round": best_round,
        "selected_checkpoint_test": selected_blk,
        "sensitivity_val_roc_selected_test": {
            "round": int(roc_round),
            **roc_blk,
        },
        "selection_disclosure": (
            f"shipped best-val-F1 rule selected round {best_round} "
            f"(val F1 {best_val_f1:.4f}); the val-ROC-selected "
            f"sensitivity row selects round {roc_round} — both "
            f"disclosed, neither test-based (register item B1 "
            f"hygiene, born with these arms)"),
        "test_trajectory": traj,
    }


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def _reference_block():
    """The comparison columns, READ from the committed artefacts at run
    time — never hand-typed (the v4.0 convention)."""
    out = {}
    try:
        d = json.load(open(os.path.join("outputs",
                                        "raw_substrate_eval.json")))
        for k in ("fedavg_mlp", "fedprox_mlp", "centralized_mlp"):
            t = d["results"].get(k, {}).get("test", {})
            if t:
                out[k] = {"roc_auc": float(t["roc_auc"]),
                          "pr_auc": float(t["pr_auc"])}
    except Exception:
        pass
    try:
        d = json.load(open(os.path.join("outputs",
                                        "raw_lstm_eval.json")))
        for k in ("fedavg_lstm", "fedprox_lstm", "central_lstm"):
            t = d["results"].get(k, {}).get("test", {})
            if t:
                out[k] = {"roc_auc": float(t["roc_auc"]),
                          "pr_auc": float(t["pr_auc"])}
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=RAW_DEFAULT,
                    help="raw benchmark dir (substrate build cache miss)")
    ap.add_argument("--output", default=OUT_DEFAULT)
    ap.add_argument("--only", choices=ALGOS, default=None,
                    help="run a single arm (default: both)")
    args = ap.parse_args()

    t0 = time.time()
    if torch is None:
        sys.exit("[nobn] torch is required (federated training); the "
                 "torch-less battery guards cover this runner "
                 "statically via tests/test_raw_bn_diagnostic.py")

    cfg.USE_LSTM = False
    cfg.USE_SCAFFOLD = True

    _load_torch_stack()

    # ── 1. substrate + shards (identical recipe, from cache) ───────────
    d = build(args.raw_dir, CACHE)
    X_train, y_train = d["X_train"], d["y_train"]
    X_val, y_val = d["X_val"], d["y_val"]
    X_test, y_test = d["X_test"], d["y_test"]
    print(f"[nobn] substrate: train {len(y_train):,} "
          f"(pos {y_train.mean():.2%}) | val {len(y_val):,} "
          f"(pos {y_val.mean():.2%}) | test {len(y_test):,} "
          f"(pos {y_test.mean():.2%})", flush=True)

    shards = partition_data_dirichlet(
        X_train, y_train, alpha=1.0, n_clients=cfg.N_CLIENTS,
        seed=cfg.SEED, min_samples=cfg.MIN_SAMPLES_PER_CLIENT)
    print(f"[nobn] shards rebuilt (Dirichlet alpha=1.0, seed "
          f"{cfg.SEED}): sizes {[len(s[1]) for s in shards]}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]

    algos = (args.only,) if args.only else ALGOS
    reports = {}
    for algo in algos:
        print(f"\n[nobn] {algo} (no-BN, {cfg.N_ROUNDS} rounds, frozen "
              f"protocol) ...", flush=True)
        reports[algo] = run_nobn_arm(algo, shards, X_val, y_val, X_test,
                                     y_test, device, input_dim)

    merged = {"runner": "experiments/run_raw_nobn.py"}
    if os.path.exists(args.output):
        with open(args.output) as f:
            prev = json.load(f)
        merged.update(prev)
    for algo, rep in reports.items():
        merged[algo] = rep
    merged["reference_columns"] = _reference_block()
    merged["purpose"] = (
        "R-FS9-R10 master change register item A3: no-BatchNorm MLP "
        "federation (FedAvg and FedProx) on the raw substrate — the "
        "encoder-conditional vs BatchNorm-under-federation-conditional "
        "attribution control. Architecture-only intervention (the "
        "run_nobn_control.py construction: BatchNorm1d -> Identity); "
        "evaluation identical to run_raw_substrate.py section 5 for "
        "direct comparability with the BN raw arms. The "
        "reference_columns block is read from the committed artefacts "
        "at run time — the attribution question is answerable from "
        "this artefact alone.")
    merged["elapsed_s"] = time.time() - t0
    _atomic_json(merged, args.output)
    print(f"\n[nobn] report -> {args.output} "
          f"({time.time() - t0:.0f}s total)")
    for algo, rep in reports.items():
        t = rep["selected_checkpoint_test"]["test"]
        print(f"[nobn] {algo}: selected r{rep['best_round']} "
              f"ROC {t['roc_auc']:.3f} PR {t['pr_auc']:.3f}")
    print("[nobn] done.")


if __name__ == "__main__":
    main()
