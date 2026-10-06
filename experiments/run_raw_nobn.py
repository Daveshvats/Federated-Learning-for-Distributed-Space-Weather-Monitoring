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

v4.9.3 errata (RUNLOG ask #9, first execution 2026-10-06): the
val-ROC sensitivity-row selection crashed AFTER the fedavg arm
had already completed all 50 rounds —
max(val_roc_by_round, key=val_roc_by_round) passed the dict
ITSELF as the key function (TypeError: 'dict' object is not
callable).  Fixed and extracted to the module-level
torch-free select_round_by_val_roc() so the battery exercises
the real selection semantics (a dynamic regression guard in
tests/test_raw_bn_diagnostic.py, not just a string pin).  The
round-resumable state file means the completed fedavg arm is
NOT lost: the re-run resumes at round 51 and goes straight to
evaluation (fedprox then trains from its absent state file).

Output: outputs/raw_nobn_eval.json (merged across arms; re-run
either arm with --only).  GPU preferred (the owner's box), CPU
works (round-resumable — the same budget the ask-#4 retrain
already ran).  Deterministic given the frozen seed-42 protocol
and the cached substrate.

Usage:
    python experiments/run_raw_nobn.py            (both arms)
    python experiments/run_raw_nobn.py --only fedavg

v4.12 external-review items 2/3 (the owner-side run kit):
    python experiments/run_raw_nobn.py --region-disjoint
        Both arms re-run with a WITHIN-FOLD REGION-DISJOINT validation
carve (item 2): whole NOAA active regions move to validation, so no
region appears on both sides of the train/validation boundary — the
leakage class the provenance audit documents benchmark-wide, and the
leading candidate for the val-up/test-down ROC divergence that
prevalence cannot explain. Output:
outputs/raw_nobn_region_disjoint.json; round state in
nobnrd_<algo>_state.pt (never collides with the seed-42 arms).
    python experiments/run_raw_nobn.py --seed 43
        Both arms re-run at seed 43 (item 3): model initialisation and
the Dirichlet shard draw reseeded, the frozen RANDOM validation carve
kept identical (the seed-replication semantics of the LSTM arms,
Section 6.6). Output: outputs/raw_nobn_eval_seed43.json; round state
in nobns43_<algo>_state.pt. Refuses to build a fresh substrate under
a non-default seed (the cache must exist) so the carve stays frozen.

v4.12.1 errata (the first owner-side execution, 2026-10-06, surfaced
two latent defects the synthetic smoke test could not catch):
(a) the region map was read from the SAMPLED audit meta
    provenance/train_meta_slim.csv.gz — 97,764 rows spanning
    partitions 1..5, its pool_row indexing the slim file's own order —
    so a real run would have aborted at the coverage gate (97,764 of
    255,820 pool rows) even with the raw dir present. The region
    source is now the parse metadata's `ar` column: the SAME
    p{p}_meta.csv files that supply the labels, full pool coverage by
    construction (the slim meta stays an audit artefact, never a
    split input).
(b) the raw-dir default is a POSIX path (/tmp/swansf_raw); on the
    owner's Windows box it resolves against the current drive and the
    failure was a bare FileNotFoundError. The split now pre-flights
    all four p{1..4}_meta.csv files and exits with a guided message
    naming --raw-dir and the regeneration command
    (provenance/swansf_parse_partition.py --meta-only). The seed-43
    state namespace is nobns43_<algo>_state.pt (this docstring and
    the run card previously wrote nobn_s43_; the code was always
    nobns43_). The seed-43 kit needed no code change: its first owner
    execution reached round-1 training and was interrupted
    (KeyboardInterrupt by the owner), not failed.
"""
import argparse
import csv
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
from experiments.raw_substrate import (                     # noqa: E402
    build, CACHE, TRAIN_PARTS, load_labels)

FPR_TARGETS = (0.005, 0.01, 0.02, 0.05)
ALGOS = ("fedavg", "fedprox")
OUT_DEFAULT = os.path.join("outputs", "raw_nobn_eval.json")
OUT_REGION_DISJOINT = os.path.join("outputs",
                                   "raw_nobn_region_disjoint.json")
RAW_DEFAULT = "/tmp/swansf_raw"
# v4.12.1: no META_PATH any more — the region map comes from the parse
# metadata (the `ar` column of the same p{p}_meta.csv files that
# supply the labels), NOT from the sampled audit meta. The slim meta
# (provenance/train_meta_slim.csv.gz) is an audit artefact: 97,764
# sampled rows across partitions 1..5, its pool_row indexing its own
# order — it can never be a full-pool region map (255,820 rows).

# v4.12: namespaces the round state files so the region-disjoint and
# seed-43 re-runs never collide with (or resume) the seed-42 random-
# carve arms' checkpoints; the empty string is the historical v4.9.4
# namespace.
STATE_TAG = ""

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
# ── the selection helper (v4.9.3: module-level and torch-free so ──
# ── the battery exercises the REAL semantics; the inline       ──
# ── original crashed the owner's ask-#9 run after 50 rounds)   ──

def select_round_by_val_roc(history):
    """The validation-ROC-selected round (the B1 sensitivity row).

    v4.9.3 errata: the inline original was
        roc_round = max(val_roc_by_round, key=val_roc_by_round)
    which passes the DICT itself as the key function — TypeError:
    'dict' object is not callable — raised only AFTER a full
    50-round arm had trained (RUNLOG ask #9, 2026-10-06).  The
    dict's bound .get is the callable; ties resolve to the
    EARLIEST round (dict insertion order = monitored-round
    order), deterministically.  Module-level and torch-free so
    tests/test_raw_bn_diagnostic.py exercises it directly."""
    if not history:
        raise ValueError("select_round_by_val_roc: empty history — "
                         "no monitored rounds exist (the frozen "
                         "protocol monitors every 5th round plus "
                         "the final one)")
    val_roc_by_round = {h["round"]: float(h["roc_auc"])
                        for h in history}
    return max(val_roc_by_round, key=val_roc_by_round.get)


# the faithful loop (mirrors run_nobn_control.py: the _run_fl_loop
# primitives — local_train_* / aggregate / evaluate_model — with
# per-round checkpoint capture and the resume discipline)
# ─────────────────────────────────────────────────────────────────────────────

def run_nobn_arm(algo, shards, X_val, y_val, X_test, y_test, device,
                 input_dim):
    STATE_PATH = os.path.join(
        CACHE, f"nobn{STATE_TAG}_{algo}_state.pt")

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
            if st.get("arch") != "nobn" or \
                    st.get("state_tag", "") != STATE_TAG or \
                    st.get("seed") != cfg.SEED:
                sys.exit(f"[{algo}-noBN] {STATE_PATH} does not belong "
                         f"to this run's namespace (arch={st.get('arch')!r}, "
                         f"tag={st.get('state_tag')!r}, "
                         f"seed={st.get('seed')!r} vs "
                         f"({STATE_TAG!r}, {cfg.SEED})) — refusing to "
                         f"resume a foreign checkpoint.")
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
            "state_tag": STATE_TAG,
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
    roc_round = select_round_by_val_roc(history)   # v4.9.3 fix

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
                     "validation": ("region-disjoint (whole NOAA "
                                     "active regions; v4.12 item 2)"
                                     if STATE_TAG == "rd" else
                                     ("random stratified carve, frozen "
                                      "seed-42 recipe"
                                      + (f", seed {cfg.SEED} re-init "
                                         "and shard draw (v4.12 item 3)"
                                         if STATE_TAG else ""))),
                     "note": ("frozen raw protocol, architecture-only "
                              "intervention; state file "
                              f"nobn{STATE_TAG}_{algo}_state.pt carries "
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


# ── v4.12 item 2: the within-fold region-disjoint validation carve ──
# ── (torch-free so the battery can exercise the split directly)  ──
# ── v4.12.1: the region map comes from the parse metadata (`ar`),  ──
# ── not the sampled audit meta — see the errata in the docstring.  ──

def _guided_raw_exit(raw_dir, missing):
    """v4.12.1: a missing raw parse metadata file is a GUIDED exit,
    not a bare FileNotFoundError (the owner's first execution hit the
    POSIX default /tmp/swansf_raw on Windows and got a traceback)."""
    files = ", ".join(os.path.basename(m) for m in missing)
    sys.exit(
        f"[rd] region-disjoint mode needs the raw parse metadata "
        f"(p1..p4_meta.csv); missing: {files}\n"
        f"       looked in raw_dir = {raw_dir!r} (a POSIX-style path "
        f"resolves against the current drive on Windows)\n"
        f"       fix: pass --raw-dir <the directory containing "
        f"p1..p4_meta.csv>\n"
        f"       files gone? regenerate from the public benchmark "
        f"(Harvard Dataverse, doi:10.7910/DVN/EBCFKM):\n"
        f"         python provenance/swansf_parse_partition.py "
        f"<partition_dir> <out_prefix> --meta-only\n"
        f"       (the substrate cache provides X; the npz is not needed "
        f"— full recipe in docs/RUN_CARD_v4.12.md)")


def _load_region_ids(raw_dir, p):
    """NOAA active-region ids for partition p in pool row order — the
    `ar` column of the parse metadata. v4.12.1: this is the region
    source (the same files that supply the labels, so pool coverage
    is complete by construction; the slim audit meta is NOT a split
    input)."""
    meta = os.path.join(raw_dir, f"p{p}_meta.csv")
    if not os.path.exists(meta):
        _guided_raw_exit(raw_dir, [meta])
    ar = []
    with open(meta) as f:
        rdr = csv.DictReader(f)
        if "ar" not in (rdr.fieldnames or []):
            sys.exit(f"[rd] {meta} has no 'ar' column — the parse "
                     f"metadata predates the current schema; "
                     f"regenerate it with provenance/"
                     f"swansf_parse_partition.py")
        for row in rdr:
            ar.append(int(row["ar"]))
    return np.asarray(ar, dtype=np.int64)


def region_disjoint_split(X_train, y_train, X_val, y_val, raw_dir,
                          seed=42, val_split=None):
    """Rebuild the training POOL, then carve validation at the level of
    WHOLE NOAA active regions.

    The shipped random carve (raw_substrate.py: train_test_split,
    stratified by label) can share active regions between training
    and validation — the leakage class the provenance audit
    documents benchmark-wide, and the leading candidate for the
    FedAvg no-BN val-up/test-down ROC divergence (a rank metric that
    prevalence cannot move). This function:
      1. rebuilds y_pool in POOL order (P1..P4 row order) from the
         raw parse metadata (cheap CSV reads);
      2. re-derives the frozen random carve (same seed / test_size /
         stratify) and VERIFIES it by a label round-trip against the
         cached arrays — a wrong reconstruction exits loudly;
      3. scatters the cached X_train/X_val back into pool order;
      4. maps every pool row to its NOAA active region (the parse
         metadata's `ar` column — v4.12.1: the same files that
         supplied the labels, full coverage by construction);
      5. assigns whole regions to validation (deterministic seed-42
         region permutation, target = the random carve's size)
         until no region spans the boundary.
    Returns (X_tr, y_tr, X_va, y_va, split_stats).
    """
    from sklearn.model_selection import train_test_split

    # v4.12.1: pre-flight the four parse metadata files BEFORE any
    # work — a missing raw dir is a guided exit, never a traceback
    missing = [os.path.join(raw_dir, f"p{p}_meta.csv")
               for p in TRAIN_PARTS
               if not os.path.exists(os.path.join(raw_dir,
                                                  f"p{p}_meta.csv"))]
    if missing:
        _guided_raw_exit(raw_dir, missing)

    if val_split is None:
        val_split = cfg.VAL_SPLIT
    n_pool = int(len(y_train) + len(y_val))

    # 1. pool-order labels from the raw parse metadata
    y_pool = np.concatenate(
        [np.asarray(load_labels(raw_dir, p)) for p in TRAIN_PARTS])
    if len(y_pool) != n_pool:
        sys.exit(f"[rd] pool size mismatch: meta labels {len(y_pool):,} "
                 f"vs cached train+val {n_pool:,} — the cache and the "
                 f"raw dir disagree; aborting rather than mis-splitting")

    # 2. re-derive the frozen random carve + verify by round-trip
    val_idx = train_test_split(
        np.arange(n_pool), test_size=val_split,
        random_state=seed, stratify=y_pool)[1]
    val_set = set(val_idx.tolist())
    train_idx = np.array([i for i in range(n_pool)
                          if i not in val_set])
    if not (np.array_equal(y_pool[train_idx], y_train)
            and np.array_equal(y_pool[val_idx], y_val)):
        sys.exit("[rd] carve round-trip FAILED: the re-derived random "
                 "split does not reproduce the cached y_train/y_val — "
                 "your cache was built under a different carve; aborting "
                 "rather than mis-splitting")

    # 3. scatter the cached arrays back into pool order
    X_pool = np.empty((n_pool, X_train.shape[1]), dtype=np.float32)
    X_pool[train_idx] = X_train
    X_pool[val_idx] = X_val

    # 4. region map (pool_row -> NOAA active region), v4.12.1: from the
    #    SAME parse metadata that supplied the labels — full pool
    #    coverage by construction (the v4.12 code read the sampled
    #    audit meta here and would have aborted at 97,764 of 255,820
    #    pool rows; the slim meta is an audit artefact, not a split
    #    input)
    region_of = np.concatenate(
        [_load_region_ids(raw_dir, p) for p in TRAIN_PARTS])
    if len(region_of) != n_pool:
        sys.exit(f"[rd] region rows {len(region_of):,} vs pool rows "
                 f"{n_pool:,} — the parse metadata and the cache "
                 f"disagree; aborting rather than mis-splitting")
    n_bad_ar = int((region_of < 0).sum())
    if n_bad_ar:
        sys.exit(f"[rd] {n_bad_ar:,} pool rows carry ar = -1 "
                 f"(unparsed instance filenames) — those rows cannot "
                 f"be region-assigned; aborting rather than "
                 f"mis-splitting")

    # 5. deterministic whole-region assignment
    regions = np.unique(region_of)
    rows_by_region = {int(r): np.where(region_of == r)[0]
                      for r in regions}
    target = len(val_idx)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(regions))
    val_regions, n_val = [], 0
    for oi in order:
        rid = int(regions[oi])
        if n_val >= target:
            break
        val_regions.append(rid)
        n_val += len(rows_by_region[rid])
    val_mask = np.isin(region_of, np.asarray(val_regions,
                                              dtype=np.int64))
    train_mask = ~val_mask

    # 6. hard disjointness guarantee
    assert set(region_of[val_mask].tolist()).isdisjoint(
        set(region_of[train_mask].tolist())), \
        "region-disjoint split violated its own invariant"

    split_stats = {
        "mode": "region-disjoint (whole NOAA active regions)",
        "region_source": "parse metadata `ar` column (p1..p4_meta.csv; "
                         "v4.12.1 — full pool coverage by construction)",
        "seed_region_permutation": seed,
        "n_regions_total": int(len(regions)),
        "n_regions_val": len(val_regions),
        "val_regions_sample": sorted(val_regions)[:20],
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "val_target_rows_random_carve": target,
        "train_pos": float(y_pool[train_mask].mean()),
        "val_pos": float(y_pool[val_mask].mean()),
        "random_carve_val_pos": float(y_val.mean()),
        "disjointness": "verified: no region id on both sides",
    }
    return (X_pool[train_mask], y_pool[train_mask],
            X_pool[val_mask], y_pool[val_mask], split_stats)


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
    ap.add_argument("--output", default=None,
                    help="output json (default: raw_nobn_eval.json / "
                         "raw_nobn_region_disjoint.json / "
                         "raw_nobn_eval_seed<N>.json by mode)")
    ap.add_argument("--only", choices=ALGOS, default=None,
                    help="run a single arm (default: both)")
    ap.add_argument("--region-disjoint", action="store_true",
                    help="v4.12 item 2: region-disjoint validation "
                         "carve (whole NOAA active regions)")
    ap.add_argument("--seed", type=int, default=None,
                    help="v4.12 item 3: reseed model init + Dirichlet "
                         "shard draw (the frozen random carve is kept "
                         "identical; requires the substrate cache)")
    args = ap.parse_args()

    global STATE_TAG
    t0 = time.time()
    if torch is None:
        sys.exit("[nobn] torch is required (federated training); the "
                 "torch-less battery guards cover this runner "
                 "statically via tests/test_raw_bn_diagnostic.py")

    cfg.USE_LSTM = False
    cfg.USE_SCAFFOLD = True

    if args.seed is not None and args.seed != cfg.SEED:
        # the seed replication must ride the SAME frozen carve, so
        # the substrate cache must already exist — refuse to build a
        # new substrate under a non-default seed
        if not os.path.exists(os.path.join(CACHE, "data.npz")):
            sys.exit(f"[nobn] --seed {args.seed} requires the frozen "
                     f"substrate cache (data/cache/rawsubstrate/"
                     f"data.npz) — build it at seed {cfg.SEED} first")
        STATE_TAG = f"s{args.seed}"
        print(f"[nobn] seed override: {cfg.SEED} -> {args.seed} "
              f"(init + shard draw reseeded; validation carve "
              f"unchanged; state namespace nobn{STATE_TAG}_*)",
              flush=True)
        cfg.SEED = args.seed
    elif args.region_disjoint:
        STATE_TAG = "rd"

    if args.output is None:
        if STATE_TAG == "rd":
            args.output = OUT_REGION_DISJOINT
        elif STATE_TAG:
            args.output = os.path.join(
                "outputs", f"raw_nobn_eval_seed{cfg.SEED}.json")
        else:
            args.output = OUT_DEFAULT

    _load_torch_stack()

    # ── 1. substrate + shards (identical recipe, from cache) ───────────
    d = build(args.raw_dir, CACHE)
    X_train, y_train = d["X_train"], d["y_train"]
    X_val, y_val = d["X_val"], d["y_val"]
    X_test, y_test = d["X_test"], d["y_test"]

    split_stats = None
    if args.region_disjoint:
        X_train, y_train, X_val, y_val, split_stats = \
            region_disjoint_split(
                d["X_train"], d["y_train"], d["X_val"], d["y_val"],
                raw_dir=args.raw_dir, seed=42)
        print(f"[nobn] REGION-DISJOINT carve: "
              f"{split_stats['n_regions_val']} of "
              f"{split_stats['n_regions_total']} regions -> val "
              f"({split_stats['val_rows']:,} rows, "
              f"pos {split_stats['val_pos']:.2%}; random carve was "
              f"{split_stats['val_target_rows_random_carve']:,} rows "
              f"at {split_stats['random_carve_val_pos']:.2%}); "
              f"train {split_stats['train_rows']:,} rows "
              f"(pos {split_stats['train_pos']:.2%})", flush=True)

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
    if split_stats is not None:
        merged["validation_split"] = split_stats
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
        "this artefact alone."
        + (" v4.12 item 2: region-disjoint validation carve — the "
           "divergence diagnosis of Section 6.4 (val ROC rising "
           "while test ROC falls cannot be a prevalence effect; the "
           "two candidate mechanisms, a region-sharing carve and "
           "partition-5 recency, are separated by this run)."
           if split_stats is not None else ""))
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
