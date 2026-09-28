"""
federated_learning.py  (v3.0 — improvements branch)
────────────────────────────────────────────────────
Federated training loops: FedAvg, FedProx, SCAFFOLD.

v3.0 FIXES (audit findings B3, B5, B6, B8, B15):
  - MONITORING ON VALIDATION: every-5-round evaluation and SCAFFOLD's
    best-checkpoint tracking now run against the VALIDATION set.
    The test set is never passed into a training loop.
  - PLAIN FEDAVG IS ACTUALLY FEDAVG: both arms previously used
    distribution-aware (DA-FL) weighted aggregation, so "FedAvg vs
    FedProx" comparisons were confounded (B6). AGGREGATION_STRATEGY
    config selects 'plain' (vanilla size-weighted, default) or 'dafl'
    (ablation arm).
  - global_pos_rate is COMPUTED from the training shards (B8) instead
    of the hardcoded 0.4887.
  - client selection uses len(shards) (B15) so --clients N works.
  - run-level seed: torch.manual_seed(seed) before model init for
    multi-seed reproducibility (B11).
  - USE_SMOTE: per-client SMOTE wired into local training (B7 — it was
    dead code before).

Retained from v2.3 (documented fixes that remain correct):
  - SCAFFOLD uses SGD (control-variate math requires fixed step size)
  - SCAFFOLD std(correction=0) NaN fix for single-element tensors
  - focal alpha clamp (0.05, 0.25)
"""

import os

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional

import config as cfg
from config import (N_ROUNDS, LOCAL_EPOCHS, FRACTION_FIT, LR,
                    CLIENT_NAMES, DEFAULT_THRESHOLD,
                    USE_LSTM, USE_FED_FOCAL, FOCAL_GAMMA,
                    USE_MIXUP, MIXUP_ALPHA, EVAL_BATCH_SIZE)
from model import (SolarMLP, SolarLSTM, get_weights, set_weights,
                   clone_model, make_fresh_model, get_device, is_lstm_model)
from evaluation import compute_all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# MIXUP AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def mixup_data(X: np.ndarray, y: np.ndarray, alpha: float = 0.4,
               rng: Optional[np.random.RandomState] = None):
    """Mixup within minority (flare) class only. 30% more positive samples."""
    if not USE_MIXUP or alpha <= 0:
        return X, y
    rng = rng or np.random
    flare_idx = np.where(y == 1)[0]
    if len(flare_idx) < 2:
        return X, y
    n_aug = min(len(flare_idx), int(len(flare_idx) * 0.3))
    X_aug, y_aug = [], []
    for _ in range(n_aug):
        i, j = rng.choice(len(flare_idx), 2, replace=False)
        lam = rng.beta(alpha, alpha)
        X_aug.append(lam * X[flare_idx[i]] + (1 - lam) * X[flare_idx[j]])
        y_aug.append(1)
    if X_aug:
        X = np.vstack([X, np.array(X_aug, dtype=X.dtype)])
        y = np.concatenate([y, np.array(y_aug, dtype=y.dtype)])
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def get_criterion(device, current_round=0, total_rounds=50,
                  focal_alpha: float = 0.25, global_pos_rate=None):
    """Fed-Focal or DynamicFocalLoss. focal_alpha=0.25 is the effective max
    (FedFocalLoss clamps to (0.05, 0.25) — audit B17).
    global_pos_rate: the ACTUAL training prevalence (B13 — replaces the
    hardcoded 0.4887 that silently broke whenever the split changed)."""
    if USE_FED_FOCAL:
        from losses import FedFocalLoss
        criterion = FedFocalLoss(
            gamma=FOCAL_GAMMA,
            alpha=focal_alpha,
            reduction='mean'
        ).to(device)
        criterion.set_round_info(current_round, total_rounds)
        if global_pos_rate is not None:
            criterion.set_global_pos_rate(global_pos_rate)
        return criterion
    else:
        from losses import DynamicFocalLoss
        criterion = DynamicFocalLoss(gamma=2.0, base_alpha=0.25).to(device)
        if global_pos_rate is not None:
            criterion.set_global_pos_rate(global_pos_rate)
        return criterion


def _make_loader(X, y, rng=None):
    """v3.0.1 FastLoader: batch-level slicing instead of per-sample
    __getitem__ collation (DataLoader collation of 512 single-row tensors
    dominated round time — ~4x slowdown). Same semantics: batch 512,
    reshuffled per epoch, last partial batch kept."""
    class _FastLoader:
        def __init__(self, X, y, batch_size=512):
            self.X = torch.as_tensor(X, dtype=torch.float32)
            self.y = torch.as_tensor(y, dtype=torch.float32)
            self.batch_size = batch_size
            self.n = len(self.y)

        def __iter__(self):
            idx = torch.randperm(self.n)
            for i in range(0, self.n, self.batch_size):
                j = idx[i:i + self.batch_size]
                yield self.X[j], self.y[j]

        def __len__(self):
            return (self.n + self.batch_size - 1) // self.batch_size

    return _FastLoader(X, y)


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATION — plain (vanilla FedAvg) and DA-FL (ablation)
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_weights_plain(client_weights: list, client_sizes: list) -> list:
    """Vanilla size-weighted FedAvg aggregation (McMahan et al., 2017)."""
    total = sum(client_sizes)
    averaged = []
    for layer_idx in range(len(client_weights[0])):
        layer_avg = np.zeros_like(client_weights[0][layer_idx], dtype=np.float64)
        for w, sz in zip(client_weights, client_sizes):
            layer_avg += (sz / total) * w[layer_idx].astype(np.float64)
        averaged.append(layer_avg.astype(np.float32))
    return averaged


def aggregate_weights_dafl(client_weights: list, client_sizes: list,
                           client_labels: list,
                           global_pos_rate: float,
                           round_num: int = 0, log_interval: int = 10) -> list:
    """
    Distribution-aware FedAvg (ablation arm; NOT the default).
    global_pos_rate must be COMPUTED from the training pool (B8).
    """
    n_clients = len(client_sizes)
    total_size = sum(client_sizes)
    max_weight = 2.0 / n_clients

    da_weights = []
    should_log = (round_num == 1 or round_num % log_interval == 0)

    if should_log:
        print(f"      [DA-FL] Round {round_num} weights (cap=2/{n_clients}):")

    for i, (size, y_c) in enumerate(zip(client_sizes, client_labels)):
        size_w = size / total_size
        pos_rate = float(np.mean(y_c))
        raw_phi = pos_rate / global_pos_rate if global_pos_rate > 0 else 1.0
        phi = np.sqrt(raw_phi)
        combined = min(size_w * phi, max_weight)
        da_weights.append(combined)
        if should_log:
            print(f"        Client {i}: size_w={size_w:.3f}, phi={phi:.3f}, "
                  f"final_w={combined:.3f} (pos_rate={pos_rate:.3f})")

    total = sum(da_weights)
    if total > 0:
        da_weights = [w / total for w in da_weights]

    averaged = []
    for layer_idx in range(len(client_weights[0])):
        layer_avg = np.zeros_like(client_weights[0][layer_idx], dtype=np.float64)
        for w, dw in zip(client_weights, da_weights):
            layer_avg += dw * w[layer_idx].astype(np.float64)
        averaged.append(layer_avg.astype(np.float32))
    return averaged


def aggregate(client_weights, client_sizes, client_labels, global_pos_rate,
              strategy=None, round_num=0):
    """Dispatch on config.AGGREGATION_STRATEGY (default: plain)."""
    strategy = strategy or cfg.AGGREGATION_STRATEGY
    if strategy == "dafl":
        return aggregate_weights_dafl(client_weights, client_sizes,
                                      client_labels, global_pos_rate,
                                      round_num=round_num)
    return aggregate_weights_plain(client_weights, client_sizes)


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def _maybe_smote(X, y, seed):
    """Per-client SMOTE (wired in v3.0; was dead code — audit B7)."""
    if not getattr(cfg, "USE_SMOTE", False):
        return X, y
    from data_preparation import apply_smote
    return apply_smote(X, y, seed=seed)


def local_train_fedavg(model, X, y, epochs=LOCAL_EPOCHS,
                       current_round=0, total_rounds=50,
                       seed=None, global_pos_rate=None) -> nn.Module:
    device = next(model.parameters()).device
    model.train()
    rng = np.random.RandomState(seed if seed is not None else cfg.SEED)
    X, y = _maybe_smote(X, y, seed=(seed if seed is not None else cfg.SEED))
    X, y = mixup_data(X, y, alpha=MIXUP_ALPHA, rng=rng)
    loader = _make_loader(X, y)
    criterion = get_criterion(device, current_round, total_rounds,
                              focal_alpha=0.25,
                              global_pos_rate=global_pos_rate)
    pos_rate = float(y.mean())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=1, eta_min=LR * 0.1)
    for epoch in range(epochs):
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb, client_pos_rate=pos_rate)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        scheduler.step()
    return model


def local_train_fedprox(model, global_model, X, y, epochs=LOCAL_EPOCHS,
                        mu=cfg.MU, current_round=0, total_rounds=50,
                        seed=None, global_pos_rate=None) -> nn.Module:
    device = next(model.parameters()).device
    model.train()
    rng = np.random.RandomState(seed if seed is not None else cfg.SEED)
    X, y = _maybe_smote(X, y, seed=(seed if seed is not None else cfg.SEED))
    X, y = mixup_data(X, y, alpha=MIXUP_ALPHA, rng=rng)
    loader = _make_loader(X, y)
    criterion = get_criterion(device, current_round, total_rounds,
                              focal_alpha=0.25,
                              global_pos_rate=global_pos_rate)
    pos_rate = float(y.mean())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=1, eta_min=LR * 0.1)
    global_params = [p.data.detach().clone().to(device) for p in global_model.parameters()]
    for epoch in range(epochs):
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            focal = criterion(model(Xb), yb, client_pos_rate=pos_rate)
            prox = torch.tensor(0.0, device=device)
            for lp, gp in zip(model.parameters(), global_params):
                prox = prox + ((lp - gp) ** 2).sum()
            loss = focal + (mu / 2.0) * prox
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        scheduler.step()
    return model


def local_train_scaffold(model, X, y, c_global, c_local,
                         epochs=LOCAL_EPOCHS, current_round=0,
                         total_rounds=50, seed=None) -> Tuple[nn.Module, list]:
    """
    SCAFFOLD local training (SGD — control variate math requires it).
    Retained from v2.3 with the std(correction=0) NaN fix.
    """
    device = next(model.parameters()).device
    model.train()
    rng = np.random.RandomState(seed if seed is not None else cfg.SEED)
    X, y = _maybe_smote(X, y, seed=(seed if seed is not None else cfg.SEED))
    X, y = mixup_data(X, y, alpha=MIXUP_ALPHA, rng=rng)
    loader = _make_loader(X, y)
    criterion = get_criterion(device, current_round, total_rounds, focal_alpha=0.25)
    pos_rate = float(y.mean())

    scaffold_lr = LR * 0.5
    optimizer = torch.optim.SGD(model.parameters(), lr=scaffold_lr,
                                momentum=0.9, weight_decay=1e-4)

    initial_params = [p.data.detach().clone() for p in model.parameters()]
    n_steps = 0

    for epoch in range(epochs):
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb, client_pos_rate=pos_rate)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            with torch.no_grad():
                for i, p in enumerate(model.parameters()):
                    if p.grad is not None and i < len(c_global) and i < len(c_local):
                        corr = c_global[i].to(device) - c_local[i].to(device)
                        grad_norm = p.grad.data.norm().item()
                        corr_norm = corr.norm().item()
                        if corr_norm > 0 and grad_norm > 0:
                            corr = corr * (grad_norm / corr_norm) * 0.1
                        p.grad.data.add_(corr)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            n_steps += 1

    new_c_local = []
    with torch.no_grad():
        for i, (p_init, p_new) in enumerate(zip(initial_params, model.parameters())):
            if n_steps > 0:
                delta = (p_new.data - p_init.to(device)) / (n_steps * scaffold_lr)
            else:
                delta = torch.zeros_like(p_new.data)
            if i < len(c_local) and i < len(c_global):
                new_c = c_local[i].to(device) + delta - c_global[i].to(device)
            else:
                new_c = delta
            std = p_new.data.std(correction=0).item() if p_new.numel() > 1 \
                else p_new.data.abs().item()
            clip = max(3.0 * std, 0.1)
            new_c_local.append(torch.clamp(new_c, -clip, clip).cpu().detach())

    return model, new_c_local


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION (monitoring on VALIDATION; full metric set)
# ─────────────────────────────────────────────────────────────────────────────

def get_model_probs(model, X, device, batch_size=EVAL_BATCH_SIZE):
    """Batched inference (GPU-memory safe) -> probabilities."""
    model.eval()
    n = len(X)
    all_probs = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            Xb = torch.tensor(X[s:e], dtype=torch.float32).to(device)
            probs = torch.sigmoid(model(Xb)).cpu().numpy().flatten()
            all_probs[s:e] = probs
            del Xb
            if device.type == "cuda":
                torch.cuda.empty_cache()
    nan_c = int(np.isnan(all_probs).sum())
    inf_c = int(np.isinf(all_probs).sum())
    if nan_c > 0 or inf_c > 0:
        print(f"  [Warning] {nan_c} NaN + {inf_c} Inf probabilities -> replaced with 0.5")
        all_probs = np.nan_to_num(all_probs, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(all_probs, 1e-7, 1 - 1e-7)


def evaluate_model(model, X, y, threshold=DEFAULT_THRESHOLD,
                   batch_size=EVAL_BATCH_SIZE, beta=2.0) -> Dict:
    """Full metric set via evaluation.compute_all_metrics (fixes B4)."""
    device = next(model.parameters()).device
    probs = get_model_probs(model, X, device, batch_size=batch_size)
    return compute_all_metrics(y, probs, threshold, beta=beta)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED ROUND LOOP
# ─────────────────────────────────────────────────────────────────────────────

def _compute_global_pos_rate(shards):
    """B8: computed, never hardcoded."""
    y_all = np.concatenate([np.asarray(y) for _, y in shards])
    return float(y_all.mean())


def _run_fl_loop(shards, X_monitor, y_monitor, n_rounds, algorithm, mu=0.0,
                 use_lstm=None, eval_batch_size=EVAL_BATCH_SIZE, seed=None,
                 warm_start_model=None, resume_path=None):
    """
    Generic FL round loop. X_monitor/y_monitor MUST be the VALIDATION set
    (audit B5): monitoring metrics and any checkpoint selection run on
    validation only. The test set is never accepted here.

    resume_path: optional .pt file for round-level crash recovery. The full
    loop state (global/best weights, RNG states, control variates, history,
    next round) is saved atomically after EVERY round, so a killed process
    resumes mid-FL instead of restarting. On completion the file carries
    done=True with the final model — callers can treat its existence +
    done flag as the phase-complete signal.
    """
    seed = cfg.SEED if seed is None else seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    sample_X = shards[0][0]
    if use_lstm is None:
        use_lstm = cfg.USE_LSTM
    input_dim = sample_X.shape[2] if sample_X.ndim == 3 else sample_X.shape[1]

    if warm_start_model is not None:
        global_model = clone_model(warm_start_model)
        device = next(global_model.parameters()).device
    else:
        global_model, device = make_fresh_model(input_dim, use_lstm=use_lstm)

    global_pos_rate = _compute_global_pos_rate(shards)
    n_shards = len(shards)  # B15: authoritative client count

    history: List[Dict] = []
    c_global = [torch.zeros_like(p) for p in global_model.parameters()]
    c_locals = {i: [torch.zeros_like(p) for p in global_model.parameters()]
                for i in range(n_shards)}
    best_val_f1, best_weights = 0.0, None
    start_round = 1

    # ── mid-run recovery from a previous killed process ───────────────────
    if resume_path and os.path.exists(resume_path):
        try:
            st = torch.load(resume_path, weights_only=False)
            same = (st.get("algorithm") == algorithm and
                    st.get("mu", 0.0) == mu and st.get("seed") == seed and
                    st.get("n_rounds") == n_rounds and
                    st.get("agg") == cfg.AGGREGATION_STRATEGY and
                    st.get("smote") == cfg.USE_SMOTE and
                    st.get("focal") == cfg.USE_FED_FOCAL and
                    st.get("use_lstm", False) == bool(use_lstm))
            if same and not st.get("done"):
                set_weights(global_model, st["global_weights"])
                history = st["history"]
                best_val_f1 = st["best_val_f1"]
                best_weights = st["best_weights"]
                rng.set_state(st["rng_state"])
                np.random.set_state(st["np_rng_state"])
                torch.set_rng_state(st["torch_rng_state"])
                if algorithm == "scaffold":
                    c_global = st["c_global"]
                    c_locals = st["c_locals"]
                start_round = st["next_round"]
                print(f"  [{algorithm}] resuming from round {start_round} "
                      f"(crash recovery)")
        except Exception as e:
            print(f"  [{algorithm}] resume state unreadable ({e}) — "
                  "restarting from round 1")

    algo_label = algorithm.upper() if algorithm != "fedprox" else \
        f"FedProx (mu={mu})"
    print("=" * 60)
    print(f" {algo_label} [{device}] | {'LSTM' if use_lstm else 'MLP'} | "
          f"agg={cfg.AGGREGATION_STRATEGY} | pos_rate={global_pos_rate:.3f}")
    print(f" monitoring: VALIDATION ({len(y_monitor):,} samples)")
    print("=" * 60)

    def _save_fl_state(next_round, done=False):
        """Atomic round-level state snapshot (crash recovery)."""
        if not resume_path:
            return
        state = {
            "algorithm": algorithm, "mu": mu, "seed": seed,
            "n_rounds": n_rounds, "agg": cfg.AGGREGATION_STRATEGY,
            "smote": cfg.USE_SMOTE, "focal": cfg.USE_FED_FOCAL,
            "use_lstm": bool(use_lstm), "next_round": next_round,
            "done": done,
            "global_weights": get_weights(global_model),
            "history": history, "best_val_f1": best_val_f1,
            "best_weights": best_weights,
            "rng_state": rng.get_state(),
            "np_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
        }
        if algorithm == "scaffold":
            state["c_global"] = c_global
            state["c_locals"] = c_locals
        if done:
            state["model"] = global_model
        tmp = resume_path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, resume_path)

    for rnd in range(start_round, n_rounds + 1):
        selected = rng.choice(n_shards,
                              max(1, int(FRACTION_FIT * n_shards)),
                              replace=False)
        client_weights, client_sizes, client_labels, updated_c = [], [], [], []

        for cid in selected:
            X_c, y_c = shards[cid]
            if len(X_c) == 0:
                continue
            local = clone_model(global_model)
            if algorithm == "fedavg":
                local = local_train_fedavg(local, X_c, y_c,
                                           current_round=rnd,
                                           total_rounds=n_rounds, seed=seed,
                                           global_pos_rate=global_pos_rate)
            elif algorithm == "fedprox":
                local = local_train_fedprox(local, global_model, X_c, y_c,
                                            mu=mu, current_round=rnd,
                                            total_rounds=n_rounds, seed=seed,
                                            global_pos_rate=global_pos_rate)
            elif algorithm == "scaffold":
                local, new_c = local_train_scaffold(
                    local, X_c, y_c, c_global, c_locals[cid],
                    current_round=rnd, total_rounds=n_rounds, seed=seed)
                updated_c.append(new_c)
                c_locals[cid] = new_c
            else:
                raise ValueError(f"unknown algorithm: {algorithm}")

            client_weights.append(get_weights(local))
            client_sizes.append(len(X_c))
            client_labels.append(y_c)
            del local
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if not client_weights:
            continue

        new_weights = aggregate(client_weights, client_sizes, client_labels,
                                global_pos_rate, round_num=rnd)

        # NaN guard (SCAFFOLD historically)
        if any(np.isnan(w).any() for w in new_weights):
            print(f"  [{algorithm}] NaN at round {rnd} — reverting round")
            if best_weights is not None:
                set_weights(global_model, best_weights)
            continue

        if algorithm == "scaffold" and updated_c:
            total = sum(client_sizes)
            new_c_global = []
            for li in range(len(updated_c[0])):
                c_avg = torch.zeros_like(updated_c[0][li], dtype=torch.float32)
                for cl, sz in zip(updated_c, client_sizes):
                    c_avg += cl[li].float() * (sz / total)
                new_c_global.append(c_avg)
            c_global = new_c_global

        set_weights(global_model, new_weights)

        # ── monitoring + checkpoint selection: VALIDATION ONLY (B3/B5) ──
        if rnd % 5 == 0 or rnd == n_rounds:
            metrics = evaluate_model(global_model, X_monitor, y_monitor,
                                     batch_size=eval_batch_size)
            history.append({"round": rnd,
                            **{k: v for k, v in metrics.items() if k != "preds"}})
            print(f"  Round {rnd:>3} | val F1: {metrics['f1']:.3f} | "
                  f"val Recall: {metrics['recall']:.3f} | "
                  f"val ROC-AUC: {metrics['roc_auc']:.3f}")
            if metrics['f1'] > best_val_f1:  # B3: validation checkpointing
                best_val_f1 = metrics['f1']
                best_weights = [w.copy() for w in new_weights]

        # crash-recovery snapshot: after every completed round
        _save_fl_state(rnd + 1)

    if best_weights is not None and best_val_f1 > 0:
        set_weights(global_model, best_weights)
        print(f"\n  [{algorithm}] Restored best VALIDATION checkpoint "
              f"(val F1={best_val_f1:.3f})")

    _save_fl_state(n_rounds + 1, done=True)

    return global_model, history


def run_fedavg(shards, X_val, y_val, n_rounds=N_ROUNDS,
               use_lstm=None, eval_batch_size=EVAL_BATCH_SIZE, seed=None,
               resume_path=None):
    return _run_fl_loop(shards, X_val, y_val, n_rounds, "fedavg",
                        use_lstm=use_lstm, eval_batch_size=eval_batch_size,
                        seed=seed, resume_path=resume_path)


def run_fedprox(shards, X_val, y_val, n_rounds=N_ROUNDS, mu=cfg.MU,
                use_lstm=None, eval_batch_size=EVAL_BATCH_SIZE, seed=None,
                resume_path=None):
    return _run_fl_loop(shards, X_val, y_val, n_rounds, "fedprox", mu=mu,
                        use_lstm=use_lstm, eval_batch_size=eval_batch_size,
                        seed=seed, resume_path=resume_path)


def run_scaffold(shards, X_val, y_val, n_rounds=N_ROUNDS,
                 use_lstm=None, eval_batch_size=EVAL_BATCH_SIZE,
                 warm_start_model=None, seed=None, resume_path=None):
    if not cfg.USE_SCAFFOLD:
        print("[SCAFFOLD] Disabled in config.")
        return None, []
    return _run_fl_loop(shards, X_val, y_val, n_rounds, "scaffold",
                        use_lstm=use_lstm, eval_batch_size=eval_batch_size,
                        seed=seed, warm_start_model=warm_start_model,
                        resume_path=resume_path)
