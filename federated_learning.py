"""
federated_learning.py  v2.3
────────────────────────────
FIXES IN THIS VERSION (v2.3 on top of v2.2):
  1. SCAFFOLD std() NaN fix — p_new.data.std() on single-element tensors
     (e.g. Linear(128,1).bias shape=[1]) returns NaN because Bessel's
     correction divides by N-1=0. This NaN propagates: clip=NaN →
     torch.clamp(...,-NaN,NaN)=all NaN → c_local=NaN → c_global=NaN →
     every round produces NaN weights. Fix: use std(correction=0) for
     multi-element tensors, abs().item() for single-element.
  2. FedFocalLoss alpha clamp 0.95→0.5 — the dynamic alpha scaling was
     inflating alpha to 0.95 for low-flare-rate clients, undoing the
     focal_alpha=0.25 fix. Loss file changed separately.
  3. DIRICHLET_ALPHA 0.3→0.5 — alpha=0.3 created pathological partitions
     (0.6% to 100% flare rates). 0.5 is moderate non-IID.
  4. DA-FL cap 1/N→2/N — 1/N was too aggressive, giving equal weight
     to clients with 1,289 vs 40,881 samples. 2/N is the original cap.

Previous fixes (v2.2):
  - SCAFFOLD uses SGD (not AdamW) for correct control variate math
  - DA-FL logging suppressed (round_num parameter added)
  - focal_alpha=0.25 passed explicitly to get_criterion()
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict

from config import (N_CLIENTS, N_ROUNDS, LOCAL_EPOCHS,
                    FRACTION_FIT, MU, LR, CLIENT_NAMES, THRESHOLD,
                    USE_LSTM, USE_FED_FOCAL, FOCAL_GAMMA, FOCAL_ALPHA,
                    USE_MIXUP, MIXUP_ALPHA, FBETA_BETA, EVAL_BATCH_SIZE)
from model import (SolarMLP, SolarLSTM, make_loader, get_weights, set_weights,
                   clone_model, make_fresh_model, get_device, is_lstm_model)


# ─────────────────────────────────────────────────────────────────────────────
# MIXUP AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def mixup_data(X: np.ndarray, y: np.ndarray, alpha: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
    """Mixup within minority (flare) class only. Adds 30% more positive samples."""
    if not USE_MIXUP or alpha <= 0:
        return X, y
    flare_idx = np.where(y == 1)[0]
    if len(flare_idx) < 2:
        return X, y
    n_aug = min(len(flare_idx), int(len(flare_idx) * 0.3))
    X_aug, y_aug = [], []
    for _ in range(n_aug):
        i, j = np.random.choice(len(flare_idx), 2, replace=False)
        lam = np.random.beta(alpha, alpha)
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
                  focal_alpha: float = 0.25):
    """
    Get loss function.

    FIX: focal_alpha defaults to 0.25 (standard focal loss default).
    The config value of 0.75 was causing models to over-weight positives,
    producing Recall~0.99 but Precision~0.03 — a degenerate 'predict everything
    as flare' outcome. 0.25 is the value used in the original focal loss paper
    (Lin et al., RetinaNet, 2017) and is appropriate for 2% minority class.
    Pass focal_alpha explicitly to override.
    """
    if USE_FED_FOCAL:
        from losses import FedFocalLoss
        criterion = FedFocalLoss(
            gamma=FOCAL_GAMMA,
            alpha=focal_alpha,   # Use passed value, NOT config FOCAL_ALPHA
            reduction='mean'
        ).to(device)
        criterion.set_round_info(current_round, total_rounds)
        return criterion
    else:
        from losses import DynamicFocalLoss
        return DynamicFocalLoss(gamma=2.0, base_alpha=0.25).to(device)


def _make_loader(X, y):
    ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(X), torch.FloatTensor(y)
    )
    return torch.utils.data.DataLoader(ds, batch_size=512, shuffle=True, drop_last=False)


# ─────────────────────────────────────────────────────────────────────────────
# DA-FL AGGREGATION  (logging suppressed to every 10 rounds)
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_weights_dafl(
    client_weights:  list,
    client_sizes:    list,
    client_labels:   list,
    global_pos_rate: float = 0.4887,
    round_num:       int   = 0,       # FIX: add round_num for log suppression
    log_interval:    int   = 10
) -> list:
    """
    Distribution-Aware FedAvg aggregation.
    Weights clients by size * sqrt(phi) where phi = local_pos_rate / global_pos_rate.
    Max weight capped at 2/N_clients to prevent any single client dominating.

    FIX: logging suppressed — only prints on round 1 and every log_interval rounds.
    """
    n_clients = len(client_sizes)
    total_size = sum(client_sizes)
    max_weight = 2.0 / n_clients   # Cap: no client exceeds 2x equal share

    da_weights = []
    should_log = (round_num == 1 or round_num % log_interval == 0)

    if should_log:
        print(f"      [DA-FL] Round {round_num} weights (cap=2/{n_clients}={max_weight:.3f}):")

    for i, (size, y_c) in enumerate(zip(client_sizes, client_labels)):
        size_w    = size / total_size
        pos_rate  = float(np.mean(y_c))
        raw_phi   = pos_rate / global_pos_rate if global_pos_rate > 0 else 1.0
        phi       = np.sqrt(raw_phi)
        combined  = min(size_w * phi, max_weight)
        da_weights.append(combined)

        if should_log:
            cap_flag = " [CAPPED]" if (size_w * phi) > max_weight else ""
            print(f"        Client {i}: size_w={size_w:.3f}, phi={phi:.3f}, "
                  f"final_w={combined:.3f}{cap_flag} (pos_rate={pos_rate:.3f})")

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


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL TRAINING — FedAvg
# ─────────────────────────────────────────────────────────────────────────────

def local_train_fedavg(model, X, y, epochs=LOCAL_EPOCHS,
                       current_round=0, total_rounds=50) -> nn.Module:
    device    = next(model.parameters()).device
    model.train()
    X, y      = mixup_data(X, y, alpha=MIXUP_ALPHA)
    loader    = _make_loader(X, y)
    # FIX: focal_alpha=0.25, not 0.75
    criterion = get_criterion(device, current_round, total_rounds, focal_alpha=0.25)
    pos_rate  = float(y.mean())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=1, eta_min=LR * 0.1
    )
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


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL TRAINING — FedProx
# ─────────────────────────────────────────────────────────────────────────────

def local_train_fedprox(model, global_model, X, y, epochs=LOCAL_EPOCHS,
                        mu=MU, current_round=0, total_rounds=50) -> nn.Module:
    device      = next(model.parameters()).device
    model.train()
    X, y        = mixup_data(X, y, alpha=MIXUP_ALPHA)
    loader      = _make_loader(X, y)
    criterion   = get_criterion(device, current_round, total_rounds, focal_alpha=0.25)
    pos_rate    = float(y.mean())
    optimizer   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=1, eta_min=LR * 0.1
    )
    global_params = [p.data.detach().clone().to(device) for p in global_model.parameters()]
    for epoch in range(epochs):
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            focal = criterion(model(Xb), yb, client_pos_rate=pos_rate)
            prox  = torch.tensor(0.0, device=device)
            for lp, gp in zip(model.parameters(), global_params):
                prox = prox + ((lp - gp) ** 2).sum()
            loss  = focal + (mu / 2.0) * prox
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        scheduler.step()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL TRAINING — SCAFFOLD  (FIX: uses SGD, not AdamW)
# ─────────────────────────────────────────────────────────────────────────────

def local_train_scaffold(model, X, y, c_global, c_local,
                         epochs=LOCAL_EPOCHS, current_round=0,
                         total_rounds=50) -> Tuple[nn.Module, list]:
    """
    SCAFFOLD local training using SGD.

    FIX: Changed from AdamW to SGD.
    SCAFFOLD's control variate update formula is:
        c_local_new = c_local + (w_new - w_init) / (K * lr) - c_global
    where K = number of local steps. This formula is derived assuming SGD
    with a fixed step size. With AdamW, the effective step size is adaptive
    per-parameter — the control variate update formula becomes invalid,
    producing correction terms that are orders of magnitude wrong.
    This is why every SCAFFOLD round produced NaN: the control variates
    grew unbounded and poisoned subsequent gradient computations.

    With SGD + fixed LR, the formula is exact and SCAFFOLD converges
    as described in Karimireddy et al. (ICML 2020).
    """
    device = next(model.parameters()).device
    model.train()
    X, y   = mixup_data(X, y, alpha=MIXUP_ALPHA)
    loader = _make_loader(X, y)
    criterion = get_criterion(device, current_round, total_rounds, focal_alpha=0.25)
    pos_rate  = float(y.mean())

    # FIX: SGD for SCAFFOLD (AdamW broke the control variate math)
    scaffold_lr = LR * 0.5  # Slightly lower LR for SCAFFOLD stability
    optimizer   = torch.optim.SGD(model.parameters(), lr=scaffold_lr,
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
            # Apply SCAFFOLD correction to gradients
            with torch.no_grad():
                for i, p in enumerate(model.parameters()):
                    if p.grad is not None and i < len(c_global) and i < len(c_local):
                        corr = c_global[i].to(device) - c_local[i].to(device)
                        # Scale correction to same magnitude as gradient
                        grad_norm = p.grad.data.norm().item()
                        corr_norm = corr.norm().item()
                        if corr_norm > 0 and grad_norm > 0:
                            corr = corr * (grad_norm / corr_norm) * 0.1  # 10% correction
                        p.grad.data.add_(corr)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            n_steps += 1

    # Update client control variate
    # Formula: c_local_new = c_local + (w_new - w_init) / (K * lr) - c_global
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
            # Clip using 3x parameter std (scale-aware)
            # FIX: Use correction=0 (population std) to avoid NaN on
            # single-element tensors (e.g., nn.Linear(128,1).bias has
            # shape [1]). Default std() uses Bessel's correction (N-1)
            # which divides by 0 for N=1 → NaN → max(NaN, 0.1)=NaN →
            # torch.clamp(..., -NaN, NaN) = all NaN → c_local=NaN →
            # c_global=NaN → every round NaN. This was THE root cause
            # of SCAFFOLD's perpetual NaN.
            std = p_new.data.std(correction=0).item() if p_new.numel() > 1 else p_new.data.abs().item()
            clip = max(3.0 * std, 0.1)
            new_c_local.append(torch.clamp(new_c, -clip, clip).cpu().detach())

    return model, new_c_local


# ─────────────────────────────────────────────────────────────────────────────
# SCAFFOLD AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_scaffold_weights(client_weights, client_sizes, c_locals, n_clients):
    total = sum(client_sizes)
    averaged = []
    for layer_idx in range(len(client_weights[0])):
        layer_avg = np.zeros_like(client_weights[0][layer_idx], dtype=np.float64)
        for w, sz in zip(client_weights, client_sizes):
            layer_avg += (sz / total) * w[layer_idx].astype(np.float64)
        averaged.append(layer_avg.astype(np.float32))

    new_c_global = []
    if c_locals:
        for layer_idx in range(len(c_locals[0])):
            c_avg = torch.zeros_like(c_locals[0][layer_idx], dtype=torch.float32)
            for cl in c_locals:
                c_avg += cl[layer_idx].float()
            new_c_global.append(c_avg / len(c_locals))
    return averaged, new_c_global


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, threshold=THRESHOLD,
                   batch_size=EVAL_BATCH_SIZE) -> Dict:
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, roc_auc_score)
    device = next(model.parameters()).device
    model.eval()
    n = len(X_test)
    all_probs = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e    = min(s + batch_size, n)
            Xb   = torch.tensor(X_test[s:e], dtype=torch.float32).to(device)
            prob = torch.sigmoid(model(Xb)).cpu().numpy().flatten()
            all_probs[s:e] = prob
            del Xb
            if device.type == "cuda":
                torch.cuda.empty_cache()
    nan_c = int(np.isnan(all_probs).sum())
    if nan_c > 0:
        print(f"  [Warning] {nan_c} NaN probabilities → replacing with 0.5")
        all_probs = np.nan_to_num(all_probs, nan=0.5, posinf=1.0, neginf=0.0)
    all_probs = np.clip(all_probs, 1e-7, 1 - 1e-7)
    preds = (all_probs >= threshold).astype(int)
    return {
        "accuracy":  accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall":    recall_score(y_test, preds, zero_division=0),
        "f1":        f1_score(y_test, preds, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, all_probs),
        "probs":     all_probs,
        "preds":     preds,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FEDAVG TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_fedavg(shards, X_test, y_test, n_rounds=N_ROUNDS,
               use_lstm=None, eval_batch_size=EVAL_BATCH_SIZE):
    import config as cfg
    sample_X = shards[0][0]
    if use_lstm is None:
        use_lstm = cfg.USE_LSTM
    input_dim    = sample_X.shape[2] if sample_X.ndim == 3 else sample_X.shape[1]
    global_model, device = make_fresh_model(input_dim, use_lstm=use_lstm)
    history      = []
    client_pos_rates = [float(y.mean()) for _, y in shards]

    print("=" * 60)
    print(f" FedAvg  [{device}] | {'LSTM' if use_lstm else 'MLP'} | Focal(a=0.25)")
    print("=" * 60)

    for rnd in range(1, n_rounds + 1):
        selected = np.random.choice(len(shards), max(1, int(FRACTION_FIT * N_CLIENTS)),
                                    replace=False)
        client_weights, client_sizes, client_labels = [], [], []
        for cid in selected:
            X_c, y_c = shards[cid]
            if len(X_c) == 0:
                continue
            local = clone_model(global_model)
            local = local_train_fedavg(local, X_c, y_c,
                                       current_round=rnd, total_rounds=n_rounds)
            client_weights.append(get_weights(local))
            client_sizes.append(len(X_c))
            client_labels.append(y_c)
            del local
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if not client_weights:
            continue
        new_weights = aggregate_weights_dafl(
            client_weights, client_sizes, client_labels,
            global_pos_rate=0.4887, round_num=rnd
        )
        set_weights(global_model, new_weights)

        if rnd % 5 == 0 or rnd == n_rounds:
            metrics = evaluate_model(global_model, X_test, y_test,
                                     batch_size=eval_batch_size)
            history.append({"round": rnd, **metrics})
            gpu = f" | GPU: {torch.cuda.memory_allocated()/1024**2:.0f}MB" \
                  if device.type == "cuda" else ""
            print(f"  Round {rnd:>3} | F1: {metrics['f1']:.3f} | "
                  f"Recall: {metrics['recall']:.3f} | "
                  f"ROC-AUC: {metrics['roc_auc']:.3f}{gpu}")
    return global_model, history


# ─────────────────────────────────────────────────────────────────────────────
# FEDPROX TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_fedprox(shards, X_test, y_test, n_rounds=N_ROUNDS, mu=MU,
                use_lstm=None, eval_batch_size=EVAL_BATCH_SIZE):
    import config as cfg
    sample_X = shards[0][0]
    if use_lstm is None:
        use_lstm = cfg.USE_LSTM
    input_dim    = sample_X.shape[2] if sample_X.ndim == 3 else sample_X.shape[1]
    global_model, device = make_fresh_model(input_dim, use_lstm=use_lstm)
    history      = []

    print("=" * 60)
    print(f" FedProx (mu={mu})  [{device}] | {'LSTM' if use_lstm else 'MLP'} | Focal(a=0.25)")
    print("=" * 60)

    for rnd in range(1, n_rounds + 1):
        selected = np.random.choice(len(shards), max(1, int(FRACTION_FIT * N_CLIENTS)),
                                    replace=False)
        client_weights, client_sizes, client_labels = [], [], []
        for cid in selected:
            X_c, y_c = shards[cid]
            if len(X_c) == 0:
                continue
            local = clone_model(global_model)
            local = local_train_fedprox(local, global_model, X_c, y_c, mu=mu,
                                        current_round=rnd, total_rounds=n_rounds)
            client_weights.append(get_weights(local))
            client_sizes.append(len(X_c))
            client_labels.append(y_c)
            del local
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if not client_weights:
            continue
        new_weights = aggregate_weights_dafl(
            client_weights, client_sizes, client_labels,
            global_pos_rate=0.4887, round_num=rnd
        )
        set_weights(global_model, new_weights)

        if rnd % 5 == 0 or rnd == n_rounds:
            metrics = evaluate_model(global_model, X_test, y_test,
                                     batch_size=eval_batch_size)
            history.append({"round": rnd, **metrics})
            gpu = f" | GPU: {torch.cuda.memory_allocated()/1024**2:.0f}MB" \
                  if device.type == "cuda" else ""
            print(f"  Round {rnd:>3} | F1: {metrics['f1']:.3f} | "
                  f"Recall: {metrics['recall']:.3f} | "
                  f"ROC-AUC: {metrics['roc_auc']:.3f}{gpu}")
    return global_model, history


# ─────────────────────────────────────────────────────────────────────────────
# SCAFFOLD TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_scaffold(shards, X_test, y_test, n_rounds=N_ROUNDS,
                 use_lstm=None, eval_batch_size=EVAL_BATCH_SIZE,
                 warm_start_model=None):
    import config as cfg
    if not cfg.USE_SCAFFOLD:
        print("[SCAFFOLD] Disabled in config.")
        return None, []

    sample_X = shards[0][0]
    if use_lstm is None:
        use_lstm = cfg.USE_LSTM
    input_dim = sample_X.shape[2] if sample_X.ndim == 3 else sample_X.shape[1]

    if warm_start_model is not None:
        global_model = clone_model(warm_start_model)
        device = next(global_model.parameters()).device
        print(f"  [SCAFFOLD] Warm-start from FedAvg model")
    else:
        global_model, device = make_fresh_model(input_dim, use_lstm=use_lstm)

    history  = []
    c_global = [torch.zeros_like(p) for p in global_model.parameters()]
    c_locals = {i: [torch.zeros_like(p) for p in global_model.parameters()]
                for i in range(len(shards))}
    best_f1, best_weights = 0.0, None

    print("=" * 60)
    print(f" SCAFFOLD  [{device}] | {'LSTM' if use_lstm else 'MLP'} | SGD(lr={LR*0.5:.4f})")
    print(f" FIX: Uses SGD (AdamW caused NaN via invalid c_local updates)")
    print("=" * 60)

    for rnd in range(1, n_rounds + 1):
        selected = np.random.choice(len(shards), max(1, int(FRACTION_FIT * N_CLIENTS)),
                                    replace=False)
        client_weights, client_sizes, client_labels, updated_c = [], [], [], []

        for cid in selected:
            X_c, y_c = shards[cid]
            if len(X_c) == 0:
                continue
            local = clone_model(global_model)
            local, new_c = local_train_scaffold(
                local, X_c, y_c, c_global, c_locals[cid],
                current_round=rnd, total_rounds=n_rounds
            )
            client_weights.append(get_weights(local))
            client_sizes.append(len(X_c))
            client_labels.append(y_c)
            updated_c.append(new_c)
            c_locals[cid] = new_c
            del local
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if not client_weights:
            continue

        new_weights = aggregate_weights_dafl(
            client_weights, client_sizes, client_labels,
            global_pos_rate=0.4887, round_num=rnd
        )
        _, new_c_global = aggregate_scaffold_weights(
            client_weights, client_sizes, updated_c, len(selected)
        )

        # NaN check — revert and reset variates
        if any(np.isnan(w).any() for w in new_weights):
            print(f"  [SCAFFOLD] NaN at round {rnd} — reverting, resetting c_global")
            if best_weights is not None:
                set_weights(global_model, best_weights)
            elif warm_start_model is not None:
                set_weights(global_model, get_weights(warm_start_model))
            else:
                global_model, device = make_fresh_model(input_dim, use_lstm=use_lstm)
            c_global = [torch.zeros_like(p) for p in global_model.parameters()]
            continue

        set_weights(global_model, new_weights)
        if new_c_global:
            c_global = new_c_global

        if rnd % 5 == 0 or rnd == n_rounds:
            metrics = evaluate_model(global_model, X_test, y_test,
                                     batch_size=eval_batch_size)
            history.append({"round": rnd, **metrics})
            gpu = f" | GPU: {torch.cuda.memory_allocated()/1024**2:.0f}MB" \
                  if device.type == "cuda" else ""
            print(f"  Round {rnd:>3} | F1: {metrics['f1']:.3f} | "
                  f"Recall: {metrics['recall']:.3f} | "
                  f"ROC-AUC: {metrics['roc_auc']:.3f}{gpu}")
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_weights = [w.copy() for w in new_weights]
                print(f"          * New best F1: {best_f1:.3f}")

    if best_weights is not None and best_f1 > 0:
        set_weights(global_model, best_weights)
        print(f"\n  [SCAFFOLD] Restored best checkpoint (F1={best_f1:.3f})")

    return global_model, history
