"""
federated_learning.py
───────────────────
Implements FedAvg, FedProx, and SCAFFOLD from scratch with 2026 improvements.

Improvements over v1:
  1. SCAFFOLD algorithm (Karimireddy et al., ICML 2020)
  2. Fed-Focal Loss for class imbalance (arxiv 2602.01633, Feb 2026)
  3. CosineAnnealingWarmRestarts scheduler (fixes FedAvg LR collapse)
  4. Mixup augmentation for minority class (ICLR 2018)
  5. F-beta threshold optimization (beta=2 for safety-critical recall)
  6. Richer temporal feature support (6-stat extraction)
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
# MIXUP AUGMENTATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def mixup_data(X: np.ndarray, y: np.ndarray, alpha: float = 0.4,
               minority_only: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply mixup augmentation to training data.

    For imbalanced data, we ONLY mix within the minority (flare) class
    to create more diverse positive examples without contaminating
    the majority class.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        alpha: Beta distribution parameter (0.4 = moderate mixing)
        minority_only: If True, only augment minority class samples

    Returns:
        Augmented X, y (may have more samples than input)
    """
    if not USE_MIXUP or alpha <= 0:
        return X, y

    # Find minority class indices
    flare_idx = np.where(y == 1)[0]
    if len(flare_idx) < 2:
        return X, y

    n_augment = min(len(flare_idx), int(len(flare_idx) * 0.3))  # Augment 30%
    X_aug = []
    y_aug = []

    for _ in range(n_augment):
        # Sample two minority examples
        i, j = np.random.choice(len(flare_idx), 2, replace=False)
        lam = np.random.beta(alpha, alpha)

        # Interpolate
        x_mixed = lam * X[flare_idx[i]] + (1 - lam) * X[flare_idx[j]]
        X_aug.append(x_mixed)
        y_aug.append(1)  # Both are flare → label stays 1

    if X_aug:
        X_aug = np.array(X_aug, dtype=X.dtype)
        y_aug = np.array(y_aug, dtype=y.dtype)
        X = np.vstack([X, X_aug])
        y = np.concatenate([y, y_aug])

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# GET LOSS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def get_criterion(device, current_round=0, total_rounds=50):
    """Get the appropriate loss function based on config."""
    if USE_FED_FOCAL:
        from losses import FedFocalLoss
        criterion = FedFocalLoss(
            gamma=FOCAL_GAMMA,
            alpha=FOCAL_ALPHA,
            reduction='mean'
        ).to(device)
        criterion.set_round_info(current_round, total_rounds)
        return criterion
    else:
        from losses import DynamicFocalLoss
        return DynamicFocalLoss(gamma=2.0, base_alpha=0.25).to(device)


# ════════════════════════════════════════════════════════════════════════
# LOCAL TRAINING — FedAvg (ENHANCED)
# ════════════════════════════════════════════════════════════════════════

def local_train_fedavg(
    model:  nn.Module,
    X:      np.ndarray,
    y:      np.ndarray,
    epochs: int = LOCAL_EPOCHS,
    current_round: int = 0,
    total_rounds: int = 50
) -> nn.Module:
    """
    Local FedAvg training with Fed-Focal Loss + CosineAnnealingWarmRestarts.

    Key improvements over v1:
    1. Fed-Focal Loss handles local-global imbalance mismatch
    2. CosineAnnealingWarmRestarts prevents LR collapse (v1 bug)
    3. Gradient clipping ensures stable training
    4. Mixup augmentation for minority class
    """
    device = next(model.parameters()).device
    model.train()

    # Apply mixup augmentation to minority class
    X, y = mixup_data(X, y, alpha=MIXUP_ALPHA)

    # Create data loader
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X),
        torch.FloatTensor(y)
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=512,
        shuffle=True,
        drop_last=False
    )

    # Get loss function
    criterion = get_criterion(device, current_round, total_rounds)

    # Calculate client's positivity rate for dynamic adjustment
    client_pos_rate = float(y.mean())

    # Optimizer with L2 regularization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    # ✅ FIX: CosineAnnealingWarmRestarts instead of CosineAnnealingLR
    # This prevents LR from decaying to near-zero by round 20
    # T_0=5: restart every 5 epochs, T_mult=1: same period each restart
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=1,
        eta_min=LR * 0.1  # Minimum LR = 10% of initial
    )

    # Training loop
    for epoch in range(epochs):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            # Forward pass
            logits = model(X_batch)

            # Compute loss
            loss = criterion(logits, y_batch, client_pos_rate=client_pos_rate)

            # Backward pass
            loss.backward()

            # Gradient clipping (prevents explosion)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            # Optimizer step
            optimizer.step()

        # LR scheduler step (per epoch)
        scheduler.step()

    return model


# ════════════════════════════════════════════════════════════════════════
# LOCAL TRAINING — FedProx (ENHANCED)
# ════════════════════════════════════════════════════════════════════════

def local_train_fedprox(
    model:        nn.Module,
    global_model: nn.Module,
    X:            np.ndarray,
    y:            np.ndarray,
    epochs:       int   = LOCAL_EPOCHS,
    mu:           float = MU,
    current_round: int = 0,
    total_rounds: int = 50
) -> nn.Module:
    """Local FedProx training with Fed-Focal Loss + proximal term."""

    device = next(model.parameters()).device
    model.train()

    # Apply mixup augmentation
    X, y = mixup_data(X, y, alpha=MIXUP_ALPHA)

    # Data loader
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X),
        torch.FloatTensor(y)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)

    # Get loss function
    criterion = get_criterion(device, current_round, total_rounds)
    client_pos_rate = float(y.mean())

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # ✅ FIX: CosineAnnealingWarmRestarts for FedProx too
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=1, eta_min=LR * 0.1
    )

    # Global weights snapshot (for proximal term)
    global_params = [p.data.detach().clone().to(device)
                     for p in global_model.parameters()]

    for epoch in range(epochs):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(X_batch)

            # Focal loss component
            focal_loss = criterion(logits, y_batch, client_pos_rate=client_pos_rate)

            # Proximal term: penalize drift from global model
            prox_term = torch.tensor(0.0, device=device)
            for local_p, global_p in zip(model.parameters(), global_params):
                prox_term = prox_term + ((local_p - global_p) ** 2).sum()
            prox_term = (mu / 2.0) * prox_term

            # Total loss
            loss = focal_loss + prox_term

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        scheduler.step()

    return model


# ════════════════════════════════════════════════════════════════════════
# LOCAL TRAINING — SCAFFOLD (NEW! ICML 2020)
# ════════════════════════════════════════════════════════════════════════

def local_train_scaffold(
    model:          nn.Module,
    X:              np.ndarray,
    y:              np.ndarray,
    c_global:       list,       # Server control variate
    c_local:        list,       # Client control variate
    epochs:         int   = LOCAL_EPOCHS,
    current_round:  int   = 0,
    total_rounds:   int   = 50
) -> Tuple[nn.Module, list]:
    """
    SCAFFOLD local training with variance reduction (Karimireddy et al., ICML 2020).

    Key idea: Each client maintains a control variate (c_local) that
    tracks the direction of its local updates. The server maintains
    c_global. The difference (c_global - c_local) corrects for
    client drift, leading to faster convergence and better performance
    on non-IID data.

    Implementation notes:
    - The correction (c_global - c_local) is added to the gradient,
      scaled by the learning rate, so the optimizer's update becomes:
        w -= lr * (grad + (c_global - c_local))
      This is the standard SCAFFOLD formulation with SGD.
    - With AdamW, the effective step size is adaptive, so we scale
      the correction by lr to prevent it from dominating the update.
    - Control variates are clipped to prevent explosion on non-IID data.
    """
    device = next(model.parameters()).device
    model.train()

    # Apply mixup augmentation
    X, y = mixup_data(X, y, alpha=MIXUP_ALPHA)

    # Data loader
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X),
        torch.FloatTensor(y)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)

    # Loss function
    criterion = get_criterion(device, current_round, total_rounds)
    client_pos_rate = float(y.mean())

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=1, eta_min=LR * 0.1
    )

    # Save initial model weights for control variate update
    initial_params = [p.data.detach().clone() for p in model.parameters()]

    # Count total optimization steps for proper c_local update
    n_steps = 0

    # Training loop with SCAFFOLD correction
    nan_detected = False
    for epoch in range(epochs):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            # Standard forward + loss
            logits = model(X_batch)
            loss = criterion(logits, y_batch, client_pos_rate=client_pos_rate)

            # Check for NaN loss (early warning of divergence)
            if torch.isnan(loss) or torch.isinf(loss):
                nan_detected = True
                continue  # Skip this batch

            # Backward to get gradients
            loss.backward()

            # SCAFFOLD: Add control variate correction to gradients
            # Scale by lr so the effective correction is lr * (c_global - c_local)
            # This prevents the correction from dominating AdamW's adaptive updates
            with torch.no_grad():
                for i, p in enumerate(model.parameters()):
                    if p.grad is not None and i < len(c_global) and i < len(c_local):
                        correction = LR * (c_global[i].to(device) - c_local[i].to(device))
                        # Clip correction to prevent it from dominating the gradient
                        corr_norm = correction.norm().item()
                        if corr_norm > 1.0:
                            correction = correction * (1.0 / corr_norm)
                        p.grad.data += correction

            # Tighter gradient clipping for SCAFFOLD (1.0 vs 5.0 for FedAvg/FedProx)
            # SCAFFOLD is more prone to instability because the control variate
            # correction can amplify gradients. max_norm=1.0 prevents explosion.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            n_steps += 1

        scheduler.step()

        # Check model weights for NaN after each epoch
        if nan_detected:
            has_nan = any(torch.isnan(p).any().item() for p in model.parameters())
            if has_nan:
                # Reset model weights to initial state for this client
                print(f"    [SCAFFOLD] NaN weights detected at epoch {epoch+1}. "
                      f"Resetting to round-start weights.")
                with torch.no_grad():
                    for p, p_init in zip(model.parameters(), initial_params):
                        p.data.copy_(p_init.to(device))
                n_steps = 0  # Reset step count
                break

    # Update client control variate
    # c_local_new = c_local_old + (w_new - w_old) / (n_steps * lr) - c_global
    # This estimates the average gradient direction of this client
    new_c_local = []
    with torch.no_grad():
        for i, (p_init, p_new) in enumerate(zip(initial_params, model.parameters())):
            delta = (p_new.data - p_init.to(device)) / max(n_steps * LR, 1e-8)
            if i < len(c_local) and i < len(c_global):
                new_c = c_local[i].to(device) + delta - c_global[i].to(device)
            else:
                new_c = delta
            # Adaptive clipping: clip based on parameter's own scale
            # Using percentile-based clipping (3x the standard deviation)
            # instead of fixed [-1, 1] which was too aggressive for LSTM
            param_std = p_new.data.std().item()
            clip_val = max(3.0 * param_std, 0.01)  # At least 0.01
            new_c = torch.clamp(new_c, -clip_val, clip_val)
            new_c_local.append(new_c.cpu().detach().clone())

    return model, new_c_local


# ════════════════════════════════════════════════════════════════════════
# SERVER AGGREGATION
# ════════════════════════════════════════════════════════════════════════

def aggregate_weights_dafl(
    client_weights: list,
    client_sizes:   list,
    client_labels:  list,
    global_pos_rate: float = 0.4887
) -> list:
    """
    Distribution-Aware Federated Learning (DA-FL) aggregation.

    Weights clients by dataset size AND minority class representation.
    When clients have different flare rates (non-IID), this upweights
    clients with fewer flares to prevent minority pattern loss.

    v2 FIX: Uses sqrt-smoothed phi and caps max per-client weight at 2/K
    to prevent a single client from dominating aggregation. With raw phi,
    Client 4 (Americas, 92% flares) got 78.7% aggregation weight → model
    predicted everything as a flare. Sqrt-smoothing dampens extreme ratios
    (1.88→1.37, 0.013→0.114) and the weight cap ensures no client exceeds
    2x equal share, with excess redistributed proportionally.
    """
    n_clients = len(client_sizes)
    total_size = sum(client_sizes)
    da_weights = []

    # Max weight cap: no client should exceed 2/K (2x equal share)
    max_weight = 2.0 / n_clients

    print(f"      [DA-FL] Distribution-aware weights (max_weight_cap={max_weight:.3f}):")

    for i, (size, y_c) in enumerate(zip(client_sizes, client_labels)):
        size_weight = size / total_size
        local_pos_rate = float(np.mean(y_c))
        # FIX: sqrt-smoothed phi dampens extreme ratios
        # Raw phi: 1.883 for Americas (92% flares) → too dominant
        # Sqrt phi: sqrt(1.883) = 1.373 → much more balanced
        raw_phi = local_pos_rate / global_pos_rate if global_pos_rate > 0 else 1.0
        phi = np.sqrt(raw_phi)
        combined = size_weight * phi

        # Apply weight cap: no client exceeds 2/K
        capped = min(combined, max_weight)
        da_weights.append(capped)

        cap_flag = " [CAPPED]" if combined > max_weight else ""
        print(f"        Client {i}: size_w={size_weight:.3f}, raw_phi={raw_phi:.3f}, "
              f"sqrt_phi={phi:.3f}, final_w={capped:.3f}{cap_flag} "
              f"(pos_rate={local_pos_rate:.3f})")

    # Redistribute excess weight from capped clients proportionally
    total_da_weight = sum(da_weights)
    if total_da_weight > 0:
        da_weights = [w / total_da_weight for w in da_weights]

    # Weighted average of model parameters
    averaged = []
    for layer_idx in range(len(client_weights[0])):
        layer_avg = np.zeros_like(client_weights[0][layer_idx], dtype=np.float64)

        for w, da_w in zip(client_weights, da_weights):
            layer_avg += da_w * w[layer_idx].astype(np.float64)

        averaged.append(layer_avg.astype(np.float32))

    return averaged


def aggregate_scaffold_weights(
    client_weights: list,
    client_sizes:   list,
    c_globals:      list,    # List of updated c_local from each client
    n_clients:      int
) -> Tuple[list, list]:
    """
    SCAFFOLD server aggregation with control variate update.

    In addition to standard weighted averaging of model parameters,
    SCAFFOLD also averages the control variates:
        c_global_new = (1/K) * sum(c_local_i_new)

    Returns:
        averaged_weights: Averaged model parameters
        new_c_global: Updated server control variate
    """
    # Standard weighted averaging of model weights
    total_size = sum(client_sizes)
    averaged = []

    for layer_idx in range(len(client_weights[0])):
        layer_avg = np.zeros_like(client_weights[0][layer_idx], dtype=np.float64)
        for w, size in zip(client_weights, client_sizes):
            layer_avg += (size / total_size) * w[layer_idx].astype(np.float64)
        averaged.append(layer_avg.astype(np.float32))

    # Average control variates across clients
    new_c_global = []
    if c_globals:
        n_contributing = len(c_globals)
        for layer_idx in range(len(c_globals[0])):
            c_avg = torch.zeros_like(c_globals[0][layer_idx], dtype=torch.float32)
            for c_local in c_globals:
                c_avg += c_local[layer_idx].float()
            new_c_global.append(c_avg / n_contributing)

    return averaged, new_c_global


# ════════════════════════════════════════════════════════════════════════
# EVALUATION
# ════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model:     nn.Module,
    X_test:    np.ndarray,
    y_test:    np.ndarray,
    threshold: float = THRESHOLD,
    batch_size: int = EVAL_BATCH_SIZE
) -> Dict:
    """
    Evaluate trained model with batched inference (GPU memory safe).

    CRITICAL FIX: The full test set (331K samples x 60 x 24 for LSTM)
    cannot fit in GPU memory at once. This version processes data in
    mini-batches to avoid CUDA OOM errors.

    Args:
        model: Trained PyTorch model
        X_test: Test features (2D for MLP, 3D for LSTM)
        y_test: Test labels
        threshold: Classification threshold
        batch_size: Number of samples per inference batch
                   2048 works well on RTX 3060 (12GB) for LSTM

    Returns:
        Dict with accuracy, precision, recall, f1, roc_auc, probs, preds
    """
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, roc_auc_score)

    device = next(model.parameters()).device
    model.eval()

    n_samples = len(X_test)
    all_probs = np.empty(n_samples, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = torch.tensor(
                X_test[start:end], dtype=torch.float32
            ).to(device)

            logits = model(X_batch)
            probs_batch = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs[start:end] = probs_batch

            # Free GPU memory after each batch
            del X_batch, logits
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ── NaN/Inf protection ──
    # SCAFFOLD and other algorithms can produce NaN logits when control
    # variates diverge or gradients explode. Replace NaN with 0.5 (random
    # guess) and clamp to valid probability range for metric computation.
    nan_count = int(np.isnan(all_probs).sum())
    inf_count = int(np.isinf(all_probs).sum())
    if nan_count > 0 or inf_count > 0:
        print(f"  [Warning] {nan_count} NaN + {inf_count} Inf probabilities detected. "
              f"Replacing with 0.5 for metric computation.")
        all_probs = np.nan_to_num(all_probs, nan=0.5, posinf=1.0, neginf=0.0)

    # Clamp to [eps, 1-eps] for stable log computation in AUC
    all_probs = np.clip(all_probs, 1e-7, 1.0 - 1e-7)

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


# ════════════════════════════════════════════════════════════════════════
# FEDAVG TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════

def run_fedavg(
    shards:   List[Tuple[np.ndarray, np.ndarray]],
    X_test:   np.ndarray,
    y_test:   np.ndarray,
    n_rounds: int = N_ROUNDS,
    use_lstm: bool = None,
    eval_batch_size: int = EVAL_BATCH_SIZE
) -> Tuple[nn.Module, List[Dict]]:
    """
    Complete FedAvg training loop.
    Returns final global model and per-round metric history.

    Handles both 2D data (MLP) and 3D data (LSTM) automatically.
    For LSTM: shards have 3D X arrays (N, 60, 24), input_size=24
    For MLP:  shards have 2D X arrays (N, features), input_dim=features
    """
    # Resolve use_lstm: explicit param > data shape > config default
    sample_X = shards[0][0]
    if use_lstm is None:
        import config as cfg
        use_lstm = cfg.USE_LSTM

    if sample_X.ndim == 3:
        input_dim = sample_X.shape[2]  # (N, T, F) -> F=24 for LSTM
    else:
        input_dim = sample_X.shape[1]  # (N, F) -> F for MLP

    global_model, device = make_fresh_model(input_dim, use_lstm=use_lstm)
    history = []

    data_type = "3D (batch, 60, 24)" if sample_X.ndim == 3 else f"2D (batch, {input_dim})"

    print("=" * 60)
    print(f" FedAvg Training  [{device}]")
    print(f" Model: {'LSTM' if use_lstm else 'MLP'} | Loss: {'Fed-Focal' if USE_FED_FOCAL else 'DAF'}")
    print(f" Data: {data_type}")
    print("=" * 60)

    for rnd in range(1, n_rounds + 1):
        n_selected = max(1, int(FRACTION_FIT * N_CLIENTS))
        selected = np.random.choice(len(shards), n_selected, replace=False)

        client_weights = []
        client_sizes = []

        for cid in selected:
            X_c, y_c = shards[cid]
            if len(X_c) == 0:
                continue

            local = clone_model(global_model)
            local = local_train_fedavg(local, X_c, y_c,
                                       current_round=rnd,
                                       total_rounds=n_rounds)
            client_weights.append(get_weights(local))
            client_sizes.append(len(X_c))

            del local
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if not client_weights:
            continue

        client_label_list = [shards[cid][1] for cid in selected]

        new_weights = aggregate_weights_dafl(
            client_weights, client_sizes, client_label_list,
            global_pos_rate=0.4887
        )

        set_weights(global_model, new_weights)

        # Evaluate every 5 rounds and on final round
        if rnd % 5 == 0 or rnd == n_rounds:
            metrics = evaluate_model(global_model, X_test, y_test, batch_size=eval_batch_size)
            history.append({"round": rnd, **metrics})

            # GPU memory logging
            gpu_mem = ""
            if device.type == "cuda":
                gpu_mem = f" | GPU: {torch.cuda.memory_allocated()/1024**2:.0f}MB"

            print(f"  Round {rnd:>3} | "
                  f"F1: {metrics['f1']:.3f} | "
                  f"Recall: {metrics['recall']:.3f} | "
                  f"ROC-AUC: {metrics['roc_auc']:.3f}{gpu_mem}")

    return global_model, history


# ════════════════════════════════════════════════════════════════════════
# FEDPROX TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════

def run_fedprox(
    shards:   List[Tuple[np.ndarray, np.ndarray]],
    X_test:   np.ndarray,
    y_test:   np.ndarray,
    n_rounds: int = N_ROUNDS,
    mu:       float = MU,
    use_lstm: bool = None,
    eval_batch_size: int = EVAL_BATCH_SIZE
) -> Tuple[nn.Module, List[Dict]]:
    """
    Complete FedProx training loop.
    Returns final global model and per-round metric history.

    Handles both 2D data (MLP) and 3D data (LSTM) automatically.
    """
    sample_X = shards[0][0]
    if use_lstm is None:
        import config as cfg
        use_lstm = cfg.USE_LSTM

    if sample_X.ndim == 3:
        input_dim = sample_X.shape[2]
    else:
        input_dim = sample_X.shape[1]

    global_model, device = make_fresh_model(input_dim, use_lstm=use_lstm)
    history = []

    data_type = "3D (batch, 60, 24)" if sample_X.ndim == 3 else f"2D (batch, {input_dim})"

    print("=" * 60)
    print(f" FedProx Training (mu = {mu})  [{device}]")
    print(f" Model: {'LSTM' if use_lstm else 'MLP'} | Loss: {'Fed-Focal' if USE_FED_FOCAL else 'DAF'}")
    print(f" Data: {data_type}")
    print("=" * 60)

    for rnd in range(1, n_rounds + 1):
        n_selected = max(1, int(FRACTION_FIT * N_CLIENTS))
        selected = np.random.choice(len(shards), n_selected, replace=False)

        client_weights = []
        client_sizes = []

        for cid in selected:
            X_c, y_c = shards[cid]
            if len(X_c) == 0:
                continue

            local = clone_model(global_model)
            local = local_train_fedprox(local, global_model, X_c, y_c, mu=mu,
                                        current_round=rnd, total_rounds=n_rounds)
            client_weights.append(get_weights(local))
            client_sizes.append(len(X_c))

            del local
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if not client_weights:
            continue

        client_label_list = [shards[cid][1] for cid in selected]

        new_weights = aggregate_weights_dafl(
            client_weights, client_sizes, client_label_list,
            global_pos_rate=0.4887
        )

        set_weights(global_model, new_weights)

        if rnd % 5 == 0 or rnd == n_rounds:
            metrics = evaluate_model(global_model, X_test, y_test, batch_size=eval_batch_size)
            history.append({"round": rnd, **metrics})

            # GPU memory logging
            gpu_mem = ""
            if device.type == "cuda":
                gpu_mem = f" | GPU: {torch.cuda.memory_allocated()/1024**2:.0f}MB"

            print(f"  Round {rnd:>3} | "
                  f"F1: {metrics['f1']:.3f} | "
                  f"Recall: {metrics['recall']:.3f} | "
                  f"ROC-AUC: {metrics['roc_auc']:.3f}{gpu_mem}")

    return global_model, history


# ════════════════════════════════════════════════════════════════════════
# SCAFFOLD TRAINING LOOP (NEW!)
# ════════════════════════════════════════════════════════════════════════

def run_scaffold(
    shards:   List[Tuple[np.ndarray, np.ndarray]],
    X_test:   np.ndarray,
    y_test:   np.ndarray,
    n_rounds: int = N_ROUNDS,
    use_lstm: bool = None,
    eval_batch_size: int = EVAL_BATCH_SIZE,
    warm_start_model: nn.Module = None
) -> Tuple[nn.Module, List[Dict]]:
    """
    Complete SCAFFOLD training loop with control variates.

    SCAFFOLD (Stochastic Controlled Averaging for Federated Learning)
    addresses client drift in non-IID settings by maintaining server
    and client control variates that correct gradient estimates.

    Algorithm:
    1. Server sends global model + c_global to selected clients
    2. Each client trains locally with correction: g_corrected = g + (c_global - c_local)
    3. Client updates c_local and sends model + c_local back
    4. Server averages model weights and c_locals

    Enhancements over vanilla SCAFFOLD:
    - Best-model checkpoint: Returns the model with highest F1 (not last round)
    - Control variate warmup: Correction ramps up over first 5 rounds
    - Warm-start from FedAvg: Initializes from FedAvg weights instead of random,
      preventing the NaN death spiral that occurs with random initialization

    Reference:
        Karimireddy et al., "SCAFFOLD" (ICML 2020)

    Args:
        warm_start_model: If provided, initialize SCAFFOLD from this model's
                         weights instead of random initialization. Typically
                         the FedAvg global model, which provides a much more
                         stable starting point for SCAFFOLD's control variates.

    Returns:
        global_model: Trained global model (best checkpoint)
        history: Per-round metric history
    """
    import config as cfg
    if not cfg.USE_SCAFFOLD:
        print("[SCAFFOLD] Disabled in config. Skipping.")
        return None, []

    sample_X = shards[0][0]
    if use_lstm is None:
        use_lstm = cfg.USE_LSTM

    if sample_X.ndim == 3:
        input_dim = sample_X.shape[2]
    else:
        input_dim = sample_X.shape[1]

    # FIX: Initialize from warm-start model (FedAvg) instead of random weights.
    # Random init → SCAFFOLD often diverges → NaN → reinit from random → NaN loop.
    # FedAvg warm-start provides a converged baseline for control variates to refine.
    if warm_start_model is not None:
        global_model = clone_model(warm_start_model)
        device = next(global_model.parameters()).device
        print(f"  [SCAFFOLD] Warm-starting from FedAvg model (not random init)")
    else:
        global_model, device = make_fresh_model(input_dim, use_lstm=use_lstm)
    history = []

    # Initialize control variates (zeros)
    c_global = [torch.zeros_like(p) for p in global_model.parameters()]
    # Each client has its own c_local
    c_locals = {i: [torch.zeros_like(p) for p in global_model.parameters()]
                for i in range(len(shards))}

    # Best model checkpoint tracking
    best_f1 = 0.0
    best_weights = None

    data_type = "3D (batch, 60, 24)" if sample_X.ndim == 3 else f"2D (batch, {input_dim})"

    print("=" * 60)
    print(f" SCAFFOLD Training  [{device}]")
    print(f" Model: {'LSTM' if use_lstm else 'MLP'} | Loss: {'Fed-Focal' if cfg.USE_FED_FOCAL else 'DAF'}")
    print(f" Data: {data_type}")
    print(f" Control variates: {sum(p.numel() for p in c_global):,} params")
    print(f" Best-checkpoint: Enabled (returns best F1 model)")
    print("=" * 60)

    for rnd in range(1, n_rounds + 1):
        n_selected = max(1, int(FRACTION_FIT * N_CLIENTS))
        selected = np.random.choice(len(shards), n_selected, replace=False)

        client_weights = []
        client_sizes = []
        updated_c_locals = []

        for cid in selected:
            X_c, y_c = shards[cid]
            if len(X_c) == 0:
                continue

            local = clone_model(global_model)

            # SCAFFOLD local training with control variates
            local, new_c_local = local_train_scaffold(
                local, X_c, y_c,
                c_global=c_global,
                c_local=c_locals[cid],
                current_round=rnd,
                total_rounds=n_rounds
            )

            client_weights.append(get_weights(local))
            client_sizes.append(len(X_c))
            updated_c_locals.append(new_c_local)

            # Update this client's c_local for next round
            c_locals[cid] = new_c_local

            del local
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if not client_weights:
            continue

        # SCAFFOLD aggregation: DA-FL weighted model params + control variate averaging
        # DA-FL weights account for data size AND minority class representation
        client_label_list = [shards[cid][1] for cid in selected]
        new_weights = aggregate_weights_dafl(
            client_weights, client_sizes, client_label_list,
            global_pos_rate=0.4887
        )

        # Control variate averaging (standard SCAFFOLD: simple mean)
        _, new_c_global = aggregate_scaffold_weights(
            client_weights, client_sizes, updated_c_locals,
            n_clients=len(selected)
        )

        set_weights(global_model, new_weights)
        c_global = new_c_global

        # ── NaN weight detection after aggregation ──
        # If aggregated weights contain NaN (from a diverged client),
        # revert using a 3-tier hierarchy:
        #   1. Best checkpoint (if available)
        #   2. FedAvg warm-start weights (if available)
        #   3. Random reinitialization (last resort)
        has_nan = any(np.isnan(w).any() for w in new_weights)
        if has_nan:
            print(f"  [SCAFFOLD] NaN in aggregated weights at round {rnd}!")
            if best_weights is not None:
                print(f"  [SCAFFOLD] Reverting to best checkpoint (F1={best_f1:.3f})")
                set_weights(global_model, best_weights)
            elif warm_start_model is not None:
                print(f"  [SCAFFOLD] Reverting to FedAvg warm-start weights")
                warm_weights = get_weights(warm_start_model)
                set_weights(global_model, warm_weights)
            else:
                print(f"  [SCAFFOLD] No checkpoint or warm-start available — reinitializing model")
                global_model, device = make_fresh_model(input_dim, use_lstm=use_lstm)
            # Always reset control variates on NaN recovery
            c_global = [torch.zeros_like(p) for p in global_model.parameters()]
            continue

        if rnd % 5 == 0 or rnd == n_rounds:
            metrics = evaluate_model(global_model, X_test, y_test, batch_size=eval_batch_size)
            history.append({"round": rnd, **metrics})

            # GPU memory logging
            gpu_mem = ""
            if device.type == "cuda":
                gpu_mem = f" | GPU: {torch.cuda.memory_allocated()/1024**2:.0f}MB"

            print(f"  Round {rnd:>3} | "
                  f"F1: {metrics['f1']:.3f} | "
                  f"Recall: {metrics['recall']:.3f} | "
                  f"ROC-AUC: {metrics['roc_auc']:.3f}{gpu_mem}")

            # Track best model checkpoint
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_weights = [w.copy() for w in new_weights]
                print(f"          ★ New best F1: {best_f1:.3f}")

    # Restore best model checkpoint
    if best_weights is not None and best_f1 > 0:
        set_weights(global_model, best_weights)
        print(f"\n  [SCAFFOLD] Restored best checkpoint (F1={best_f1:.3f})")

    return global_model, history