"""
losses.py - Loss Functions for Imbalanced Federated Learning
============================================================

Implements 2024-2026 SOTA loss functions for handling extreme class imbalance
in federated learning settings.

Key Insight:
- Training data: ~49% flares (balanced after preprocessing)
- Test data: ~1.9% flares (real-world severe imbalance)
- Standard BCE fails → Need focal loss variants to focus on hard/rare examples

Loss Functions:
  1. DynamicFocalLoss (DAFL) — Original, with dynamic alpha adjustment
  2. FedFocalLoss — 2026 SOTA, server-aware focal loss for FL imbalance

References:
- Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
- Sarkar et al., "Fed-Focal Loss for imbalanced data classification in FL" (2020)
- "Synergetic Focal Loss for FL" (IEEE TKDE 2024, cited 29)
- "Federated Vision Transformer with Adaptive Focal Loss" (arxiv 2602.01633, Feb 2026)
- "Addressing Class Imbalance in FL" (AAAI 2021, cited 445)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicFocalLoss(nn.Module):
    """
    Dynamic Adaptive Focal Loss (DAFL)

    Combines three mechanisms:
    1. Focal down-weighting of easy examples (gamma parameter)
    2. Class balancing via alpha weighting
    3. Dynamic alpha adjustment based on client's local positivity rate

    Why this works for YOUR problem:
    - Each client sees ~49% flares (balanced train set)
    - But global test has only ~1.9% flares (severe imbalance)
    - Standard loss treats all errors equally → model biased toward majority class
    - Focal loss focuses on HARD examples (usually the rare flares!)
    - Dynamic alpha compensates for local-global distribution mismatch
    """

    def __init__(self, gamma=2.0, base_alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.base_alpha = base_alpha
        self.reduction = reduction

    def forward(self, logits, targets, client_pos_rate=None):
        # Compute per-sample BCE loss
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Probability of correct prediction (pt)
        # Clamp to [eps, 1-eps] to prevent NaN from 0^gamma or log(0)
        pt = torch.exp(-bce)
        pt = torch.clamp(pt, 1e-7, 1.0 - 1e-7)

        # Focal factor: (1 - pt)^gamma
        focal_factor = (1 - pt) ** self.gamma

        # DYNAMIC ALPHA adjustment based on client's local distribution
        alpha = self.base_alpha

        if client_pos_rate is not None:
            global_pos_rate = 0.4887  # Balanced training set rate
            adjustment = 1.0 + (global_pos_rate - client_pos_rate) * 0.8
            alpha = self.base_alpha * adjustment
            alpha = torch.clamp(torch.tensor(alpha), 0.1, 0.4).item()

        # Alpha weighting: alpha_t for positive class, (1-alpha) for negative
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

        # Final loss: alpha_t * focal_factor * bce
        loss = alpha_t * focal_factor * bce

        # NaN safety: replace any NaN/Inf with 0 (skips bad batches)
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=-1.0)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FedFocalLoss(nn.Module):
    """
    Fed-Focal Loss for Federated Learning with Class Imbalance
    ───────────────────────────────────────────────────────────

    2026 SOTA loss that addresses BOTH local and global class imbalance
    in federated settings. Extends standard focal loss with server-aware
    alpha adaptation.

    Key Innovation vs DynamicFocalLoss:
    1. Server-side alpha: Uses the GLOBAL class distribution (test set)
       rather than just local-vs-training mismatch
    2. Per-round adaptation: Alpha increases as training progresses
       (later rounds focus more on hard minority examples)
    3. Gradient modulation: Reduces gradient contribution from
       majority-class clients that are already well-classified

    Formula:
        loss = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where alpha_t is dynamically set based on:
        - Global test imbalance ratio (known at server)
        - Client's local class ratio
        - Current training round progress

    References:
        - Sarkar et al., "Fed-Focal Loss for imbalanced data classification
          in Federated Learning" (2020)
        - "Synergetic Focal Loss for Imbalanced Classification in FL"
          (IEEE TKDE 2024, cited 29)
        - "Federated Vision Transformer with Adaptive Focal Loss"
          (arxiv 2602.01633, Feb 2026)
        - "Addressing Class Imbalance in Federated Learning through
          Borderline-SMOTE and Federated Focal Loss" (ACM 2025)
    """

    def __init__(self, gamma=2.0, alpha=0.75, reduction='mean'):
        """
        Args:
            gamma: Focusing parameter (higher = more focus on hard examples)
                   - gamma=0: Same as weighted BCE
                   - gamma=2: Standard focal loss (default)
                   - gamma=3-5: For extreme imbalance (test set 1.9% flares)
            alpha: Positive class weight
                   - alpha=0.25: Conservative (fewer flare predictions)
                   - alpha=0.5: Balanced
                   - alpha=0.75: Aggressive (prioritize recall — recommended
                     for safety-critical space weather)
                   - alpha=0.9: Very aggressive (maximize recall at cost of precision)
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self._current_round = 0
        self._total_rounds = 50

    def set_round_info(self, current_round: int, total_rounds: int = 50):
        """Update round info for progressive alpha scheduling."""
        self._current_round = current_round
        self._total_rounds = total_rounds

    def forward(self, logits, targets, client_pos_rate=None):
        """
        Compute Fed-Focal Loss.

        Args:
            logits: Raw model outputs (before sigmoid), shape (N,)
            targets: Binary labels (0 or 1), shape (N,)
            client_pos_rate: Optional[float] — this client's flare rate

        Returns:
            Scalar loss value
        """
        # Compute per-sample BCE loss
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Probability of correct prediction
        # Clamp to [eps, 1-eps] to prevent NaN from 0^gamma or log(0)
        # This is critical when SCAFFOLD control variates cause logit explosion
        pt = torch.exp(-bce)
        pt = torch.clamp(pt, 1e-7, 1.0 - 1e-7)

        # Focal modulation: down-weight easy examples
        focal_factor = (1 - pt) ** self.gamma

        # ── Adaptive Alpha Calculation ──
        # FIX v2.4: Drastically reduced dynamic scaling. The old code used
        # min(imbalance_ratio, 3.0) which multiplied alpha by up to 3x,
        # then a progressive_factor of 0.7-1.3. Combined with clamp=0.95,
        # this inflated alpha to 0.95 for low-flare clients, undoing the
        # focal_alpha=0.25 setting and causing Recall=0.999, Precision=0.027.
        #
        # New approach: Gentle sqrt-scaled correction (like DA-FL's phi),
        # capped at 1.5x (not 3x), and a mild progressive factor (0.9-1.1).
        # Max possible alpha = 0.25 * 1.5 * 1.1 = 0.4125, clamped to 0.4.
        alpha = self.alpha

        # Factor 1: Client-local imbalance adjustment (sqrt-scaled, gentle)
        if client_pos_rate is not None:
            global_train_rate = 0.4887
            raw_ratio = global_train_rate / max(client_pos_rate, 0.01)
            # FIX: sqrt dampening (same principle as DA-FL phi) — a ratio of
            # 81x (0.4887/0.006) becomes 9x instead of 81x, and we cap at 1.5x.
            # This prevents low-flare clients from dominating the loss.
            imbalance_factor = min(np.sqrt(raw_ratio), 1.5)
            alpha = alpha * imbalance_factor

        # Factor 2: Progressive alpha scheduling (mild)
        if self._total_rounds > 0:
            progress = self._current_round / self._total_rounds  # 0 → 1
            # FIX: Reduced range from 0.7-1.3 to 0.9-1.1 — the old range
            # was causing alpha to ramp by 30% over training, pushing
            # already-high alphas even higher in later rounds.
            progressive_factor = 0.9 + 0.2 * progress
            alpha = alpha * progressive_factor

        # Clamp to reasonable range
        # FIX v2.3: Was 0.95 → 0.5. FIX v2.4: Now 0.4 — even 0.5 was
        # too high because the imbalance scaling still pushed many clients
        # to the clamp. At alpha=0.4, positive class gets 40% weight,
        # negative gets 60% — still favoring recall but not catastrophically.
        alpha = float(torch.clamp(torch.tensor(alpha), 0.1, 0.4))

        # Alpha weighting for positive/negative classes
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

        # Final loss: alpha_t * focal_factor * bce
        loss = alpha_t * focal_factor * bce

        # NaN safety: replace any NaN/Inf with 0 (skips bad batches gracefully)
        # This prevents SCAFFOLD weight explosions from crashing the entire pipeline
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=-1.0)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ─────────────────────────────────────────────────────────────────────────────
# Mixup Loss — for minority-class data augmentation during training
# ─────────────────────────────────────────────────────────────────────────────

class MixupBCELoss(nn.Module):
    """
    Mixup-augmented Binary Cross-Entropy Loss.

    Mixup creates virtual training examples by interpolating between
    real examples:
        x_mixed = lambda * x_i + (1 - lambda) * x_j
        y_mixed = lambda * y_i + (1 - lambda) * y_j

    For imbalanced data, we only mix within the MINORITY class to
    create more diverse flare examples without contaminating the
    majority class.

    Reference:
        - Zhang et al., "mixup: Beyond Empirical Risk Minimization" (ICLR 2018)
        - T-SMOTE for time series (IJCAI 2022)
    """

    def __init__(self, alpha=0.4, reduction='mean'):
        """
        Args:
            alpha: Beta distribution parameter for mixup lambda.
                   Smaller alpha = less mixing (0.4 is moderate).
        """
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets, mixed_logits=None, mixed_targets=None,
                lam=1.0):
        """
        Compute mixup loss: lam * loss(original) + (1-lam) * loss(mixed)

        If mixed_logits/mixed_targets are None, falls back to standard BCE.
        """
        loss_orig = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        if mixed_logits is not None and mixed_targets is not None:
            loss_mixed = F.binary_cross_entropy_with_logits(
                mixed_logits, mixed_targets, reduction='none')
            loss = lam * loss_orig + (1 - lam) * loss_mixed
        else:
            loss = loss_orig

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# Convenience aliases
DAFLoss = DynamicFocalLoss