"""
losses.py - Dynamic Adaptive Focal Loss for Imbalanced Federated Learning
=========================================================================

Implements 2024-2026 SOTA loss functions for handling extreme class imbalance
in federated learning settings.

Key Insight:
- Your training data: ~49% flares (balanced after preprocessing)
- Your test data: ~1.9% flares (real-world severe imbalance)
- Standard BCE fails here → Need focal loss to focus on hard/rare examples

References:
- Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
- "Synergetic Focal Loss for FL" (2024, arXiv)
- "Dynamic Adaptive Focal Loss" (2025-2026 medical FL papers)
"""

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
        """
        Args:
            gamma: Focusing parameter (higher = more focus on hard examples)
                   - gamma=0: Same as standard BCE
                   - gamma=2: Recommended default (from original Focal Loss paper)
                   - gamma=5: Very aggressive focusing (use if imbalance is extreme)
            
            base_alpha: Weight for positive class (flares)
                       - alpha < 0.5: Model more conservative (fewer flare predictions)
                       - alpha = 0.25: Default (good starting point)
                       - alpha > 0.5: Model more aggressive (more flare predictions)
            
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.gamma = gamma
        self.base_alpha = base_alpha
        self.reduction = reduction
        
    def forward(self, logits, targets, client_pos_rate=None):
        """
        Compute dynamic focal loss.
        
        Args:
            logits: Raw model outputs (before sigmoid), shape (N,)
            targets: Binary labels (0 or 1), shape (N,)
            client_pos_rate: Optional[float]
                           This client's flare rate (0.0 to 1.0)
                           If provided, dynamically adjusts alpha
                           If None, uses base_alpha
        
        Returns:
            Scalar loss value
        """
        # Compute per-sample BCE loss
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Probability of correct prediction (pt)
        # pt is high when model is confident and correct
        # pt is low when model is wrong or uncertain
        pt = torch.exp(-bce)
        
        # Focal factor: (1 - pt)^gamma
        # - When pt → 1 (easy example): factor → 0 (down-weighted)
        # - When pt → 0 (hard example): factor → 1 (kept at full weight)
        focal_factor = (1 - pt) ** self.gamma
        
        # DYNAMIC ALPHA adjustment based on client's local distribution
        alpha = self.base_alpha
        
        if client_pos_rate is not None:
            # Global training rate is ~0.4887 (balanced)
            # Test rate is ~0.019 (severely imbalanced)
            # If client_pos_rate differs from global, adjust alpha
            
            global_pos_rate = 0.4887  # Your balanced training set rate
            
            # Adjustment formula:
            # - If client has FEWER flares than global: increase alpha (focus more on flares)
            # - If client has MORE flares than global: decrease alpha
            adjustment = 1.0 + (global_pos_rate - client_pos_rate) * 0.8
            
            alpha = self.base_alpha * adjustment
            
            # Clamp to reasonable range [0.1, 0.9] to avoid extremes
            alpha = torch.clamp(torch.tensor(alpha), 0.1, 0.9).item()
        
        # Alpha weighting: alpha_t for positive class, (1-alpha) for negative
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        
        # Final loss: alpha_t * focal_factor * bce
        loss = alpha_t * focal_factor * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# Convenience alias
DAFLoss = DynamicFocalLoss