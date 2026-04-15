"""
model.py
─────────
Defines model architectures for federated solar flare prediction.

Models:
  1. SolarMLP  — 3-hidden-layer MLP (baseline)
  2. SolarLSTM — Bidirectional LSTM for temporal dynamics (2026 SOTA)

Why not XGBoost for federation?
  FedAvg/FedProx/SCAFFOLD work by averaging *numerical weight tensors*.
  XGBoost stores trees (discrete structures) which cannot be averaged.
  A centralised XGBoost is still run as an upper-bound baseline.

References:
  - LSTM for solar flare: arxiv 2507.05313v1 (2025)
  - Transformer for solar flare: arxiv 2510.23400 (2025)
  - SCAFFOLD: Karimireddy et al., "SCAFFOLD: Stochastic Controlled
    Averaging for Federated Learning" (ICML 2020)
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import (INPUT_DIM, HIDDEN_DIMS, DROPOUT, LR, BATCH_SIZE, RANDOM_STATE,
                    USE_LSTM, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS,
                    LSTM_DROPOUT, LSTM_BIDIRECTIONAL)


# ─────────────────────────────────────────────────────────────────────────────
# Global device — set once, used everywhere
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_device():
    return DEVICE


# ─────────────────────────────────────────────────────────────────────────────
# Dataset wrapper
# ─────────────────────────────────────────────────────────────────────────────

class SolarDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        SolarDataset(X, y),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        drop_last=False
    )


# ─────────────────────────────────────────────────────────────────────────────
# MLP architecture (baseline)
# ─────────────────────────────────────────────────────────────────────────────

class SolarMLP(nn.Module):
    """
    3-hidden-layer MLP with BatchNorm and Dropout.
    Output: single raw logit (binary classification).
    Architecture: INPUT → 128 → 64 → 32 → 1

    Outputs RAW LOGITS (no sigmoid) — consistent with SolarLSTM.
    The loss functions (FocalLoss, FedFocalLoss) use
    binary_cross_entropy_with_logits which applies sigmoid internally.
    The evaluate_model function also applies sigmoid for probability output.

    Compatible with FedAvg/FedProx/SCAFFOLD weight averaging.
    """

    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        dims = [input_dim] + HIDDEN_DIMS

        layers = []
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
            ]
        # Single output neuron → raw logit (no sigmoid)
        # Sigmoid is applied externally by loss/eval functions
        layers.append(nn.Linear(dims[-1], 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# LSTM architecture (2026 SOTA for temporal solar flare prediction)
# ─────────────────────────────────────────────────────────────────────────────

class SolarLSTM(nn.Module):
    """
    LSTM-based model for solar flare prediction with temporal dynamics.

    Architecture:
      Input (batch, seq_len, input_size)
        → LSTM (hidden_size, num_layers, bidirectional)
        → Last hidden state
        → FC head (hidden → 64 → 1)

    Key advantages over MLP:
      1. Preserves temporal ordering of 60 timesteps (no information loss)
      2. Captures long-range dependencies in magnetic field evolution
      3. Learns when features change (derivative info), not just static values

    References:
      - "Solar Flare Prediction Using LSTM and DLSTM" (arxiv 2507.05313, 2025)
      - "An Interpretable LSTM Network for Solar Flare Prediction" (2026)
      - "Comprehensive review of deep learning for solar irradiance" (2026)
    """

    def __init__(self, input_size=24, hidden_size=LSTM_HIDDEN_SIZE,
                 num_layers=LSTM_NUM_LAYERS, dropout=LSTM_DROPOUT,
                 bidirectional=LSTM_BIDIRECTIONAL):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # Determine the actual feature dim after LSTM
        lstm_output_dim = hidden_size * (2 if bidirectional else 1)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1)
        )

        # Xavier initialization for stable training
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0.0)
                # Set forget gate bias to 1.0 (helps with long-term memory)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)

        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: 3D input tensor of shape (batch, seq_len, input_size)
               The LSTM expects temporal data with shape (batch, 60, 24).

        Returns:
            Raw logits of shape (batch,)
        """
        # LSTM requires 3D input: (batch, seq_len, input_size)
        if x.dim() == 2:
            raise ValueError(
                f"SolarLSTM expects 3D input (batch, seq_len, input_size), "
                f"but got 2D input of shape {x.shape}. "
                f"When USE_LSTM=True, the data pipeline should provide 3D arrays. "
                f"Check that load_and_scale_3d_data() is being used."
            )

        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state from top layer
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_hidden = h_n[-1]  # (batch, hidden_size)

        # Classification head
        logits = self.head(last_hidden).squeeze(-1)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_weights(model) -> list:
    """Return a list of numpy arrays (one per parameter tensor)."""
    return [p.data.cpu().numpy().copy() for p in model.parameters()]


def set_weights(model, weights: list) -> None:
    """Load a list of numpy arrays into model parameters in-place."""
    with torch.no_grad():
        for p, w in zip(model.parameters(), weights):
            p.data.copy_(torch.tensor(w, dtype=torch.float32))


def clone_model(model) -> nn.Module:
    """Deep copy of the model, placed on the same device as the original."""
    device = next(model.parameters()).device
    input_dim = None

    # Detect model type and input dim
    if isinstance(model, SolarMLP):
        input_dim = model.net[0].in_features
        new_model = SolarMLP(input_dim=input_dim)
    elif isinstance(model, SolarLSTM):
        input_dim = model.lstm.input_size
        new_model = SolarLSTM(
            input_size=input_dim,
            hidden_size=model.hidden_size,
            num_layers=model.num_layers,
            dropout=0.0,  # Don't add dropout during clone to avoid warnings
            bidirectional=model.bidirectional
        )
    else:
        # Generic fallback
        new_model = type(model)(**{k: v for k, v in model.__init__.__code__.co_varnames
                                    if k != 'self' and k in model.__dict__})

    new_model.load_state_dict(copy.deepcopy(model.state_dict()))
    return new_model.to(device)


def make_fresh_model(input_dim: int = INPUT_DIM, use_lstm: bool = None):
    """
    Create a new model and move it to GPU if available.

    Args:
        input_dim: Number of input features (used only for MLP)
        use_lstm: If True, create SolarLSTM; if False, create SolarMLP
                  If None, reads USE_LSTM from config

    Returns
    -------
    model  : SolarMLP or SolarLSTM (already on DEVICE)
    device : torch.device
    """
    if use_lstm is None:
        import config as cfg
        use_lstm = cfg.USE_LSTM

    torch.manual_seed(RANDOM_STATE)

    if use_lstm:
        print(f"[Model] Creating SolarLSTM (hidden={LSTM_HIDDEN_SIZE}, "
              f"layers={LSTM_NUM_LAYERS}, bidir={LSTM_BIDIRECTIONAL})")
        model = SolarLSTM(input_size=24).to(DEVICE)
    else:
        print(f"[Model] Creating SolarMLP (input_dim={input_dim})")
        model = SolarMLP(input_dim=input_dim).to(DEVICE)

    if DEVICE.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        mem_mb = torch.cuda.memory_allocated() / 1024 ** 2
        print(f"[Model] GPU: {gpu_name} | allocated: {mem_mb:.1f} MB")
    else:
        print("[Model] CUDA not available — using CPU. "
              "Install CUDA PyTorch build for GPU acceleration.")

    # Print model size
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Parameters: {n_params:,}")

    return model, DEVICE


def is_lstm_model(model: nn.Module) -> bool:
    """Check if a model is an LSTM model (needs 3D input)."""
    return isinstance(model, SolarLSTM)


class PersonalizedSolarMLP(nn.Module):
    """
    MLP with personalized head per client.

    Architecture:
    - Shared backbone (trained via FL, averaged across clients)
    - Personal head (kept local, never shared)

    Reference:
    - "Ditto: Fair and Robust Federated Learning" (ICML 2021)
    - "pFedFDA: Personalized Federated Learning for Heterogeneous Data"
    """

    def __init__(self, input_dim=24, hidden_dims=[128, 64, 32], dropout=0.3):
        super().__init__()

        # Shared backbone (will be averaged in FL)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

        # Personal head (client-specific, NOT shared)
        self.personal_head = nn.Sequential(
            nn.Linear(hidden_dims[2], 16),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout * 0.5),
            nn.Linear(16, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        features = self.backbone(x)
        output = self.personal_head(features)
        return output.squeeze()

    def get_backbone_params(self):
        """Return only backbone parameters (for FL averaging)"""
        return list(self.backbone.parameters())

    def get_personal_params(self):
        """Return personal head parameters (local only)"""
        return list(self.personal_head.parameters())