"""
model.py
─────────
Defines the MLP used by every client and the central server.
Why MLP (not XGBoost) for federation?
  FedAvg/FedProx work by averaging *numerical weight tensors*.
  XGBoost stores trees (discrete structures) which cannot be averaged.
  The MLP gives up a small amount of raw accuracy for full FL compatibility.
  A centralised XGBoost is still run as an upper-bound baseline.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import INPUT_DIM, HIDDEN_DIMS, DROPOUT, LR, BATCH_SIZE, RANDOM_STATE


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
# MLP architecture
# ─────────────────────────────────────────────────────────────────────────────

class SolarMLP(nn.Module):
    """
    3-hidden-layer MLP with BatchNorm and Dropout.
    Output: single sigmoid neuron (binary classification probability).
    Architecture: INPUT → 128 → 64 → 32 → 1

    Uses BCELoss — one output neuron, probability in [0, 1].
    Compatible with FedAvg/FedProx weight averaging.
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
        # Single output neuron + sigmoid → probability
        layers.append(nn.Linear(dims[-1], 1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BUG FIX: must call self.net(x), not return self.net
        return self.net(x).squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_weights(model: SolarMLP) -> list:
    """Return a list of numpy arrays (one per parameter tensor)."""
    return [p.data.cpu().numpy().copy() for p in model.parameters()]


def set_weights(model: SolarMLP, weights: list) -> None:
    """Load a list of numpy arrays into model parameters in-place."""
    with torch.no_grad():
        for p, w in zip(model.parameters(), weights):
            p.data.copy_(torch.tensor(w, dtype=torch.float32))


def clone_model(model: SolarMLP) -> SolarMLP:
    """Deep copy of the model, placed on the same device as the original."""
    device = next(model.parameters()).device
    new_model = SolarMLP(input_dim=model.net[0].in_features)
    new_model.load_state_dict(copy.deepcopy(model.state_dict()))
    return new_model.to(device)


def make_fresh_model(input_dim: int = INPUT_DIM):
    """
    Create a new SolarMLP and move it to GPU if available.

    Returns
    -------
    model  : SolarMLP  (already on DEVICE)
    device : torch.device

    ALWAYS unpack both values:
        model, device = make_fresh_model()
    """
    torch.manual_seed(RANDOM_STATE)
    model  = SolarMLP(input_dim=input_dim).to(DEVICE)

    if DEVICE.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        mem_mb   = torch.cuda.memory_allocated() / 1024 ** 2
        print(f"[Model] GPU: {gpu_name} | allocated: {mem_mb:.1f} MB")
    else:
        print("[Model] CUDA not available — using CPU. "
              "Install CUDA PyTorch build for GPU acceleration.")

    return model, DEVICE