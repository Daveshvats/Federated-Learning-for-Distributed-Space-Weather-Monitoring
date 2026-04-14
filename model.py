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
        layers.append(nn.Linear(dims[-1], 1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers used by federated_learning.py
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
    """Deep copy of the model (used to snapshot the global model)."""
    new_model = SolarMLP(input_dim=model.net[0].in_features)
    new_model.load_state_dict(copy.deepcopy(model.state_dict()))
    return new_model


def make_fresh_model(input_dim: int = INPUT_DIM) -> SolarMLP:
    torch.manual_seed(RANDOM_STATE)
    return SolarMLP(input_dim=input_dim)
