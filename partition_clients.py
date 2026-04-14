"""
partition_clients.py
─────────────────────
Splits the global training set into N_CLIENTS shards.

Partitioning strategy: HARPNUM_MOD (active-region ID modulo N_CLIENTS).
This simulates each regional observatory monitoring a different subset of
solar active regions — the defining privacy motivation of this paper.

The resulting shards are intentionally non-IID (different class ratios per
client) to stress-test FedProx's heterogeneity handling vs. plain FedAvg.
"""

import numpy as np
from typing import List, Tuple

from config import N_CLIENTS, FEATURE_COLS, RANDOM_STATE, CLIENT_NAMES
from data_preparation import apply_smote


ClientShard = Tuple[np.ndarray, np.ndarray]  # (X_local, y_local)


def partition_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    df_train_meta: np.ndarray   # HARPNUM_MOD column for the train indices
) -> List[ClientShard]:
    """
    Assign each training sample to a client based on HARPNUM_MOD.
    Applies SMOTE independently on each client shard so no client
    shares synthetic minority samples with another.

    Returns a list of (X_local, y_local) tuples, one per client.
    """
    shards: List[ClientShard] = []

    for client_id in range(N_CLIENTS):
        mask = (df_train_meta == client_id)
        X_c = X_train[mask]
        y_c = y_train[mask]

        if len(X_c) == 0:
            print(f"[Partition] Warning: client {client_id} has 0 samples — skipping.")
            shards.append((np.empty((0, X_train.shape[1])), np.empty(0, dtype=int)))
            continue

        # Per-client SMOTE (minority oversampling stays local)
        X_c, y_c = apply_smote(X_c, y_c)

        flare_pct = y_c.mean() * 100
        print(f"[Partition] {CLIENT_NAMES[client_id]:<28} | "
              f"n={len(X_c):>5} | flare rate: {flare_pct:.1f}%")
        shards.append((X_c.astype(np.float32), y_c.astype(int)))

    print()
    return shards


def partition_data_dirichlet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 0.5
) -> List[ClientShard]:
    """
    Alternative: Dirichlet-based non-IID partitioning (alpha controls
    heterogeneity; lower alpha = more skewed per-client label distribution).
    Use this if your real SWAN-SF file lacks a HARPNUM column.
    """
    np.random.seed(RANDOM_STATE)
    n_classes = len(np.unique(y_train))
    shards: List[ClientShard] = [[] for _ in range(N_CLIENTS)]

    for c in range(n_classes):
        idx_c = np.where(y_train == c)[0]
        np.random.shuffle(idx_c)
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, N_CLIENTS))
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, proportions)
        for k, split in enumerate(splits):
            shards[k].extend(split.tolist())

    result: List[ClientShard] = []
    for client_id, idx in enumerate(shards):
        if len(idx) == 0:
            result.append((np.empty((0, X_train.shape[1])), np.empty(0, dtype=int)))
            continue
        X_c = X_train[idx]
        y_c = y_train[idx]
        X_c, y_c = apply_smote(X_c, y_c)
        flare_pct = y_c.mean() * 100
        print(f"[Partition/Dir] {CLIENT_NAMES[client_id]:<28} | "
              f"n={len(X_c):>5} | flare rate: {flare_pct:.1f}%")
        result.append((X_c.astype(np.float32), y_c.astype(int)))

    print()
    return result
