"""
communication_cost.py  (v3.0 — improvements branch)
────────────────────────────────────────────────────
Communication-cost instrumentation (revision Stage 15).

Federated deployment viability depends on:
    * bytes per client per round (uplink: delta/weights; downlink: global)
    * total bytes over the run
    * simulated client-dropout robustness (partial participation)

All sizes are measured on REAL parameter tensors of the trained model —
no hand-waved numbers. float32 is the wire format baseline; optional
fp16/int8 quantisation deltas are reported as future compression.
"""

import numpy as np


def model_size_bytes(model, dtype_bytes=4):
    """Total bytes of all parameters at the given precision."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * dtype_bytes, total_params


def measure_communication(model, n_clients, n_rounds, dtype_bytes=4,
                          fraction_fit=1.0):
    """
    Round-trip cost model:
      downlink: server -> each participating client: full global weights
      uplink:   each participating client -> server: full local weights
    (FedAvg transmits full weights; delta compression is future work —
    noted in the report.)
    """
    size_bytes, total_params = model_size_bytes(model, dtype_bytes)

    participating = max(1, int(fraction_fit * n_clients))

    per_client_per_round = {
        "downlink_bytes": size_bytes,
        "uplink_bytes": size_bytes,
        "roundtrip_bytes": 2 * size_bytes,
    }
    per_round = {
        "participating_clients": participating,
        "total_bytes": 2 * size_bytes * participating,
    }
    total = {
        "rounds": n_rounds,
        "total_bytes_all_clients": per_round["total_bytes"] * n_rounds,
        "total_mb_all_clients": per_round["total_bytes"] * n_rounds / 1e6,
        "per_client_mb_full_run": 2 * size_bytes * n_rounds / 1e6,
        "parameters": total_params,
        "dtype_bytes": dtype_bytes,
        "note": "full-weight FedAvg transmission; delta/quantisation "
                "compression not implemented (future work)",
    }

    print(f"[Comm] model: {total_params:,} params "
          f"({size_bytes/1e6:.2f} MB @ {dtype_bytes*8}-bit)")
    print(f"[Comm] per client per round: {2*size_bytes/1e6:.2f} MB "
          f"(up+down)")
    print(f"[Comm] full run ({n_rounds} rounds, {participating} clients): "
          f"{total['total_mb_all_clients']:.1f} MB total, "
          f"{total['per_client_mb_full_run']:.1f} MB per client")

    return {
        "per_client_per_round": per_client_per_round,
        "per_round": per_round,
        "total": total,
    }


def simulate_dropout(shards, n_rounds, drop_rate=0.2, seed=0):
    """
    Failure-mode simulation (Stage 15): with probability drop_rate a
    client misses a round. Returns the participation matrix so robustness
    experiments can replay identical schedules across algorithms.

    NOTE: the FL loop itself already tolerates partial participation
    (FRACTION_FIT); this utility produces the schedule for controlled
    experiments comparing algorithms under the SAME dropout pattern.
    """
    rng = np.random.RandomState(seed)
    n_clients = len(shards)
    schedule = (rng.rand(n_rounds, n_clients) >= drop_rate).astype(int)
    # guarantee at least one participant per round
    for r in range(n_rounds):
        if schedule[r].sum() == 0:
            schedule[r, rng.randint(n_clients)] = 1
    stats = {
        "drop_rate": drop_rate,
        "mean_participation": float(schedule.mean()),
        "min_clients_per_round": int(schedule.sum(axis=1).min()),
        "schedule_shape": list(schedule.shape),
    }
    return schedule, stats
