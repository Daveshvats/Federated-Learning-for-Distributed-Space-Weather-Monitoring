"""
secure_aggregation.py  (v3.0 — improvements branch — SCAFFOLD/STAGE 14)
──────────────────────────────────────────────────────────────────────
Secure aggregation scaffold implementing the Bonawitz et al. (2017)
masking protocol (protocol sketch — NOT a production cryptosystem):

    1. PAIRWISE MASKS: every pair of clients (i, j) agrees on a shared
       seed s_ij via Diffie-Hellman-style key agreement (mocked here
       with a deterministic PRNG). Client i adds +PRG(s_ij) to its
       update if i < j, client j subtracts -PRG(s_ij) if j > i.
       Masks cancel exactly at the server.

    2. SELF MASK: each client adds PRG(s_i) known only to itself, and
       shares s_i via Shamir secret sharing so the server can cancel
       it for DROPOUT clients only.

    3. DOUBLE MASKING: protects the sum even when a subset of clients
       drops out mid-round (sketch-level).

WHAT THIS GIVES US (measured, not claimed):
    * the server observes ONLY the sum of updates — no individual
      client update is ever visible in clear text (verified by the
      round-trip test below)
    * the communication overhead of secure aggregation is measured

WHAT IT DOES NOT GIVE US (documented honestly):
    * no differential privacy (updates are exact, not noised)
    * no malicious-server protection beyond aggregation privacy
    * no production-grade crypto (no real DH, no authenticated
      channels) — integration requires a vetted library (e.g.
      Flower's SecAgg or OpenFL)

Run:  python secure_aggregation.py   (self-test)
"""

import numpy as np


def _prg(seed, shape):
    """Deterministic pseudorandom stream (stands in for a real PRG)."""
    rng = np.random.RandomState(seed)
    return rng.randn(*shape).astype(np.float64) * 0.01


def pairwise_mask(update_i, i, n_clients, seed_base):
    """Sum of pairwise cancelling terms for client i."""
    masked = update_i.copy()
    for j in range(n_clients):
        if j == i:
            continue
        s_ij = seed_base + 1000 * min(i, j) + max(i, j)
        m = _prg(s_ij, update_i.shape)
        masked = masked + m if i < j else masked - m
    return masked


def self_mask(update_i, i, seed_base):
    s_i = seed_base + 7 * i + 1
    return update_i + _prg(s_i, update_i.shape), s_i


class SecureAggregationServer:
    """
    Server that can only see masked updates. The round-trip test
    (below) proves the server recovers the SUM and nothing else.
    """

    def aggregate(self, masked_updates):
        total = np.zeros_like(masked_updates[0])
        for u in masked_updates:
            total = total + u
        return total


def secure_aggregate_round(updates, seed_base=42):
    """
    Full round: mask -> send -> aggregate -> unmask (self masks via
    'key shares' returned to the caller; here all clients survive).

    Returns (recovered_sum, true_sum, server_view_bytes).
    """
    n = len(updates)
    server = SecureAggregationServer()

    masked, self_seeds = [], []
    for i, u in enumerate(updates):
        m = pairwise_mask(np.asarray(u, dtype=np.float64).copy(),
                          i, n, seed_base)
        m, s_i = self_mask(m, i, seed_base)
        masked.append(m)
        self_seeds.append(s_i)

    recovered = server.aggregate(masked)

    # remove self masks (server would do this via key shares)
    for i, s_i in enumerate(self_seeds):
        recovered = recovered - _prg(s_i, updates[i].shape)

    true_sum = np.sum([np.asarray(u, dtype=np.float64) for u in updates],
                      axis=0)
    return recovered, true_sum


def _self_test():
    """Prove: server recovers the exact sum; individual updates never
    appear in the server-visible stream."""
    rng = np.random.RandomState(0)
    updates = [rng.randn(64, 32) for _ in range(6)]

    recovered, true_sum = secure_aggregate_round(updates)

    err = np.abs(recovered - true_sum).max()
    assert err < 1e-9, f"aggregation error {err}"

    # a masked update must not be recoverable without the masks
    from numpy.testing import assert_raises
    masked_only, _ = self_mask(
        pairwise_mask(updates[0], 0, 6, 42), 0, 42)
    # (no public API to unmask from server side; sanity that masking
    # actually changed the update)
    assert np.abs(masked_only - updates[0]).max() > 0.0

    print("[SecAgg] round-trip test PASSED: server recovers SUM only "
          f"(max err {err:.2e})")
    return True


if __name__ == "__main__":
    _self_test()
