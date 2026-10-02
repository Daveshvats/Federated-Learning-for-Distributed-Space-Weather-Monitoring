"""
partition_clients.py  (v3.0 — improvements branch)
──────────────────────────────────────────────────
Partition training data into regional client shards.

v3.0 FIXES (audit finding B2, P0):
  - Dirichlet partitioning is now DISJOINT BY CONSTRUCTION. The old
    implementation sampled indices independently for every
    (client, class) pair with np.random.choice, so the same underlying
    sample could land in two different clients, and int() truncation
    silently dropped samples. Both problems invalidate the
    "mutually exclusive sovereign shards" simulation premise.
  - Allocation now uses a single permutation per class, split at
    cumulative Dirichlet proportions -> every sample is assigned to
    EXACTLY ONE client and total coverage is 100%.
  - The index assignment is returned (and auditable) so
    leakage_audit/audit_leakage.py can verify disjointness at runtime.
  - Seeds are parameters (multi-seed experiments), not module constants.
  - Client selection bug in the FL loops (B15) is fixed by returning
    shard lists whose length is authoritative.

New in v3.0:
  - partition_data_geographic(): observatory-informed partition for
    physical-realism experiments (Stage 11). Uses active-region IDs
    when available; falls back to disjoint Dirichlet with a warning
    documenting that the cleaned SWAN-SF export does not carry
    per-sample region IDs.
"""

import numpy as np
from collections import Counter

CLIENT_NAMES = [
    # Plain neutral identifiers (v4.1): simulated shards, not regions
    # or institutions (mirrors config.py).
    "Client A",
    "Client B",
    "Client C",
    "Client D",
    "Client E",
    "Client F",
]


# ─────────────────────────────────────────────────────────────────────────────
# CORE: disjoint Dirichlet index allocation (the auditable primitive)
# ─────────────────────────────────────────────────────────────────────────────

def allocate_indices_dirichlet(y, alpha=1.0, n_clients=6, seed=0,
                               min_samples=0, max_tries=50):
    """
    Disjoint Dirichlet class-skew allocation.

    For each class c, draw one Dirichlet proportion vector p_c over
    clients, permute the class's sample indices once, and split the
    permutation at cumulative cut-points floor(cumsum(p_c) * n_c).
    Guarantees:
        * every sample assigned exactly once  (coverage = 100%)
        * no sample appears in two clients     (overlap = 0)
        * realised per-class proportions ~ Dirichlet(alpha)

    A minimum-samples-per-client constraint is enforced by rejection
    resampling of the Dirichlet draw (bounded by max_tries), falling
    back to the best draw found.

    Returns
    -------
    assignment : np.ndarray shape (len(y),), values in [0, n_clients)
        client id for every sample index (position-aligned with y).
    """
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    n = len(y)
    classes = np.unique(y)
    assignment = np.full(n, -1, dtype=int)

    best_assignment, best_deficit = None, np.inf

    for _ in range(max_tries):
        assignment.fill(-1)
        for c in classes:
            idx_c = np.where(y == c)[0]
            n_c = len(idx_c)
            # Dirichlet proportions over clients for this class
            props = rng.dirichlet([alpha] * n_clients)
            # cumulative cut-points over a FIXED permutation
            perm = rng.permutation(n_c)
            cuts = np.floor(np.cumsum(props) * n_c).astype(int)
            cuts[-1] = n_c  # exact coverage
            start = 0
            for k, end in enumerate(cuts):
                assignment[idx_c[perm[start:end]]] = k
                start = end

        # min-samples check
        sizes = np.bincount(assignment, minlength=n_clients)
        deficit = int((sizes < min_samples).sum()) if min_samples > 0 else 0
        if deficit == 0:
            return assignment
        if deficit < best_deficit:
            best_deficit, best_assignment = deficit, assignment.copy()

    if best_assignment is not None and min_samples > 0:
        print(f"[Partition] WARNING: could not satisfy min_samples={min_samples} "
              f"with alpha={alpha} after {max_tries} draws; "
              f"using best draw ({best_deficit} clients below minimum).")
    if best_assignment is not None:
        return best_assignment
    return assignment


def _audit_stats(assignment, y, n_clients, verbose=True):
    """Print and return disjointness/coverage statistics (audit hook)."""
    sizes = np.bincount(assignment, minlength=n_clients)
    coverage = (assignment >= 0).sum() / len(y)
    lines = []
    for k in range(n_clients):
        mask = assignment == k
        rate = y[mask].mean() * 100 if mask.sum() else 0.0
        lines.append(f"[Partition] {CLIENT_NAMES[k]:30s} | "
                     f"n={sizes[k]:>7,} | flare rate: {rate:5.1f}%")
    stats = {
        "sizes": sizes.tolist(),
        "coverage": coverage,
        "total_assigned": int(sizes.sum()),
        "n_train": int(len(y)),
    }
    if verbose:
        print("\n".join(lines))
        print(f"[Partition] coverage: {sizes.sum():,}/{len(y):,} "
              f"({coverage:.1%}) — disjoint by construction")
    return stats


def partition_indices_dirichlet(X_train, y_train, alpha=1.0, n_clients=6,
                                seed=0, min_samples=100, verbose=True):
    """
    Returns (assignment, stats): the auditable primitive used by
    audit_leakage.py and run_multiseed.py.
    """
    y = np.asarray(y_train)
    assignment = allocate_indices_dirichlet(y, alpha=alpha, n_clients=n_clients,
                                            seed=seed, min_samples=min_samples)
    stats = _audit_stats(assignment, y, n_clients, verbose=verbose)
    return assignment, stats


def partition_data_dirichlet(X_train, y_train, alpha=1.0, n_clients=6,
                             seed=0, min_samples=100, return_indices=False,
                             verbose=True):
    """
    Backward-compatible shard API: list of (X_client, y_client) tuples.

    v3.0: disjoint by construction, seed-parameterised, optional
    index return for auditing.
    """
    assignment, stats = partition_indices_dirichlet(
        X_train, y_train, alpha=alpha, n_clients=n_clients, seed=seed,
        min_samples=min_samples, verbose=verbose)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    shards = []
    for k in range(n_clients):
        mask = assignment == k
        if mask.sum() == 0:
            print(f"[Partition] WARNING: {CLIENT_NAMES[k]} received 0 samples")
            shards.append((X_train[mask], y_train[mask]))
            continue
        # shuffle within client (deterministic per seed+client)
        rng = np.random.RandomState(seed * 1000 + k)
        idx = np.where(mask)[0]
        idx = idx[rng.permutation(len(idx))]
        shards.append((X_train[idx], y_train[idx]))

    if return_indices:
        return shards, assignment
    return shards


# ─────────────────────────────────────────────────────────────────────────────
# Geographic / observatory-informed partition (Stage 11 — physical realism)
# ─────────────────────────────────────────────────────────────────────────────

def partition_data_geographic(X_train, y_train, region_ids=None, n_clients=6,
                              seed=0, alpha_fallback=1.0, verbose=True):
    """
    Observatory-informed partition.

    If per-sample active-region IDs are available, regions are assigned to
    clients (hash-consistent, so a region's windows never split across
    clients — this also removes region-level leakage *within* the training
    federation), producing physically heterogeneous shards.

    The cleaned SWAN-SF export does NOT carry region IDs, so the default
    behaviour falls back to disjoint Dirichlet and prints an explicit
    warning: physical heterogeneity is then SIMULATED label skew, not real
    observatory coverage (documented limitation, audit claim #7).
    """
    if region_ids is None:
        print("[Partition] WARNING: no region IDs available — geographic "
              "partition falls back to SIMULATED Dirichlet label skew. "
              "Real observatory coverage requires raw SWAN-SF metadata.")
        return partition_data_dirichlet(X_train, y_train,
                                        alpha=alpha_fallback, n_clients=n_clients,
                                        seed=seed, verbose=verbose)

    region_ids = np.asarray(region_ids)
    unique_regions = np.unique(region_ids)
    rng = np.random.RandomState(seed)
    # hash-consistent region -> client map
    region_to_client = {r: int(rng.randint(n_clients)) for r in unique_regions}
    assignment = np.array([region_to_client[r] for r in region_ids], dtype=int)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    _audit_stats(assignment, y_train, n_clients, verbose=verbose)

    shards = []
    for k in range(n_clients):
        mask = assignment == k
        shards.append((X_train[mask], y_train[mask]))
    return shards


# ─────────────────────────────────────────────────────────────────────────────
# Legacy balanced partition (kept for fallback; already disjoint)
# ─────────────────────────────────────────────────────────────────────────────

def partition_data(X_train, y_train, harpnum_mod=None, n_clients=6,
                   min_samples_per_client=100, verbose=True, seed=0):
    """
    Legacy entry point. HARPNUM-based partitioning is DEPRECATED: the
    previous implementation partitioned on a fabricated HARPNUM_MOD
    column (audit finding B13). It now always returns a balanced
    stratified partition unless genuinely real HARP numbers are passed
    in (uniqueness check enforced).
    """
    if harpnum_mod is not None:
        harpnum_mod = np.asarray(harpnum_mod).flatten()
        unique = np.unique(harpnum_mod)
        genuinely_real = (len(unique) > 10 * n_clients and
                          not np.all(np.sort(unique) == np.arange(len(unique))))
        if not genuinely_real:
            if verbose:
                print("[Partition] HARPNUM vector looks synthetic — "
                      "ignoring it (audit B13) and using balanced partition.")
            harpnum_mod = None

    if harpnum_mod is None:
        return _partition_balanced_random(X_train, y_train, n_clients,
                                          verbose=verbose, seed=seed)

    client_ids = harpnum_mod % n_clients
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    shards = []
    for cid in range(n_clients):
        mask = client_ids == cid
        if mask.sum() < min_samples_per_client:
            continue
        shards.append((X_train[mask], y_train[mask]))
    if not shards:
        return _partition_balanced_random(X_train, y_train, n_clients,
                                          verbose=verbose, seed=seed)
    return shards


def _partition_balanced_random(X_train, y_train, n_clients=6, verbose=True,
                               seed=0):
    """
    Stratified balanced fallback (disjoint: contiguous chunks of one
    permutation per class). seed-parameterised in v3.0.
    """
    rng = np.random.RandomState(seed)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    if verbose:
        print("\n      [Fallback] Using STRATIFIED BALANCED partition...")

    flare_mask = y_train == 1
    X_flare, y_flare = X_train[flare_mask], y_train[flare_mask]
    X_noflare, y_noflare = X_train[~flare_mask], y_train[~flare_mask]

    n_flares, n_noflares = len(y_flare), len(y_noflare)
    if verbose:
        print(f"      Total flares: {n_flares:,}, Non-flares: {n_noflares:,}")

    perm_f = rng.permutation(n_flares)
    perm_nf = rng.permutation(n_noflares)
    X_flare, y_flare = X_flare[perm_f], y_flare[perm_f]
    X_noflare, y_noflare = X_noflare[perm_nf], y_noflare[perm_nf]

    # split points (exact coverage, disjoint)
    f_splits = np.array_split(np.arange(n_flares), n_clients)
    nf_splits = np.array_split(np.arange(n_noflares), n_clients)

    shards = []
    for k in range(n_clients):
        X_c = np.vstack([X_flare[f_splits[k]], X_noflare[nf_splits[k]]])
        y_c = np.concatenate([y_flare[f_splits[k]], y_noflare[nf_splits[k]]])
        perm = rng.permutation(len(y_c))
        shards.append((X_c[perm], y_c[perm]))
        if verbose:
            print(f"[Partition] {CLIENT_NAMES[k]:30s} | "
                  f"n={len(y_c):>7,} | flare rate: {y_c.mean()*100:5.1f}%")

    if verbose:
        total = sum(len(s[1]) for s in shards)
        print(f"\n      ✓ Distributed {total:,} samples across {len(shards)} clients")
    return shards


if __name__ == '__main__':
    # Quick self-test with synthetic labels
    rng = np.random.RandomState(0)
    y = rng.binomial(1, 0.05, 20000)
    X = rng.randn(20000, 10)
    assignment, stats = partition_indices_dirichlet(X, y, alpha=1.0, seed=42)
    sizes = np.bincount(assignment, minlength=6)
    assert sizes.sum() == len(y), "coverage broken"
    assert (assignment >= 0).all(), "unassigned samples"
    print("\nSELF-TEST PASSED: disjoint, full coverage")
