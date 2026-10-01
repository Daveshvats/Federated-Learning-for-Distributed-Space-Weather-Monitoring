"""
data_preparation.py  (v3.0 — improvements branch)
──────────────────────────────────────────────────
Loads the SWAN-SF benchmark dataset (Harvard Dataverse / cleaned pkl files)
if present, otherwise generates a physics-based synthetic fallback.

v3.0 FIXES:
  - REAL FEATURE NAMES PROPAGATE (audit B12): stat-prefixed names such as
    'mean_TOTUSJH' are kept end-to-end, so SHAP plots show physical
    feature names instead of 'feature_N'. The old code renamed them to
    feature_0..feature_143, destroying interpretability.
  - FABRICATED HARPNUM_MOD REMOVED (audit B13): the random client-id
    column masquerading as HARP numbers is now an opt-in
    '_SIM_CLIENT_ID' and is never selected as a model feature.
  - IMMUTABLE TRAIN/VAL/TEST CONTRACT (audit B10): preprocess() now
    returns a validation split carved ONLY from the training pool.
    The test set is touched exactly once, at final evaluation.
    Index arrays are returned so leakage_audit can verify exclusivity.
  - Double-normalization handling made explicit via config flag
    (CLEANED_ALREADY_NORMALIZED) instead of duck-typing on '_split'.

Evaluation contract (do not violate anywhere else in the pipeline):

    TRAIN     -> local FL client training
    VAL       -> client selection monitoring, F-beta threshold search,
                 calibration fitting, SCAFFOLD checkpointing,
                 hyperparameter/model selection
    TEST      -> final evaluation ONLY (one pass, frozen threshold)
"""

import os
import numpy as np
import pandas as pd
from collections import namedtuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

import config as cfg
from config import (
    DATA_PATH, FEATURE_COLS, LABEL_COL,
    N_SAMPLES, FLARE_RATIO, N_CLIENTS,
    TEST_SPLIT, SMOTE_RATIO, RANDOM_STATE
)

Splits = namedtuple("Splits", [
    "X_train", "y_train", "X_val", "y_val", "X_test", "y_test",
    "scaler", "features", "train_idx", "val_idx", "test_idx",
])


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD OR GENERATE
# ─────────────────────────────────────────────────────────────────────────────

def load_or_generate_data() -> pd.DataFrame:
    """
    Load pipeline:
      1. Try cleaned pkl partitions (load_cleaned_data.py)
      2. Try raw SWAN-SF CSV
      3. Fall back to physics-based synthetic data
    Returns a flat 2-D DataFrame with feature columns + 'label' + '_split'.
    """

    # ── Priority 1: cleaned pkl dataset ────────────────────────────────────
    try:
        from config import USE_CLEANED_DATA, CLEANED_DATA_DIR, COMBINE_PARTITIONS
        if not USE_CLEANED_DATA:
            raise FileNotFoundError("USE_CLEANED_DATA is False")
        print("\n" + "=" * 66)
        print("  SF-9 DATA LOADING PIPELINE")
        print("=" * 66 + "\n")
        print("[Priority 1] Attempting to load CLEANED dataset...")
        from load_cleaned_data import load_cleaned_partition
        from config import FLATTEN_METHOD

        X_tr, y_tr, X_te, y_te, feat_names = load_cleaned_partition(
            combine_all_partitions=COMBINE_PARTITIONS,
            data_dir=CLEANED_DATA_DIR,
            flatten_method=FLATTEN_METHOD
        )
        return _arrays_to_dataframe(X_tr, y_tr, X_te, y_te, feat_names)
    except Exception as e:
        print(f"[Priority 1] Cleaned data load failed: {e}")

    # ── Priority 2: raw CSV ─────────────────────────────────────────────────
    if os.path.exists(DATA_PATH):
        print(f"[Priority 2] Loading raw SWAN-SF CSV from '{DATA_PATH}' ...")
        df = pd.read_csv(DATA_PATH)
        core = set(FEATURE_COLS[:8])
        if core.issubset(df.columns) and LABEL_COL in df.columns:
            print(f"[Data] {len(df):,} samples | "
                  f"flare rate: {df[LABEL_COL].mean()*100:.2f}%")
            return df

    # ── Priority 3: synthetic fallback ─────────────────────────────────────
    print("[Priority 3] No real data found — generating physics-based synthetic dataset.")
    return _generate_synthetic()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _arrays_to_dataframe(X_tr, y_tr, X_te, y_te, feat_names):
    """
    Convert pre-split arrays (from the cleaned pkl loader) to a single
    DataFrame with a '_split' column so preprocess() can respect the
    original train/test boundary.

    v3.0 (B12): feature names from the loader (e.g. 'mean_TOTUSJH',
    'std_TOTPOT') are KEPT AS-IS. They are real, ordered, and
    interpretable; renaming them to feature_N was the cause of the
    unreadable SHAP figures.

    v3.0 (B13): no fabricated HARPNUM column. An optional '_SIM_CLIENT_ID'
    (random regional assignment) can be attached for demo purposes via
    add_simulated_client_ids(); it is never selected as a feature.
    """
    n_feat = X_tr.shape[1]
    feat_names = list(feat_names) if feat_names is not None else \
        [f"feature_{i}" for i in range(n_feat)]

    # sanity: names must be unique and match width
    if len(feat_names) != n_feat:
        print(f"[Data] feature-name count {len(feat_names)} != data width "
              f"{n_feat}; falling back to positional names")
        feat_names = [f"feature_{i}" for i in range(n_feat)]

    cols = feat_names + [LABEL_COL]
    df_tr = pd.DataFrame(np.column_stack([X_tr, y_tr]), columns=cols)
    df_tr["_split"] = "train"
    df_te = pd.DataFrame(np.column_stack([X_te, y_te]), columns=cols)
    df_te["_split"] = "test"
    df = pd.concat([df_tr, df_te], ignore_index=True)

    df[LABEL_COL] = df[LABEL_COL].astype(int)
    print(f"\n[Data] Cleaned dataset loaded: {len(df):,} samples "
          f"({n_feat} features, names preserved for interpretability)")
    return df


def add_simulated_client_ids(df, n_clients=6, seed=42):
    """
    OPT-IN demo utility (audit B13): attach a random regional client id.
    This is a SIMULATION aid, not a HARP-based partition, and is never
    used as a model feature. The real partitioning is Dirichlet (see
    partition_clients.py).
    """
    rng = np.random.RandomState(seed)
    df = df.copy()
    df["_SIM_CLIENT_ID"] = rng.randint(n_clients, size=len(df))
    return df


def _generate_synthetic() -> pd.DataFrame:
    """
    Physics-based synthetic dataset with intentional class overlap.
    (Same generator as v2.5 — 24 correct Cleaned-SWAN-SF features.)
    Used ONLY when no real data is found; clearly labelled so no paper
    table can ever silently report synthetic numbers.
    """
    np.random.seed(RANDOM_STATE)
    n_flare = int(N_SAMPLES * FLARE_RATIO)
    n_quiet = N_SAMPLES - n_flare
    records = []

    for label, n, mag in [(0, n_quiet, 1.0), (1, n_flare, 7.0)]:
        R_VALUE  = np.random.lognormal(np.log(1.8*mag), 0.6, n)
        TOTUSJH  = np.random.lognormal(np.log(4e21 * mag),   0.9, n)
        TOTBSQ   = np.random.lognormal(np.log(9e22*mag), 0.8, n)
        TOTPOT   = np.random.lognormal(np.log(8e31 * mag),   1.0, n)
        TOTUSJZ  = np.random.lognormal(np.log(9e11 * mag),   0.8, n)
        ABSNJZH  = np.abs(np.random.normal(9e11 * mag, 4e11 * mag, n))
        SAVNCPP  = np.random.lognormal(np.log(80 * mag),     0.5, n)
        USFLUX   = np.random.lognormal(np.log(8e21 * mag),   0.9, n)
        TOTFZ    = np.random.normal(0, 9e21*mag, n)
        MEANPOT  = np.clip(np.random.normal(250*mag, 90, n), 0, None)
        EPSX     = np.random.normal(0, 9e21*mag, n)
        EPSY     = np.random.normal(0, 9e21*mag, n)
        EPSZ     = np.random.normal(0, 9e21*mag, n)
        MEANSHR  = np.random.normal(18*mag, 14, n)
        SHRGT45  = np.random.beta(2*mag, 5, n) * 100
        MEANGAM  = np.random.normal(4*mag, 3, n)
        MEANGBT  = np.random.lognormal(np.log(45*mag), 0.6, n)
        MEANGBZ  = np.random.normal(0, 28*mag, n)
        MEANGBH  = np.random.lognormal(np.log(38*mag), 0.7, n)
        MEANJZH  = np.random.normal(0, 4e7*mag, n)
        TOTFY    = np.random.normal(0, 9e21*mag, n)
        MEANJZD  = np.random.normal(0, 9e6*mag, n)
        MEANALP  = np.random.normal(0, 0.4*mag, n)
        TOTFX    = np.random.normal(0, 9e21*mag, n)

        if label == 1:  # 15% overlap to avoid 100% accuracy on synthetic data
            ov = np.random.rand(n) < 0.15
            TOTUSJH[ov] /= 5; TOTPOT[ov] /= 5
            SHRGT45[ov] /= 3; R_VALUE[ov] /= 4

        for i in range(n):
            records.append({
                "R_VALUE": R_VALUE[i], "TOTUSJH": TOTUSJH[i],
                "TOTBSQ": TOTBSQ[i], "TOTPOT": TOTPOT[i],
                "TOTUSJZ": TOTUSJZ[i], "ABSNJZH": ABSNJZH[i],
                "SAVNCPP": SAVNCPP[i], "USFLUX": USFLUX[i],
                "TOTFZ": TOTFZ[i], "MEANPOT": MEANPOT[i],
                "EPSX": EPSX[i], "EPSY": EPSY[i],
                "EPSZ": EPSZ[i], "MEANSHR": MEANSHR[i],
                "SHRGT45": SHRGT45[i], "MEANGAM": MEANGAM[i],
                "MEANGBT": MEANGBT[i], "MEANGBZ": MEANGBZ[i],
                "MEANGBH": MEANGBH[i], "MEANJZH": MEANJZH[i],
                "TOTFY": TOTFY[i], "MEANJZD": MEANJZD[i],
                "MEANALP": MEANALP[i], "TOTFX": TOTFX[i],
                LABEL_COL: label,
            })

    df = (pd.DataFrame(records)
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True))
    df["_split"] = None        # random stratified split in preprocess()
    df["_SYNTHETIC"] = True    # provenance flag (never silently mixed)

    os.makedirs("data", exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"[Data] SYNTHETIC fallback dataset saved -> {DATA_PATH} "
          f"({len(df):,} samples, flare rate {df[LABEL_COL].mean()*100:.2f}%)")
    print("[Data] WARNING: synthetic data — NOT for paper tables.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PREPROCESSING  (immutable train/val/test contract)
# ─────────────────────────────────────────────────────────────────────────────

_PROTECTED = {LABEL_COL, "_split", "_SYNTHETIC", "_SIM_CLIENT_ID",
              "HARPNUM_MOD", "_HARPNUM_MOD"}


def _select_feature_columns(df):
    """
    v3.0 (B12): ordered, name-preserving selection.
      1. exact FEATURE_COLS matches (raw/synthetic path)
      2. stat-prefixed names from the cleaned loader (mean_TOTUSJH, ...)
      3. positional feature_N fallback (last resort, logged loudly)
    """
    named = [c for c in FEATURE_COLS if c in df.columns]
    if named:
        return named, "base24"
    stat_prefixed = [c for c in df.columns
                     if any(c.startswith(p + "_") for p in
                            ("mean", "std", "max", "min", "trend", "slope"))
                     and c.split("_", 1)[1] in FEATURE_COLS]
    if stat_prefixed:
        return stat_prefixed, "stat_prefixed"
    positional = [c for c in df.columns if c.startswith("feature_")]
    if positional:
        positional = sorted(positional, key=lambda c: int(c.split("_")[1]))
        return positional, "positional_fallback"
    numeric = [c for c in df.columns if c not in _PROTECTED]
    return numeric, "numeric_fallback"


def preprocess(df: pd.DataFrame, val_fraction=0.16) -> Splits:
    """
    Scale and split into TRAIN / VALIDATION / TEST with an immutable
    contract (audit B10):

      * If the cleaned dataset provides its own train/test boundary
        ('_split' column), it is respected exactly; the validation set
        is carved OUT OF TRAIN ONLY by stratified split.
      * Otherwise a stratified 64/16/20 split is created.
      * Returns index arrays so leakage_audit can verify exclusivity.

    Validation may select: hyperparameters, thresholds, calibration,
    checkpoints, model selection.
    Test does: nothing except the single final evaluation.
    """
    print("\n" + "=" * 66)
    print("  PREPROCESSING PIPELINE (v3.0 immutable split contract)")
    print("=" * 66 + "\n")

    available, mode = _select_feature_columns(df)
    print(f"[Preprocess] Selected {len(available)} feature columns (mode: {mode})")
    if mode.endswith("fallback"):
        print("[Preprocess] WARNING: physical feature names unavailable — "
              "SHAP labels will degrade.")

    X_all = df[available].values.astype(np.float32)
    y_all = df[LABEL_COL].values.astype(int)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    has_pre_split = "_split" in df.columns and df["_split"].notna().any()
    rng = np.random.RandomState(cfg.SEED)

    if has_pre_split:
        print("[Preprocess] Using pre-existing train/test boundary from cleaned dataset.")
        train_pos = np.where(df["_split"].values == "train")[0]
        test_pos = np.where(df["_split"].values != "train")[0]
        # carve validation out of TRAIN ONLY (stratified)
        tr_labels = y_all[train_pos]
        val_rel = train_test_split(
            train_pos, test_size=val_fraction,
            random_state=cfg.SEED, stratify=tr_labels)[1]
        val_set = set(val_rel.tolist())
        train_idx = np.array([i for i in train_pos if i not in val_set])
        val_idx = np.asarray(val_rel)
        test_idx = np.asarray(test_pos)
    else:
        print(f"[Preprocess] No pre-existing split — stratified "
              f"{1 - TEST_SPLIT - val_fraction:.0%}/{val_fraction:.0%}/{TEST_SPLIT:.0%} split.")
        first = train_test_split(
            np.arange(len(y_all)), test_size=TEST_SPLIT,
            random_state=cfg.SEED, stratify=y_all)
        train_val_idx, test_idx = first[0], first[1]
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_fraction / (1 - TEST_SPLIT),
            random_state=cfg.SEED, stratify=y_all[train_val_idx])

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]

    # ── normalization policy (explicit, audit: double-normalization note) ──
    scaler = StandardScaler()
    if has_pre_split and getattr(cfg, "CLEANED_ALREADY_NORMALIZED", True):
        # Cleaned SWAN-SF export is already LSBZM-normalized; fitting only.
        print("[Preprocess] Cleaned data already LSBZM-normalized — StandardScaler fitted, NOT applied.")
        scaler.fit(X_train)
    else:
        print("[Preprocess] Applying StandardScaler (fit on TRAIN only).")
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

    print(f"[Preprocess] Train: {len(X_train):,} | flare rate: {y_train.mean()*100:.2f}%")
    print(f"[Preprocess] Val:   {len(X_val):,} | flare rate: {y_val.mean()*100:.2f}%  "
          f"(selection-only)")
    print(f"[Preprocess] Test:  {len(X_test):,} | flare rate: {y_test.mean()*100:.2f}%  "
          f"(final evaluation ONLY)\n")

    return Splits(X_train, y_train, X_val, y_val, X_test, y_test,
                  scaler, available, train_idx, val_idx, test_idx)


# ─────────────────────────────────────────────────────────────────────────────
# 3b.  REGION-DISJOINT VALIDATION SPLIT  (review R7 — hard-gate protocol)
# ─────────────────────────────────────────────────────────────────────────────

def region_disjoint_val_split(y_train, region_ids, val_fraction=0.16,
                              seed=42):
    """
    Carve a validation set from the training pool such that NO active
    region contributes windows to both train and validation (review
    finding R7: windows of one physical region must not straddle the
    selection boundary, or validation metrics are optimistically biased
    by region overlap).

    Protocol
    --------
    1. Group window indices by region id.
    2. Order regions by |region positivity - pooled positivity| (most
       representative first), with a seeded tie-break shuffle, so the
       validation set is approximately prevalence-stratified at REGION
       granularity (not window granularity).
    3. Greedily move whole regions into validation until it reaches
       ~val_fraction of windows, choosing the region that keeps the
       running validation prevalence closest to the pooled rate.

    The test set keeps the dataset's own (time-based) boundary; when
    region ids are available for BOTH pools, the runtime leakage audit
    additionally reports train-vs-test region overlap
    (leakage_audit.check_region_leakage).

    Returns (train_idx, val_idx, report) where report documents region
    counts and the realised validation prevalence.
    """
    y_train = np.asarray(y_train).astype(int)
    region_ids = np.asarray(region_ids)
    n = len(y_train)
    if len(region_ids) != n:
        raise ValueError("region_ids must align with the training pool")

    rng = np.random.RandomState(seed)
    unique_regions = np.unique(region_ids)
    region_indices = {r: np.where(region_ids == r)[0] for r in unique_regions}
    region_sizes = {r: int(len(region_indices[r])) for r in unique_regions}
    region_pos = {r: int(y_train[region_indices[r]].sum()) for r in unique_regions}

    pooled_rate = float(y_train.mean())
    order = sorted(
        unique_regions,
        key=lambda r: (abs(region_pos[r] / region_sizes[r] - pooled_rate),
                       rng.rand()))  # seeded tie-break

    target_val_windows = int(round(val_fraction * n))
    val_regions, n_val, pos_val = [], 0, 0
    remaining = list(order)
    while remaining and n_val < target_val_windows:
        # pick the region whose addition keeps the running validation
        # prevalence closest to the pooled rate (greedy whole-region
        # stratification; running sums keep this O(R^2) arithmetic)
        best_r, best_cost = None, np.inf
        for r in remaining:
            nn, pp = region_sizes[r], region_pos[r]
            cand_rate = (pos_val + pp) / (n_val + nn)
            cost = (abs(cand_rate - pooled_rate)
                    + abs(n_val + nn - target_val_windows) / n * 0.25)
            if cost < best_cost:
                best_cost, best_r = cost, r
        val_regions.append(best_r)
        n_val += region_sizes[best_r]
        pos_val += region_pos[best_r]
        remaining.remove(best_r)

    val_idx = (np.concatenate([region_indices[r] for r in val_regions])
               if val_regions else np.array([], int))
    val_mask = np.zeros(n, dtype=bool)
    val_mask[val_idx] = True
    train_idx = np.where(~val_mask)[0]

    report = {
        "n_regions_total": int(len(unique_regions)),
        "n_regions_val": int(len(val_regions)),
        "n_regions_train": int(len(unique_regions) - len(val_regions)),
        "n_windows_train": int(len(train_idx)),
        "n_windows_val": int(len(val_idx)),
        "val_prevalence": float(y_train[val_idx].mean()) if len(val_idx) else None,
        "train_prevalence": float(y_train[train_idx].mean()),
        "region_disjoint": True,
        "protocol": "greedy prevalence-matching whole-region allocation",
        "seed": int(seed),
    }
    print(f"[RegionSplit] {report['n_regions_val']}/{report['n_regions_total']} "
          f"regions -> validation ({report['n_windows_val']:,} windows, "
          f"prevalence {report['val_prevalence']:.3f}); train "
          f"{report['n_windows_train']:,} windows "
          f"(prevalence {report['train_prevalence']:.3f})")
    return train_idx, val_idx, report


# ─────────────────────────────────────────────────────────────────────────────
# 4.  3D DATA (LSTM path) — same immutable contract
# ─────────────────────────────────────────────────────────────────────────────

def load_and_scale_3d_data(val_fraction=0.16):
    """
    Load 3D data for LSTM models WITH a validation split carved from
    the training pool only (same contract as preprocess()).

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, scaler
    """
    from config import CLEANED_DATA_DIR, COMBINE_PARTITIONS
    from load_cleaned_data import load_cleaned_3d

    X_train, y_train, X_test, y_test = load_cleaned_3d(
        data_dir=CLEANED_DATA_DIR,
        combine_all_partitions=COMBINE_PARTITIONS
    )

    # validation carved from TRAIN only (stratified)
    from sklearn.model_selection import train_test_split
    tr_idx, val_idx = train_test_split(
        np.arange(len(y_train)), test_size=val_fraction,
        random_state=cfg.SEED, stratify=y_train)
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_train, y_train = X_train[tr_idx], y_train[tr_idx]

    # Cleaned data is already LSBZM-normalized — scaler fit-only (API compat)
    N_train, T, F = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(N_train * T, F))

    print(f"\n[3D] Train: {X_train.shape} | flare rate: {y_train.mean()*100:.2f}%")
    print(f"[3D] Val:   {X_val.shape} | flare rate: {y_val.mean()*100:.2f}% (selection-only)")
    print(f"[3D] Test:  {X_test.shape} | flare rate: {y_test.mean()*100:.2f}% (final only)\n")

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SMOTE (per-client balancing — NOW ACTUALLY WIRED, audit B7)
# ─────────────────────────────────────────────────────────────────────────────

def apply_smote(X: np.ndarray, y: np.ndarray, seed=None):
    """
    Apply SMOTE to a single client shard. Dead code in the original
    pipeline (never called — audit B7). The ablation matrix
    (experiments/run_ablations.py) now calls this under USE_SMOTE=True.

    v3.0: accepts an explicit seed for multi-seed reproducibility.
    v3.8: 3D-aware — LSTM shards (n, T, F) are flattened to (n, T*F),
    resampled, and reshaped back, so the natural-prevalence SMOTE
    ablation works for the sequence arms (imblearn's SMOTE is 2D-only
    and would otherwise raise, hit the except-branch, and silently
    return the shard UNCHANGED — a no-op ablation).
    """
    if seed is None:
        seed = cfg.SEED
    n_minority = int(y.sum())

    if n_minority < 2:
        return X, y

    n_majority = int(len(y) - n_minority)
    # skip gracefully when the minority is already at/above target ratio
    if n_majority > 0 and n_minority / n_majority >= SMOTE_RATIO:
        return X, y

    k = min(5, n_minority - 1)

    # 3D LSTM shards: flatten -> SMOTE -> reshape (synthetic windows are
    # feature-wise interpolations of two real minority windows)
    is_3d = (X.ndim == 3)
    if is_3d:
        n, T, F = X.shape
        X_in = np.asarray(X.reshape(n, T * F), dtype=np.float32)
    else:
        X_in = X

    try:
        sm = SMOTE(
            sampling_strategy=SMOTE_RATIO,
            random_state=seed,
            k_neighbors=k
        )
        X_res, y_res = sm.fit_resample(X_in, y)
    except Exception as e:
        print(f"[SMOTE] Error: {e} — returning original shard unchanged.")
        return X, y

    if is_3d:
        X_res = X_res.reshape(-1, T, F)
    if len(X_res) != len(X):
        print(f"[SMOTE] shard rebalanced: n {len(X):,} -> {len(X_res):,} "
              f"(pos {n_minority:,} -> {int(np.asarray(y_res).sum()):,}, "
              f"ratio target {SMOTE_RATIO}, k={k})")
    return X_res, y_res
