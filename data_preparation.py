"""
data_preparation.py
───────────────────
Loads the SWAN-SF benchmark dataset (Harvard Dataverse / cleaned pkl files)
if present, otherwise generates a physics-based synthetic fallback.

SWAN-SF Download:
  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EBCFKM
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from config import (
    DATA_PATH, FEATURE_COLS, LABEL_COL,
    N_SAMPLES, FLARE_RATIO, N_CLIENTS,
    TEST_SPLIT, SMOTE_RATIO, RANDOM_STATE
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD OR GENERATE
# ─────────────────────────────────────────────────────────────────────────────

def load_or_generate_data() -> pd.DataFrame:
    """
    Load pipeline:
      1. Try cleaned pkl partitions (load_cleaned_data.py)
      2. Try raw SWAN-SF CSV
      3. Fall back to physics-based synthetic data
    Returns a flat 2-D DataFrame with FEATURE_COLS + 'label'.
    """

    # ── Priority 1: cleaned pkl dataset ────────────────────────────────────
    try:
        from config import USE_CLEANED_DATA, CLEANED_DATA_DIR, COMBINE_PARTITIONS
        if USE_CLEANED_DATA:
            print("\n╔" + "═"*66 + "╗")
            print("║" + "      SF-9 DATA LOADING PIPELINE".center(66) + "║")
            print("╚" + "═"*66 + "╝\n")
            print("[Priority 1] Attempting to load CLEANED dataset...")
            # WITH THIS:
        from load_cleaned_data import load_cleaned_partition
        from config import FLATTEN_METHOD  # Add this import at top

        X_tr, y_tr, X_te, y_te, feat_names = load_cleaned_partition(
            combine_all_partitions=COMBINE_PARTITIONS,
            data_dir=CLEANED_DATA_DIR,
            flatten_method=FLATTEN_METHOD  # ← ADD THIS PARAMETER!
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
            if "HARPNUM_MOD" not in df.columns:
                df["HARPNUM_MOD"] = (
                    df["HARPNUM"] % N_CLIENTS if "HARPNUM" in df.columns
                    else np.random.randint(0, N_CLIENTS, len(df))
                )
            if "TIME_SINCE_LAST_FLARE" not in df.columns:
                df["TIME_SINCE_LAST_FLARE"] = _compute_time_since_flare(df)
            return df

    # ── Priority 3: synthetic fallback ─────────────────────────────────────
    print("[Priority 3] No real data found — generating physics-based synthetic dataset.")
    return _generate_synthetic()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _arrays_to_dataframe(X_tr, y_tr, X_te, y_te, feat_names):
    """
    Convert pre-split arrays (from cleaned pkl loader) back to a single
    DataFrame with a '_split' column so preprocess() can respect the
    original train/test boundary.
    """
    n_feat = X_tr.shape[1]
    if len(feat_names) != n_feat:
        feat_names = [f'feature_{i}' for i in range(n_feat)]

    cols = feat_names + [LABEL_COL]

    df_tr = pd.DataFrame(
        np.column_stack([X_tr, y_tr]), columns=cols
    )
    df_tr["_split"] = "train"

    df_te = pd.DataFrame(
        np.column_stack([X_te, y_te]), columns=cols
    )
    df_te["_split"] = "test"

    df = pd.concat([df_tr, df_te], ignore_index=True)

    # ════════════════════════════════════════════════════════════════════
    # 🔧 IMPROVED: Add HARPNUM_MOD with VARIATION (not just sequential)
    # This helps partitioning create more realistic distributions
    # ════════════════════════════════════════════════════════════════════
    if "HARPNUM_MOD" not in df.columns:
        np.random.seed(42)
        n_total = len(df)
        
        # Create varied client assignments (not perfectly uniform)
        # Simulates different observatories having different amounts of data
        base_distribution = [0.22, 0.20, 0.18, 0.16, 0.14, 0.10]  # Americas has most
        
        client_ids = []
        cumulative = 0
        for client_id, proportion in enumerate(base_distribution):
            n_for_client = int(n_total * proportion)
            if client_id == len(base_distribution) - 1:
                n_for_client = n_total - cumulative  # Last client gets remainder
            client_ids.extend([client_id] * n_for_client)
            cumulative += n_for_client
        
        # Trim/pad to exact length
        client_ids = client_ids[:n_total]
        if len(client_ids) < n_total:
            client_ids.extend([0] * (n_total - len(client_ids)))
        
        # Shuffle to mix train/test
        client_ids = np.array(client_ids)
        perm = np.random.permutation(n_total)
        client_ids = client_ids[perm]
        
        # Re-sort to maintain train/test order (train first, then test)
        train_indices = df[df["_split"] == "train"].index.tolist()
        test_indices = df[df["_split"] == "test"].index.tolist()
        
        final_client_ids = np.zeros(n_total, dtype=int)
        final_client_ids[train_indices] = client_ids[:len(train_indices)]
        final_client_ids[test_indices] = client_ids[len(train_indices):]
        
        df["HARPNUM_MOD"] = final_client_ids
        
        print(f"      ✓ Generated HARPNUM_MOD with realistic distribution:")
        for cid, name in enumerate(["Americas", "Europe", "Asia-Pacific", 
                                    "South Asia", "East Asia", "Oceania"]):
            count = (df["HARPNUM_MOD"][:len(X_tr)] == cid).sum()
            print(f"          {name}: {count:,} train samples")

    df[LABEL_COL] = df[LABEL_COL].astype(int)
    print(f"\n✅ SUCCESS: Cleaned dataset loaded!")
    return df


def _generate_synthetic() -> pd.DataFrame:
    """Physics-based synthetic dataset with intentional class overlap."""
    np.random.seed(RANDOM_STATE)
    n_flare = int(N_SAMPLES * FLARE_RATIO)
    n_quiet = N_SAMPLES - n_flare
    records = []

    for label, n, mag in [(0, n_quiet, 1.0), (1, n_flare, 7.0)]:
        TOTUSJH  = np.random.lognormal(np.log(4e21 * mag),   0.9, n)
        TOTPOT   = np.random.lognormal(np.log(8e31 * mag),   1.0, n)
        TOTUSJZ  = np.random.lognormal(np.log(9e11 * mag),   0.8, n)
        ABSNJZH  = np.abs(np.random.normal(9e11 * mag, 4e11 * mag, n))
        SAVNCPP  = np.random.lognormal(np.log(80 * mag),     0.5, n)
        USFLUX   = np.random.lognormal(np.log(8e21 * mag),   0.9, n)
        AREA_ACR = np.random.lognormal(np.log(400 * mag),    0.7, n)
        MEANPOT  = np.clip(np.random.normal(250*mag, 90, n), 0, None)
        SHRGT45  = np.random.beta(2*mag, 5, n) * 100
        MEANSHR  = np.random.normal(18*mag, 14, n)
        MEANGAM  = np.random.normal(4*mag, 3, n)
        MEANGBT  = np.random.lognormal(np.log(45*mag), 0.6, n)
        MEANGBZ  = np.random.normal(0, 28*mag, n)
        MEANGBH  = np.random.lognormal(np.log(38*mag), 0.7, n)
        MEANJZH  = np.random.normal(0, 4e7*mag, n)
        TOTBSQ   = np.random.lognormal(np.log(9e22*mag), 0.8, n)
        MEANJZD  = np.random.normal(0, 9e6*mag, n)
        MEANALP  = np.random.normal(0, 0.4*mag, n)
        R_VALUE  = np.random.lognormal(np.log(1.8*mag), 0.6, n)
        EPSY     = np.random.normal(0, 9e21*mag, n)
        EPSX     = np.random.normal(0, 9e21*mag, n)
        EPSZ     = np.random.normal(0, 9e21*mag, n)
        TSF      = np.random.exponential(max(1, 48 / mag), n)

        # 15% overlap to avoid 100% accuracy on synthetic data
        if label == 1:
            ov = np.random.rand(n) < 0.15
            TOTUSJH[ov] /= 5; TOTPOT[ov] /= 5
            SHRGT45[ov] /= 3; R_VALUE[ov] /= 4

        client_ids = (
            np.random.choice(N_CLIENTS, size=n, p=[0.25,0.20,0.20,0.15,0.12,0.08])
            if label == 1
            else np.array([i % N_CLIENTS for i in range(n)])
        )

        for i in range(n):
            records.append({
                "TOTUSJH": TOTUSJH[i], "TOTPOT": TOTPOT[i],
                "TOTUSJZ": TOTUSJZ[i], "ABSNJZH": ABSNJZH[i],
                "SAVNCPP": SAVNCPP[i], "USFLUX": USFLUX[i],
                "AREA_ACR": AREA_ACR[i], "MEANPOT": MEANPOT[i],
                "SHRGT45": SHRGT45[i], "MEANSHR": MEANSHR[i],
                "MEANGAM": MEANGAM[i], "MEANGBT": MEANGBT[i],
                "MEANGBZ": MEANGBZ[i], "MEANGBH": MEANGBH[i],
                "MEANJZH": MEANJZH[i], "TOTBSQ": TOTBSQ[i],
                "MEANJZD": MEANJZD[i], "MEANALP": MEANALP[i],
                "R_VALUE": R_VALUE[i], "EPSY": EPSY[i],
                "EPSX": EPSX[i], "EPSZ": EPSZ[i],
                "HARPNUM_MOD": client_ids[i],
                "TIME_SINCE_LAST_FLARE": TSF[i],
                LABEL_COL: label,
            })

    df = (pd.DataFrame(records)
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True))
    df["_split"] = None   # will use random split in preprocess()

    os.makedirs("data", exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"[Data] Synthetic dataset saved → {DATA_PATH}")
    print(f"[Data] {len(df):,} samples | flares: {df[LABEL_COL].sum()} "
          f"({df[LABEL_COL].mean()*100:.2f}%)\n")
    return df


def _compute_time_since_flare(df: pd.DataFrame) -> np.ndarray:
    tsf = np.random.exponential(48, len(df))
    if LABEL_COL in df.columns:
        tsf[df[LABEL_COL] == 1] = np.random.exponential(
            6, (df[LABEL_COL] == 1).sum()
        )
    return tsf


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    """
    Scale and split into train/test.
    Respects the '_split' column if present (cleaned dataset has fixed splits).

    Returns
    -------
    X_train, X_test : np.ndarray  (scaled float32)
    y_train, y_test : np.ndarray  (int)
    scaler          : fitted StandardScaler
    features        : list[str]
    """
    print("\n╔" + "═"*66 + "╗")
    print("║" + "        PREPROCESSING PIPELINE".center(66) + "║")
    print("╚" + "═"*66 + "╝\n")

    available = [c for c in FEATURE_COLS if c in df.columns]
    # Also accept generic feature_N columns from cleaned loader
    if not available:
        available = [c for c in df.columns
                     if c.startswith("feature_") or c in FEATURE_COLS]

    X_all = df[available].values.astype(np.float32)
    y_all = df[LABEL_COL].values.astype(int)

    # Replace any inf/nan
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    if "_split" in df.columns and df["_split"].notna().any():
        # Respect pre-existing split from cleaned dataset
        print("[Preprocess] Using pre-existing train/test split from cleaned dataset.")
        train_mask = df["_split"].values == "train"
        X_train, X_test = X_all[train_mask], X_all[~train_mask]
        y_train, y_test = y_all[train_mask], y_all[~train_mask]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=TEST_SPLIT,
            random_state=RANDOM_STATE, stratify=y_all
        )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    print(f"[Preprocess] Train: {len(X_train):,} | "
          f"flare rate: {y_train.mean()*100:.2f}%")
    print(f"[Preprocess] Test:  {len(X_test):,} | "
          f"flare rate: {y_test.mean()*100:.2f}%\n")

    return X_train, X_test, y_train, y_test, scaler, available


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SMOTE  (applied per-client, not globally)
# ─────────────────────────────────────────────────────────────────────────────

def apply_smote(X: np.ndarray, y: np.ndarray):
    """
    Apply SMOTE to a single client shard.
    BUG FIX: removed 'n_jobs' — not a valid SMOTE parameter in any version
    of imbalanced-learn. Use k_neighbors only.
    """
    n_minority = y.sum()

    if n_minority < 2:
        # Cannot run SMOTE with fewer than 2 minority samples
        return X, y

    # k_neighbors must be < n_minority
    k = min(5, n_minority - 1)

    try:
        sm = SMOTE(
            sampling_strategy=SMOTE_RATIO,
            random_state=RANDOM_STATE,
            k_neighbors=k
            # NOTE: do NOT pass n_jobs here — it is not a SMOTE parameter
        )
        X_res, y_res = sm.fit_resample(X, y)
        return X_res, y_res
    except Exception as e:
        print(f"[SMOTE] Error: {e} — returning original shard unchanged.")
        return X, y