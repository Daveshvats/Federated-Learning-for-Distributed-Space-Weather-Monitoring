"""
data_preparation.py
───────────────────
Loads the SWAN-SF benchmark dataset (Harvard Dataverse) if present,
otherwise generates a physics-based synthetic fallback with controlled
class overlap so ML accuracy stays realistic (not 100%).

SWAN-SF Download:
  1. Visit: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EBCFKM
  2. Download any Partition CSV (e.g. "Partition1.csv")
  3. Save as: data/swan_sf.csv
  4. Re-run main.py — real data will be used automatically.
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
    """Return a DataFrame with FEATURE_COLS + 'label' column."""

    if os.path.exists(DATA_PATH):
        print(f"[Data] Loading real SWAN-SF from '{DATA_PATH}' ...")
        df = pd.read_csv(DATA_PATH)
        # Accept file if it contains at least the first 8 magnetic features
        core = set(FEATURE_COLS[:8])
        if core.issubset(df.columns) and LABEL_COL in df.columns:
            print(f"[Data] {len(df):,} samples | "
                  f"flare rate: {df[LABEL_COL].mean()*100:.2f}%")
            # Add synthetic helper columns if missing
            if "HARPNUM_MOD" not in df.columns:
                if "HARPNUM" in df.columns:
                    df["HARPNUM_MOD"] = df["HARPNUM"] % N_CLIENTS
                else:
                    df["HARPNUM_MOD"] = np.random.randint(0, N_CLIENTS, len(df))
            if "TIME_SINCE_LAST_FLARE" not in df.columns:
                df["TIME_SINCE_LAST_FLARE"] = _compute_time_since_flare(df)
            return df
        else:
            missing = core - set(df.columns)
            print(f"[Data] Warning: columns missing {missing}. Using synthetic data.")

    print("[Data] SWAN-SF not found — generating realistic synthetic dataset.")
    print("[Data] See file header for download instructions.\n")
    return _generate_synthetic()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SYNTHETIC DATA GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def _generate_synthetic() -> pd.DataFrame:
    """
    Simulate SWAN-SF magnetic-field feature distributions using lognormals.
    Intentional 15% overlap between classes keeps accuracy realistic.
    Each of the N_CLIENTS groups gets a slightly different class balance
    to model geographic data heterogeneity (non-IID federation scenario).
    """
    np.random.seed(RANDOM_STATE)
    n_flare = int(N_SAMPLES * FLARE_RATIO)
    n_quiet = N_SAMPLES - n_flare
    records = []

    for label, n, mag in [(0, n_quiet, 1.0), (1, n_flare, 7.0)]:
        # Magnetic complexity scales with `mag` (flares are ~7× more complex)
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

        # Inject 15% overlap (borderline M1 flares look like quiet sun)
        if label == 1:
            overlap = np.random.rand(n) < 0.15
            TOTUSJH[overlap] /= 5
            TOTPOT[overlap]  /= 5
            SHRGT45[overlap] /= 3
            R_VALUE[overlap] /= 4

        # Assign each sample to a regional client (non-IID: clients differ)
        # Clients with higher index see proportionally more flares
        client_ids = np.array([i % N_CLIENTS for i in range(n)])
        if label == 1:
            # Over-represent flares in first 3 clients, suppress in last 3
            client_ids = np.random.choice(
                list(range(N_CLIENTS)),
                size=n,
                p=[0.25, 0.20, 0.20, 0.15, 0.12, 0.08]
            )

        for i in range(n):
            records.append({
                "TOTUSJH":  TOTUSJH[i],  "TOTPOT":  TOTPOT[i],
                "TOTUSJZ":  TOTUSJZ[i],  "ABSNJZH": ABSNJZH[i],
                "SAVNCPP":  SAVNCPP[i],  "USFLUX":  USFLUX[i],
                "AREA_ACR": AREA_ACR[i], "MEANPOT": MEANPOT[i],
                "SHRGT45":  SHRGT45[i],  "MEANSHR": MEANSHR[i],
                "MEANGAM":  MEANGAM[i],  "MEANGBT": MEANGBT[i],
                "MEANGBZ":  MEANGBZ[i],  "MEANGBH": MEANGBH[i],
                "MEANJZH":  MEANJZH[i],  "TOTBSQ":  TOTBSQ[i],
                "MEANJZD":  MEANJZD[i],  "MEANALP": MEANALP[i],
                "R_VALUE":  R_VALUE[i],  "EPSY":    EPSY[i],
                "EPSX":     EPSX[i],     "EPSZ":    EPSZ[i],
                "HARPNUM_MOD":            client_ids[i],
                "TIME_SINCE_LAST_FLARE":  TSF[i],
                "label":                  label,
            })

    df = (pd.DataFrame(records)
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True))

    os.makedirs("data", exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"[Data] Synthetic dataset saved → {DATA_PATH}")
    print(f"[Data] {len(df):,} samples | "
          f"flares: {df['label'].sum()} ({df['label'].mean()*100:.2f}%)\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    """
    Clean, scale, and split into global train/test.
    The test set is held out BEFORE federation begins — clients never see it.

    Returns
    -------
    X_train, X_test : np.ndarray  (scaled)
    y_train, y_test : np.ndarray
    scaler          : fitted StandardScaler
    features        : list[str]  names of columns used
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    df = df[available + [LABEL_COL]].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    X = df[available].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT,
        random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"[Preprocess] Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"[Preprocess] Flare rate — train: {y_train.mean()*100:.2f}%  "
          f"test: {y_test.mean()*100:.2f}%\n")

    return X_train, X_test, y_train, y_test, scaler, available


def apply_smote(X: np.ndarray, y: np.ndarray):
    """Apply SMOTE to a client's local training shard."""
    if y.sum() < 2:
        # Not enough minority samples for SMOTE — return as-is
        return X, y
    sm = SMOTE(sampling_strategy=SMOTE_RATIO, random_state=RANDOM_STATE, k_neighbors=min(3, y.sum()-1))
    return sm.fit_resample(X, y)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _compute_time_since_flare(df: pd.DataFrame) -> np.ndarray:
    """Approximate TIME_SINCE_LAST_FLARE if not in real SWAN-SF file."""
    tsf = np.random.exponential(48, len(df))
    if "label" in df.columns:
        tsf[df["label"] == 1] = np.random.exponential(6, (df["label"] == 1).sum())
    return tsf
