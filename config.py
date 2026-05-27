"""
SF-9: Federated Learning for Distributed Space Weather Monitoring
─────────────────────────────────────────────────────────────────
All hyperparameters and paths in one file.
Change values here; every other script reads from here.

v2.5 — Dataset audit fixes (Cleaned SWAN-SF alignment):
  - FEATURE_COLS: Replaced 3 non-existent features (AREA_ACR, HARPNUM_MOD,
    TIME_SINCE_LAST_FLARE) with actual features from the Cleaned SWAN-SF
    dataset (TOTFZ, TOTFY, TOTFX — Total Lorentz Force components)
  - FEATURE_COLS: Fixed attribute order to match the actual dataset
    (from https://github.com/samresume/Cleaned-SWANSF-Dataset)
  - Removed double StandardScaler on already LSBZM-normalized data
  - Plus all v2.3/v2.4 fixes: SCAFFOLD NaN, focal alpha, Dirichlet, etc.
"""

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_PATH   = "data/swan_sf.csv"   # Place downloaded SWAN-SF CSV here
OUTPUT_DIR  = "outputs"            # All plots, logs, and model checkpoints

# ── Dataset ───────────────────────────────────────────────────────────────
N_SAMPLES        = 10000           # Synthetic fallback: total rows
FLARE_RATIO      = 0.06            # ~6% flare rate (matches real SWAN-SF)
RANDOM_STATE     = 42

# SWAN-SF magnetic-field feature names — EXACT order from Cleaned SWAN-SF Dataset
# Source: https://github.com/samresume/Cleaned-SWANSF-Dataset
# Attributes Order (as stored in the 3D pkl files, index 0..23):
#   Index 0:  R_VALUE   — Sum of positive/negative polarity flux correlation
#   Index 1:  TOTUSJH   — Total unsigned current helicity
#   Index 2:  TOTBSQ    — Total magnitude of Lorentz force
#   Index 3:  TOTPOT    — Total photospheric magnetic free energy
#   Index 4:  TOTUSJZ   — Total unsigned vertical current
#   Index 5:  ABSNJZH   — Absolute value of net current helicity
#   Index 6:  SAVNCPP   — Sum of absolute value of net current per polarity
#   Index 7:  USFLUX    — Total unsigned flux
#   Index 8:  TOTFZ     — Total Lorentz force Z-component (was wrongly labeled AREA_ACR)
#   Index 9:  MEANPOT   — Mean photospheric magnetic free energy
#   Index 10: EPSX      — Sum of epsilon X-component (was wrongly ordered)
#   Index 11: EPSY      — Sum of epsilon Y-component
#   Index 12: EPSZ      — Sum of epsilon Z-component
#   Index 13: MEANSHR   — Mean shear angle
#   Index 14: SHRGT45   — Fraction of area with shear > 45 deg
#   Index 15: MEANGAM   — Mean angle of field from radial
#   Index 16: MEANGBT   — Mean gradient of total field
#   Index 17: MEANGBZ   — Mean gradient of Bz (vertical)
#   Index 18: MEANGBH   — Mean gradient of Bh (horizontal)
#   Index 19: MEANJZH   — Mean current helicity (Bz contribution)
#   Index 20: TOTFY     — Total Lorentz force Y-component (was wrongly labeled HARPNUM_MOD)
#   Index 21: MEANJZD   — Mean vertical current density
#   Index 22: MEANALP   — Mean alpha parameter
#   Index 23: TOTFX     — Total Lorentz force X-component (was wrongly labeled TIME_SINCE_LAST_FLARE)
#
# v2.5 FIX: Previous version had 3 features that don't exist in the Cleaned dataset:
#   AREA_ACR, HARPNUM_MOD, TIME_SINCE_LAST_FLARE
# These were replaced with the actual features present in the data:
#   TOTFZ, TOTFY, TOTFX (Lorentz Force components — physically important for flares!)
FEATURE_COLS = [
    "R_VALUE",  "TOTUSJH",  "TOTBSQ",   "TOTPOT",
    "TOTUSJZ",  "ABSNJZH",  "SAVNCPP",  "USFLUX",
    "TOTFZ",    "MEANPOT",  "EPSX",     "EPSY",
    "EPSZ",     "MEANSHR",  "SHRGT45",  "MEANGAM",
    "MEANGBT",  "MEANGBZ",  "MEANGBH",  "MEANJZH",
    "TOTFY",    "MEANJZD",  "MEANALP",  "TOTFX"
]
LABEL_COL   = "label"

# ════════════════════════════════════════════════════════════════════════
# 🔧 FIX #1: Changed from "HARPNUM" to "HARPNUM_MOD" to match main.py!
# ══════════════════════════════════════════════════════════════════════
HARPNUM_COL = "HARPNUM_MOD"    # ← MUST MATCH WHAT main.py EXPECTS!

# ── Federated Learning ────────────────────────────────────────────────────
N_CLIENTS       = 6        # Regional observatories simulated
N_ROUNDS        = 50       # Total communication rounds
LOCAL_EPOCHS    = 3        # Local training epochs per round (3 = faster, good enough)
FRACTION_FIT    = 1.0      # Fraction of clients per round (1.0 = all)
MU              = 0.01     # FedProx proximal coefficient

# Labels used in plots and logs
CLIENT_NAMES = [
    "Americas (NASA/NOAA)",
    "Europe (ESA/PROBA-2)",
    "Asia-Pacific (JAXA)",
    "South Asia (ISRO)",
    "East Asia (KASI)",
    "Oceania (BoM)"
]

# ── Model Architecture ──────────────────────────────────────────────────
INPUT_DIM   = len(FEATURE_COLS)
HIDDEN_DIMS = [128, 64, 32]
DROPOUT     = 0.3
LR          = 0.001
BATCH_SIZE  = 256

# ── LSTM Model ─────────────────────────────────────────────────────────────
USE_LSTM         = True          # Use LSTM instead of MLP (preserves temporal dynamics)
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS  = 2
LSTM_DROPOUT     = 0.3
LSTM_BIDIRECTIONAL = False

# ── SCAFFOLD Algorithm ────────────────────────────────────────────────────
USE_SCAFFOLD     = True          # Add SCAFFOLD as 3rd FL algorithm
SCAFFOLD_LR      = 0.001

# ── Fed-Focal Loss ────────────────────────────────────────────────────────
USE_FED_FOCAL    = True          # Use Fed-Focal Loss instead of DynamicFocalLoss
FOCAL_GAMMA      = 2.0           # Focusing parameter
FOCAL_ALPHA      = 0.25          # FIX: was 0.75 → over-predicted flares (F1~0.07)
                        # 0.25 is standard. federated_learning.py also
                        # hardcodes 0.25 directly for safety.

# ── Richer Temporal Features ──────────────────────────────────────────────
FLATTEN_METHOD   = "concat_stats_enhanced"  # 6-stat extraction: mean/std/max/min/trend/slope

# ── Non-IID Partitioning ─────────────────────────────────────────────────
DIRICHLET_ALPHA  = 0.5           # FIX: was 0.3 → created 0.6%-100% flare-rate clients
                        # 0.5 = moderate non-IID without pathological extremes
FORCE_NON_IID    = True          # Force Dirichlet partitioning even with cleaned data

# ── F-beta Threshold Optimization ─────────────────────────────────────────
FBETA_BETA       = 2.0           # β=2 weights recall 2x more than precision

# ── Mixup Augmentation ───────────────────────────────────────────────────
USE_MIXUP        = True
MIXUP_ALPHA      = 0.4           # Beta distribution parameter (0.4 = moderate mixing)

# ── Preprocessing ─────────────────────────────────────────────────────────
TEST_SPLIT  = 0.20     # Global held-out test set (never seen during federation)
SMOTE_RATIO = 0.25     # Minority class fraction after SMOTE per client

# ── Evaluation ────────────────────────────────────────────────────────────
# Lower threshold maximises Recall — missing a flare is worse than a false alarm
# NOTE: threshold=0.35 is used for F1 evaluation; F-beta optimization finds the
# optimal threshold separately (typically 0.15-0.40 range for safety-critical recall)
THRESHOLD   = 0.35

# ── Cleaned Dataset Settings ────────────────────────────────────────────────
USE_CLEANED_DATA = True           # Set to False to use original merged data
CLEANED_DATA_DIR = "data/cleaned" # Path to cleaned dataset folder
COMBINE_PARTITIONS = True         # Use all 5 partitions (recommended)

# ── GPU ACCELERATION ──
USE_CUDA = True          # Enable CUDA
PIN_MEMORY = True        # Faster GPU memory transfer
EVAL_BATCH_SIZE = 2048   # Batch size for evaluation (lower if OOM; 2048 for RTX 3060 12GB)