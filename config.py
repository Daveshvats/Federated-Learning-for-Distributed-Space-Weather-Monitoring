"""
SF-9: Federated Learning for Distributed Space Weather Monitoring
─────────────────────────────────────────────────────────────────
All hyperparameters and paths in one file.
Change values here; every other script reads from here.

v2.0 — Enhanced with 2026 SOTA techniques:
  - LSTM temporal model
  - SCAFFOLD algorithm
  - Fed-Focal Loss
  - Richer temporal features (6-stat extraction)
  - Non-IID Dirichlet partitioning
  - F-beta threshold optimization
  - CosineAnnealingWarmRestarts scheduler
  - Mixup augmentation
"""

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_PATH   = "data/swan_sf.csv"   # Place downloaded SWAN-SF CSV here
OUTPUT_DIR  = "outputs"            # All plots, logs, and model checkpoints

# ── Dataset ───────────────────────────────────────────────────────────────
N_SAMPLES        = 10000           # Synthetic fallback: total rows
FLARE_RATIO      = 0.06            # ~6% flare rate (matches real SWAN-SF)
RANDOM_STATE     = 42

# SWAN-SF magnetic-field feature names (Harvard Dataverse column headers)
FEATURE_COLS = [
    "TOTUSJH",  "TOTPOT",   "TOTUSJZ",  "ABSNJZH",
    "SAVNCPP",  "USFLUX",   "AREA_ACR", "MEANPOT",
    "SHRGT45",  "MEANSHR",  "MEANGAM",  "MEANGBT",
    "MEANGBZ",  "MEANGBH",  "MEANJZH",  "TOTBSQ",
    "MEANJZD",  "MEANALP",  "R_VALUE",  "EPSY",
    "EPSX",     "EPSZ",     "HARPNUM_MOD", "TIME_SINCE_LAST_FLARE"
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
FOCAL_ALPHA      = 0.75          # Positive class weight (higher = more recall)

# ── Richer Temporal Features ──────────────────────────────────────────────
FLATTEN_METHOD   = "concat_stats_enhanced"  # 6-stat extraction: mean/std/max/min/trend/slope

# ── Non-IID Partitioning ─────────────────────────────────────────────────
DIRICHLET_ALPHA  = 0.3           # Lower = more non-IID (0.5 moderate, 0.1 extreme)
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