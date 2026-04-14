"""
SF-9: Federated Learning for Distributed Space Weather Monitoring
─────────────────────────────────────────────────────────────────
All hyperparameters and paths in one file.
Change values here; every other script reads from here.
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
LOCAL_EPOCHS    = 5        # Local training epochs per round
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

# ── MLP Model ─────────────────────────────────────────────────────────────
INPUT_DIM   = len(FEATURE_COLS)
HIDDEN_DIMS = [128, 64, 32]
DROPOUT     = 0.3
LR          = 0.001
BATCH_SIZE  = 64

# ── Preprocessing ─────────────────────────────────────────────────────────
TEST_SPLIT  = 0.20     # Global held-out test set (never seen during federation)
SMOTE_RATIO = 0.25     # Minority class fraction after SMOTE per client

# ── Evaluation ────────────────────────────────────────────────────────────
# Lower threshold maximises Recall — missing a flare is worse than a false alarm
THRESHOLD   = 0.70

# ── Cleaned Dataset Settings ────────────────────────────────────────────────
USE_CLEANED_DATA = True           # Set to False to use original merged data
CLEANED_DATA_DIR = "data/cleaned" # Path to cleaned dataset folder
COMBINE_PARTITIONS = True         # Use all 5 partitions (recommended)

# ── GPU ACCELERATION ──
USE_CUDA = True          # Enable CUDA
PIN_MEMORY = True      # Faster GPU memory transfer
BATCH_SIZE      = 256      # INCREASED from 64 (better GPU utilization)
LOCAL_EPOCHS    = 3        # Reduced from 5 (faster)