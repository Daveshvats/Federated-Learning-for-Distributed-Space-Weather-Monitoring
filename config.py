"""
SF-9: Federated Learning for Distributed Space Weather Monitoring
─────────────────────────────────────────────────────────────────
config.py v3.0 (improvements branch) — single configuration source.

v3.0 consolidation (audit B16/B17):
  - one VERSION constant, snapshotted into every run's results.json
  - SEED is the single root seed; multi-seed runs use SEED + k
  - evaluation protocol switches are explicit:
      AGGREGATION_STRATEGY : 'plain' (vanilla size-weighted FedAvg) or
                             'dafl'  (distribution-aware — ablation arm)
      USE_SMOTE            : per-client SMOTE balancing (ablation arm;
                             was dead code in v2.x, audit B7)
      CALIBRATION_METHOD   : 'none' | 'prior_shift' | 'platt' |
                             'isotonic' | 'temperature'
      USE_FED_FOCAL        : Fed-Focal loss (ablation arm)
  - threshold search happens on VALIDATION only (audit B1); the frozen
    threshold is stored in results.json and applied once to test
  - normalization policy flag CLEANED_ALREADY_NORMALIZED replaces
    duck-typed detection of double normalization

Historical notes (v2.6) retained below for traceability of the original
runs; the FL-stability rationale (Dirichlet alpha, focal alpha clamp)
still applies.
"""

# ── Identity / provenance ─────────────────────────────────────────────────
VERSION          = "3.0.1-improvements"   # v3.0.1: FastLoader (batch-sliced
#   local training, ~27% faster rounds), B13 global_pos_rate passed into the
#   focal losses instead of hardcoded 0.4887, round-level FL crash recovery,
#   manifest prevalence aggregates all partitions (B26).
PAPER_ID         = "SF-9"
RUN_ID           = None        # None -> auto-generated timestamp at runtime
SEED             = 42          # single root seed (multi-seed: SEED + k)
RANDOM_STATE     = SEED        # backward-compatible alias

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_PATH        = "data/swan_sf.csv"
OUTPUT_DIR       = "outputs"
RESULTS_JSON     = "outputs/results.json"      # machine-readable results (B18)
RUN_MANIFEST     = "outputs/run_manifest.json" # config + env snapshot (B18)

# ── Dataset ───────────────────────────────────────────────────────────────
N_SAMPLES        = 10000
FLARE_RATIO      = 0.06
FEATURE_COLS = [
    "R_VALUE",  "TOTUSJH",  "TOTBSQ",   "TOTPOT",
    "TOTUSJZ",  "ABSNJZH",  "SAVNCPP",  "USFLUX",
    "TOTFZ",    "MEANPOT",  "EPSX",     "EPSY",
    "EPSZ",     "MEANSHR",  "SHRGT45",  "MEANGAM",
    "MEANGBT",  "MEANGBZ",  "MEANGBH",  "MEANJZH",
    "TOTFY",    "MEANJZD",  "MEANALP",  "TOTFX"
]
LABEL_COL        = "label"

# ── Evaluation protocol (immutable contract) ──────────────────────────────
TEST_SPLIT       = 0.20     # global held-out test (touched exactly once)
VAL_SPLIT        = 0.16     # carved from TRAIN only — all selection happens here
CLEANED_ALREADY_NORMALIZED = True   # cleaned SWAN-SF is LSBZM-normalized

# ── Federated Learning ────────────────────────────────────────────────────
N_CLIENTS        = 6
N_ROUNDS         = 50
LOCAL_EPOCHS     = 10
FRACTION_FIT     = 1.0
MU               = 0.01
DIRICHLET_ALPHA  = 1.0      # v2.6: 1.0 = moderate non-IID (20%-60% rates)
FORCE_NON_IID    = True
MIN_SAMPLES_PER_CLIENT = 100

CLIENT_NAMES = [
    "Americas (NASA/NOAA)",
    "Europe (ESA/PROBA-2)",
    "Asia-Pacific (JAXA)",
    "South Asia (ISRO)",
    "East Asia (KASI)",
    "Oceania (BoM)"
]

# ── Aggregation / loss ablation switches ──────────────────────────────────
AGGREGATION_STRATEGY = "plain"   # 'plain' (vanilla FedAvg) | 'dafl' (ablation)
USE_FED_FOCAL    = True
FOCAL_GAMMA      = 2.0
FOCAL_ALPHA      = 0.25          # NOTE: FedFocalLoss internally clamps to
                                 # (0.05, 0.25); 0.25 is the effective max (B17)
USE_SMOTE        = False         # per-client SMOTE (ablation arm; B7)
SMOTE_RATIO      = 0.25
USE_MIXUP        = False
MIXUP_ALPHA      = 0.4

# ── Threshold & calibration protocol ──────────────────────────────────────
FBETA_BETA       = 2.0           # recall-weighted F-beta
CALIBRATION_METHOD = "prior_shift"   # 'none'|'prior_shift'|'platt'|'isotonic'|'temperature'
THRESHOLD_GRID   = (0.05, 0.95, 0.005)  # search grid ON VALIDATION
DEFAULT_THRESHOLD = 0.35         # monitoring-only default (never final)

# ── Model architecture ────────────────────────────────────────────────────
INPUT_DIM   = len(FEATURE_COLS)
HIDDEN_DIMS = [128, 64, 32]
DROPOUT     = 0.3
LR          = 0.0005
BATCH_SIZE  = 256

# ── LSTM ──────────────────────────────────────────────────────────────────
USE_LSTM          = True
LSTM_HIDDEN_SIZE  = 128
LSTM_NUM_LAYERS   = 2
LSTM_DROPOUT      = 0.3
LSTM_BIDIRECTIONAL = False

# ── SCAFFOLD ──────────────────────────────────────────────────────────────
USE_SCAFFOLD     = False
SCAFFOLD_LR      = 0.001

# ── Temporal features ─────────────────────────────────────────────────────
FLATTEN_METHOD   = "concat_stats_enhanced"

# ── Statistical validation ────────────────────────────────────────────────
N_SEEDS          = 5            # multi-seed runs (audit B11)
CONFIDENCE       = 0.95         # CI level for reporting

# ── Cleaned dataset settings ──────────────────────────────────────────────
USE_CLEANED_DATA = True
CLEANED_DATA_DIR = "data/cleaned"
COMBINE_PARTITIONS = True

# ── GPU ───────────────────────────────────────────────────────────────────
USE_CUDA = True
PIN_MEMORY = True
EVAL_BATCH_SIZE = 2048
