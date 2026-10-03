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
VERSION          = "3.1.0-improvements"   # v3.1: review-2 protocol hardening
#   (R4 validation-frozen FPR thresholds, R7 region-disjoint split code,
#   R14 untouched client holdouts, R15 training-budget audit, R13/R19
#   event-level metrics, R6 calibration-comparison runner, neutral client
#   labels R8). Paper: v3.2 manuscript revision.
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
#   CONFIGURATION PROVENANCE (review R3): the headline run and ALL
#   statistical studies (multiseed, ablations, client eval) use this
#   preregistered alpha=1.0. The mu/alpha sweep (experiments/run_sweep.py)
#   selected alpha=5.0, mu=0.01 as the validation-optimal configuration;
#   adopting it as the headline configuration requires the queued re-run.
#   Both values are recorded in run_manifest.json.
DIRICHLET_ALPHA_SWEEP_WINNER = 5.0   # validation-selected (sweep, 15 rounds)
OPERATING_FPR_TARGETS = (0.005, 0.01, 0.02, 0.05)  # R4 deployment budgets
FORCE_NON_IID    = True
MIN_SAMPLES_PER_CLIENT = 100

CLIENT_NAMES = [
    # Plain neutral identifiers (v4.1, R-FS9-R1 C2): clients are
    # simulated shards of a public benchmark, not institutions and
    # not regions — no geographic naming anywhere.
    "Client A",
    "Client B",
    "Client C",
    "Client D",
    "Client E",
    "Client F",
]

# ── Aggregation / loss ablation switches ──────────────────────────────────
AGGREGATION_STRATEGY = "plain"   # 'plain' (vanilla FedAvg) | 'dafl' (ablation)
USE_FED_FOCAL    = True
FOCAL_GAMMA      = 2.0
FOCAL_ALPHA      = 0.25          # NOTE: FedFocalLoss internally clamps to
LOSS_VARIANT     = "fed_focal"   # "fed_focal" | "weighted_bce" | "bce"
#   ablation-only switch: weighted_bce/bce swap in BCEWithWeightLoss
#   (fixed alpha 0.25 / 0.5, no adaptive machinery) so the loss ablation
#   actually isolates the focal component
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
# BATCH_SIZE (256) deleted at v4.5 — R-FS9-R5 R7-2: the federated
# local loader has trained at LOCAL_BATCH_SIZE = 512 (federated_
# learning.py, single source of truth) since v3.0.1; this stale
# constant never reached the loader and only mis-fed the budget
# report in main.py.

# ── LSTM ──────────────────────────────────────────────────────────────────
USE_LSTM          = True
LSTM_HIDDEN_SIZE  = 128
LSTM_NUM_LAYERS   = 2
LSTM_DROPOUT      = 0.3
LSTM_BIDIRECTIONAL = False

# ── SCAFFOLD ──────────────────────────────────────────────────────────────
USE_SCAFFOLD     = False
# SCAFFOLD_LR (0.001) deleted at v4.5 — R-FS9-R5 R7-3: dead constant,
# imported nowhere; the scaffold path uses LR * 0.5 (federated_
# learning.py local_train_scaffold), now disclosed in the paper.

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
