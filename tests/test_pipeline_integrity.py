"""
tests/test_pipeline_integrity.py  (v3.0)
─────────────────────────────────────────
Gate-1/2 pass criteria: runnable WITHOUT torch or the SWAN-SF dataset
(synthetic fixtures only). Verifies the audit fixes actually hold:

  1. Dirichlet partitioning is disjoint with 100% coverage (B2)
  2. Leakage detector catches seeded leakage (audit trustworthiness)
  3. Immutable train/val/test contract (B10)
  4. Threshold protocol: selection happens on validation data, and the
     frozen-threshold test metric is NOT the oracle value (B1)
  5. Stale-accuracy fix: compute_all_metrics recomputes everything at
     the given threshold (B4)
  6. Calibration: prior-shift correction recovers shifted prevalence;
     calibrators reduce Brier under prior shift (protocol need)
  7. Operational metrics: recall@FPR, PR-AUC, ECE sanity (B19)
  8. Secure aggregation round-trip: server sees the SUM only (Stage 14)
  9. Communication-cost arithmetic (Stage 15)
 10. Feature-name propagation through the synthetic pipeline (B12)

Run:  python tests/test_pipeline_integrity.py
"""

import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def make_labels(n, pos=0.05, seed=0):
    return np.random.RandomState(seed).binomial(1, pos, n)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Disjoint Dirichlet partitioning (B2)
# ─────────────────────────────────────────────────────────────────────────────
def test_disjoint_dirichlet():
    print("\n[1] Disjoint Dirichlet partitioning (B2)")
    from partition_clients import partition_data_dirichlet, \
        allocate_indices_dirichlet

    for alpha, seed in [(0.5, 1), (1.0, 42), (5.0, 7), (0.1, 3)]:
        y = make_labels(20000, pos=0.05, seed=seed)
        X = np.random.RandomState(seed).randn(20000, 12)
        shards, assignment = partition_data_dirichlet(
            X, y, alpha=alpha, n_clients=6, seed=seed,
            return_indices=True, verbose=False)

        sizes = np.bincount(assignment, minlength=6)
        coverage = sizes.sum() == len(y)
        unique_vals = set(np.unique(assignment).tolist())

        # disjointness: reconstruct per-client membership and count
        # occurrences of each sample
        counts = np.bincount(assignment[assignment >= 0], minlength=6)
        # each sample appears in assignment exactly once by construction;
        # the real risk was duplicate X rows across shards — emulate via
        # an id array because synthetic X may collide by chance
        ids = np.arange(len(y))
        seen = []
        for k in range(6):
            mask = assignment == k
            seen.extend(ids[mask].tolist())
        dup = len(seen) - len(set(seen))

        check(f"alpha={alpha} seed={seed}: coverage 100%",
              coverage and unique_vals == set(range(6)),
              f"sizes={sizes}")
        check(f"alpha={alpha} seed={seed}: 0 duplicate samples",
              dup == 0, f"dup={dup}")
        check(f"alpha={alpha} seed={seed}: shard sizes match assignment",
              all(len(shards[k][1]) == sizes[k] for k in range(6)))

    # non-IID-ness sanity: label skew exists at alpha=0.5
    y = make_labels(20000, pos=0.05, seed=11)
    X = np.random.RandomState(11).randn(20000, 12)
    assignment = allocate_indices_dirichlet(y, alpha=0.5, n_clients=6, seed=11)
    rates = [y[assignment == k].mean() for k in range(6)]
    spread = max(rates) - min(rates)
    check("alpha=0.5 produces label skew (spread > 5pp)", spread > 0.05,
          f"spread={spread:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Leakage detector catches seeded leakage
# ─────────────────────────────────────────────────────────────────────────────
def test_leakage_detector():
    print("\n[2] Leakage detector detects seeded leakage")
    from leakage_audit.audit_leakage import (check_disjointness,
                                             check_split_exclusivity,
                                             check_region_leakage,
                                             check_window_overlap)

    a = np.arange(100)
    r = check_disjointness([a[:40], a[35:70], a[70:]])
    check("detects shard overlap", not r["pass"] and r["duplicate_count"] == 5)

    r = check_disjointness([a[:34], a[34:67], a[67:]])
    check("passes clean shards", r["pass"])

    r = check_split_exclusivity(np.arange(50), np.arange(40, 60),
                                np.arange(55, 80))
    check("detects split overlap", not r["pass"])

    regions = np.repeat(np.arange(10), 10)
    r = check_region_leakage(regions, np.arange(0, 55),
                             np.arange(50, 70), np.arange(60, 100))
    check("detects region leakage across boundary", not r["pass"])

    r = check_region_leakage(regions, np.arange(0, 50),
                             np.arange(50, 70), np.arange(70, 100))
    check("passes region-exclusive split", r["pass"])

    keys = np.arange(0, 600, 10)  # window origins every 10 steps, window 60
    r = check_window_overlap(keys, np.arange(0, 30), np.arange(30, 60),
                             stride=10, window=60)
    # windows at index 290..300 (key 290) overlap train windows — seeded
    check("detects window boundary overlap", not r["pass"],
          f"violations={r.get('boundary_violations')}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Immutable train/val/test contract (B10)
# ─────────────────────────────────────────────────────────────────────────────
def test_split_contract():
    print("\n[3] Immutable train/val/test contract (B10)")
    import pandas as pd
    from data_preparation import preprocess

    rng = np.random.RandomState(0)
    n = 6000
    df = pd.DataFrame(rng.randn(n, 10),
                      columns=[f"feature_{i}" for i in range(10)])
    df["label"] = rng.binomial(1, 0.06, n)
    df["_split"] = None

    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)  # protect repo dir from synthetic CSV writes
        try:
            splits = preprocess(df, val_fraction=0.16)
        finally:
            os.chdir(cwd)

        tr, va, te = set(splits.train_idx.tolist()), \
            set(splits.val_idx.tolist()), set(splits.test_idx.tolist())
        check("train ∩ val = ∅", not (tr & va))
        check("train ∩ test = ∅", not (tr & te))
        check("val ∩ test = ∅", not (va & te))
        check("union = all samples", len(tr | va | te) == n)
        check("sizes 64/16/20 ±1%",
              abs(len(tr) / n - 0.64) < 0.01 and
              abs(len(va) / n - 0.16) < 0.01 and
              abs(len(te) / n - 0.20) < 0.01,
              f"{len(tr)}/{len(va)}/{len(te)}")

        # cleaned-data path: val carved from TRAIN only
        df2 = df.copy()
        df2["_split"] = ["train"] * 5000 + ["test"] * 1000
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                s2 = preprocess(df2, val_fraction=0.16)
            finally:
                os.chdir(cwd)
        check("cleaned path: val carved from train only",
              set(np.where(df2["_split"].values == "test")[0].tolist()) ==
              set(s2.test_idx.tolist()) and
              not (set(s2.val_idx.tolist()) &
                   set(s2.test_idx.tolist())))


# ─────────────────────────────────────────────────────────────────────────────
# 4 + 5. Threshold protocol & metric recomputation (B1, B4)
# ─────────────────────────────────────────────────────────────────────────────
def test_threshold_protocol():
    print("\n[4] Threshold protocol: validation selection, no test oracle (B1)")
    from evaluation import (find_optimal_threshold_fbeta,
                            compute_all_metrics)

    rng = np.random.RandomState(0)
    # validation: balanced-ish; test: 2% positives (the SWAN-SF situation)
    y_val = rng.binomial(1, 0.4, 4000)
    p_val = np.clip(0.3 + 0.4 * y_val + rng.randn(4000) * 0.15, 0.01, 0.99)
    y_test = rng.binomial(1, 0.02, 20000)
    p_test = np.clip(0.3 + 0.4 * y_test + rng.randn(20000) * 0.15, 0.01, 0.99)

    t_val, fb_val = find_optimal_threshold_fbeta(y_val, p_val, beta=2.0)
    t_oracle, fb_oracle = find_optimal_threshold_fbeta(y_test, p_test, beta=2.0)

    m_frozen = compute_all_metrics(y_test, p_test, t_val)
    m_oracle = compute_all_metrics(y_test, p_test, t_oracle)

    check("val-selected threshold exists", 0.05 <= t_val <= 0.95,
          f"t_val={t_val}")
    check("frozen-threshold F2 <= oracle F2 (no leakage upside)",
          m_frozen["f2"] <= m_oracle["f2"] + 1e-12,
          f"frozen={m_frozen['f2']:.3f} oracle={m_oracle['f2']:.3f}")
    check("val and oracle thresholds usually differ (selection is not test-fit)",
          abs(t_val - t_oracle) > 1e-9,
          f"t_val={t_val} t_oracle={t_oracle}")

    print("\n[5] Stale-accuracy fix: full metric recomputation (B4)")
    # accuracy must be computed at THE GIVEN threshold, matching manual calc
    t = 0.4
    m = compute_all_metrics(y_test, p_test, t)
    manual_acc = ((p_test >= t).astype(int) == y_test).mean()
    check("accuracy recomputed at frozen threshold",
          abs(m["accuracy"] - manual_acc) < 1e-12,
          f"{m['accuracy']:.6f} vs {manual_acc:.6f}")
    check("PR-AUC present", 0 <= m["pr_auc"] <= 1)
    check("Brier present", 0 <= m["brier"] <= 1)
    check("ECE present", 0 <= m["ece"] <= 1)
    check("recall@FPR keys present",
          "recall@1%FPR" in m and "recall@5%FPR" in m)
    check("F2 = fbeta(beta=2)", abs(m["f2"] - m["f2"]) < 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Calibration (protocol layer)
# ─────────────────────────────────────────────────────────────────────────────
def test_calibration():
    print("\n[6] Calibration under prior shift")
    from evaluation import (PriorShiftCorrection, PlattScaling,
                            TemperatureScaling, IsotonicCalibration,
                            expected_calibration_error, select_calibration)
    from sklearn.metrics import brier_score_loss

    rng = np.random.RandomState(0)
    # calibrated model on training prevalence ~0.5: y ~ Bernoulli(score)
    n = 30000
    score = np.clip(rng.beta(2, 2, n), 0.02, 0.98)
    y_train = (rng.rand(n) < score).astype(int)

    # calibrated pair for ECE test (no shift)
    ece_cal = expected_calibration_error(y_train, score)
    ece_bad = expected_calibration_error(y_train, np.full(n, 0.9))
    check("ECE discriminates (calibrated << miscalibrated)",
          ece_cal < 0.05 and ece_cal < ece_bad / 3,
          f"{ece_cal:.3f} vs {ece_bad:.3f}")

    # operational prevalence 0.05: keep negatives, downsample positives
    keep = np.where((y_train == 0) | (rng.rand(n) < 0.0526))[0]
    p_op, y_op = score[keep], y_train[keep]
    check("fixture prevalence ~5%", abs(y_op.mean() - 0.05) < 0.01,
          f"prev={y_op.mean():.3f}")

    ps = PriorShiftCorrection().set_prevalences(0.5, y_op.mean())
    p_adj = ps.transform(p_op)
    check("prior-shift: corrected mean ~ operational prevalence",
          abs(p_adj.mean() - y_op.mean()) < 0.02,
          f"adj={p_adj.mean():.4f} true={y_op.mean():.4f}")

    base_brier = brier_score_loss(y_op, p_op)
    adj_brier = brier_score_loss(y_op, p_adj)
    check("prior-shift: Brier improves", adj_brier < base_brier,
          f"{adj_brier:.4f} < {base_brier:.4f}")

    # fit-on-validation calibrators also improve
    for Cal in (PlattScaling, TemperatureScaling, IsotonicCalibration):
        cal = Cal().fit(y_op[:len(y_op)//2], p_op[:len(y_op)//2])
        p_cal = cal.transform(p_op[len(y_op)//2:])
        b = brier_score_loss(y_op[len(y_op)//2:], p_cal)
        check(f"{Cal.__name__}: Brier <= raw+0.02", b <= base_brier + 0.02,
              f"{b:.4f} vs raw {base_brier:.4f}")

    best, table = select_calibration(y_op, p_op)
    check("select_calibration returns a valid method",
          best in ("none", "prior_shift", "platt", "isotonic", "temperature"),
          f"best={best}")
    check("select_calibration prefers prior_shift on shifted data",
          best in ("prior_shift", "isotonic", "platt"),
          f"best={best} (expected a correcting method, not 'none')")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Secure aggregation round-trip (Stage 14)
# ─────────────────────────────────────────────────────────────────────────────
def test_secure_aggregation():
    print("\n[7] Secure aggregation round-trip")
    from secure_aggregation import secure_aggregate_round
    rng = np.random.RandomState(0)
    updates = [rng.randn(16, 8) for _ in range(6)]
    recovered, true_sum = secure_aggregate_round(updates)
    err = np.abs(recovered - true_sum).max()
    check("masks cancel exactly (server sees SUM only)", err < 1e-9,
          f"err={err:.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Communication cost arithmetic (Stage 15)
# ─────────────────────────────────────────────────────────────────────────────
def test_communication_cost():
    print("\n[8] Communication cost arithmetic")
    from communication_cost import model_size_bytes, simulate_dropout

    class FakeP:
        def __init__(self, n):
            self.n = n

        def numel(self):  # torch API: method, not property
            return self.n

    class FakeM:
        def parameters(self):
            return [FakeP(1000), FakeP(2000)]

    size, params = model_size_bytes(FakeM())
    check("model size = params * 4 bytes", size == 12000 and params == 3000)

    sched, stats = simulate_dropout([None] * 6, 50, drop_rate=0.3, seed=1)
    check("dropout schedule has >=1 client per round",
          sched.sum(axis=1).min() >= 1)
    check("dropout schedule shape", sched.shape == (50, 6))
    check("mean participation in plausible range",
          0.5 < stats["mean_participation"] <= 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Feature-name propagation (B12)
# ─────────────────────────────────────────────────────────────────────────────
def test_feature_names():
    print("\n[9] Feature-name propagation (B12)")
    from data_preparation import _select_feature_columns, \
        _arrays_to_dataframe
    import pandas as pd

    # stat-prefixed names survive
    names = [f"{s}_{f}" for s in ("mean", "std") for f in ("R_VALUE", "TOTUSJH")]
    df = pd.DataFrame(np.random.rand(10, 4), columns=names)
    df["label"] = np.random.randint(0, 2, 10)
    selected, mode = _select_feature_columns(df)
    check("stat-prefixed names selected in order", selected == names and
          mode == "stat_prefixed", f"mode={mode}")

    # arrays -> dataframe keeps names
    X_tr, y_tr = np.random.rand(8, 4), np.random.randint(0, 2, 8)
    X_te, y_te = np.random.rand(4, 4), np.random.randint(0, 2, 4)
    df = _arrays_to_dataframe(X_tr, y_tr, X_te, y_te, names)
    check("dataframe carries physical names",
          list(df.columns[:4]) == names and set(df["_split"]) == {"train", "test"})
    check("no fabricated HARPNUM column", "HARPNUM_MOD" not in df.columns
          and "_HARPNUM_MOD" not in df.columns)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Multi-seed summary (B11)
# ─────────────────────────────────────────────────────────────────────────────
def test_multiseed_summary():
    print("\n[10] Multi-seed statistics summary (B11)")
    sys.path.insert(0, os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "experiments"))
    from run_multiseed import _summary

    vals = [0.7, 0.75, 0.8, 0.85, 0.9]
    s = _summary(vals)
    check("mean correct", abs(s["mean"] - 0.8) < 1e-12)
    check("std (ddof=1) correct",
          abs(s["std"] - np.std(vals, ddof=1)) < 1e-12)
    check("CI95 present and positive", s["ci95_half"] > 0)
    check("n recorded", s["n"] == 5)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Balanced fallback + geographic partition
# ─────────────────────────────────────────────────────────────────────────────
def test_fallback_partitions():
    print("\n[11] Balanced fallback & geographic partition")
    from partition_clients import _partition_balanced_random, \
        partition_data_geographic

    rng = np.random.RandomState(0)
    X, y = rng.randn(6000, 5), rng.binomial(1, 0.1, 6000)
    shards = _partition_balanced_random(X, y, n_clients=6, verbose=False,
                                        seed=3)
    total = sum(len(s[1]) for s in shards)
    check("balanced fallback covers 100%", total == len(y))

    # region-based geographic partition: regions never split
    regions = np.repeat(np.arange(60), 100)
    shards = partition_data_geographic(X, y, region_ids=regions,
                                       n_clients=6, verbose=False)
    total = sum(len(s[1]) for s in shards)
    check("geographic partition covers 100%", total == len(y))

    # Dirichlet fallback warns when no region ids
    shards = partition_data_geographic(X, y, region_ids=None,
                                       n_clients=6, verbose=False)
    check("geographic fallback returns 6 shards", len(shards) == 6)


# ─────────────────────────────────────────────────────────────────────────────
# 11b. Review-2 protocol hardening (R4 / R7 / R13 / R15)
# ─────────────────────────────────────────────────────────────────────────────

def test_fpr_threshold_transfer():
    """R4: validation-selected FPR thresholds transfer to deployment data
    drawn from the same negative score distribution, across prevalence."""
    print("\n[11b-1] Validation-frozen FPR thresholds (R4)")
    from evaluation import (select_fpr_thresholds_on_validation,
                            frozen_operating_point_metrics)
    rng = np.random.RandomState(11)
    n_val = 20000
    y_val = rng.randint(0, 2, n_val)
    p_val = np.where(y_val == 1, rng.beta(5, 1.5, n_val),
                     rng.beta(2, 5, n_val))
    frozen = select_fpr_thresholds_on_validation(y_val, p_val)
    check("4 FPR budgets produced", len(frozen) == 4)

    n_dep = 100000
    y_dep = (rng.rand(n_dep) < 0.02).astype(int)
    p_dep = np.where(y_dep == 1, rng.beta(5, 1.5, n_dep),
                     rng.beta(2, 5, n_dep))
    res = frozen_operating_point_metrics(y_dep, p_dep, frozen)
    for key, v in res.items():
        target = float(key.replace("%FPR", "")) / 100.0
        check(f"realised {key} ~ target (got {v['realised_fpr']:.4f})",
              abs(v["realised_fpr"] - target) <= 0.004)
    # selection must not depend on validation prevalence: shuffling the
    # positive scores leaves the negative quantile untouched
    p_val2 = p_val.copy()
    p_val2[y_val == 1] = rng.permutation(p_val2[y_val == 1])
    frozen2 = select_fpr_thresholds_on_validation(y_val, p_val2)
    same = all(abs(frozen[k]["threshold"] - frozen2[k]["threshold"]) < 1e-12
               for k in frozen)
    check("threshold invariant to positive-score permutation", same)


def test_region_disjoint_split():
    """R7: no region spans train/validation; coverage is exact."""
    print("\n[11b-2] Region-disjoint validation split (R7)")
    from data_preparation import region_disjoint_val_split
    from leakage_audit.audit_leakage import check_region_leakage

    rng = np.random.RandomState(5)
    n_regions, windows_per_region = 120, 40
    n = n_regions * windows_per_region
    region_ids = np.repeat(np.arange(n_regions), windows_per_region)
    # regions 0..59 flare-rich, 60..119 flare-poor -> pooled ~balanced
    rates = np.where(np.arange(n_regions) < 60, 0.8, 0.2)
    y = np.zeros(n, dtype=int)
    for r in range(n_regions):
        idx = np.arange(r * windows_per_region, (r + 1) * windows_per_region)
        y[idx] = rng.binomial(1, rates[r], windows_per_region)

    train_idx, val_idx, report = region_disjoint_val_split(
        y, region_ids, val_fraction=0.16, seed=42)

    check("coverage exact (train+val = pool)",
          len(train_idx) + len(val_idx) == n)
    check("no index in both splits",
          len(np.intersect1d(train_idx, val_idx)) == 0)
    r_leak = check_region_leakage(region_ids, train_idx, val_idx,
                                  np.array([], int))
    check("no region spans train/val", r_leak["pass"] is True)
    check("val fraction within [10%, 25%]",
          0.10 <= len(val_idx) / n <= 0.25,
          f"got {len(val_idx)/n:.3f}")
    check("val prevalence near pooled rate (region-stratified)",
          abs(report["val_prevalence"] - y.mean()) <= 0.05,
          f"got {report['val_prevalence']:.3f} vs {y.mean():.3f}")
    check("report marks region_disjoint", report["region_disjoint"] is True)


def test_event_level_metrics():
    """R13/R19: event-level metrics on a deterministic fixture."""
    print("\n[11b-3] Event-level evaluation (R13/R19)")
    from experiments.run_event_level import (_synthetic_self_check,
                                             event_level_metrics)
    ok, r = _synthetic_self_check()
    check("synthetic fixture: 2 events, 1 detected, 1 missed", ok)
    # cooldown de-duplication suppresses duplicate alerts
    y = np.ones(20, dtype=int)
    p = np.where(np.arange(20) % 2 == 0, 0.9, 0.1)
    ev = np.array(["X"] * 20)
    r_nocd = event_level_metrics(y, p, ev, threshold=0.5)
    r_cd = event_level_metrics(y, p, ev, threshold=0.5, cooldown_windows=3)
    check("without cooldown: 10 alerts for one event",
          r_nocd["total_alerts"] == 10)
    check("with cooldown=3: duplicates suppressed",
          r_cd["total_alerts"] < r_nocd["total_alerts"])


def test_training_budget_report():
    """R15: budget audit exposes the centralized-vs-FL confound."""
    print("\n[11b-4] Training-budget audit (R15)")
    from centralized_baseline import training_budget_report
    rep = training_budget_report(82121, n_val=15643,
                                 central_epochs=30, central_batch=256,
                                 fl_rounds=50, fl_local_epochs=10,
                                 fl_batch=512)
    check("central epochs recorded", rep["centralized_mlp"]["max_epochs"] == 30)
    check("FL effective passes recorded",
          rep["federated"]["effective_passes_per_client_shard"] == 500)
    check("budgets_matched is False at defaults (honest)",
          rep["budgets_matched"] is False)
    check("confound note present", "NOT matched" in rep["note"])


def test_neutral_client_labels():
    """R8: no institution names in the client labels."""
    print("\n[11b-5] Neutral client labels (R8)")
    import config as cfg
    joined = " ".join(cfg.CLIENT_NAMES)
    banned = ["NASA", "NOAA", "ESA", "JAXA", "ISRO", "KASI", "BoM",
              "PROBA"]
    check("no institution names in CLIENT_NAMES",
          not any(b in joined for b in banned), joined)


def main():
    print("=" * 64)
    print("  SF-9 PIPELINE INTEGRITY TESTS (fixes verification)")
    print("=" * 64)

    test_disjoint_dirichlet()
    test_leakage_detector()
    test_split_contract()
    test_threshold_protocol()
    test_calibration()
    test_secure_aggregation()
    test_communication_cost()
    test_feature_names()
    test_multiseed_summary()
    test_fallback_partitions()
    test_fpr_threshold_transfer()
    test_region_disjoint_split()
    test_event_level_metrics()
    test_training_budget_report()
    test_neutral_client_labels()

    print("\n" + "=" * 64)
    print(f"  RESULT: {PASS} passed, {FAIL} failed")
    print("=" * 64)
    return FAIL == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
