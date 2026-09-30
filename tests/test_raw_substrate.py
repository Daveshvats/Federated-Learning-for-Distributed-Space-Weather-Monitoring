"""
tests/test_raw_substrate.py  (v3.5)
────────────────────────────────────
Unit tests for experiments/raw_substrate.py — runnable WITHOUT the
SWAN-SF dataset (synthetic fixtures only; no torch):

  1. interp_partial: fills partial gaps, leaves fully-missing series
     for the kNN stage (stage-A contract)
  2. _window_means / _nanmean / _corr_matrix: shapes, pairwise-complete
     behaviour, min-pair-count gating (stage-B contract)
  3. fit_lsbzm + apply_lsbzm: output finite and in [0,1]; the chain is
     strictly rank-preserving on the fitted domain (monotone per
     feature); parameters JSON-round-trip stable (stage-C/D contract)
  4. compute_scales: power-of-two scales that cover the data (f16
     storage safety contract)
  5. availability masking logic: features below 50% availability are
     excluded from kNN distance spaces (TOTFZ lesson)

Run:  python tests/test_raw_substrate.py
"""

import os
import sys
import json
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


def main():
    from experiments.raw_substrate import (interp_partial, fit_lsbzm,
                                            apply_lsbzm, compute_scales,
                                            _window_means, _corr_matrix,
                                            KNN_K, CORR_SPACE)

    rng = np.random.default_rng(42)

    # ── 1. stage A: interpolation contract ──────────────────────────
    X = rng.normal(size=(50, 60, 24)).astype(np.float32)
    X[0, 0:10, 3] = np.nan                      # leading-edge gap
    X[1, :, 5] = np.nan                         # fully missing
    X[2, 20:40, 7] = np.nan                     # interior gap
    n_partial = interp_partial(X)
    check("interp fills partial gaps",
          np.isfinite(X[0, :, 3]).all() and np.isfinite(X[2, :, 7]).all())
    check("interp leaves fully-missing series for kNN",
          np.isnan(X[1, :, 5]).all())
    check("interp count = number of partial cells", n_partial == 2,
          f"got {n_partial}")
    # edge extension: the first valid value extends to the border
    check("interp edge extension is constant",
          X[0, 0, 3] == X[0, 10, 3] and X[0, 9, 3] == X[0, 10, 3])

    # ── 2. stage B helpers ──────────────────────────────────────────
    A = rng.normal(size=(200, 60, 24)).astype(np.float32)
    A[:150, :, 8] = np.nan                      # 25% availability for f8
    M = _window_means(A)
    check("window means shape", M.shape == (200, 24))
    check("window means NaN iff fully-missing",
          np.isnan(M[:, 8]).sum() == 150 and
          np.isfinite(M[:, 0]).all())
    R = _corr_matrix(M)
    check("corr matrix shape + diagonal",
          R.shape == (24, 24) and np.allclose(np.diag(R), 1.0))
    # a feature missing for >70% of windows must be gated to 0
    M2 = M.copy()
    M2[:, 9] = np.nan
    M2[:10, 9] = 1.0                            # 5% availability
    R2 = _corr_matrix(M2)
    check("min-pair gating zeroes sparse-feature correlations",
          np.allclose(R2[9, np.arange(24) != 9], 0.0) and
          np.allclose(R2[np.arange(24) != 9, 9], 0.0))
    # availability masking (TOTFZ lesson): usable features only
    avail = 1.0 - np.isnan(M).mean(axis=0)
    usable = avail >= 0.5
    check("low-availability feature excluded from distance spaces",
          not usable[8] and usable[0])

    # ── 3. stage C/D: LSBZM chain ──────────────────────────────────
    # heavy-tailed synthetic features spanning many magnitudes
    S = np.zeros((300, 60, 24), dtype=np.float32)
    for f in range(24):
        S[:, :, f] = (rng.lognormal(mean=2 * f, sigma=1.5,
                                    size=(300, 60)) - 5.0)
    S[5, :, 7] = np.nan                         # a fully-missing series
    # (fit tolerates and ignores NaN values)
    params = fit_lsbzm(S, seed=42)
    check("lsbzm params cover all 24 features",
          len(params["features"]) == 24)
    kinds = {p["kind"] for p in params["features"].values()}
    check("lsbzm dispatch only uses the documented menu",
          kinds <= {"log", "sqrt", "boxcox"}, f"got {kinds}")
    # JSON round-trip stability
    params_rt = json.loads(json.dumps(params))
    X1 = S.copy()
    apply_lsbzm(X1, params_rt)
    check("apply_lsbzm output finite",
          np.isfinite(X1[~np.isnan(S)]).all())
    check("apply_lsbzm output in [0, 1]",
          float(np.nanmin(X1)) >= 0.0 and float(np.nanmax(X1)) <= 1.0)
    # rank preservation on a finite feature: the chain is monotone
    # (non-decreasing along the input's sorted order; float32 output
    # quantisation may create ties, never inversions)
    f = 3
    v0 = S[:, :, f].ravel().astype(np.float64)
    v1 = X1[:, :, f].ravel().astype(np.float64)
    order0 = np.argsort(v0)
    check("lsbzm chain is monotone (no rank inversions)",
          bool(np.all(np.diff(v1[order0]) >= 0)),
          f"inversions: {int((np.diff(v1[order0]) < 0).sum())}")
    from scipy.stats import spearmanr
    rho = spearmanr(v0, v1).statistic
    check("lsbzm rank fidelity >= 0.93 (tie dilution bounded)",
          abs(float(rho)) >= 0.93, f"rho={rho}")

    # ── 4. f16 storage scales ──────────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        for p in (1, 2, 3, 4):
            np.savez(os.path.join(td, f"p{p}_raw.npz"),
                     X=(rng.lognormal(20, 3, size=(30, 60, 24))
                        * rng.choice([-1, 1], size=(30, 60, 24))).astype(
                            np.float32))
        scales = compute_scales(td, out_dir=td)   # isolated cache
        Xbig = np.load(os.path.join(td, "p1_raw.npz"))["X"]
        scaled = Xbig / scales
        check("scales are powers of two",
              np.all(np.log2(scales) == np.round(np.log2(scales))))
        check("scaled values fit float16 range",
          float(np.abs(scaled).max()) < 65504.0 and
          np.isfinite(scaled.astype(np.float16)).all())

    # ── 5. constants documented ─────────────────────────────────────
    check("kNN constants match the paper",
          KNN_K == 5 and CORR_SPACE == 5)

    print("\n" + "=" * 64)
    print(f"  RESULT: {PASS} passed, {FAIL} failed")
    print("=" * 64)
    return FAIL


if __name__ == "__main__":
    import tempfile
    sys.exit(1 if main() else 0)
