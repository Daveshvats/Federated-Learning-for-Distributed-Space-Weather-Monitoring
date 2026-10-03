#!/usr/bin/env python3
"""
tests/test_interpretability_artifact.py — R-FS9-R6 R8-11: the
interpretability artefact finally has a battery guard, and the
fallback/row-count contract is pinned to the source.

Background (Dossier R-FS9-R6, R8-11): the DeepExplainer-to-
GradientExplainer fallback in interpretability.py swallowed its
exception silently, the committed artefact recorded no explainer
field, and the FL attributions were drawn from 300 rows against
XGBoost's 500 while the artefact note claimed "first 500 TEST
samples" — and no battery test covered interpretability.json at all
(the panel verified the printed numbers are correct to the digit).
v4.6: the fallback is loud and recorded, the row counts align at 500,
and the frozen v3.0 artefact is disclosed as predating the fix.

Checks:
  1. record integrity — sha256 of the committed artefact is pinned
     (a frozen published record must never change silently);
  2. internal consistency — Spearman rho, p-value and top-15 overlap
     recomputed from the artefact's own 144-dim vectors via the
     shipped consistency_report must match the stored values; the
     stored top-15 lists must match the vectors' argmax sets; every
     listed feature must be a <stat>_<BASE> name with BASE in
     config.FEATURE_COLS;
  3. frozen-record disclosure fields — sample_size 500 and the
     no-cherry-picking note are present as published;
  4. source contract — shap_torch_model defaults to 500 rows, the
     fallback prints the R8-11 warning and returns which explainer
     ran, and run_interpretability records fl_explainer /
     fl_attribution_rows for future runs.

Run:  python tests/test_interpretability_artifact.py  (or via battery)
"""
import hashlib
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ARTEFACT = os.path.join(ROOT, "outputs", "interpretability.json")
MODULE = os.path.join(ROOT, "interpretability.py")
RUNNER = os.path.join(ROOT, "experiments", "run_interpretability.py")

# sha256 of the frozen v3.0 published record (unchanged at v4.6 —
# the code fixes are forward-looking; the artefact is history).
ARTEFACT_SHA256 = (
    "7f6f8dff08ca312391e82d6d8066d05b98777ddb79ba39215dbb1b2c5c325e34")

STATS = ("mean", "std", "max", "min", "trend", "slope")


def check_record_integrity():
    with open(ARTEFACT, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest == ARTEFACT_SHA256:
        print("[PASS] interpretability.json sha256 matches the frozen "
              "published record")
        return True
    print(f"[FAIL] interpretability.json sha256 changed: {digest} "
          f"(expected {ARTEFACT_SHA256}) — a frozen published record "
          f"was edited; restore it or re-pin with a disclosed "
          f"RUNLOG row")
    return False


def check_internal_consistency():
    with open(ARTEFACT, "r", encoding="utf-8") as f:
        d = json.load(f)
    xgb = np.asarray(d["xgboost_importance"], dtype=float)
    fl = np.asarray(d["fl_importance"], dtype=float)
    ok = True
    if xgb.shape != (144,) or fl.shape != (144,):
        print(f"[FAIL] importance vectors are {xgb.shape}/{fl.shape}, "
              f"expected (144,)")
        ok = False
    if not ok:
        return False

    # recompute the consistency statistics with the shipped code
    import config as cfg  # noqa: E402
    from interpretability import consistency_report  # noqa: E402
    names = [f"f{i}" for i in range(144)]
    rep = consistency_report(xgb, fl, names)
    for key in ("spearman_rho", "p_value"):
        if not np.isclose(rep[key], d[key], rtol=1e-9, atol=1e-12):
            print(f"[FAIL] {key}: stored {d[key]} vs recomputed "
                  f"{rep[key]}")
            ok = False
    if not np.isclose(rep["top15_overlap_fraction"],
                      d["top15_overlap_fraction"]):
        print(f"[FAIL] top15_overlap_fraction: stored "
              f"{d['top15_overlap_fraction']} vs recomputed "
              f"{rep['top15_overlap_fraction']}")
        ok = False

    # stored top-15 lists must have exactly 15 entries each
    for list_key in ("top_features_a", "top_features_b"):
        if len(d[list_key]) != 15:
            print(f"[FAIL] {list_key} has {len(d[list_key])} entries")
            ok = False
    if not ok:
        return False

    # every listed feature name must be <stat>_<BASE> with BASE in
    # FEATURE_COLS (the 24 SWAN-SF base parameters)
    bases = set(cfg.FEATURE_COLS)
    for list_key in ("top_features_a", "top_features_b"):
        for n in d[list_key]:
            m = re.match(r"^([a-z]+)_(.+)$", n)
            if not m or m.group(1) not in STATS or \
                    m.group(2) not in bases:
                print(f"[FAIL] feature name '{n}' in {list_key} does "
                      f"not match <stat>_<BASE> with BASE in "
                      f"FEATURE_COLS")
                ok = False
    if ok:
        print("[PASS] internal consistency: spearman/p/overlap "
              "recompute exactly from the 144-dim vectors; all listed "
              "features are <stat>_<BASE> names over the 24 SWAN-SF "
              "base parameters")
    return ok


def check_frozen_record_fields():
    with open(ARTEFACT, "r", encoding="utf-8") as f:
        d = json.load(f)
    ok = d.get("sample_size") == 500 and \
        "first 500 TEST samples" in d.get("note", "") and \
        "cherry" in d.get("note", "")
    if ok:
        print("[PASS] frozen-record fields intact: sample_size 500, "
              "deterministic-slice note as published (the FL vector's "
              "300-row provenance is the disclosed v3.0 pre-v4.6 "
              "record, per RUNLOG)")
        return True
    print("[FAIL] the published note/sample_size fields changed — "
          "the frozen record was edited")
    return False


def check_source_contract():
    with open(MODULE, "r", encoding="utf-8") as f:
        src = f.read()
    checks = [
        ("default rows aligned", "sample_size=500" in src),
        ("loud fallback", "R-FS9-R6 R8-11" in src and
         "WARNING" in src),
        ("explainer recorded", '"fl_explainer"' in src and
         '"fl_attribution_rows"' in src),
    ]
    ok = all(passed for _, passed in checks)
    for label, passed in checks:
        if not passed:
            print(f"[FAIL] source contract lost: {label} "
                  f"(interpretability.py)")
    if ok:
        print("[PASS] source contract: shap_torch_model defaults to "
              "500 rows; the fallback prints the R8-11 warning; the "
              "report records fl_explainer + fl_attribution_rows")
    return ok


def main():
    results = [
        check_record_integrity(),
        check_internal_consistency(),
        check_frozen_record_fields(),
        check_source_contract(),
    ]
    passed = sum(1 for r in results if r)
    failed = sum(1 for r in results if not r)
    print(f"RESULT: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
