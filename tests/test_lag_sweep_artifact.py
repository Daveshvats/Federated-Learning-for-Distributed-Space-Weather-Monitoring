"""
tests/test_lag_sweep_artifact.py  (v4.3 — Dossier R-FS9-R3 N3)
─────────────────────────────────────────────────────
Validates outputs/lag_definition_sweep.json against
outputs/standard_metrics.json — the cross-artefact agreement that,
before this test, lived only inside the generation script
(experiments/run_lag_definition_sweep.py), so a silent future edit
that broke the sweep's agreement with the metrics artefact would not
have failed the battery.

Everything is read at run time from the two committed artefacts; no
number is hand-copied into this file. Run standalone or via
tests/run_battery.py.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS, FAIL = 0, 0

SWEEP = os.path.join(ROOT, "outputs", "lag_definition_sweep.json")
METRICS = os.path.join(ROOT, "outputs", "standard_metrics.json")


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def _strict_load(path):
    """json.load that refuses NaN/Infinity tokens (strict-JSON rule)."""
    with open(path) as f:
        return json.load(f, parse_constant=lambda c: (_ for _ in ()).throw(
            ValueError(f"non-strict JSON constant {c!r} in {path}")))


def main():
    for p in (SWEEP, METRICS):
        check(f"artefact present: {os.path.basename(p)}",
              os.path.exists(p))
    sweep = _strict_load(SWEEP)
    metrics = _strict_load(METRICS)

    defs = sweep["definitions"]
    by_id = {d["id"]: d for d in defs}

    print("\n[S1] structure of the sweep artefact")
    check(f"9 definitions (8 candidates + 1 shipped), got {len(defs)}",
          len(defs) == 9)
    check("role split is 8 candidate + 1 shipped",
          sum(1 for d in defs if d.get("role") == "candidate") == 8 and
          sum(1 for d in defs if d.get("role") == "shipped") == 1)
    check("every definition carries pooled_test, p5_test, "
          "p5_clears_floor, definition string",
          all(all(k in d for k in
                  ("definition", "pooled_test", "p5_test",
                   "p5_clears_floor")) for d in defs))

    print("\n[S2] shipped rule reproduces block C (the sweep's own "
          "self-check, now battery-guarded)")
    shipped = by_id["shipped_nearest_35"]
    lag_c = (metrics["block_c_baselines"]["p5_test"]
             ["persistence_24h_lag"])
    for field in ("tss", "hss"):
        check(f"shipped p5 {field} == block C lagged persistence "
              f"({shipped['p5_test'][field]:.6f})",
              abs(shipped["p5_test"][field] - lag_c[field]) < 1e-9,
              f"block C says {lag_c[field]}")
    check(f"shipped p5 n_evaluated == block C n "
          f"({shipped['p5_test']['n_evaluated']})",
          shipped["p5_test"]["n_evaluated"] == lag_c["n_evaluated"],
          f"block C says {lag_c['n_evaluated']}")

    print("\n[S3] clears-the-floor lists recomputed from block A "
          "(run-time cross-reference, no hand-copied arms)")
    arms = metrics["block_a_derived"]["leakage_free_fold"]
    arm_tss = {k: v["fbeta_threshold"]["tss"] for k, v in arms.items()}
    arm_hss = {k: v["fbeta_threshold"]["hss"] for k, v in arms.items()}
    known_arms = set(arm_tss)
    for d in defs:
        floor_tss, floor_hss = d["p5_test"]["tss"], d["p5_test"]["hss"]
        want_tss = sorted(a for a in known_arms
                          if arm_tss[a] > floor_tss)
        want_hss = sorted(a for a in known_arms
                          if arm_hss[a] > floor_hss)
        got_tss = sorted(d["p5_clears_floor"]["tss"])
        got_hss = sorted(d["p5_clears_floor"]["hss"])
        check(f"{d['id']}: clears-TSS list matches block A "
              f"({', '.join(got_tss) or 'none'})",
              got_tss == want_tss,
              f"recomputed {want_tss}")
        check(f"{d['id']}: clears-HSS list matches block A "
              f"({', '.join(got_hss) or 'none'})",
              got_hss == want_hss,
              f"recomputed {want_hss}")

    print("\n[S4] invariance across the point-lag family + the "
          "disclosed lookback exception")
    point_lag = [d for d in defs if d["id"] != "any_flare_in_lookback"]
    lookback = by_id["any_flare_in_lookback"]
    ref = sorted(point_lag[0]["p5_clears_floor"]["tss"])
    check("all point-lag definitions agree on the clears-TSS list",
          all(sorted(d["p5_clears_floor"]["tss"]) == ref
              for d in point_lag))
    check("all point-lag definitions agree that no arm clears at HSS",
          all(not d["p5_clears_floor"]["hss"] for d in point_lag))
    check("lookback variant stores its own (different) n_evaluated "
          "— the disclosed semantic footnote",
          lookback["p5_test"]["n_evaluated"] !=
          shipped["p5_test"]["n_evaluated"] and
          lookback["p5_test"]["n_evaluated"] > 0)

    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    return FAIL == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
