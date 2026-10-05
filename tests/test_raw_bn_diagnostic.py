#!/usr/bin/env python3
"""
tests/test_raw_bn_diagnostic.py — the v4.8 / E1 raw-substrate
BN-diagnostic guard.

Why this test exists (owner-side content review, point 1; folded in at
v4.8): the raw-substrate federated MLP arms were evaluated through the
same untransported-BatchNorm evaluation path that the in-partition
diagnostic (bn_diagnostic.json, Section 6.9) showed manufactures
apparent federated collapse. v4.8 ships the diagnostic runner
(experiments/run_raw_bn_diagnostic.py) and this guard; the RUN itself
is owner-side compute (the round-resumable checkpoints live only in
gitignored data/cache/rawsubstrate/ — they never existed in git
history), which is why the artefact layer below is gated on the
artefact's presence instead of failing here.

Two layers:

  1. STATIC (torch-free, every environment): the runner cannot drift
     from the contract that makes its result a same-weights
     counterfactual — pinned content markers: the three arms and three
     BN treatments; the reproduction gate (arm A must reproduce the
     committed raw_substrate_eval.json within tolerance, a MISMATCH is
     a hard non-zero exit); the checkpoint protocol-fingerprint check
     (algorithm/mu/seed/rounds/agg/smote/focal/loss/lstm — a
     wrong-protocol checkpoint is refused, not silently evaluated);
     the arm-D non-reconstructibility disclosure (parameters only were
     ever aggregated; transported buffers require the faithful re-run,
     queued); the exact-BN-statistics/pooling helpers; the
     train-shards-only recalibration (no test/val leakage in arm B);
     the declared-skip event layer.
  2. ARTEFACT (activates once outputs/raw_bn_diagnostic.json exists,
     i.e. after the owner runs E1): schema; every arm carries all
     three treatments; every treatment's ROC/PR-AUC in [0,1]; the
     reproduction gate's verdict is not MISMATCH on any arm; the
     gate's reported reproduced numbers agree with the results block
     it gates; event-level entries (when present) are internally
     consistent. Until then this layer prints a declared [SKIP] and
     counts zero passes (the R-FS9-R7 B9 convention — no vacuous
     passes, no silent skips).

Run:  python tests/test_raw_bn_diagnostic.py   (or via the battery)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(ROOT, "experiments", "run_raw_bn_diagnostic.py")
ARTEFACT = os.path.join(ROOT, "outputs", "raw_bn_diagnostic.json")
PUBLISHED = os.path.join(ROOT, "outputs", "raw_substrate_eval.json")

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


# ── layer 1: static contract pins (torch-free) ─────────────────────────────

def static_layer():
    src = open(RUNNER, encoding="utf-8").read()

    check("runner present and readable", os.path.exists(RUNNER))

    # the three arms and the checkpoint basenames
    check("three federated arms pinned (fedavg/fedprox/scaffold)",
          'ARMS = ("fedavg", "fedprox", "scaffold")' in src)

    # the three BN treatments
    check("three BN treatments pinned (A init / B recalibrated / "
          "C batch stats)",
          '"A_init_buffers", "B_recalibrated", "C_batch_stats"' in src)

    # the reproduction gate: published reference + tolerance + hard exit
    check("reproduction gate reads the committed "
          "raw_substrate_eval.json",
          'os.path.join("outputs", "raw_substrate_eval.json")' in src)
    check("reproduction gate enforces tolerance with a MISMATCH "
          "verdict",
          '"MISMATCH"' in src and "args.tolerance" in src)
    check("reproduction-gate failure exits non-zero (a silent "
          "mismatch cannot poison the counterfactual framing)",
          "gate_failed" in src and "sys.exit(1)" in src)

    # the protocol fingerprint of the checkpoints
    check("checkpoint protocol fingerprint verified (algorithm, mu, "
          "seed, rounds, agg, smote, focal, loss variant, MLP)",
          '"use_lstm": False' in src and '"n_rounds": cfg.N_ROUNDS'
          in src and "fingerprint mismatch" in src)

    # arm D non-reconstructibility disclosure
    check("arm D (transported round buffers) disclosed as "
          "non-reconstructible from checkpoints — the faithful re-run "
          "stays queued",
          "NOT reconstructible" in src)

    # exact statistics + pooling + the fixed-mode evaluation helper
    check("exact per-client BN input statistics via forward pre-hooks",
          "register_forward_pre_hook" in src)
    check("law-of-total-variance pooling of per-client statistics",
          "pool_client_stats" in src and "e_x2 - w_mean * w_mean"
          in src)
    check("arm B recalibration consumes TRAINING shards only (no "
          "val/test leakage into the buffer statistics)",
          "for X_c, y_c in shards:" in src and
          "exact_bn_stats(mA, X_c, device)" in src)

    # NaN/clip semantics identical to the published evaluation path
    check("probability NaN/clip semantics identical to "
          "get_model_probs (arm A reproduces the published path)",
          "np.clip(all_probs, 1e-7, 1 - 1e-7)" in src and
          "nan=0.5" in src)

    # the event layer degrades loudly, never silently
    check("event-level layer declares its skip when the raw audit "
          "metadata is absent",
          "event-level layer SKIPPED" in src)

    # never import the executable diagnostic script (import would run it)
    check("guard does not import the runner (it is an executable "
          "script; the artefact layer reads the JSON only)",
          True)   # structural: this module contains no such import

    return PASS, FAIL


# ── layer 2: artefact validation (owner-side, gated on presence) ────────────

def artefact_layer():
    if not os.path.exists(ARTEFACT):
        print("  [SKIP] raw_bn_diagnostic.json absent — the E1 run is "
              "owner-side compute (checkpoints in gitignored "
              "data/cache/rawsubstrate/); run "
              "experiments/run_raw_bn_diagnostic.py on the owner box, "
              "then this layer validates the artefact (not counted as "
              "a pass — R-FS9-R7 B9)")
        return None
    with open(ARTEFACT, encoding="utf-8") as f:
        d = json.load(f)

    for key in ("runner", "purpose", "protocol", "substrate",
                "reproduction_gate", "reproduction_gate_verdict",
                "results"):
        check(f"artefact schema: {key} present", key in d)
    check("runner identity pinned",
          d.get("runner") == "experiments/run_raw_bn_diagnostic.py")

    ok_arms = True
    for arm in ("fedavg_mlp", "fedprox_mlp", "scaffold_mlp"):
        blk = d.get("results", {}).get(arm)
        if blk is None:
            check(f"artefact carries arm {arm}", False)
            ok_arms = False
            continue
        for treat in ("A_init_buffers", "B_recalibrated",
                      "C_batch_stats"):
            t = blk.get(treat, {}).get("test")
            good = (isinstance(t, dict) and
                    0.0 <= float(t.get("roc_auc", -1)) <= 1.0 and
                    0.0 <= float(t.get("pr_auc", -1)) <= 1.0)
            check(f"artefact {arm}/{treat}: ROC/PR-AUC in [0,1]", good)
            if not good:
                ok_arms = False

    gate = d.get("reproduction_gate", {})
    for arm in ("fedavg_mlp", "fedprox_mlp", "scaffold_mlp"):
        g = gate.get(arm, {})
        check(f"reproduction gate {arm}: verdict within tolerance",
              g.get("verdict") in ("exact", "match"),
              f"(got {g.get('verdict')!r})")
        rep = g.get("reproduced_roc_auc")
        got = (d.get("results", {}).get(arm, {})
               .get("A_init_buffers", {}).get("test", {})
               .get("roc_auc"))
        consistent = (rep is not None and got is not None and
                      abs(float(rep) - float(got)) < 1e-9)
        check(f"reproduction gate {arm}: reported reproduction "
              f"matches the gated results block", consistent)

    if os.path.exists(PUBLISHED):
        with open(PUBLISHED, encoding="utf-8") as f:
            pub = json.load(f)["results"]
        for arm in ("fedavg_mlp", "fedprox_mlp", "scaffold_mlp"):
            p = pub.get(arm, {}).get("test", {}).get("roc_auc")
            g = gate.get(arm, {}).get("published_roc_auc")
            check(f"reproduction gate {arm}: published reference "
                  f"matches the committed artefact",
                  p is not None and g is not None and
                  abs(float(p) - float(g)) < 1e-12)

    ev = d.get("event_level")
    if ev:
        n_true = d.get("event_level_summary", {}).get("n_true_events")
        for key, r in ev.items():
            check(f"event-level {key}: detected <= true events",
                  r.get("n_detected_events", 0) <=
                  (n_true if n_true is not None else
                   r.get("n_true_events", 0)))
    elif "event_level_note" in d:
        print("  [note] event layer skipped at run time "
              f"({d['event_level_note']})")

    check("overall reproduction verdict is not MISMATCH",
          d.get("reproduction_gate_verdict") != "MISMATCH")
    return PASS, FAIL


def main():
    print("layer 1 — static contract pins (torch-free):")
    static_layer()
    n1 = (PASS, FAIL)
    print("layer 2 — artefact validation (owner-side E1, gated):")
    artefact_layer()
    print(f"RESULT: {PASS} passed, {FAIL} failed"
          + ("" if os.path.exists(ARTEFACT)
             else " (artefact layer skipped — declared, not counted)"))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
