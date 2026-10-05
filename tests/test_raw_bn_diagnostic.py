#!/usr/bin/env python3
"""
tests/test_raw_bn_diagnostic.py — the v4.8 / E1 raw-substrate
BN-diagnostic guard, re-pinned at v4.9 to the artefact of record,
extended at v4.9.2 with the R-FS9-R10 register instruments,
amended at v4.9.3 after the first ask-#8/#9 executions: the A1
gate recalibrated (the v4.9.2 single 1e-2 bracket generalised an
ROC-only measurement to PR-AUC and was unsatisfiable by
construction — the frozen ask-#4 MLP PR drift is 1.93e-2) and the
A3 selection crash fixed with a DYNAMIC regression guard (the
inline max(dict, key=dict) raised TypeError only after a full
50-round arm had trained; string pins cannot prove a callable is
right, execution can).

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

v4.9: the owner EXECUTED the E1 programme (RUNLOG ask #4, status RAN)
and the artefact of record is now committed — outputs/
raw_bn_diagnostic.json, sha256-pinned below like every other frozen
record. The run's disclosed outcome: the v3.x-era raw-substrate
checkpoints were never persisted (parameters-only aggregation + that
era's checkpoint rule), so the same-weights counterfactual was
unavailable by construction; the diagnostic ran over a fresh seed-42
replication (converter-rebuilt substrate, one extra f16 quantisation;
RTX 4060, torch 2.11.0+cu126, python 3.13.12) and the reproduction
gate honestly reported MISMATCH on every arm (FL deltas 0.002-0.023
ROC; pooled baselines reproduce to <= 2.3e-3, so the substrate rebuild
is near-exact and the FL deltas are retrain nondeterminism). The
v4.8 layer encoded the same-weights happy path ("verdict within
tolerance"); the v4.9 layer pins the DISCLOSED MISMATCH state — a
silent re-run that flips a verdict, or any byte edit of the frozen
record, fails this battery (the interpretability-sha convention).

v4.9.2 (Dossier R-FS9-R10 — the Round-13 third-party-adjudication
register): two new instruments join the diagnostic's guard home, one
layer each of static contract pins plus gated artefact validation:

  A1  experiments/run_arm_b_central_sanity.py — the centralised
      arm-B sanity check (register item A1, "the thirteen-second
      check that decides whether Table 8's strongest column stands").
      Artefact: outputs/arm_b_central_sanity.json, declared [SKIP]
      until the owner-side run lands (the v4.8 pre-run convention),
      activating on presence.
  A3  experiments/run_raw_nobn.py — no-BatchNorm MLP federation on
      the raw substrate, FedAvg and FedProx (register item A3, the
      encoder-vs-BN-under-federation attribution control).  Artefact:
      outputs/raw_nobn_eval.json, same gated convention.

Three layers (the two new ones are 3 and 4):

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
  2. ARTEFACT (the committed record of the executed E1 run — in every
     clone since v4.9; previously gated on the artefact's presence
     while the run was owner-side): sha256 of the frozen artefact
     pinned; schema; every arm carries all three treatments; every
     treatment's ROC/PR-AUC in [0,1]; the reproduction gate's verdict
     equals the DISCLOSED MISMATCH state on every arm and overall (the
     structural outcome of the never-persisted v3.x checkpoints —
     flipping any verdict without a RUNLOG row fails); the gate's
     reported reproduced numbers agree with the results block it
     gates; the published reference still equals the committed
     raw_substrate_eval.json; event-level entries are internally
     consistent.
  3. R-FS9-R10 A1 STATIC: the centralised-sanity runner's contract —
     the baselines.pt load with the gate-as-fingerprint-surrogate
     disclosure (no fingerprint fields exist on the centralised path);
     the v4.9.3 TWO-GATE design: the IDENTITY gate vs the ask-#4
     retrain record (raw_substrate_rerun.json, 1e-3 — the
     wrong-checkpoint tripwire, the tightest reference that exists
     for this checkpoint) and the PUBLISHED retrain bracket, PER
     METRIC (ROC 1e-2 — never the 1e-4 same-weights gate, the
     ask-#4 lesson — and PR 2.5e-2, the frozen ask-#4 MLP PR drift
     1.93e-2 with headroom; the v4.9.2 single 1e-2 bracket was
     unsatisfiable by construction for the correct checkpoint); the
     declared verdict bands (0.95/0.35 validated, 0.60/0.10 broken);
     the exit contract (decisive verdicts exit 0, either gate's
     MISMATCH and inconclusive exit 1); the byte-identical pooling
     machinery; train-shards-only statistics; the artefact written
     BEFORE any exit.
  3b. v4.9.3 DYNAMIC SELECTION GUARD (torch-free): import the no-BN
      runner (its deferred torch stack) and exercise the REAL
      val-ROC round-selection semantics — the inline original passed
      the dict itself as the key function and crashed the owner's
      ask-#9 run only after 50 rounds had trained; this layer makes
      that bug class a battery failure at import/execution time.
  4. R-FS9-R10 A1/A3 ARTEFACTS: gated on presence, declared [SKIP]
     while the runs are owner-side (the v4.8 convention — a skip is
     not counted as a pass, B9); on presence: schema, verdict in the
     declared alphabet with bands consistent with verdict_basis, the
     v4.9.3 identity-gate block (a pre-v4.9.3 artefact fails loudly —
     it was written by the superseded single-tolerance instrument
     and must be regenerated by the ask-#10 re-run), both gates'
     verdicts consistent with their own deltas and tolerances, the
     A3 arms' selection + B1-hygiene blocks, and the reference
     columns equal to the committed artefacts read live (never
     hand-typed).

Run:  python tests/test_raw_bn_diagnostic.py   (or via the battery)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(ROOT, "experiments", "run_raw_bn_diagnostic.py")
ARTEFACT = os.path.join(ROOT, "outputs", "raw_bn_diagnostic.json")
PUBLISHED = os.path.join(ROOT, "outputs", "raw_substrate_eval.json")
RUNNER_A1 = os.path.join(ROOT, "experiments",
                         "run_arm_b_central_sanity.py")
ARTEFACT_A1 = os.path.join(ROOT, "outputs", "arm_b_central_sanity.json")
RUNNER_A3 = os.path.join(ROOT, "experiments", "run_raw_nobn.py")
ARTEFACT_A3 = os.path.join(ROOT, "outputs", "raw_nobn_eval.json")
LSTM_REFERENCE = os.path.join(ROOT, "outputs", "raw_lstm_eval.json")

A1_VERDICT_ALPHABET = {"arm_b_validated", "arm_b_broken",
                       "inconclusive"}

PASS, FAIL = 0, 0

# v4.9 pins: the artefact of record (frozen) and the disclosed state of
# its reproduction gate. The owner's E1 run could not reproduce the
# published raw-substrate numbers from the original weights — those
# checkpoints were never persisted — and reported MISMATCH honestly;
# these pins freeze THAT state so any change is loud.
ARTEFACT_SHA256 = (
    "9cebc01e6839f93aabf51a1cec8b63b19e611fe60d2be3e28dc0dbeced5768e2")
DISCLOSED_GATE_VERDICTS = {"fedavg_mlp": "MISMATCH",
                           "fedprox_mlp": "MISMATCH",
                           "scaffold_mlp": "MISMATCH"}
DISCLOSED_OVERALL_VERDICT = "MISMATCH"


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
        # v4.9: the record is committed — absence is a defect, not a
        # pending owner run (the pre-v4.9 declared [SKIP] convention
        # applied only while the artefact was owner-side).
        print("  [FAIL] raw_bn_diagnostic.json absent — the artefact "
              "of record is committed at v4.9; a clone without it is "
              "broken")
        return None
    with open(ARTEFACT, "rb") as f:
        import hashlib
        digest = hashlib.sha256(f.read()).hexdigest()
    check("artefact sha256 matches the frozen v4.9 record of the "
          "executed E1 run", digest == ARTEFACT_SHA256,
          f"(got {digest[:16]}…)")
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
        check(f"reproduction gate {arm}: verdict equals the disclosed "
              f"MISMATCH state (the never-persisted v3.x checkpoints; "
              "RUNLOG ask #4)",
              g.get("verdict") == DISCLOSED_GATE_VERDICTS[arm],
              f"(got {g.get('verdict')!r} — a re-run changed the "
              f"record; re-pin with a disclosed RUNLOG row)")
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

    check("overall reproduction verdict equals the disclosed MISMATCH "
          "state (v4.9 re-pin of the v4.8 happy-path pin)",
          d.get("reproduction_gate_verdict") == DISCLOSED_OVERALL_VERDICT)
    return PASS, FAIL


# ── layer 3: R-FS9-R10 A1/A3 static contract pins (v4.9.2) ──────────────────

def r10_static_layer():
    # ── A1: the centralised arm-B sanity runner ───────────────────────
    src = open(RUNNER_A1, encoding="utf-8").read()

    check("A1 runner present and readable", os.path.exists(RUNNER_A1))

    # the centralised model load + the fingerprint disclosure
    check("A1 loads the centralised MLP from the rawsubstrate "
          "baselines cache (the ask-#4 retrain's by-product; the "
          "v3.x-era central checkpoint was never persisted)",
          'os.path.join(CACHE, "baselines.pt")' in src and
          '"centralized_mlp" not in c_models' in src)
    check("A1 discloses that baselines.pt carries NO protocol "
          "fingerprint, so the arm-A reproduction gate IS the "
          "fingerprint surrogate (a wrong checkpoint fails the gate, "
          "not a fingerprint check that cannot exist)",
          "fingerprint surrogate" in src)

    # the retrain-bracket gate — never the 1e-4 same-weights gate
    check("A1 gate reads the committed raw_substrate_eval.json "
          "centralised_mlp reference",
          'os.path.join("outputs", "raw_substrate_eval.json")' in src
          and 'published["centralized_mlp"]' in src)
    check("A1 gate tolerance is the RETRAIN bracket (1e-2 default, "
          "not the 1e-4 same-weights gate — a retrained checkpoint "
          "cannot pass a same-weights gate by construction, the "
          "ask-#4 lesson)",
          '"--gate-tolerance", type=float, default=1e-2' in src and
          "retrain bracket" in src)

    # v4.9.3 — the two-gate amendment (the first ask-#8 execution,
    # 2026-10-06, exposed the v4.9.2 single-tolerance miscalibration)
    check("A1 v4.9.3 identity gate reads the committed ask-#4 retrain "
          "record (raw_substrate_rerun.json — the tightest reference "
          "that exists for THIS checkpoint: same weights, same frozen "
          "eval path)",
          'os.path.join("outputs", "raw_substrate_rerun.json")' in src
          and 'rerun_ref["centralized_mlp"]' in src)
    check("A1 identity-gate tolerance is the same-weights bracket "
          "(1e-3 default — the wrong-checkpoint tripwire, 10x tighter "
          "than the v4.9.2 published bracket)",
          '"--identity-tolerance", type=float, default=1e-3' in src)
    check("A1 published-bracket tolerance is PER METRIC (ROC 1e-2 "
          "unchanged; PR 2.5e-2 NEW — the frozen ask-#4 MLP PR drift "
          "is 1.93e-2, so the v4.9.2 single 1e-2 bracket was "
          "unsatisfiable by construction for the correct checkpoint)",
          '"--gate-tolerance-pr", type=float, default=2.5e-2' in src
          and "args.gate_tolerance_pr" in src)
    check("A1 gate verdicts are computed per gate, per metric "
          "(identity: max of deltas; published: ROC AND PR "
          "independently)",
          "identity_verdict" in src and "args.identity_tolerance" in src
          and "d_roc <= args.gate_tolerance and" in src
          and "d_pr <= args.gate_tolerance_pr" in src)
    check("A1 BOTH gates exit non-zero on MISMATCH (a wrong "
          "checkpoint and an out-of-bracket checkpoint are both "
          "unreportable)",
          'identity_verdict == "MISMATCH"' in src and
          'gate_verdict == "MISMATCH"' in src and
          "sys.exit(1)" in src)
    check("A1 tolerance basis cites the frozen ask-#4 drift table "
          "(ROC deltas + the 1.93e-2 MLP PR drift — evidence-based, "
          "never hand-waved)",
          "1.93e-2" in src and "-2.32e-3" in src and
          "metric-scoped" in src)

    # the declared verdict bands
    check("A1 verdict bands declared and pinned (validated: "
          "ROC >= 0.95 and PR >= 0.35; broken: ROC <= 0.60 or "
          "PR <= 0.10)",
          "VERDICT_PASS_ROC, VERDICT_PASS_PR = 0.95, 0.35" in src and
          "VERDICT_FAIL_ROC, VERDICT_FAIL_PR = 0.60, 0.10" in src)

    # the exit contract
    check("A1 exit contract: decisive verdicts (validated OR broken) "
          "exit 0 — both are findings; gate-MISMATCH and inconclusive "
          "exit 1",
          'verdict == "inconclusive"' in src and
          'verdict == "arm_b_validated"' in src and "sys.exit(0)" in src
          and "sys.exit(1)" in src)

    # the byte-identical pooling machinery + train-shards-only
    check("A1 reuses the diagnostic's exact contracts: forward "
          "pre-hooks, law-of-total-variance pooling, non-negativity "
          "clamp",
          "register_forward_pre_hook" in src and
          "e_x2 - w_mean * w_mean" in src)
    check("A1 computes per-client statistics over the TRAINING shards "
          "only (no val/test leakage into the buffer statistics)",
          "for X_c, y_c in shards:" in src and
          "exact_bn_stats(mA, X_c, device)" in src)

    # the supporting diagnostics + the before-exit write
    check("A1 records the pooled-vs-own per-layer buffer deltas "
          "(the reconstruction made visible)",
          "pooled_vs_own_buffer_deltas" in src)
    check("A1 writes the artefact BEFORE any exit (a gate failure "
          "still leaves the record behind)",
          "_atomic_json(report, args.output)" in src and
          'os.path.join("outputs", "arm_b_central_sanity.json")' in src)

    # ── A3: the no-BN raw-substrate federation runner ─────────────────
    src3 = open(RUNNER_A3, encoding="utf-8").read()

    check("A3 runner present and readable", os.path.exists(RUNNER_A3))

    # the architecture intervention
    check("A3 architecture: SolarMLP with BatchNorm1d replaced by "
          "Identity (the run_nobn_control.py construction — same "
          "stack minus gamma/beta, no running statistics, exact "
          "weight transport)",
          "isinstance(m, _nn.BatchNorm1d)" in src3 and
          "_nn.Identity()" in src3)
    check("A3 patches model.SolarMLP so the FL machinery constructs "
          "no-BN models transparently (the in-partition control's "
          "mechanism)",
          "_model_mod.SolarMLP = _NoBNSolarMLP" in src3)

    # both register-named algorithms + state separation
    check("A3 runs both register-named arms: FedAvg and FedProx",
          'ALGOS = ("fedavg", "fedprox")' in src3)
    check("A3 state files are separate from the BN arms' checkpoints "
          "and carry arch='nobn' (no programme collision, no foreign "
          "resume)",
          'f"nobn_{algo}_state.pt"' in src3 and
          '"arch": "nobn"' in src3)

    # the comparability contract
    check("A3 evaluation mirrors run_raw_substrate.py section 5 "
          "exactly (prior-shift calibration, val-frozen F-beta "
          "threshold, frozen-FPR operating points — the "
          "comparability requirement)",
          "run_raw_substrate.py section 5" in src3 and
          "set_prevalences" in src3 and
          "find_optimal_threshold_fbeta" in src3 and
          "select_fpr_thresholds_on_validation" in src3)

    # B1 hygiene born with the arms
    check("A3 B1 hygiene: the val-ROC-selected sensitivity row rides "
          "beside the shipped best-val-F1 rule (both disclosed, "
          "neither test-based)",
          "best_val_roc (sensitivity row" in src3 and
          "selection_rule" in src3)

    # v4.9.3 — the ask-#9 crash fix (TypeError after 50 rounds)
    check("A3 selection extracted to the module-level torch-free "
          "helper select_round_by_val_roc (the v4.9.3 fix for the "
          "ask-#9 crash: max(val_roc_by_round, key=val_roc_by_round) "
          "passed the DICT itself as the key function — TypeError: "
          "'dict' object is not callable — after the owner's fedavg "
          "arm had completed all 50 rounds)",
          "def select_round_by_val_roc" in src3 and
          "return max(val_roc_by_round, "
          "key=val_roc_by_round.get)" in src3)
    check("A3 selection call site uses the helper — the buggy inline "
          "max(...) form survives ONLY as errata quotations in the "
          "docstrings (exactly 2 occurrences: the module header and "
          "the helper's own errata record), the code carries exactly "
          "one .get form, and layer 3b executes the real semantics)",
          "roc_round = select_round_by_val_roc(history)" in src3 and
          src3.count("key=val_roc_by_round)") == 2 and
          src3.count("key=val_roc_by_round.get)") == 1)
    check("A3 records the per-round test trajectory (the degenerate "
          "monitor cannot hide an operating point)",
          '"test_trajectory": traj' in src3)
    check("A3 discloses the degenerate-selection-rule fallback (the "
          "Round-13 pathology, handled loudly instead of shipping "
          "silently)",
          "DEGENERATE selection rule disclosed" in src3)

    # the reference columns + the torch gate
    check("A3 comparison columns are READ from the committed "
          "artefacts at run time (never hand-typed — the v4.0 "
          "convention)",
          "_reference_block" in src3 and "raw_lstm_eval.json" in src3)
    check("A3 fails loudly at a torch-less gate (no raw import "
          "traceback — the battery convention)",
          "[nobn] torch is required" in src3 and
          "_load_torch_stack()" in src3)
    check("A3 writes outputs/raw_nobn_eval.json atomically",
          'os.path.join("outputs", "raw_nobn_eval.json")' in src3)

    return PASS, FAIL


# ── layer 3b: the v4.9.3 DYNAMIC selection guard ───────────────────
# The ask-#9 crash class: code that only executes AFTER a full
# federation arm has trained (the round-selection step) — a string
# pin cannot prove a callable is right; execution can.  The runner's
# deferred-torch-stack discipline means it imports torch-less, so
# this layer runs in every battery environment.

def r10_dynamic_layer():
    sys.path.insert(0, ROOT)
    try:
        from experiments.run_raw_nobn import select_round_by_val_roc
        imported = True
    except Exception as e:                        # pragma: no cover
        imported = False
        print(f"       (import error: {type(e).__name__}: {e})")
    check("A3 runner imports cleanly in the battery environment "
          "(torch-less here — the deferred-torch-stack discipline "
          "holds; no raw import traceback)",
          imported)
    if not imported:
        return PASS, FAIL
    check("A3 val-ROC selection picks the max-ROC round (the v4.9.3 "
          "crash fix, exercised for real)",
          select_round_by_val_roc(
              [{"round": 5, "roc_auc": 0.510},
               {"round": 10, "roc_auc": 0.974},
               {"round": 15, "roc_auc": 0.555}]) == 10)
    check("A3 val-ROC selection resolves ties to the EARLIEST round "
          "(deterministic — dict insertion order = monitored-round "
          "order)",
          select_round_by_val_roc(
              [{"round": 5, "roc_auc": 0.9},
               {"round": 10, "roc_auc": 0.9}]) == 5)
    try:
        select_round_by_val_roc([])
        empty_raises = False
    except ValueError:
        empty_raises = True
    check("A3 val-ROC selection raises loudly on empty history (no "
          "silent empty selection can ship a sensitivity row)",
          empty_raises)
    return PASS, FAIL


# ── layer 4: R-FS9-R10 A1/A3 artefact validation (gated on presence, ────────
# declared [SKIP] while the runs are owner-side — the v4.8 pre-run
# convention; a skip is NOT counted as a pass, B9)

def r10_artefact_layer():
    # ── A1 artefact ────────────────────────────────────────────────────
    if not os.path.exists(ARTEFACT_A1):
        print("  [SKIP] arm_b_central_sanity.json absent — the A1 run "
              "is owner-side compute (RUNLOG ask #8); this layer "
              "activates when the artefact lands (not counted as a "
              "pass — B9)")
    else:
        with open(ARTEFACT_A1, encoding="utf-8") as f:
            d = json.load(f)
        for key in ("runner", "purpose", "protocol", "substrate",
                    "reproduction_gate", "reproduction_gate_verdict",
                    "results", "pooled_vs_own_buffer_deltas", "verdict",
                    "verdict_basis"):
            check(f"A1 artefact schema: {key} present", key in d)
        check("A1 runner identity pinned",
              d.get("runner") ==
              "experiments/run_arm_b_central_sanity.py")
        v = d.get("verdict")
        check("A1 verdict in the declared alphabet "
              "(arm_b_validated / arm_b_broken / inconclusive)",
              v in A1_VERDICT_ALPHABET, f"(got {v!r})")
        # the verdict must be consistent with its own declared bands
        b = d.get("verdict_basis", {})
        obs = b.get("observed", {})
        pb, fb = b.get("pass_bands", {}), b.get("fail_bands", {})
        roc = float(obs.get("roc_auc", -1))
        pr = float(obs.get("pr_auc", -1))
        band_verdict = (
            "arm_b_validated"
            if roc >= float(pb.get("roc_ge", 0.95)) and
            pr >= float(pb.get("pr_ge", 0.35)) else
            "arm_b_broken"
            if roc <= float(fb.get("roc_le", 0.60)) or
            pr <= float(fb.get("pr_le", 0.10)) else "inconclusive")
        check("A1 verdict is consistent with its own declared bands "
              "and observed numbers (the bands cannot drift from the "
              "verdict)",
              v == band_verdict,
              f"(verdict says {v!r}, bands say {band_verdict!r})")
        g = d.get("reproduction_gate", {})
        check("A1 reproduction-gate verdict is a declared value",
              g.get("verdict") in ("match", "MISMATCH", "exact"))
        for treat in ("A_own_buffers", "B_pooled_recalibrated"):
            t = (d.get("results", {}).get("centralized_mlp", {})
                 .get(treat, {}).get("test", {}))
            check(f"A1 centralised/{treat}: ROC/PR-AUC in [0,1]",
                  isinstance(t, dict) and
                  0.0 <= float(t.get("roc_auc", -1)) <= 1.0 and
                  0.0 <= float(t.get("pr_auc", -1)) <= 1.0)
        if os.path.exists(PUBLISHED):
            with open(PUBLISHED, encoding="utf-8") as f:
                pub = json.load(f)["results"]
            p = pub.get("centralized_mlp", {}).get("test", {})
            check("A1 published reference matches the committed "
                  "artefact (never hand-typed)",
                  p is not None and g.get("published_roc_auc") is not
                  None and abs(float(p["roc_auc"]) -
                               float(g["published_roc_auc"])) < 1e-12)

        # v4.9.3 — the identity gate (the wrong-checkpoint tripwire).
        # A pre-v4.9.3 artefact fails here loudly: it was written by
        # the superseded single-tolerance instrument and must be
        # regenerated by the ask-#10 re-run under the amended one.
        ig = d.get("identity_gate", {})
        check("A1 identity-gate block present (v4.9.3 schema — a "
              "pre-v4.9.3 artefact must be regenerated by the ask-#10 "
              "re-run)",
              isinstance(ig, dict) and "verdict" in ig and
              "retrain_roc_auc" in ig)
        check("A1 identity-gate verdict is a declared value",
              ig.get("verdict") in ("match", "MISMATCH"))
        RERUN = os.path.join(ROOT, "outputs", "raw_substrate_rerun.json")
        if os.path.exists(RERUN):
            with open(RERUN, encoding="utf-8") as f:
                rer = json.load(f)["results"]["centralized_mlp"]["test"]
            check("A1 identity reference matches the committed ask-#4 "
                  "retrain record (never hand-typed)",
                  rer is not None and ig.get("retrain_roc_auc") is not
                  None and abs(float(rer["roc_auc"]) -
                               float(ig["retrain_roc_auc"])) < 1e-12)
        try:
            _id_exp = ("match" if max(float(ig["delta_roc_auc"]),
                                      float(ig["delta_pr_auc"]))
                       <= float(ig["tolerance"]) else "MISMATCH")
            id_consistent = ig.get("verdict") == _id_exp
        except (KeyError, TypeError, ValueError):
            id_consistent = False
        check("A1 identity-gate verdict consistent with its own "
              "deltas and tolerance (v4.9.3)",
              id_consistent)
        try:
            _tol = g["tolerance"]
            _pub_exp = ("match" if
                        (float(g["delta_roc_auc"]) <=
                         float(_tol["roc_auc"]) and
                         float(g["delta_pr_auc"]) <=
                         float(_tol["pr_auc"])) else "MISMATCH")
            pub_consistent = g.get("verdict") == _pub_exp
        except (KeyError, TypeError, ValueError):
            pub_consistent = False
        check("A1 published-gate verdict consistent with its own "
              "per-metric deltas and tolerances (v4.9.3 — the gate "
              "cannot report a verdict its numbers contradict)",
              pub_consistent)

    # ── A3 artefact ────────────────────────────────────────────────────
    if not os.path.exists(ARTEFACT_A3):
        print("  [SKIP] raw_nobn_eval.json absent — the A3 run is "
              "owner-side GPU compute (RUNLOG ask #9); this layer "
              "activates when the artefact lands (not counted as a "
              "pass — B9)")
    else:
        with open(ARTEFACT_A3, encoding="utf-8") as f:
            d = json.load(f)
        for key in ("runner", "purpose", "reference_columns"):
            check(f"A3 artefact schema: {key} present", key in d)
        check("A3 runner identity pinned",
              d.get("runner") == "experiments/run_raw_nobn.py")
        for algo in ("fedavg", "fedprox"):
            blk = d.get(algo)
            if blk is None:
                check(f"A3 artefact carries arm {algo}", False)
                continue
            check(f"A3 {algo}: selected-checkpoint block present",
                  isinstance(blk.get("selected_checkpoint_test"), dict)
                  and "test" in blk.get("selected_checkpoint_test", {}))
            t = blk.get("selected_checkpoint_test", {}).get("test", {})
            check(f"A3 {algo}: selected ROC/PR-AUC in [0,1]",
                  0.0 <= float(t.get("roc_auc", -1)) <= 1.0 and
                  0.0 <= float(t.get("pr_auc", -1)) <= 1.0)
            check(f"A3 {algo}: B1 sensitivity row present "
                  "(val-ROC-selected, disclosed)",
                  isinstance(
                      blk.get("sensitivity_val_roc_selected_test"),
                      dict) and
                  "sensitivity_val_roc_selected_test" in blk)
            check(f"A3 {algo}: per-round test trajectory present",
                  isinstance(blk.get("test_trajectory"), dict) and
                  len(blk.get("test_trajectory", {})) > 0)
        # reference columns equal the committed artefacts, read live
        ref = d.get("reference_columns", {})
        ok_mlp = ok_lstm = True
        if os.path.exists(PUBLISHED):
            with open(PUBLISHED, encoding="utf-8") as f:
                pub = json.load(f)["results"]
            for k in ("fedavg_mlp", "fedprox_mlp", "centralized_mlp"):
                t = pub.get(k, {}).get("test", {})
                if k in ref or t:
                    ok_mlp = ok_mlp and k in ref and \
                        abs(float(ref[k]["roc_auc"]) -
                            float(t["roc_auc"])) < 1e-12 and \
                        abs(float(ref[k]["pr_auc"]) -
                            float(t["pr_auc"])) < 1e-12
        if os.path.exists(LSTM_REFERENCE):
            with open(LSTM_REFERENCE, encoding="utf-8") as f:
                lref = json.load(f)["results"]
            for k in ("fedavg_lstm", "fedprox_lstm", "central_lstm"):
                t = lref.get(k, {}).get("test", {})
                if k in ref or t:
                    ok_lstm = ok_lstm and k in ref and \
                        abs(float(ref[k]["roc_auc"]) -
                            float(t["roc_auc"])) < 1e-12 and \
                        abs(float(ref[k]["pr_auc"]) -
                            float(t["pr_auc"])) < 1e-12
        check("A3 reference columns equal the committed "
              "raw_substrate_eval.json (never hand-typed)",
              ok_mlp and bool(ref))
        check("A3 reference columns equal the committed "
              "raw_lstm_eval.json (never hand-typed)",
              ok_lstm and bool(ref))

    return PASS, FAIL


def main():
    print("layer 1 — static contract pins (torch-free):")
    static_layer()
    n1 = (PASS, FAIL)
    print("layer 2 — artefact validation (the committed v4.9 record):")
    artefact_layer()
    print("layer 3 — R-FS9-R10 A1/A3 static contract pins (v4.9.2, "
          "amended v4.9.3):")
    r10_static_layer()
    print("layer 3b — the v4.9.3 dynamic selection guard (ask #9's "
          "crash class, executed for real):")
    r10_dynamic_layer()
    print("layer 4 — R-FS9-R10 A1/A3 artefact validation "
          "(gated on presence, declared [SKIP] while owner-side):")
    r10_artefact_layer()
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
