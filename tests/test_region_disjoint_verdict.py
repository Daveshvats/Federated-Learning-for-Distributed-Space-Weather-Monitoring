#!/usr/bin/env python3
"""
tests/test_region_disjoint_verdict.py — the v4.12 external-review
item-2 verdict guard (the within-fold region-disjoint validation
re-run of the no-BatchNorm raw-substrate arms).

The v4.12 response committed two owner-side compute items. Item 3
(the seed-43 replication) executed, landed, and integrated at
v4.13/v4.13.1. Item 2 — the divergence diagnosis of Section 6.4:
FedAvg's validation ROC-AUC rising while its test ROC-AUC falls,
which a prevalence difference cannot explain (ROC-AUC is a rank
statistic), leaving two candidate mechanisms: a region-sharing
random validation carve, or partition-5 temporal recency —
EXECUTED on the owner GPU on 2026-10-06 (3,119 s, RTX 4060;
``python experiments/run_raw_nobn.py --region-disjoint
--raw-dir raw``) and its artefact was committed by the owner
directly (commit 17af7e4: outputs/raw_nobn_region_disjoint.json,
742 lines, the Windows CRLF form, 25,722 bytes — the first
owner-side push of a full-fidelity artefact rather than a paste).
This module byte-pins that artefact, re-derives every number the
paper's v4.14 integration cites, re-derives the verdict against
the run card's decision rules using the frozen seed-42 random-
carve artefact as the comparator, and pins the paper text at
every integration site (full edition, trimmed edition,
conclusion, appendix version row, run card).

The verdict (run card rule 1, "divergence disappears"): CONFIRMED.
Under the whole-region carve FedAvg's validation ROC-AUC
\emph{declines} (0.980 -> 0.974) in parallel with the test decline
(0.975 -> 0.960); the random carve's rising monitor (0.974 ->
0.977) was region-sharing leakage; partition-5 recency is
rejected. Rule 2 also passes: the shipped checkpoint (round 35)
is no longer the weakest test round of its own trajectory, and
the validation-ROC sensitivity rule now selects round 5 —
simultaneously the best test round of the entire trajectory. The
card's "shipped numbers likely improve" guess did NOT
materialise (both deltas negative, ~0.001): the leak inflated the
monitor's trend, not the validation-based selections.

Landing note (recorded for the provenance trail): the artefact
was pasted to the analysis session before the owner push, and the
pasted reconstruction differed from the pushed bytes in exactly
four deep-decimal digits (FedProx rounds 20/25 recall@FPR
transpositions) and in line endings — the push is the artefact of
record and this module pins its CRLF/LF-stable hash. The
selected-checkpoint ROC (0.9617175...) and the trajectory r35
entry (0.961713) differ by 4.6e-6 IN THE OWNER'S OWN FILE — two
separate post-hoc evaluation passes, not corruption; the paper
cites the selected-block value.

Layers (all torch-free; pure file/JSON parsing):
  1. artefact byte-pin + protocol facts (a missing file is a
     FAILURE, not a SKIP — the artefact is committed);
  2. the validation-split disclosure block (whole-region carve
     facts, prevalence, disjointness, pool identity);
  3. the verdict re-derived from the artefact + the frozen
     seed-42 random-carve comparator (the run card's decision
     rules, the monitor-pathology pins, the stability pins);
  4. paper tex pins at every v4.14 integration site (full +
     trimmed editions, conclusion, appendix, run card), including
     the queued-list removals and the retired recipes.

Run:  python tests/test_region_disjoint_verdict.py  (or via the
battery)
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARTEFACT_RD = os.path.join(ROOT, "outputs",
                           "raw_nobn_region_disjoint.json")
ARTEFACT_S42 = os.path.join(ROOT, "outputs", "raw_nobn_eval.json")

# The owner-pushed form: CRLF line endings, 25,722 bytes, 742
# lines, no trailing newline after the final brace. The hash is
# pinned CRLF/LF-stable (the seed-43 convention), so a future
# .gitattributes renormalisation cannot silently break the pin.
JSON_SHA256_LF = (
    "86e8080f2cd8455df4df8889494ce36db2c811892baec96041075f36e6c"
    "3ae1b"
)
JSON_BYTES_CRLF = 25722        # the owner-pushed Windows form
JSON_ELAPSED_S = 3118.538515806198   # the console's 3,119 s

PAPER_RESULTS = os.path.join(ROOT, "paper", "sections",
                             "sec_results.tex")
PAPER_LIMITS = os.path.join(ROOT, "paper", "sections",
                            "sec_limitations.tex")
PAPER_CONCL = os.path.join(ROOT, "paper", "sections",
                           "sec_conclusion.tex")
PAPER_APPX = os.path.join(ROOT, "paper", "sections",
                          "sec_appendix.tex")
TRIM_RESULTS = os.path.join(ROOT, "paper_trimmed", "sections",
                            "sec_results.tex")
TRIM_DISCUSS = os.path.join(ROOT, "paper_trimmed", "sections",
                            "sec_discussion.tex")
RUN_CARD = os.path.join(ROOT, "docs", "RUN_CARD_v4.12.md")

FAILED = []
N_CHECKS = 0


def check(name, cond, note=""):
    global N_CHECKS
    N_CHECKS += 1
    line = ("[PASS] " if cond else "[FAIL] ") + name
    if note and not cond:
        line += f"  {note}"
    print(line)
    if not cond:
        FAILED.append(name)


def _read(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _hist(arm):
    return {h["round"]: h for h in arm["history"]}


def _traj(arm):
    return {int(k): v for k, v in arm["test_trajectory"].items()}


# ── layer 1: the artefact — byte pin + protocol facts ───────────

def artefact_layer():
    if not os.path.exists(ARTEFACT_RD):
        check("the region-disjoint JSON is committed (the owner "
              "push, 17af7e4)", False, "(file absent)")
        return None
    raw = open(ARTEFACT_RD, "rb").read()
    lf = raw.replace(b"\r\n", b"\n")
    digest = hashlib.sha256(lf).hexdigest()
    check("the region-disjoint JSON sha256 byte-pin (CRLF/LF-"
          "stable; the owner-pushed artefact)", digest ==
          JSON_SHA256_LF, f"(got {digest[:16]}…)")
    crlf_len = (len(raw) if raw.count(b"\r\n")
                else len(raw) + lf.count(b"\n"))
    check(f"the pushed Windows CRLF form is exactly "
          f"{JSON_BYTES_CRLF:,} bytes (742 lines, no trailing "
          f"newline)", crlf_len == JSON_BYTES_CRLF)
    with open(ARTEFACT_RD, encoding="utf-8") as f:
        d = json.load(f)
    check("elapsed_s equals the console's 3,119 s run",
          abs(float(d.get("elapsed_s", -1)) - JSON_ELAPSED_S) <
          1e-9)
    check("runner identity: experiments/run_raw_nobn.py",
          d.get("runner") == "experiments/run_raw_nobn.py")
    for algo, mu in (("fedavg", 0.0), ("fedprox", 0.01)):
        blk = d.get(algo, {})
        p = blk.get("protocol", {})
        check(f"{algo}: seed 42, mu {mu}, 28,929 params, 6 "
              f"clients, alpha 1.0, prior_shift calibration",
              p.get("seed") == 42 and p.get("mu") == mu and
              p.get("n_params") == 28929 and
              p.get("clients") == 6 and p.get("alpha") == 1.0 and
              p.get("calibration") == "prior_shift")
        check(f"{algo}: the validation marker names the "
              f"region-disjoint carve (v4.12 item 2)",
              str(p.get("validation", "")).startswith(
                  "region-disjoint (whole NOAA active regions"))
        check(f"{algo}: the state namespace is nobnrd_*",
              "nobnrd_" in str(p.get("note", "")))
    check("reference_columns is the disclosed empty block (the "
          "run reads them from committed artefacts; the "
          "comparisons here re-derive them instead)",
          d.get("reference_columns") == {})
    return d


# ── layer 2: the validation-split disclosure block ──────────────

def split_layer(d):
    vs = d.get("validation_split", {})
    check("split mode: whole NOAA active regions, disjoint",
          "region-disjoint" in str(vs.get("mode", "")) and
          "whole NOAA active regions" in str(vs.get("mode", "")))
    check("region source: the parse metadata `ar` column with "
          "full pool coverage (the v4.12.1 errata fix, exercised "
          "for real by this run)",
          "parse metadata" in str(vs.get("region_source", "")) and
          "full pool coverage" in str(vs.get("region_source", "")))
    check("deterministic whole-region assignment: seed-42 region "
          "permutation, 390 of 2,447 regions held out",
          vs.get("seed_region_permutation") == 42 and
          vs.get("n_regions_total") == 2447 and
          vs.get("n_regions_val") == 390)
    sample = vs.get("val_regions_sample", [])
    check("val region sample: 20 ids disclosed, opening "
          "[2, 14, 29, 47, 78]",
          len(sample) == 20 and sample[:5] == [2, 14, 29, 47, 78])
    check("row accounting: 214,861 train / 40,959 val / the "
          "random carve's 40,932 target rows",
          vs.get("train_rows") == 214861 and
          vs.get("val_rows") == 40959 and
          vs.get("val_target_rows_random_carve") == 40932)
    check("pool identity: the carve re-partitions the frozen "
          "255,820-row pool (train+val totals equal the frozen "
          "split's 214,888+40,932)",
          vs.get("train_rows") + vs.get("val_rows") == 255820)
    check("prevalences disclosed: val 1.4893% / random carve "
          "2.0497% / train 2.1567%",
          abs(float(vs.get("val_pos", -1)) - 0.0148929413408041) <
          1e-12 and
          abs(float(vs.get("random_carve_val_pos", -1)) -
              0.020497409626841545) < 1e-12 and
          abs(float(vs.get("train_pos", -1)) -
              0.021567432209849358) < 1e-12)
    check("disjointness statement: verified, no region id on "
          "both sides", "no region id on both sides" in
          str(vs.get("disjointness", "")))


# ── layer 3: the verdict (decision rules re-derived) ────────────

def verdict_layer(d, s42):
    fa, fp = d["fedavg"], d["fedprox"]
    fa42, fp42 = s42["fedavg"], s42["fedprox"]
    h, t = _hist(fa), _traj(fa)
    h42, t42 = _hist(fa42), _traj(fa42)

    # rule 1 — the divergence disappears
    check("RULE 1 (the divergence disappears): under the "
          "region-disjoint carve FedAvg's validation ROC-AUC "
          "DECLINES (0.9803 -> 0.9740) while the frozen random "
          "carve's monitor ROSE (0.9736 -> 0.9773)",
          h[5]["roc_auc"] > h[50]["roc_auc"] and
          abs(h[5]["roc_auc"] - 0.9803) < 5e-4 and
          h42[5]["roc_auc"] < h42[50]["roc_auc"] and
          abs(h42[5]["roc_auc"] - 0.9736) < 5e-4 and
          abs(h42[50]["roc_auc"] - 0.9773) < 5e-4)
    check("rule 1, test side: the test ROC-AUC declines in BOTH "
          "carves (0.9746 -> 0.9603 here; 0.9744 -> 0.9626 "
          "random) — the decline is carve-invariant, hence real "
          "client drift, not a validation artefact",
          t[5]["roc_auc"] > t[50]["roc_auc"] and
          t42[5]["roc_auc"] > t42[50]["roc_auc"] and
          abs(t[5]["roc_auc"] - 0.9746) < 5e-4 and
          abs(t[50]["roc_auc"] - 0.9603) < 5e-4)
    check("the seed-43 sharpening is covered by the same "
          "resolution: the random-carve monitor's rise is the "
          "effect the whole-region carve removes",
          h42[5]["roc_auc"] < h42[50]["roc_auc"])

    # rule 2 — the shipped checkpoint is no longer the weakest
    tmin_rd = min(v["roc_auc"] for v in t.values())
    tmin42 = min(v["roc_auc"] for v in t42.values())
    shipped_rd = fa["selected_checkpoint_test"]["test"]["roc_auc"]
    shipped42 = fa42["selected_checkpoint_test"]["test"]["roc_auc"]
    check("RULE 2 (no longer the weakest round): the shipped r35 "
          "(0.9617) sits 0.0014 ABOVE its own trajectory minimum "
          "(r50, 0.9603) and beats rounds 40-50",
          shipped_rd > tmin_rd + 0.001 and
          t[35]["roc_auc"] > t[40]["roc_auc"] and
          t[35]["roc_auc"] > t[50]["roc_auc"] and
          abs(shipped_rd - 0.9617) < 5e-4)
    check("the random carve's contrast: its shipped r50 was the "
          "trajectory minimum (within the 9e-6 r45/r50 tie)",
          shipped42 <= tmin42 + 1e-5)
    check("the honest-monitor sensitivity row: the validation-ROC "
          "rule now selects round 5, simultaneously the best test "
          "round of the entire trajectory (0.9746/0.4497) — under "
          "the random carve it selected round 40, mid-decay",
          fa["sensitivity_val_roc_selected_test"]["round"] == 5 and
          max(t, key=lambda r: t[r]["roc_auc"]) == 5 and
          abs(fa["sensitivity_val_roc_selected_test"]["test"]
              ["roc_auc"] - 0.9746) < 5e-4 and
          abs(fa["sensitivity_val_roc_selected_test"]["test"]
              ["pr_auc"] - 0.4497) < 5e-4 and
          fa42["sensitivity_val_roc_selected_test"]["round"] == 40)
    check("FedProx: both selection rules agree at round 50 "
          "(shipped == sensitivity)",
          fp["best_round"] == 50 and
          fp["sensitivity_val_roc_selected_test"]["round"] == 50)

    # FedProx — no divergence in either carve
    hp, tp = _hist(fp), _traj(fp)
    hp42, tp42 = _hist(fp42), _traj(fp42)
    check("FedProx shows no divergence in EITHER carve (val and "
          "test both climb: 0.9552 -> 0.9765 val / 0.9410 -> "
          "0.9743 test here; 0.930 -> 0.969 / 0.931 -> 0.975 "
          "random)",
          hp[5]["roc_auc"] < hp[50]["roc_auc"] and
          tp[5]["roc_auc"] < tp[50]["roc_auc"] and
          hp42[5]["roc_auc"] < hp42[50]["roc_auc"] and
          tp42[5]["roc_auc"] < tp42[50]["roc_auc"])

    # the shipped numbers — stability and the honest non-improvement
    fp_ship = fp["selected_checkpoint_test"]["test"]["roc_auc"]
    fp42_ship = fp42["selected_checkpoint_test"]["test"]["roc_auc"]
    check("level stability across carves: shipped selections move "
          "-0.0009 (FedAvg) and -0.0012 (FedProx) ROC-AUC — "
          "inside the ±0.005 rule with margin",
          abs(shipped_rd - shipped42) <= 0.002 and
          abs(fp_ship - fp42_ship) <= 0.002 and
          shipped_rd < shipped42 and fp_ship < fp42_ship)
    check("the card's 'likely improve' guess did NOT materialise "
          "(both deltas negative — pinned so the paper cannot "
          "claim an improvement it did not get; the leak "
          "inflated the monitor's trend, not the selections)",
          shipped_rd - shipped42 < 0 and fp_ship - fp42_ship < 0)

    # the monitor pathology — the population-dependent layer
    check("the fifth fixed-threshold-monitor confirmation: "
          "FedProx's validation F1 is exactly 0.0000 through "
          "round 25, then climbs to 0.5741 — against the random "
          "carve's 0.0922 at the SAME seed and protocol, while "
          "its test ROC moves 0.0012",
          all(float(hp[r]["f1"]) == 0.0 for r in (5, 10, 15, 20, 25))
          and abs(float(fp["best_val_f1"]) - 0.574074) < 5e-4 and
          abs(float(fp42["best_val_f1"]) - 0.092238) < 5e-4)
    check("the rank layer is the stable one: FedProx's test "
          "ROC-AUC moves 0.0012 across carves (0.9755 -> 0.9743)",
          abs(fp42_ship - 0.9755) < 5e-4 and
          abs(fp_ship - 0.9743) < 5e-4)

    # the residuals the paper discloses
    check("residual 1 — the stable level offset: r50 validation "
          "ROC 0.9740 against test 0.9603 (~0.014), a genuine "
          "held-out-region difficulty difference",
          abs((h[50]["roc_auc"] - t[50]["roc_auc"]) - 0.0137) <
          0.001)
    check("residual 2 — the PR-space divergence persists: val "
          "PR-AUC rises 0.6753 -> 0.7154 while test PR-AUC falls "
          "0.4497 -> 0.3575 (a prevalence- and population-"
          "sensitive statistic — why ROC-AUC is the diagnostic "
          "axis)",
          h[5]["pr_auc"] < h[50]["pr_auc"] and
          t[5]["pr_auc"] > t[50]["pr_auc"] and
          abs(h[50]["pr_auc"] - 0.7154) < 5e-4 and
          abs(t[50]["pr_auc"] - 0.3575) < 5e-4)
    check("the test set is untouched by the carve: the test "
          "prevalence equals the frozen artefact's bit-for-bit "
          "(0.013136071120546673)",
          fa["selected_checkpoint_test"]["test"]["prevalence"] ==
          fa42["selected_checkpoint_test"]["test"]["prevalence"])
    check("the selected-block vs trajectory 4.6e-6 ROC gap at r35 "
          "is a property of the owner's own file (two post-hoc "
          "evaluation passes), not corruption",
          abs(shipped_rd - t[35]["roc_auc"]) < 1e-5 and
          shipped_rd != t[35]["roc_auc"])


# ── layer 4: paper tex pins (every v4.14 integration site) ───────

def paper_layer(d, s42):
    tex = _read(PAPER_RESULTS)
    for phrase in (
        "The region-disjoint re-run closes the diagnosis.",
        "390 of 2{,}447",
        "40{,}959 validation rows",
        "at 1.49\\% prevalence against the random carve's 2.05\\%",
        "(0.980 to 0.974",
        "(0.975 to 0.960)",
        "0.978 to 0.989 at seed 43,",
        "FedAvg 0.962/0.367 at",
        "round 35, FedProx 0.974/0.358 at round 50",
        "$-0.001$ and",
        "(0.975/0.450) ---",
        "0.974$\\to$0.960 under the",
        "0.974$\\to$0.963 under the random",
        "$\\sim$0.014 validation--test ROC-AUC",
        "0.675$\\to$0.715 while test",
        "0.450$\\to$0.357), the expected",
        "validation F1 0.574 against the random carve's 0.092",
        "fifth independent confirmation",
        "moves by 0.001. The artefact of record",
        "raw\\_nobn\\_region\\_disjoint.json",
        "214{,}861-row",
        "3{,}119\\,s on",
        "remained, both testable with the committed "
        "region-disjoint split",
        "has since executed and separates them decisively",
    ):
        check(f"sec_results.tex carries the verdict text: "
              f"{phrase!r}", phrase in tex, "(absent)")
    check("sec_results.tex: the seed-43 JSON sentence updated to "
          "the landed state (was 'awaiting the owner push')",
          "has since landed" in tex and
          "will be frozen on arrival" not in tex)

    lim = _read(PAPER_LIMITS)
    for phrase in (
        "the fifth owner-GPU batch",
        "3{,}119\\,s, both arms; verdict:",
        "the divergence disappears --- the random carve's rising "
        "validation",
        "landed with the owner push and sha256-frozen,",
    ):
        check(f"sec_limitations.tex: {phrase!r}", phrase in lim,
              "(absent)")
    check("sec_limitations.tex: the region-disjoint re-run "
          "REMOVED from the queued list (an executed item cannot "
          "stay queued; the parse-metadata blocker sentence gone)",
          "both no-BatchNorm raw arms (the divergence diagnosis" not
          in lim and
          "blocked only on regenerating the raw parse metadata" not
          in lim)

    con = _read(PAPER_CONCL)
    check("sec_conclusion.tex: the re-run programme list now "
          "TWELVE items with the region-disjoint re-run",
          "twelve items of the re-run" in con and
          "and the region-disjoint validation re-run---have" in con)
    check("sec_conclusion.tex: the parity clause gains the "
          "validation-divergence closure",
          "diagnosis the region-disjoint re-run closes" in con)

    app = _read(PAPER_APPX)
    check("sec_appendix.tex: the v4.14 version-history row with "
          "the verdict and the sensitivity-row fact",
          "v4.14 &" in app and
          "battery version v4.14" in app and
          "0.092$\\to$0.574" in app)
    check("sec_appendix.tex: the caption range extended to "
          "v2.1--v4.14", "v2.1--v4.14" in app)

    tr = _read(TRIM_RESULTS)
    for phrase in (
        "\\paragraph{The validation divergence, resolved.}",
        "(0.980 to",
        "0.962/0.367,",
        "FedProx 0.974/0.358),",
        "(0.975/0.450).",
        "$\\sim$0.014 validation--test offset",
    ):
        check(f"trimmed edition results: {phrase!r}",
              phrase in tr, "(absent)")
    check("trimmed edition: zero process language maintained — "
          "no 'first remaining compute item', no 'prepared in the "
          "released code' pending clause",
          "first remaining compute" not in tr and
          "prepared in the released code and is the first" not in tr)

    td = _read(TRIM_DISCUSS)
    check("trimmed edition discussion: the region-sharing caveat "
          "discharged into the resolved audit clause",
          "region-disjoint re-run audits" in td and
          "monitor declines with" in td and
          "the test trajectory while the selections hold" in td)
    check("trimmed edition discussion: the pending 1.7-ks recipe "
          "clause gone", "1.7\\,kiloseconds" not in td and
          "is likewise prepared in the released" not in td)

    rc = _read(RUN_CARD)
    check("run card rev. v4.14: item 2 EXECUTED and integrated "
          "with the verdict; item 3's DONE state, the $parts "
          "recipe record, and the no-owner-action statement all "
          "intact",
          "rev. v4.14" in rc and
          "the divergence disappears" in rc and
          "parity holds" in rc and
          "JSON push" in rc and
          "DONE (v4.13.1" in rc and
          "No owner action remains" in rc and
          "$parts" in rc)
    check("run card: the item-2 push recipe RETIRED (the artefact "
          "is landed; a stale recipe is a false open item)",
          "git add outputs\\raw_nobn_region_disjoint.json" not in rc)


def main():
    print("layer 1 — the region-disjoint artefact (byte pin + "
          "protocol facts):")
    d = artefact_layer()
    s42 = None
    if d is None:
        print(f"RESULT: {N_CHECKS - len(FAILED)} passed, "
              f"{len(FAILED)} failed")
        sys.exit(1)
    print("layer 2 — the validation-split disclosure block:")
    split_layer(d)
    print("layer 3 — the verdict re-derived against the frozen "
          "seed-42 random-carve artefact:")
    if os.path.exists(ARTEFACT_S42):
        with open(ARTEFACT_S42, encoding="utf-8") as f:
            s42 = json.load(f)
        verdict_layer(d, s42)
    else:
        check("frozen seed-42 artefact present (the comparator)",
              False, "(file absent)")
    print("layer 4 — paper integration pins (full + trimmed "
          "editions, conclusion, appendix, run card):")
    if s42 is not None:
        paper_layer(d, s42)
    print(f"RESULT: {N_CHECKS - len(FAILED)} passed, "
          f"{len(FAILED)} failed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
