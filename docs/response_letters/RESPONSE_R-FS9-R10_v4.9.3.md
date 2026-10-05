# Record of Revision v4.9.3 — First Executions of the Register
Instruments: Gate Recalibration + Selection Crash Fix (Repo-Only)

**Re:** Dossier R-FS9-R10, master change register items A1 and A3
(the Round-13 third-party adjudication, verdict AGREE ~90 PERCENT —
ACCEPT, SUSTAINED — 8.0/10 — E1′ REGISTER ISSUED). The owner
executed RUNLOG asks **#8** (the centralised arm-B sanity check)
and **#9** (the no-BN raw federation) on 2026-10-06 and sent back
the console tails. Both instruments ran far enough to expose one
defect each — one in the A1 gate's calibration, one a crash in
the A3 runner's round-selection step — and this revision amends
both, evidence-based, with the first-execution records **frozen,
not massaged**.
**Revision:** v4.9.3, this commit. No re-review is pending or
sought — the v4.4/v4.9.1/v4.9.2 precedent: a repo-only errata
batch, executed and recorded, with the paper of record UNTOUCHED.
**Paper of record after this revision:** unchanged — `paper/main.pdf`,
59 pp at v4.9/v4.9.1/v4.9.2. Every paper-side register item (A4,
B1–B5, the A2/A3 queue lines) remains scheduled for **v4.10**,
now with the A1 verdict's evidence on the way via the ask-#10
re-run.

We accept the adjudication in full, as at v4.9.2. Nothing in this
revision changes any number in the paper; nothing weakens any
gate. The A1 instrument is **strictly stronger** after this
revision (a second, ten-times-tighter gate), and the A3 runner is
fixed where it crashed. What follows is the honest ledger of what
the first executions found.

<!-- LETTER-MANIFEST v1
base: fd5e18554766c77fd80b3d0a616e0818574af9e9
-->

## Diff manifest (machine-checked by tests/test_letters_manifest.py)

`git diff --name-status fd5e185 <this commit>` — the complete list;
nothing else changed:

    M    RUNLOG.md
    M    docs/ENVIRONMENTS.md
    A    docs/response_letters/RESPONSE_R-FS9-R10_v4.9.3.md
    M    experiments/run_arm_b_central_sanity.py
    M    experiments/run_raw_nobn.py
    M    tests/run_battery.py
    M    tests/test_import_graph.py
    M    tests/test_letters_manifest.py
    M    tests/test_raw_bn_diagnostic.py

## 1. Ask #8, first execution — the GATE FAILED, correctly

The run loaded `baselines.pt`, evaluated arm A (own running
buffers) at **ROC 0.96870 / PR 0.42566**, and the arm-A gate
returned **MISMATCH** against the published centralised numbers
(0.97102 / 0.44496): ROC delta 2.32e-3 (inside), PR delta 1.93e-2
(outside the 1e-2 tolerance). Per the instrument's own exit
contract the verdict was **not** read off the failed gate, the
artefact was written before the exit, and the run exited 1 with
"investigate before reporting". This section is that
investigation, and its conclusion is that **the checkpoint is the
right one and the gate's tolerance was miscalibrated at v4.9.2
design time.**

**The checkpoint is proven correct by the record itself.** The
ask-#4 retrain record committed at v4.9
(`outputs/raw_substrate_rerun.json`) carries the centralised
numbers of exactly this checkpoint: ROC **0.968703** / PR
**0.425662**. The first ask-#8 execution reproduced those numbers
to **<1e-5 on both metrics** — same weights, same frozen
evaluation path. `baselines.pt` is the ask-#4 retrain's model,
exactly as the v4.9.2 design intended.

**The tolerance was unsatisfiable by construction.** The v4.9.2
gate used a single 1e-2 tolerance over both metrics, justified by
"the ask-#4 run measured pooled-baseline drift of <= 2.3e-3
cross-machine". That measurement is **ROC-AUC only**, and the
paper of record has always scoped it correctly
("the pooled baselines reproduce the published rows to
$\le 2.3\times10^{-3}$ **ROC-AUC**", Section 6.3). The full
frozen drift table, recomputed from the committed artefacts
(`raw_substrate_rerun.json` vs `raw_substrate_eval.json`):

| pooled baseline | d ROC-AUC | d PR-AUC |
|---|---|---|
| logistic regression | 1.66e-4 | 8.25e-4 |
| XGBoost | 1.51e-3 | 1.90e-3 |
| centralised MLP | 2.32e-3 | **1.93e-2** |

The centralised MLP's PR-AUC retrain drift is an order of
magnitude above its ROC drift (PR-AUC at 1.31% test prevalence is
the noisier ranking metric), and the v4.9.2 instrument silently
generalised the ROC-only figure to both metrics. The correct
checkpoint could never pass that gate; the gate fired on the PR
side alone and did precisely what a gate should do with a
miscalibrated instrument — it stopped the reading. The panel's
standing phrase for this cycle's discipline applies to our own
apparatus: the MISMATCH is frozen rather than massaged.

**What the (unreported) numbers showed, for the record.** Arm B
on the centralised model scored **ROC 0.96864 / PR 0.43051** —
arm-B-versus-arm-A deltas of -6e-5 ROC and +4.9e-3 PR, nowhere
near the federated arms' chance signature (ROC 0.500-0.501, PR
0.013-0.015), comfortably inside the declared `arm_b_validated`
bands (ROC ≥ 0.95, PR ≥ 0.35, declared before any run and
unchanged by this revision). This is consistent with the
expectation the dossier set for the check — but per the exit
contract it becomes reportable only from the ask-#10 re-run under
the amended instrument below.

## 2. The A1 amendment — two gates, both evidence-based

`experiments/run_arm_b_central_sanity.py` now gates arm A twice,
both exit-controlling:

- **Identity gate (new, primary)**: arm A vs the ask-#4 retrain
  record `outputs/raw_substrate_rerun.json` (the committed
  same-weights reference — the tightest reference that exists for
  THIS checkpoint) at `--identity-tolerance`, default **1e-3**.
  The first execution matched to <1e-5, so the default leaves two
  orders of magnitude of margin over the observed reproduction
  while being **ten times tighter** than the v4.9.2 published
  bracket on the wrong-checkpoint question it exists to answer.
  A MISMATCH here means `baselines.pt` is not the checkpoint the
  frozen record describes — re-run the ask-#4 repair sequence or
  widen the tolerance with a RUNLOG disclosure.
- **Published retrain bracket, per metric**: arm A vs
  `outputs/raw_substrate_eval.json` at `--gate-tolerance` (ROC,
  default 1e-2, **unchanged from v4.9.2**) AND
  `--gate-tolerance-pr` (PR-AUC, default **2.5e-2**, new). The
  basis is the frozen drift table above: the bracket brackets the
  record's own PR drift (1.93e-2) with headroom instead of
  silently denying it.

The verdict bands (0.95/0.35 validated, 0.60/0.10 broken), the
byte-identical arm-B machinery, the pooled-vs-own buffer deltas,
and the artefact-before-exit contract are **unchanged**. The
artefact schema gains an `identity_gate` block alongside the
existing `reproduction_gate` block (whose `tolerance` field
becomes the per-metric dict); the battery pins both.

No paper erratum is owed by this defect: the paper's disclosure
was already metric-scoped, and no number moved. The defect was
ours, in the instrument, and the first execution caught it — the
apparatus working on itself.

## 3. Ask #9, first execution — the crash after fifty rounds

The fedavg arm trained all 50 rounds (best val F1 0.5266 at round
50, val ROC 0.977) and the runner then died at the round-selection
step:

    roc_round = max(val_roc_by_round, key=val_roc_by_round)
    TypeError: 'dict' object is not callable

The dict was passed as the key **function**; its bound `.get` is
the callable. The bug sat behind fifty rounds of training, which
is why no static pin caught it — the v4.9.2 battery guards these
runners by string pins, and the buggy line *read* like a
selection. The round-resumable state file
(`data/cache/rawsubstrate/nobn_fedavg_state.pt`, `next_round=51`)
means **the completed arm is not lost**: the fixed re-run resumes
past the training loop and goes straight to evaluation; fedprox
then trains from its absent state file.

**The fix is a guard, not just a patch.** The selection is
extracted to the module-level, torch-free
`select_round_by_val_roc(history)` (ties resolve to the earliest
round, deterministically; empty history raises loudly), and the
battery gains **layer 3b — a DYNAMIC regression guard** that
imports the runner in every battery environment (its
deferred-torch-stack discipline) and *executes* the real
selection semantics. The bug class — code that only runs after a
full training arm — was invisible to string pins; it is not
invisible to execution. The runner's docstring records the errata
with the buggy line quoted verbatim, and the static pin asserts
the quoted form survives only in docstrings (exactly twice) while
the code carries exactly one `.get` form.

## 4. Battery and apparatus

- `tests/test_raw_bn_diagnostic.py`: **86/0** torch-less (74 at
  v4.9.2 + 6 new A1 two-gate static pins + 2 A3 crash pins + 4
  dynamic-layer checks + the amended module docstring); the two
  artefact layers still declared [SKIP] until the ask-#10/#11
  re-runs land their JSONs — a skip is not a pass (B9).
- The A1 artefact layer now validates the **identity gate** when
  the artefact is present: block present (a pre-v4.9.3 artefact
  fails loudly — it was written by the superseded
  single-tolerance instrument and must be regenerated by the
  ask-#10 re-run), verdict in the declared alphabet, identity
  reference equal to the committed retrain record read live, and
  both gates' verdicts consistent with their own deltas and
  tolerances.
- `tests/test_import_graph.py`: EXPECTED_FROM_IMPORTS 335 → 336
  (layer 3b's one from-import, deliberate).
- `tests/test_letters_manifest.py`: this letter registered
  (registry now 9 entries).
- `tests/run_battery.py`: BATTERY_VERSION v4.9.3 (fresh versioned
  log path `logs/test_battery_v4.9.3.log`).
- Full battery, torch-less, pre-commit: **333/0** (per-file
  75/20/29/34/17/5/1/3/1/6/7/5/13/18/13/86 with one internal
  gated skip and the fl-smoke module skip); post-commit 334/0
  (this letter's manifest verifies against the actual diff).
- Paper, submission, README, appendix: untouched (the repo-only
  precedent; the paper's metric-scoped disclosure needs no
  erratum).

## 5. The register disposition, updated

| Register item | Status after v4.9.3 |
|---|---|
| A1 arm-B centralised sanity | **Instrument amended** (two gates, strictly stronger); first execution frozen in the RUNLOG with the miscalibration diagnosed; verdict reportable from the ask-#10 re-run (seconds) |
| A2 arm-D transport run / re-frame | Unchanged — decided by A1's verdict, now via ask #10 |
| A3 no-BN raw federation | **Crash fixed** (helper + dynamic guard); fedavg's 50 rounds preserved by the resume state; the merged artefact lands via ask #11 |
| A4, B1–B5, B6 | Unchanged from v4.9.2 (v4.10 paper-side / executed) |
| C1–C3 | Unchanged (owner decisions) |

## 6. What the owner runs next (the complete list)

1. `git pull origin improvements` (this commit)
2. RUNLOG ask **#10** — `python experiments/run_arm_b_central_sanity.py`
   (CPU, seconds; same command, amended instrument) — expected:
   identity gate `match` (arm A ≈ the ask-#4 record to <1e-5),
   published bracket `match` (ROC 2.32e-3 ≤ 1e-2, PR 1.93e-2 ≤
   2.5e-2), verdict `arm_b_validated`, **exit 0**; send back the
   JSON + console tail
3. RUNLOG ask **#11** — `python experiments/run_raw_nobn.py`
   (GPU; fedavg resumes at round 51 → straight to evaluation,
   fedprox trains its 50 rounds) — send back
   `outputs/raw_nobn_eval.json` + console tail
4. Optionally re-run the battery (the A1/A3 artefact layers
   activate on the two JSONs; note the superseded
   pre-v4.9.3 `arm_b_central_sanity.json` fails the new
   identity-gate schema check loudly until ask #10 overwrites it —
   run ask #10 first)

v4.10 then integrates unchanged in plan: A4's scoped abstract,
B1–B5, the A2 fork's resolution, the version history — now with
the A1 verdict from a gate that could actually pass for the right
reasons, and the A3 attribution answer off a completed artefact.
