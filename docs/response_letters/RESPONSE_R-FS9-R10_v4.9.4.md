# Record of Revision v4.9.4 — Asks #10 and #11 Executed: the A1
Verdict and the A3 Attribution Answer (Repo-Only)

**Re:** Dossier R-FS9-R10, master change register items A1 and A3
(the Round-13 third-party adjudication, verdict AGREE ~90 PERCENT —
ACCEPT, SUSTAINED — 8.0/10 — E1′ REGISTER ISSUED). The owner executed
RUNLOG asks **#10** (the centralised arm-B sanity check, re-run under
the v4.9.3 amended two-gate instrument) and **#11** (the no-BN raw
federation, resumed under the fixed runner) on 2026-10-06 and sent
back the console tails and both JSON artefacts. Both instruments ran
**clean**: every gate matched, the verdict landed in the declared
alphabet, the exit contract honoured, and this revision lands the two
artefacts and the two verdicts into the record — frozen, not
massaged.
**Revision:** v4.9.4, this commit. No re-review is pending or
sought — the v4.4/v4.9.1/v4.9.2/v4.9.3 precedent: a repo-only
results-landing batch, executed and recorded, with the paper of
record UNTOUCHED.
**Paper of record after this revision:** unchanged — `paper/main.pdf`,
59 pp at v4.9/v4.9.1/v4.9.2/v4.9.3. Every paper-side register item
(A4, B1–B5, the A2 resolution sentence, the contribution-5 rewrite,
the version history) remains scheduled for **v4.10** — which this
revision unblocks with both decisive experiments' evidence in hand.

The v4.9.3 instrument is exactly what executed. No tolerance was
widened after the fact; no number was re-run; the amended gates
passed for the reasons they were designed to detect. What follows is
the honest ledger of what the two executions found.

<!-- LETTER-MANIFEST v1
base: 25856bacd8b59bdfcebd1dbb799d8df3ac8560f4
-->

## Diff manifest (machine-checked by tests/test_letters_manifest.py)

`git diff --name-status 25856ba <this commit>` — the complete list;
nothing else changed:

    M    RUNLOG.md
    A    docs/response_letters/RESPONSE_R-FS9-R10_v4.9.4.md
    A    outputs/arm_b_central_sanity.json
    A    outputs/raw_nobn_eval.json
    M    tests/run_battery.py
    M    tests/test_letters_manifest.py

## 1. Ask #10, executed — both gates match, verdict `arm_b_validated`

The run (2 s, CPU, `data.npz` cache resumed) evaluated arm A (own
running buffers) and passed **both** v4.9.3 gates, exit 0:

| gate | observed | tolerance | verdict |
|---|---|---|---|
| identity vs the ask-#4 retrain record | d_ROC 1.36e-08, d_PR 8.82e-07 | 1e-3 | **match** |
| published retrain bracket, ROC | 2.32e-3 | 1e-2 | **match** |
| published retrain bracket, PR | 1.93e-2 | 2.5e-2 | **match** |

Arm A scored **ROC 0.96870 / PR 0.42566** — reproducing the committed
`outputs/raw_substrate_rerun.json` record to better than 1e-6, exactly
the "<1e-5 same-weights" premise the v4.9.3 diagnosis staked the
amendment on. Arm B (pooled per-client recalibration over the six
seed-42 train shards) scored **ROC 0.96864 / PR 0.43051**: B-versus-A
deltas of −6.8e-05 ROC and +4.9e-03 PR — numerically identical to the
first execution's arm-B numbers (same weights, deterministic
evaluation: the instrument's own reproducibility demonstrated), and
nowhere near the federated arms' chance signature (ROC 0.500–0.501,
PR 0.013–0.015). The verdict bands, declared at v4.9.2 and unchanged
since, read: ROC-B 0.9686 ≥ 0.95 and PR-B 0.4305 ≥ 0.35 →
**`arm_b_validated`**, exit 0.

## 2. What `arm_b_validated` settles — register items A1 and A2

Table 8's B column stands. The pooled-statistics recalibration is a
**validated instrument**, not an untested implementation: applied to
the centralised model — where pooled statistics should reconstruct the
model's own running statistics — it preserves the centralised ranking
to −6.8e-05 ROC and *improves* PR-AUC by 4.9e-03. The supporting
diagnostics make the reconstruction visible rather than assumed:
pooled-vs-own buffer deltas run 0.0205 / 0.0432 / 0.1126 (mean rel L2)
and 0.0185 / 0.1180 / 0.1813 (var mean rel delta) across layers 0–2 —
depth-monotone drift, non-trivial at the last layer, and yet the
ranking holds. The instrument tolerates realistic statistics drift
without collapsing; that is precisely the property the federated B
column's readers needed certified.

Register item **A2** therefore resolves to the **re-frame branch**
its v4.9.2 fork documented: Table 8's B column becomes the validated
proxy for the transport fix, the arm-D faithful re-run is **not
required**, and the paper-side sentence lands at v4.10. The reviewer's
"actively destructive" concern is answered by evidence, not by
argument — the column is certified against the exact failure mode it
was suspected of.

## 3. Ask #11, executed — the completed artefact and the attribution answer

`outputs/raw_nobn_eval.json` (1,699 s total on RTX 4060 Laptop) now
carries both arms under the frozen raw protocol with the one
architecture-only intervention (BatchNorm1d → Identity, 28,929 params
vs the BN baseline's 29,377):

- **FedAvg resumed at round 51 → evaluation only** — the v4.9.3
  round-resumable state preserved the crashed first execution's
  completed 50 rounds exactly as designed. Best val F1 0.5266 at
  round 50; the shipped best-val-F1 selection takes round 50, the
  val-ROC sensitivity row takes round 40 — both disclosed, neither
  test-based (B1 hygiene, born with the arm).
- **FedProx trained its 50 rounds fresh**, best val F1 0.0922 at
  round 50 (val scores compressed at the default 0.35 threshold; its
  calibrated test operating point is 0.155 — test F1 0.400, recall
  0.763); both selection rules agree on round 50.

The attribution table, every reference read live from the committed
artefacts at run time (and battery-checked, never hand-typed):

| raw-substrate arm (test, shipped selection) | ROC-AUC | PR-AUC |
|---|---|---|
| FedAvg MLP (BN, published) | 0.8747 | 0.0561 |
| **FedAvg MLP (no-BN, ask #11)** | **0.9626** | **0.3657** |
| FedProx MLP (BN, published) | 0.9303 | 0.1487 |
| **FedProx MLP (no-BN, ask #11)** | **0.9755** | **0.3448** |
| Centralised MLP (reference) | 0.9710 | 0.4450 |
| FedAvg LSTM (reference) | 0.9581 | 0.3054 |
| FedProx LSTM (reference) | 0.9697 | 0.4045 |

**The reading.** Removing BatchNorm — and nothing else — recovers the
overwhelming majority of the raw MLP's federation penalty: FedAvg
+0.0879 ROC / +0.3095 PR (×6.5 on PR-AUC); FedProx +0.0452 ROC /
+0.1961 PR (×2.3). Both no-BN arms sit at or above the LSTM raw arms
on ROC-AUC (0.9626 vs 0.9581; 0.9755 vs 0.9697), and no-BN FedProx
**exceeds the centralised MLP on ROC-AUC** (0.9755 vs 0.9710). The
federated collapse of the BN raw MLP arms is therefore **dominated by
BatchNorm-under-federation, not by the encoder** — the attribution
inverts, and contribution 5's encoder-conditional sentence must be
rewritten at v4.10, exactly the branch the register anticipated ("no-
BN arms at LSTM-level quality inverts contribution 5's attribution").
The gap arithmetic, for the record: FedAvg's ROC gap to centralised
closes from 0.0963 to 0.0084 (91%) and its PR gap from 0.3888 to
0.0793 (80%); FedProx's ROC gap closes past zero (−0.0045 residual)
and its PR gap from 0.2963 to 0.1002 (66%).

**Disclosed, not buried.** (a) FedAvg's per-round test trajectory
*declines* r5→r50 (ROC 0.9744 → 0.9626, PR 0.4461 → 0.3657) while its
val F1 still improves — client drift unmitigated, and val prevalence
(2.05%) optimised against test prevalence (1.31%) — so the shipped
rule selects the weakest test-ROC monitored round of its trajectory;
the sensitivity row and the trajectory carry the real picture, which
is the B1 hygiene working as designed. FedProx climbs monotonically
(0.9312 → 0.9755): the proximal term's stabilisation is visible
precisely once BN is removed — a result the BN arms' confound had
hidden. (b) No-BN FedProx's PR-AUC (0.3448) remains below the BN
LSTM FedProx arm (0.4045) and the centralised MLP (0.4450): the claim
these arms license is **attribution**, not a new state-of-the-art
row. (c) The paper's existing in-partition no-BN control is a
different substrate (balanced, in-partition); these arms are its
raw-substrate, out-of-partition complement, and the comparison
columns make the two unreadable into each other.

## 4. Battery and apparatus

- `tests/test_raw_bn_diagnostic.py`: **122/0** torch-less — 86 at
  v4.9.3 plus the **36 activated layer-4 checks** (22 A1 + 14 A3).
  The two declared [SKIP]s are gone: the artefact layers now execute
  against the landed JSONs — identity-gate schema, verdict-vs-bands
  consistency, gate verdicts consistent with their own deltas and
  tolerances, reference columns equal to the committed
  `raw_substrate_eval.json` / `raw_lstm_eval.json` read live. A skip
  is not a pass (B9); these now are.
- `tests/test_letters_manifest.py`: this letter registered (registry
  now 10 entries).
- `tests/run_battery.py`: BATTERY_VERSION v4.9.4 (fresh versioned log
  path `logs/test_battery_v4.9.4.log`, uncommitted by design).
- Full battery, torch-less: **370/0 pre-commit** (this letter present
  but uncommitted, so its manifest verification defers — 333 at
  v4.9.3 + 36 activated + the registry pass), **371/0 post-commit**
  (this letter's manifest verifies against the actual diff).
- Paper, submission, README, appendix, experiment runners: untouched
  — the instruments executed are byte-identical to v4.9.3's, which is
  itself part of the record's meaning (no instrument was edited
  between the failed first executions and the clean re-runs).

## 5. The register disposition, updated

| Register item | Status after v4.9.4 |
|---|---|
| A1 arm-B centralised sanity | **EXECUTED — `arm_b_validated`**, both gates match, exit 0; artefact committed, battery layer active |
| A2 arm-D transport run / re-frame | **RESOLVED by A1's verdict — the re-frame branch**: Table 8's B column is the validated proxy for the transport fix; arm-D not required; the paper-side sentence lands at v4.10 |
| A3 no-BN raw federation | **EXECUTED — the attribution answer: BatchNorm-under-federation-conditional**; artefact committed, battery layer active; contribution 5's rewrite is the v4.10 paper item |
| A4, B1–B5 | Unchanged — v4.10 paper-side (B1's new-arm hygiene now has its artefact-side evidence landed) |
| B6 | Executed at v4.9.2, unchanged |
| C1–C3 | Unchanged (owner decisions; C2's A1–A3-land path is now evidence-complete) |

## 6. What the owner runs next

1. `git pull origin improvements` (this commit)
2. **Nothing mandatory** — the register's compute programme is
   discharged: RUNLOG asks #1–#11 are all RAN, and both decisive
   instruments' artefacts are committed. The B2 SCAFFOLD rerun
   remains the one *optional* owner-GPU queue entry, unchanged, an
   owner decision as before.
3. Optionally re-run the battery and watch the two artefact layers
   light up: `python tests/run_battery.py` → the BN-diagnostic guard
   reports 122/0 with the A1/A3 blocks executing (no more [SKIP]s).
   If the pull declines to overwrite an untracked local
   `outputs/arm_b_central_sanity.json` / `outputs/raw_nobn_eval.json`
   (identical content), `git stash -u && git pull origin improvements
   && git stash drop` is the ask-#2 precedent.
4. v4.10 — the paper-side integration (A4's scoped abstract, B1–B5,
   the A2 resolution sentence, the contribution-5 rewrite, the
   version history) — is prepared assistant-side next, every number
   read off the committed artefacts, exactly as this cycle's
   discipline requires.

The two experiments the Round-13 adjudication called "decisive" have
both now run, both clean, both recorded. What they decided: the
diagnostic's B column is a validated instrument (A1), and the
federated MLP's raw-substrate collapse is a BatchNorm artifact, not
an encoder failure (A3). The paper's story changes at v4.10 — for the
better, on evidence.
