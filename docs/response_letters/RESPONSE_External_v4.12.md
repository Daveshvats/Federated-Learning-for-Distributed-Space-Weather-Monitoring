# Response to the external re-review of v4.11 (v4.12)

**Reviewer**: the independent external re-review of the v4.11 state
(delivered 2026-10-06; five findings, "a strong read, not a full
audit").
**Manuscript under review**: the v4.11 build of *Federated
Solar-Flare Prediction on SWAN-SF: A Benchmark Audit of Provenance,
Leakage, and Evaluation Protocols* (63 pp) and its mirrored
submission edition.
**This revision**: v4.12, all five findings accepted, executed as
below. The owner's structural decision: **keep both editions** —
the full paper as the extended record (`paper/`, 65 pp after this
revision's edits), and a new **trimmed, publishable edition**
(`paper_trimmed/`, 12 pp) which is what `submission/` now builds.

We thank the reviewer. The review verified our controls against the
committed artefacts, retired nothing that was true, and found the
one statistical error (item 2) that genuinely needed fixing. Every
point below is answered with the executed change, not a promise.

---

## Item 1 — "FedProx above the centralised MLP on ROC-AUC" is not
supportable

**Agreed, and fixed at every site.** The gap (0.975 vs 0.971,
−0.004) is a single-seed quantity sitting inside the retrain
nondeterminism our own replication record documents for exactly
these arms (the BN-diagnostic gate deltas: +0.023 FedAvg, −0.002
FedProx, −0.005 SCAFFOLD). The claim is re-worded to **"matching
the centralised MLP on ROC-AUC within recorded retrain
nondeterminism"** at all sites: the abstract (and `ABSTRACT.md`),
contribution 3 of the introduction, Section 6.4's reading paragraph
(now with the gate-delta citation inline), the conclusion (twice),
and the limitations. Additionally, Section 6.4's boundary list
gains item (d): both no-BN arms are single-seed, the recovery
swings dwarf the noise band but the *level* claims sit within it —
which is exactly why item 3's replication leads the remaining
programme.

## Item 2 — the validation story doesn't hold together
(prevalence-invariance)

**Agreed: the explanation was wrong, and it is withdrawn.** The
v4.11 passage blamed the validation prevalence (2.05%) against the
test prevalence (1.31%) for the FedAvg no-BN divergence (validation
ROC-AUC rising 0.974→0.977 while test ROC-AUC falls 0.974→0.963).
As the reviewer states, ROC-AUC is a rank statistic and is
prevalence-invariant, so that mechanism cannot produce the
divergence. The rewritten passage now says so explicitly and states
the two live candidate mechanisms: **(a)** the random validation
carve shares active regions with the training shards — the same
leakage class our own provenance audit documents benchmark-wide,
which would inflate validation while test declines; **(b)** partition
5 is the temporally latest fold, so late-round specialisation to
partitions 1–4 registers as validation gain and test loss
simultaneously. The reviewer's observation that this also explains
why the shipped checkpoint is the weakest test round of its own
trajectory is now carried verbatim in the honest-record sentence.

**The re-run is committed, not queued on paper alone.**
`experiments/run_raw_nobn.py` gains `--region-disjoint`: the
training pool is rebuilt, the frozen random carve is re-derived and
verified by a label round-trip against the cached arrays (a wrong
reconstruction aborts loudly), every pool row is mapped to its NOAA
active region via the provenance metadata, and whole regions move
to validation until no region spans the boundary (deterministic
seed-42 region permutation, target = the random carve's size,
disjointness asserted). Round state is namespaced (`nobnrd_*`) so
the frozen v4.9.4 artefacts cannot be disturbed. The split function
is torch-free and smoke-tested on synthetic fixtures
(deterministic rerun, disjointness verified, split stats
disclosed). The owner-side recipe — two commands, ~1.7 ks of GPU
compute for both arms, plus decision rules for both outcomes — is
`docs/RUN_CARD_v4.12.md`. (This sandbox has no GPU and no dataset
cache; the run itself executes owner-side, exactly as every
previous compute item in this project, and its artefact will be
pinned on arrival per the established convention.)

## Item 3 — the no-BN control is one seed

**Agreed; the replication is committed with the same kit.**
`experiments/run_raw_nobn.py` gains `--seed N`: model
initialisation and the Dirichlet shard draw are reseeded, the
frozen random validation carve is kept identical (the same
replication semantics the sequence arms' seed-43 run used), the
substrate cache is required to exist (the runner refuses to build a
new substrate under a non-default seed), state is namespaced
(`nobn_s43_*`), and the output lands in
`outputs/raw_nobn_eval_seed43.json`. Both commands are in the run
card. The paper's boundary (d) (above) states the single-seed
status and what stands and what does not until it lands: the
recovery swings (+0.088/+0.310 and +0.045/+0.196) are safe; the
level claims await seed 43.

## Item 4 — SCAFFOLD (E2) is still unfixed

**Agreed; consolidated out of the main tables and the conclusion.**
Every SCAFFOLD cell now lives in the new Appendix B ("SCAFFOLD
Arms: Consolidated Implementation-Case Results") with the full
implementation status: the sign-inverted client control-variate
update relative to Eq. 11 of the reference, the cold start at half
the global rate, the five disclosed departures, and the pending
vanilla re-run (B2, in the remaining programme). Concretely: the
SCAFFOLD rows are removed from Tables 6/7/8/10/11 (the
raw-substrate, event-level, BN-diagnostic, LSTM and replication
tables), the main-text mentions are re-pointed to the appendix, the
conclusion's SCAFFOLD-based claims (including the "one priced
exception" sentence) are removed or requalified, and every executed
value is preserved verbatim in the consolidated `tab:scaffold`
(MLP 0.768/0.057, Brier 0.137, R@2%FPR 0.175, event 54/65 at
6.4/day, BN-diagnostic A/B/C 0.763/0.500/0.757 with gate −0.005;
LSTM 0.970/0.294, ECE 0.035, event 55/65 at 0.41/day, seed-43
0.968/0.368). Nothing is deleted from the record; nothing in the
main body or the conclusion rests on the variant's numbers any
more. The method section's disclosure of the five departures stays
(a reader of the full edition must know what was measured), and the
trimmed edition carries the status as an honest limitations
qualifier.

## Item 5 — packaging went the wrong way again

**Agreed; this is the structural change of v4.12.** The two-edition
decision:

- **`paper_trimmed/` — the publishable edition (NEW, 12 pp).** Cut
  to the reviewer's ship-list: provenance and leakage, the
  leakage-free fold, persistence baselines, the two BatchNorm
  findings, calibration failure. The LSTM, SMOTE, client,
  SCAFFOLD and version-history material is not in it (the sequence
  arms appear only as the BatchNorm-free comparator rows of the
  no-BN table, one honest limitations sentence covers SCAFFOLD's
  status, and the extended record is referenced by repository
  path). **Zero process language** — verified by grep: no
  "Dossier", no "owner-side", no "owner-GPU", no "queued", no
  "v4.x" mentions, no internal-review advertising; the parity
  wording of item 1 and the validation honesty of item 2 are in it
  from birth.
- **`submission/` is now the trimmed edition.** `main.pdf` is 12 pp
  (was 63, mirroring the full paper — the reviewer's "there is no
  trimmed edition" finding, exactly). `tools/build_submission.py`
  is retargeted to `paper_trimmed/`: a verbatim, byte-verified copy
  plus a tectonic compile; the v4.6–v4.11 declared
  transformations are retired with the mirrored submission; the
  build stamp is single-sourced from `BATTERY_VERSION` into the
  submission README (the manuscript itself carries no version
  strings). The apparatus test now enforces the two-edition
  contract (per-edition rendered-bibliography pins and page pins:
  65 pp full / 12 pp submission).
- **`paper/` — the full/extended edition, kept** (the owner's
  decision; 65 pp after this revision's additions). Its process
  apparatus (the audit trail, the version-history table, the
  disclosure conventions) is retained deliberately: it is the
  internal record that makes every number in either edition
  traceable, and the reviewer's own verification of our controls
  was performed against it.
- **READMEs neutralised.** The "ACCEPT — SUSTAINED, 8.4/10" line
  and all panel/dossier verdict advertising are removed from the
  root README, `paper/README.md` and `submission/README.md`. The
  root README now leads with the two-edition structure, the
  scientific headline, and the verification entry point; the
  revision history remains where it belongs (`RUNLOG.md`,
  `docs/response_letters/` — internal ledgers, not advertising).

## What we did not do, and why

- We did not run items 2 and 3's compute here: this environment has
  no GPU and no dataset cache. The scripts, guards, namespaces and
  run card are committed; the runs execute owner-side (the same
  workflow every decisive artefact of this project used), and
  integration + pinning follows on arrival per convention.
- We did not remove the process apparatus from the FULL edition:
  the owner chose to keep it as the extended record. The reviewer's
  concern — that a reviewer or judge reads internal-review language
  as noise or worse — is answered where it applies: the submission
  edition and every README.

## Verification

Full battery re-run at v4.12 (see `RUNLOG.md` and
`docs/ENVIRONMENTS.md` row 3 for the counts and the pin changes:
paper 63→65 pp, submission 63→12 pp, both disclosed recompiles).
The submission apparatus test passes 7/7: byte-identical
regeneration from `paper_trimmed/`, the cited-keys subset
bibliography contract, per-edition rendered-bibliography pins, and
the honest-status markers on both editions.

<!-- LETTER-MANIFEST v1
base: d59f299a3b0c0fa39fd7d7e4deb0f15b6cf749a1
-->

## Diff manifest (machine-checked by tests/test_letters_manifest.py)

`git diff --name-status d59f299 <this commit>` — the complete list;
nothing else changed:

    A    docs/RUN_CARD_v4.12.md
    A    docs/response_letters/RESPONSE_External_v4.12.md
    A    paper_trimmed/figures/fig_partition.png
    A    paper_trimmed/main.tex
    A    paper_trimmed/refs.bib
    A    paper_trimmed/sections/sec_calibration.tex
    A    paper_trimmed/sections/sec_data.tex
    A    paper_trimmed/sections/sec_discussion.tex
    A    paper_trimmed/sections/sec_intro.tex
    A    paper_trimmed/sections/sec_method.tex
    A    paper_trimmed/sections/sec_related.tex
    A    paper_trimmed/sections/sec_results.tex
    A    submission/src/sections/sec_calibration.tex
    A    submission/src/sections/sec_data.tex
    M    ABSTRACT.md
    M    README.md
    M    RUNLOG.md
    M    docs/ENVIRONMENTS.md
    M    experiments/run_raw_nobn.py
    M    paper/README.md
    M    paper/main.pdf
    M    paper/main.tex
    M    paper/sections/sec_appendix.tex
    M    paper/sections/sec_conclusion.tex
    M    paper/sections/sec_intro.tex
    M    paper/sections/sec_limitations.tex
    M    paper/sections/sec_results.tex
    M    submission/README.md
    M    submission/main.pdf
    M    submission/src/main.tex
    M    submission/src/refs.bib
    M    submission/src/sections/sec_discussion.tex
    M    submission/src/sections/sec_intro.tex
    M    submission/src/sections/sec_method.tex
    M    submission/src/sections/sec_related.tex
    M    submission/src/sections/sec_results.tex
    M    tests/run_battery.py
    M    tests/test_import_graph.py
    M    tests/test_letters_manifest.py
    M    tests/test_raw_bn_diagnostic.py
    M    tests/test_submission_apparatus.py
    M    tools/build_submission.py
    D    submission/src/figures/FL_Convergence.png
    D    submission/src/figures/ROC_Curves.png
    D    submission/src/figures/SHAP_Feature_Importance.png
    D    submission/src/figures/fig_ablation.png
    D    submission/src/figures/fig_architecture.png
    D    submission/src/figures/fig_clients.png
    D    submission/src/figures/fig_multiseed.png
    D    submission/src/sections/sec_appendix.tex
    D    submission/src/sections/sec_conclusion.tex
    D    submission/src/sections/sec_experiments.tex
    D    submission/src/sections/sec_limitations.tex
    D    submission/src/sections/sec_problem.tex
