# Response to the external re-review — the v4.13 compute follow-up
(item 3 executed; item 2 unblocked)

**Reviewer**: the independent external re-review of the v4.11 state
(five findings, answered at v4.12). This revision reports the
execution of the re-review's own compute items and the integration
of their verdicts. No finding is re-opened; no claim is weakened.
**Manuscript**: the v4.13 build of *Federated Solar-Flare Prediction
on SWAN-SF: A Benchmark Audit of Provenance, Leakage, and Evaluation
Protocols* (full edition 67 pp; trimmed submission edition 12 pp).

## Item 3 — the no-BN control is one seed: **EXECUTED; parity
holds**

The owner executed the committed kit on 2026-10-06
(`python experiments/run_raw_nobn.py --seed 43`, both arms, 2,899 s,
RTX 4060; model initialisation and the Dirichlet shard draw
reseeded, the frozen random validation carve kept identical, state
namespaced `nobns43_*`, the frozen v4.9.4 artefacts untouched). The
full-fidelity artefact `outputs/raw_nobn_eval_seed43.json` (24,768
bytes) exists on the owner box and lands by owner push — at
integration time the record of the paper's numbers is the owner
console transcript, committed verbatim at
`logs/run_raw_nobn_seed43_transcript.txt` (the pasted session,
including the two failed placeholder invocations of the item-2 kit
that motivated this revision's run-card rewrite).

**The verdict, against the run card's decision rules:**

- **Level claims: replicated.** Shipped selections move by
  −0.002/+0.001 ROC-AUC (FedAvg 0.9626→0.961, FedProx
  0.9755→0.976) — inside the run card's ±0.005 rule, inside the
  0.002–0.023 retrain band, and inside the sequence arms' ±0.005
  replication band. No-BN FedAvg stays level with the FedAvg-LSTM
  sequence arm (0.961 vs 0.958); no-BN FedProx stays above the
  centralised MLP (0.976 vs 0.971). Every level claim of v4.12's
  re-wording stands with replication.
- **PR-AUC: moves −0.017/−0.024** (FedAvg 0.366→0.349, FedProx
  0.345→0.321) without overturning an ordering — the same
  magnitude class the sequence arms' own seed-43 replication
  recorded at 1.3% prevalence.
- **Both dynamics signatures reproduce**: the FedAvg val-up/test-down
  divergence (validation ROC-AUC 0.978→0.989 while the test
  trajectory falls 0.976→0.959 — the divergence *sharpens*, which
  is exactly why item 2's region-disjoint run is now the control's
  one open compute item), and FedProx's compressed early-round
  monitor (validation F1 exactly 0.0000 through round 20 before the
  late climb to 0.300).
- **What the seed does move is the thresholded validation-F1
  monitor** (best values 0.527→0.660 and 0.092→0.300) — a fourth
  independent confirmation that at 2% prevalence the fixed-threshold
  monitor is the fragile layer and the rank metrics carry the
  conclusions.
- **Cross-runner reproducibility**: the seed-43 Dirichlet shard draw
  is identical to the sequence arms' own seed-43 batch of
  Section 6.6 (the same six client sizes and flare rates,
  `logs/queue_seed43.log`), a fact the battery pins.

**Integration (v4.13)**: boundary (d) of Section 6.4 is discharged
and rewritten; the new Table 10 (`tab:rawnobnrep`, seed-42/seed-43
columns, the sequence replication table's format) is added; the
re-run programme's item list moves the replication from queued to
executed (fourth owner-GPU batch); the conclusion's parity sentence
gains the replication clause; the appendix version-history carries
the v4.13 row; the trimmed edition's two single-seed caveats are
replaced by the replication verdict (zero process language). Every
integrated number is battery-pinned to the parsed transcript lines
(new module `tests/test_seed43_replication.py`, 42 checks), and a
gated consistency layer activates on the JSON the moment the owner
push lands (the sha256 freeze follows per the v4.9.4-onwards
convention).

## Item 2 — the validation story: **kit unblocked, run pending**
(owner-side)

The v4.12.1 kit errata (region source = the parse metadata `ar`
column; pre-flighted `p1..p4_meta.csv` with a guided exit) stands
and is battery-guarded. The owner's execution attempt surfaced a
documentation defect and a data prerequisite:

- *The run card's recipe was un-pasteable in PowerShell.* Its
  `<path>`-style placeholders died as
  `ParserError: The '<' operator is reserved for future use` before
  Python ever ran (the pasted session preserves both failures).
  The v4.13 run card contains no angle brackets: one PowerShell
  variable to adjust, literal commands, a repo-relative `--raw-dir`
  (the runner resolves it against the repo root regardless of the
  invocation directory).
- *The raw parse metadata is gone.* The substrate build that
  created the owner's `data.npz` recorded its raw dir as
  `/tmp/swansf_raw` — a POSIX path, almost certainly WSL, whose
  `/tmp` is wiped on reboot; the committed owner-box inventory
  (`logs/owner_dirlisting_2026-10-06.txt`) confirms no
  `p1..p4_meta.csv`, partition directory, or `.tar.gz` anywhere in
  the repo tree. The run card now carries the full three-case
  recipe: search (PowerShell + WSL) for surviving copies; else
  regenerate metadata-only from surviving Dataverse partitions
  (minutes, ~6 MB, no npz — the substrate cache provides X); else
  re-download partitions 1–4 from the Harvard Dataverse dataset the
  provenance record cites and regenerate.

The decision rules of v4.12 are unchanged and await the run; note
that the seed-43 replication *sharpened* the question it answers
(the val-up/test-down divergence widened at seed 43).

## Items 1, 4, 5 — unchanged

The parity re-wording (item 1) is now replication-backed at every
site it was applied to; the SCAFFOLD consolidation (item 4) and the
two-edition packaging (item 5) are untouched by this revision.

## Verification

Full battery re-run at v4.13: **445/0 torch-less pre-commit,
446/0 post-commit** (this letter's own manifest check activating
at commit, the documented deferral convention) (the new
`tests/test_seed43_replication.py` contributes 42 checks; the
letters registry carries this letter; page pins 65/67 for the full
edition reflowed by the item-3 integration, 12/12 for the trimmed
edition — the B7 convention, disclosed recompiles of both PDFs).
The owner follow-ups, in cost order: push
`outputs/raw_nobn_eval_seed43.json` (10 seconds — the recipe is in
the run card), then the item-2 run (~1.7 ks GPU) once the parse
metadata is regenerated; integration of its verdict follows at
v4.14 per the established convention.

<!-- LETTER-MANIFEST v1
base: bac4162d13e000a33eb70612b916d02b5c45b5c9
-->

## Diff manifest (machine-checked by tests/test_letters_manifest.py)

`git diff --name-status bac4162 <this commit>` — the complete list;
nothing else changed:

    M    docs/ENVIRONMENTS.md
    M    docs/RUN_CARD_v4.12.md
    A    docs/response_letters/RESPONSE_External_v4.13.md
    A    logs/owner_dirlisting_2026-10-06.txt
    A    logs/run_raw_nobn_seed43_transcript.txt
    M    paper/main.pdf
    M    paper/README.md
    M    paper/sections/sec_appendix.tex
    M    paper/sections/sec_conclusion.tex
    M    paper/sections/sec_limitations.tex
    M    paper/sections/sec_results.tex
    M    paper_trimmed/sections/sec_discussion.tex
    M    paper_trimmed/sections/sec_results.tex
    M    README.md
    M    RUNLOG.md
    M    submission/README.md
    M    submission/main.pdf
    M    submission/src/sections/sec_discussion.tex
    M    submission/src/sections/sec_results.tex
    M    tests/run_battery.py
    M    tests/test_import_graph.py
    M    tests/test_letters_manifest.py
    M    tests/test_submission_apparatus.py
    A    tests/test_seed43_replication.py
