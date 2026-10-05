# Record of Revision v4.9.2 — Round-13 Register, Instruments Phase
(Repo-Only)

**Re:** Dossier R-FS9-R10, "Referee Re-Review: The Condition,
Reopened in Letter" (the Round-13 third-party adjudication of v4.9 —
verdict **AGREE, ~90 PERCENT — ACCEPT, SUSTAINED — 8.0/10 — E1′
REGISTER ISSUED**, the panel's own Round-12 rating corrected from
8.1 and its "executed in substance" wording withdrawn), verified
against the repository at `b6d1450` before this revision was drafted.
**Revision:** v4.9.2, this commit. No re-review is pending or sought
— this is the v4.4/v4.9.1-precedent form: a repo-only errata batch,
executed and recorded, with the paper of record UNTOUCHED.
**Paper of record after this revision:** unchanged — `paper/main.pdf`,
59 pp at v4.9/v4.9.1. Every paper-side register item (A4, B1–B5, the
A2/A3 queue lines) is scheduled for **v4.10**, after the A1 verdict
arrives, so that ONE paper revision integrates the register with
full knowledge of which world the thirteen-second check lands us in.

We accept the adjudication in full — the ~90-percent agreement with
the third-party reviewer, the corrected 8.0, the E1′ condition, and
the master change register as the binding work order. The register's
own dependency ordering ("the arm-B sanity check first, because it
gates the reading of everything else in Table 8; then the two
missing runs; then the abstract") is what this revision executes:
**the instruments, the queue entries, and the battery hardening —
the paper language follows the evidence, not the other way round.**

<!-- LETTER-MANIFEST v1
base: b6d1450
-->

## Diff manifest (machine-checked by tests/test_letters_manifest.py)

`git diff --name-status b6d1450 <this commit>` — the complete list;
nothing else changed:

    M    RUNLOG.md
    M    docs/ENVIRONMENTS.md
    A    docs/response_letters/RESPONSE_R-FS9-R10_v4.9.2.md
    A    experiments/run_arm_b_central_sanity.py
    A    experiments/run_raw_nobn.py
    M    tests/run_battery.py
    M    tests/test_import_graph.py
    M    tests/test_letters_manifest.py
    M    tests/test_raw_bn_diagnostic.py

## 1. What this revision is — and is not

This is the repo-only phase of the Round-13 register. It ships the
two instruments the register demands before any paper language can
be written honestly (A1 and A3), hardens the battery per B6, records
the register's queue entries in the RUNLOG (the binding queue, per
the register's "add it to the re-run programme queue now regardless
of execution order" — the paper's queue line follows at v4.10 with
everything else), and pays one erratum the register itself flagged
(the RUNLOG v4.8 letters count, B6-b).

It is deliberately NOT a paper revision. The register sequences the
abstract rewrite (A4) AFTER the decisive experiments precisely
because A1's outcome decides the wording: an abstract that says
"pooled-statistics recalibration collapses every federated arm to
chance" is either a validated scientific sentence or a withdrawn
one, and the difference costs thirteen seconds of owner compute to
learn. One paper revision (v4.10) integrating A4, B1–B5, the A1
verdict, and whatever A2/A3 evidence has landed is strictly better
than two revisions written half-blind. The C3 fallback (delete the
raw-substrate claim today) remains available as the register's
own escape hatch if the owner prefers deletion over sequencing —
an owner decision, not ours to take unilaterally.

## 2. Register item A1 — the centralised arm-B sanity check

`experiments/run_arm_b_central_sanity.py` (new, this commit): the
thirteen-second experiment the dossier calls "the cheapest and most
decisive experiment this cycle has left." Design, all disclosed in
the runner's own docstring and pinned statically by the battery:

- **Load** the centralised raw-substrate MLP from
  `data/cache/rawsubstrate/baselines.pt` — the ask-#4 retrain's
  by-product, already on the owner machine. The v3.x-era central
  checkpoint, like the v3.x FL checkpoints, was never persisted; the
  runner discloses this and uses the arm-A reproduction gate as the
  fingerprint surrogate (no fingerprint fields exist on the
  centralised path).
- **Gate**: arm A (own running buffers, the published path) must
  reproduce the committed `raw_substrate_eval.json` centralised
  numbers at the **1e-2 retrain bracket** — NOT the diagnostic's
  1e-4 same-weights gate, which a retrained checkpoint cannot pass
  by construction (the ask-#4 lesson, learned the honest way). A
  MISMATCH exits 1: never read the verdict off a wrong checkpoint.
- **The check itself**: the byte-identical arm-B machinery (forward
  pre-hooks, law-of-total-variance pooling, non-negativity clamp,
  train-shards-only) applied to the centralised model, plus the
  per-layer pooled-vs-own buffer deltas recorded as supporting
  diagnostics — the reconstruction made visible in the artefact.
- **Verdict bands, declared before the run**: `arm_b_validated`
  (ROC-B ≥ 0.95 and PR-B ≥ 0.35 — Table 8's B column stands and
  becomes the validated proxy for the transport fix, which is
  register item A2's re-frame branch), `arm_b_broken` (ROC-B ≤ 0.60
  or PR-B ≤ 0.10 — the B column withdraws, the "actively
  destructive" sentence goes, and A2's faithful arm-D re-run becomes
  mandatory), `inconclusive` (between the bands; adjudicate before
  Table 8's reading changes — exit 1, undecided is not a verdict).
- **Exit contract**: a decisive verdict exits 0 either way — both
  worlds are findings. The artefact
  (`outputs/arm_b_central_sanity.json`) is written before any exit.

RUNLOG ask **#8** carries the single command. Battery layer 4 of
`tests/test_raw_bn_diagnostic.py` activates on the artefact's
presence (the v4.8 pre-run convention, declared [SKIP] while the
run is owner-side — a skip is not a pass).

## 3. Register item A3 — no-BatchNorm federation on the raw substrate

`experiments/run_raw_nobn.py` (new, this commit): FedAvg and
FedProx, the two arms the register names, on the frozen raw
protocol with ONE intervention — SolarMLP's three BatchNorm1d
layers replaced by Identity (the `run_nobn_control.py`
construction, verbatim: same stack minus gamma/beta, no running
statistics, exact weight transport, so the shipped evaluation path
IS the exact evaluation).

- Round-resumable state at
  `data/cache/rawsubstrate/nobn_<algo>_state.pt` — separate
  filenames from the BN arms' checkpoints, `arch='nobn'` stamped
  and validated on resume (no programme collision, no foreign
  resume).
- Evaluation mirrors `run_raw_substrate.py` section 5 exactly
  (prior-shift calibration, val-frozen F-beta threshold,
  frozen-FPR operating points) — the comparability requirement.
- **B1 hygiene is born with these arms**, not retrofitted: the
  shipped best-val-F1 selection rides beside a val-ROC-selected
  sensitivity row (both disclosed, neither test-based), a
  per-round test trajectory is recorded, and the degenerate-rule
  fallback (val F1 = 0.0 at every monitored round — the exact
  pathology the Round-13 review caught on the BN arms) is handled
  loudly with the `_run_fl_loop` precedent: the final round ships,
  disclosed, and the sensitivity row plus trajectory carry the real
  picture regardless.
- The comparison columns (BN MLP arms, centralised, the LSTM arms)
  are READ from the committed artefacts at run time — never
  hand-typed — so the attribution question (encoder-conditional
  vs BatchNorm-under-federation-conditional) is answerable from
  `outputs/raw_nobn_eval.json` alone.

RUNLOG ask **#9** carries the single command (GPU, the same budget
the ask-#4 retrain already ran; run after #8 — both independent,
but #8 is seconds and decides Table 8's reading).

## 4. Register item A2 — deferred by design, decided by A1

The register itself offers the fork: "execute the buffer-transport
federation run (arm D) on the raw substrate, **or** re-frame arm B
explicitly as its proxy with A1 as the validation." We take the
fork honestly: **if A1 validates arm B, the re-frame branch is the
default at v4.10** (arm B becomes the validated proxy; the faithful
arm-D re-run remains the endorsed clean-closure option for any
external submission, and the runner for it will be written then);
**if A1 breaks arm B, the re-run becomes mandatory** and we write
it with full knowledge of what broke. Writing the arm-D runner
before the fork resolves would be sequencing the cart before the
thirteen-second horse.

## 5. Register item B6 — the three battery minors, executed

- **B6-a, recursive letters registry**: the manifest walk in
  `tests/test_letters_manifest.py` is now `os.walk`-recursive. The
  v4.8 registry closed the flat-directory filename escapes; the
  walk itself stayed `os.listdir`, so a letter in a SUBDIRECTORY
  of `docs/response_letters/` escaped classification entirely —
  the loophole the Round-13 panel verified live. Every `.md` at
  any depth is now inventoried by relative path and must appear in
  the pinned registry exactly. **Fault-injected first-hand**: a
  rogue `tmp_subdir/rogue_v4.9.md` fails exit 1 (previously
  invisible); tree restored clean afterwards.
- **B6-b, the RUNLOG v4.8 letters count**: corrected 2 → 3 in the
  v4.8 row's per-file list. Verified by re-summation before
  editing: the per-file sum was 257 against the row's own
  documented 258 pre-commit total, and the single-digit correction
  closes the gap exactly (R6 + R7 + registry = 3 at pre-commit,
  the v4.8 letter's own manifest deferred until its commit — the
  documented convention).
- **B6-c, scope-based external-module classification**: the pinned
  `EXTERNAL_MODULES` name list in `tests/test_import_graph.py` is
  replaced by a three-scope classification: **stdlib** from
  `sys.stdlib_module_names` (the interpreter's own authoritative
  set — the `gc`/`ctypes` whitelist churn ends: stdlib imports are
  never dependencies and will never again need a deliberate pin),
  **third-party** parsed from `requirements.txt` at run time (never
  hand-typed; a NEW dependency still fails deliberately until
  requirements.txt is updated — the C6 discipline kept), plus an
  **origin verification** for whatever is importable in the running
  environment (site-packages / interpreter prefix = pass; resolving
  inside the repository but outside the pinned inventory = a
  SCAN_DIRS-gap error; resolving outside every declared scope = an
  error, not an implicit external). Not installed here (the
  torch-less battery's `torch`) passes on declared scope — scope,
  not availability, is what the static layer checks.
  **Fault-injected both ways first-hand**: a rogue
  `import torchx` fails ("not stdlib scope, not a declared
  dependency"), and removing `seaborn` from requirements.txt fails
  its importer in `visualize_results.py` (the name-based list would
  have kept passing it — scope, not name). Counts re-pinned
  deliberately: 335 from-imports (+26 from the two new runners),
  43 plain local imports (+3), the 75-module inventory (+2).

## 6. The register disposition table

| Register item | Status after v4.9.2 |
|---|---|
| A1 arm-B centralised sanity | **Instrument shipped** (ask #8, seconds, owner); verdict integrates at v4.10 |
| A2 arm-D transport run / re-frame | **Fork documented** (§4); decided by A1's verdict; runner written at v4.10 if the re-run branch wins |
| A3 no-BN raw federation | **Instrument shipped** (ask #9, GPU, owner); queue entry recorded in the RUNLOG now — the paper's re-run-programme line lands at v4.10 |
| A4 abstract scoped rewording + F1 (TSS→PR-AUC) + drop the internal tag | **Scheduled v4.10** (post-A1 wording; F1 is pre-external-submission-mandatory, not pre-revision) |
| B1 checkpoint-selection hygiene | **Half-executed**: the NEW arms carry it from birth (§3); the existing raw arms' sensitivity rows + the "zero of 65" re-qualification are v4.10 paper items |
| B2 E2 SCAFFOLD rerun-or-demote + Table 8 qualifier | **Owner decision + v4.10 text**; the GPU rerun stays queued owner-GPU as before |
| B3 persistence sentence swap | **Scheduled v4.10** (two lines) |
| B4 name + SHA-pin `raw_substrate_rerun.json` in 6.3 | **Scheduled v4.10** (the artefact is committed since v4.9; the pin + the Section 6.3 sentence are one edit) |
| B5 abstract manifest-sentence rewording | **Scheduled v4.10** |
| B6 battery minors | **Executed** (all three, §5, fault-injected) |
| C1 process-language purge | **Owner decision** (the 15-page carve-out C2 and the C3 fallback interlock with it); the counts are the panel's, verified |
| C2 core-audit 15-page submission | **Owner decision** (the strategy fork: A1–A3 land → 8.4–8.6 path; flip → the 7.5 carve-out path) |
| C3 delete-the-claim fallback | **Owner decision**, available at any moment before v4.10 |

## 7. Verification

- `tests/test_raw_bn_diagnostic.py`: 74/0 in the torch-less sandbox
  (51 prior + 23 new static pins; the two new artefact layers
  declared [SKIP] until asks #8/#9 land — the v4.8 pre-run
  convention).
- `tests/test_letters_manifest.py`: 4/0; the subdirectory rogue
  fails exit 1 (fault-injected, restored).
- `tests/test_import_graph.py`: green on the re-pinned 335/43/75;
  both scope fault-injections fail as designed (restored).
- Full battery at BATTERY_VERSION v4.9.2, torch-less, pre-commit:
  the R10 letter's own manifest check defers until this commit
  exists (the v4.6/v4.7/v4.8 deferral convention); post-commit the
  manifest verifies against the actual git diff. Per-file counts
  recorded in `docs/ENVIRONMENTS.md` row 3.
- The two runners fail loudly at a torch-less gate (actionable
  message, exit 1) — never a raw import traceback.

## 8. What the owner runs next (the complete list)

1. `git pull origin improvements` (this commit)
2. RUNLOG ask **#8** — `python experiments/run_arm_b_central_sanity.py`
   (CPU, seconds) — send back `outputs/arm_b_central_sanity.json`
   + the console tail
3. RUNLOG ask **#9** — `python experiments/run_raw_nobn.py`
   (GPU, the ask-#4 budget; round-resumable) — send back
   `outputs/raw_nobn_eval.json`
4. Optionally re-run the battery (the new artefact layers activate
   on the two JSONs)

v4.10 then integrates: A4's scoped abstract (wording decided by
the A1 verdict), B1's sensitivity rows and the 38/65 companion
re-qualification, B2–B5, the A2 fork's resolution, and the version
history — one paper revision, one recompile, one re-review cycle
close.
