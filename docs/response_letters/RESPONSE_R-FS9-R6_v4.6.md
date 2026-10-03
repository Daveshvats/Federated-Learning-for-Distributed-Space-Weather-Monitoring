# Response to Dossier R-FS9-R6 — v4.6

**Re:** Referee Re-Review R-FS9-R6, "The Manuscript Beside the Record"
(Submission-Manuscript Audit · Improvements Branch · Commit 154776A)
**Verdict under response:** MAJOR REVISION — 5.9/10, fourteen findings
(one critical, five major), repair register items 1–6.
**Revision:** v4.6, this commit.
**Paper of record after this revision:** `paper/main.pdf`, 55 pp
(recompiled; zero unresolved references), sha256 c6db9fa24576a977…;
**journal submission manuscript:** regenerated from the record at
`submission/` (source + PDF, 55 pp), sha256 ddfefe1cb8083c05….

We accept the verdict, all fourteen findings, and the register in
full. The panel's central diagnosis is ours too: six rounds of
adversarial review built the record, and the submission manuscript
was built one step outside it — the fourth instance of the
hand-prepared-artefact drift species (R2-F1, R3-N1, R4-P1, and now
the manuscript). This revision's response has two halves, and the
second half is the one we ask the panel to weigh hardest: every
register item is executed, **and each defect class the round found is
now closed by a battery guard**, so the species cannot recur by
author discipline alone. The integrity battery grows from 224 checks
to 243, including six new guards written for this round's classes
(import-graph resolution, SCAFFOLD algebra pinning, criterion
plumbing, interpretability artefact, submission apparatus, letter
manifests — each detailed below).

<!-- LETTER-MANIFEST v1
base: 154776a
-->

## Diff manifest (machine-checked by tests/test_letters_manifest.py)

`git diff --name-status 154776a <this commit>` — the complete list;
nothing else changed:

    M    ABSTRACT.md
    M    centralized_baseline.py
    M    docs/ENVIRONMENTS.md
    M    RUNLOG.md
    M    docs/lit_search/q1_fl_solar_flare.json
    M    docs/lit_search/q2_fl_space_weather.json
    M    docs/lit_search/q3_fl_swansf.json
    M    docs/lit_search/q4_fl_cross_silo_geo.json
    M    docs/response_letters/README.md
    A    docs/response_letters/RESPONSE_R-FS9-R6_v4.6.md
    M    experiments/run_budget_matched.py
    M    experiments/run_federated_lstm.py
    M    experiments/run_standard_metrics.py
    M    federated_learning.py
    M    interpretability.py
    M    model.py
    M    README.md
    M    paper/main.pdf
    M    paper/main.tex
    M    paper/sections/sec_appendix.tex
    M    paper/sections/sec_conclusion.tex
    M    paper/sections/sec_discussion.tex
    M    paper/sections/sec_experiments.tex
    M    paper/sections/sec_intro.tex
    M    paper/sections/sec_limitations.tex
    M    paper/sections/sec_method.tex
    M    paper/sections/sec_related.tex
    M    paper/sections/sec_results.tex
    A    submission/README.md
    A    submission/main.pdf
    A    submission/src/figures/FL_Convergence.png
    A    submission/src/figures/ROC_Curves.png
    A    submission/src/figures/SHAP_Feature_Importance.png
    A    submission/src/figures/fig_ablation.png
    A    submission/src/figures/fig_architecture.png
    A    submission/src/figures/fig_clients.png
    A    submission/src/figures/fig_multiseed.png
    A    submission/src/figures/fig_partition.png
    A    submission/src/main.tex
    A    submission/src/refs.bib
    A    submission/src/sections/sec_appendix.tex
    A    submission/src/sections/sec_conclusion.tex
    A    submission/src/sections/sec_discussion.tex
    A    submission/src/sections/sec_experiments.tex
    A    submission/src/sections/sec_intro.tex
    A    submission/src/sections/sec_limitations.tex
    A    submission/src/sections/sec_method.tex
    A    submission/src/sections/sec_problem.tex
    A    submission/src/sections/sec_related.tex
    A    submission/src/sections/sec_results.tex
    M    tests/run_battery.py
    A    tests/test_central_criterion.py
    A    tests/test_import_graph.py
    A    tests/test_interpretability_artifact.py
    A    tests/test_letters_manifest.py
    A    tests/test_scaffold_algebra.py
    A    tests/test_submission_apparatus.py
    A    tools/build_submission.py

No output artefact, figure, log, or published number changed:
`git diff 154776a -- outputs/ logs/` is empty. `paper/main.pdf` and
`submission/main.pdf` are rebuilds of changed sources, not changed
results.

## Register item 1 — the import graph (R8-1, critical): fixed and guarded

Conceded in full, with thanks for the severity call. v4.5 deleted the
stale `cfg.BATCH_SIZE` (R7-2) while `model.py` still imported it, and
the torch-less battery was structurally blind to the class.

- **Fix.** We took the register's second option, not the first:
  `model.py`'s import now names the surviving constant
  (`EVAL_BATCH_SIZE`, which `make_loader` takes as its default)
  rather than re-defining `BATCH_SIZE = 256` in `config.py` —
  resurrecting the stale constant would have re-created the R7-2
  defect (a number that lies about the loader) while fixing R8-1.
  `model.make_loader` has no callers in the shipped code (the
  federated paths use `federated_learning._make_loader`), so the
  default change alters no published path.
- **Guard (the class, not the instance).**
  `tests/test_import_graph.py` (battery): an AST resolver that walks
  every repository source, extracts every `from <local module>
  import NAME`, and resolves each name against the target module's
  module-level bindings — **without executing the target**, so a
  missing torch can never again mask a missing symbol. All 294
  local-name imports resolve at v4.6. A second, torch-gated layer
  smoke-imports the library core when torch is present (it skips,
  with a stated reason, in the torch-less verification environment).
  The static layer is the one that matters: a future deletion of any
  still-imported name fails the battery in *every* environment,
  torch-equipped or not.

## Register item 2 — the SCAFFOLD record (R8-2, R8-3, R8-8): corrected, pinned, and executable

- **R8-2 (sign inversion), conceded and disclosed.** The methods
  section now enumerates **five** departures from the reference
  algorithm, in the enumerate-and-explain register the panel asked
  for: (i) the damped correction; (ii) gradient clipping; (iii) the
  SGD-momentum optimiser at half the global rate (previously the
  "fourth departure" parenthetical, folded into the list); (iv) the
  client control-variate update is **sign-inverted** relative to
  Eq. 11 — the paper now prints both the repository's form
  ($c_i^+ = c_i - c + (y_i^+ - y_i)/\eta K$, the negative of the
  average local gradient) and the reference form, states that a
  scalar one-client simulation confirms the inversion at ratio
  −1.0000, and explains why the arm still trains (damping and
  variate clamping bound the damage); and (v) the **cold start** (see
  R8-3). The arm's comparisons are restated plainly: they measure
  *this implementation*, not the reference algorithm.
- **R8-3 (false warm-start), conceded and withdrawn.** The
  FedAvg-initialised warm-start sentence is gone from the methods
  section, replaced by the truth the call sites and the queue log
  state: every published SCAFFOLD operating point started from a
  fresh random initialisation at half the global rate — a *harder*
  protocol than the refinement previously claimed — and the
  validation F1 pinned at zero through round thirty is quoted as the
  cold-start signature. Implementation-case qualifiers are added at
  every load-bearing narrative site (introduction highlight,
  conclusion's "priced exception", limitations, discussion — the
  discussion also gains the generalised lesson: direction errors in
  correction terms do not announce themselves).
- **Code arithmetic deliberately unchanged**, with the reason stated
  in a comment at the update site: every published operating point
  was produced by this code as-is, and a silent sign fix without
  re-running the arm would create a new record-versus-code
  contradiction. The inversion is disclosed, not patched.
- **Guard.** `tests/test_scaffold_algebra.py` (battery) pins the
  shipped algebra verbatim (the two update lines), the R8-2
  disclosure comment, the cold-start call-site record (both
  published-path callers pass no `warm_start_model`; `main.py`'s
  cleaned-fold call is the only warm-start caller), the queue log's
  behavioural signature, and the paper's disclosure markers — plus a
  torch-gated numeric simulation that reproduces the −1.0000 ratio.
  Neither the code nor the disclosure can now drift silently.

## Register item 3 — the SCAFFOLD re-run: declined in favour of the panel's cheaper form

We take the panel's own framing: "the cheaper form of this item and
fully sufficient for the claims as scoped." The narrative is now
explicitly implementation-case evidence (above), the paper states
that a sign-corrected, warm-started vanilla replication under the
frozen protocol is queued owner-GPU future work whose outcome may
differ, and no algorithm-level claim is made anywhere. The
owner-GPU queue remains available for the re-run if a future round
asks for it; until then the claims are scoped to what was run.

## Register item 4 — the submission manuscript: regenerated from the record

The hand-prepared 48-page manuscript is superseded. The journal
submission edition now lives **inside the repository** at
`submission/`, and it is a **generated artefact**:

- `tools/build_submission.py` copies the record's section sources,
  figures, and bibliography verbatim (never retyped —
  `submission/src/refs.bib` is a byte-identical copy of
  `paper/refs.bib`), composes the journal front matter (re-ordered
  title, dual-author block, submission-edition note), and applies a
  **declared, machine-checked transformation list**: T1 front-matter
  metadata; T2 the abstract pair carries metric names (made in the
  record itself at v4.6 — the generator asserts presence); T3 the
  three occurrences of the un-regenerable "estimated 74 h" CPU figure
  are dropped (R8-10) with the record-backed logged GPU hours (2.6 h
  + 8.7 h) retained; T4 nothing else. Each transformation is
  asserted to match exactly once, so a record change that invalidates
  a declared edit fails the build loudly.
- Consequently R8-6 (the bibliography), R8-7 (the floor apparatus:
  TSS/HSS/inertia/persistence, the arms-clearing statement, metric
  names on the abstract pair), R8-8 (both SCAFFOLD disclosures plus
  the new ones), R8-9 (the six prior-art citations — FedBN lineage
  and the benchmark's leakage warnings — are cited; the retired
  GIC-framing keys are absent because the record retired them), and
  R8-10 (no 74-hour estimate, no 4.4×10⁻⁴, the 5×10⁻³ fp16 bound
  present) are all discharged **by construction**: the submission
  inherits the record's repaired state rather than a retyped
  approximation of it.
- **Guard.** `tests/test_submission_apparatus.py` (battery)
  regenerates the source tree into a temp directory and
  byte-compares it with the committed `submission/src` — a stale or
  hand-edited submission fails the battery; it also verifies the
  refs byte-identity, the six citations, the apparatus markers, the
  corrected scheduler cell, and the absence of the untraceable
  constants. The drift species ends here not by promise but by
  construction.
- The regenerated edition is 55 pages (the record's full apparatus,
  including the reproducibility appendix), not the 48-page
  distillation; the submission deliberately carries everything the
  cycle earned. The owner may trim for venue limits later — by
  editing the generator's transformation list, which the battery
  then re-verifies.

## Register item 5 — the comparator contradictions (R8-4, R8-5)

- **R8-4.** The budget table's centralised cell now reads
  "constant (no scheduler)" (the register's first option — correct
  the cell; we did not implement the schedule and re-run, since the
  parity numbers are frozen artefacts), and the "identical
  optimisation" phrasing in Section 5 is corrected to "the same loss
  family and decision layer", with the optimisation difference
  pointed at the budget table.
- **R8-5.** Both halves of the register's either/or: the B13
  computed-prevalence plumbing is **extended** to all three
  centralised trainers (`centralized_baseline.py`,
  `experiments/run_budget_matched.py`,
  `experiments/run_federated_lstm.py` — `global_pos_rate` now passed
  at every call site; published operating points predate the
  alignment, exactly as R7-4 recorded for the SCAFFOLD arm) **and**
  the asymmetry is **disclosed** in the same breath as the SCAFFOLD
  disclosure in the methods section.
- **Guard.** `tests/test_central_criterion.py` (battery): an AST
  scan asserting every `get_criterion` call site in the repository
  passes `global_pos_rate` — all six call sites comply; the 0.4887
  fallback is unreachable by omission.

## Register item 6 — the minor items (R8-11, R8-12, Section 6 soft spots)

- **R8-11 (interpretability).** The DeepExplainer→GradientExplainer
  fallback is now loud (a printed warning naming the exception,
  tagged R8-11) and recorded (`fl_explainer`, `fl_attribution_rows`,
  `xgboost_attribution_rows` fields for future runs); the FL
  attribution default is aligned to 500 rows, matching the artefact
  note. The committed v3.0 artefact is frozen as the published
  record — its 300-row FL provenance is disclosed here and in the
  RUNLOG, not silently patched. New battery test
  `tests/test_interpretability_artifact.py`: sha256-pins the frozen
  artefact, recomputes its Spearman/p/overlap statistics from its
  own vectors via the shipped code, checks the feature-name family,
  and pins the source contract.
- **R8-12 (block A).** `run_standard_metrics.py` block A no longer
  embeds its denominators: `n_win` is read from
  `results.json:protocol.split.test`, the fold counts from
  `partition_disjoint_eval.json:sizes`, the raw-substrate counts from
  `raw_substrate_eval.json:substrate` (missing keys are errors, not
  silent defaults) — the R-FS9-R3 N3 sweep-test convention — with
  self-checks asserting each artefact value equals the constant every
  published block-A number was computed with. Verified: block A's
  output is byte-identical to the committed `standard_metrics.json`;
  the artefact itself is untouched (the torch-less environment
  cannot re-run block B, and no number changed).
- **Focal constants.** The progressive-factor range [0.9, 1.1] (and
  its v2.x 0.7–1.3 history) and the max($p_k$, 0.01) floor are now
  stated in the loss definition, completing it.
- **Lit-search dates.** The four `docs/lit_search/*.json` files now
  carry an embedded `query_date` (2026-09-29, the date the paper of
  record itself documents, cited as the source in each file), with
  all 30 records preserved verbatim; the paper's §2 pointer notes the
  embedding.
- **Export-controls motivation.** Not applicable to the regenerated
  submission by construction: the record's introduction (since the
  v4.1 GIC retirement) carries no data-locality/export-control claim,
  and the submission inherits that text. No GIC-framing citations
  return, because they are not in the record's bibliography at all.
- **The three adjacent works (Section 6 advisory).** We record the
  panel's suggestion with thanks and adopt the narrow framing rather
  than new citations: the regenerated submission makes no
  air-quality/seismic/SSA deployment claims (the record's text never
  did), so the suggested citations have no claim to support. Adding
  new bibliography entries that we have not verified field-by-field
  against publisher records would re-create the R8-6 species we are
  closing; if the owner later adds a deployment narrative, the
  citations should be added with the same Crossref/arXiv verification
  the record's 31 entries received.

## The anti-recurrence machinery (the panel's trajectory concern)

Six new battery guards, each closing one defect class this round
found:

| Guard (battery test) | Class it ends | Provenance |
|---|---|---|
| `test_import_graph.py` | import regressions invisible to a torch-less battery (R8-1) | 294 imports resolve by AST, environment-independent |
| `test_scaffold_algebra.py` | paper-vs-code algebra drift on the SCAFFOLD update (R8-2/R8-3) | shipped lines pinned verbatim + torch-gated −1.0000 simulation |
| `test_central_criterion.py` | criterion construction that silently re-enables the 0.4887 fallback (R8-5) | all `get_criterion` call sites AST-checked |
| `test_interpretability_artifact.py` | un-guarded artefact + silent fallback (R8-11) | sha256 pin + internal-consistency recompute |
| `test_submission_apparatus.py` | hand-prepared-artefact drift, all manuscript findings (R8-6/7/8/9/10) | regenerate-and-byte-compare vs the committed submission |
| `test_letters_manifest.py` | letter-vs-record drift (R2-F1, R3-N1, R4-P1) | every v4.6+ letter's diff manifest machine-checked against git history |

The letter you are reading carries the manifest block this last test
verifies — the complete `git diff --name-status` of this revision is
enumerated above, and the battery now fails any future letter that
understates its own diff.

## Verification

- Integrity battery at v4.6 (torch-less environment of record, row 3
  of `docs/ENVIRONMENTS.md`): **242/0** (24 torch-dependent skips),
  per-file counts in the RUNLOG row; the pre-commit run's manifest
  check defers to the post-commit run by design (both recorded).
- Paper of record recompiled (tectonic): 55 pp, zero unresolved
  references, sha256 c6db9fa24576a977…; every number, table cell,
  and figure unchanged — the diff is disclosures, corrections, and
  the apparatus row.
- Submission regenerated and rebuilt: 55 pp, apparatus test 6/6,
  regeneration byte-identical.
- `git diff 154776a -- outputs/ logs/` is empty: no artefact, log, or
  published number changed. The sweep digest
  (ceb7189826af15af…) is untouched.
- The record's own 74-hour CPU estimate remains in the repository
  paper (the record states it as an estimate and seven rounds
  accepted it as such); the *submission edition* drops it via the
  declared T3 transformation, per the register's "drop or re-derive".

We ask the panel to re-run its own instruments: the fresh-clone
first-run contract still holds (exit 0 at 243/0), the battery now
fails loudly on every class this round found, and the object under
journal review is, for the first time, generated and guarded inside
the audit trail it stands on.
