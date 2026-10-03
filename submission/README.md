# Journal Submission Edition (regenerated from the record)

This directory is the **journal-formatted submission manuscript** of
the SF9 paper, regenerated programmatically from the repository of
record at **v4.6**, per **Dossier R-FS9-R6, register item 4**
(R8-6/R8-7/R8-8/R8-9/R8-10).

## Why this exists

The previous 48-page submission manuscript was hand-prepared outside
the repository and drifted from the record: eleven defective
bibliography entries (v3.9-era), the skill-score floor apparatus and
both SCAFFOLD disclosures dropped, six prior-art citations missing,
and two untraceable constants — the fourth documented instance of the
hand-prepared-artefact drift species (R2-F1, R3-N1, R4-P1, R6-4).
This directory ends that species structurally: **the submission is a
generated artefact, and the battery verifies it.**

## Contents

| Path | What it is |
|---|---|
| `src/main.tex` | Journal front matter + the record's preamble, abstract, section order (GENERATED — do not edit) |
| `src/sections/*.tex` | Verbatim copies of `paper/sections/*.tex` (GENERATED) |
| `src/refs.bib` | **Byte-identical copy** of `paper/refs.bib` — the bibliography is never retyped (GENERATED) |
| `src/figures/*` | Byte-identical copies of `paper/figures/*` (GENERATED) |
| `main.pdf` | The built submission PDF (tectonic) |
| (repo) `tools/build_submission.py` | The generator — the ONLY place submission content is authored |

## The complete list of deviations from the record

Declared, machine-checked transformations in `tools/build_submission.py`
(each asserted to match exactly once at generation time):

- **T1** journal front matter: re-ordered title
  ("…A Provenance, Leakage, and Evaluation-Protocol Audit"),
  submission-edition note, pdftitle metadata;
- **T2** abstract pair carries metric names (made in the RECORD at
  v4.6 — the generator *asserts* presence, it does not re-edit);
- **T3** the three occurrences of the un-regenerable "estimated 74 h"
  CPU figure are dropped (R8-10) — the record-backed logged GPU hours
  (2.6 h + 8.7 h) are retained;
- **T4** nothing else — every other byte is a verbatim copy.

## Invariants (battery-enforced)

`tests/test_submission_apparatus.py` (in the integrity battery):

1. regenerating the source tree (`--source-only` into a temp dir) is
   **byte-identical** to the committed `src/` — drift is impossible
   without a failing battery;
2. `src/refs.bib` is byte-identical to `paper/refs.bib` and every
   `\cite` key resolves;
3. the floor apparatus (TSS/HSS/inertia/persistence/Brier skill, the
   arms-clearing statement, metric names on the abstract pair) is
   present;
4. both SCAFFOLD disclosures (five-departure enumeration, sign
   inversion, cold start, optimiser constants, prevalence asymmetry)
   and the corrected budget-table scheduler cell are present;
5. the six prior-art citations are cited; no GIC-framing keys; no
   untraceable constants; the 5e-3 fp16 bound present.

## Rebuild

```
python tools/build_submission.py          # regenerate src + build PDF
python tools/build_submission.py --source-only   # source tree only
```

Run after **any** change under `paper/` — then re-run the battery.
