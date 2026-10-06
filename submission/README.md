# Journal Submission Edition (the trimmed paper, regenerated from the record)

This directory is the **journal submission manuscript** of the SF9
paper (current build: **v4.13**). Since v4.12 it is the **trimmed,
publishable edition** — the 12-page paper covering exactly what an
external reviewer needs: provenance and leakage, the leakage-free
fold, persistence baselines, the two BatchNorm findings, and
calibration failure. The extended record (the full 65-page edition
with the complete experiment apparatus) lives at `paper/` and is
kept alongside by design.

The submission is GENERATED from `paper_trimmed/` by
`tools/build_submission.py`: a verbatim, byte-verified copy of the
trimmed source (main.tex, sections, figures, refs.bib) plus a
tectonic compile. There are no declared transformations — the
trimmed source IS the submission manuscript. The build stamp in
this README is written at generation time from `BATTERY_VERSION`,
so it can never trail a release; the manuscript itself carries no
version strings (a publishable edition reads as a paper, not a
changelog — the external re-review's packaging item).

## Contents

| Path | What it is |
|---|---|
| `src/main.tex` | Verbatim copy of `paper_trimmed/main.tex` (GENERATED — do not edit) |
| `src/sections/*.tex` | Verbatim copies of `paper_trimmed/sections/*.tex` (GENERATED) |
| `src/refs.bib` | **Byte-identical copy** of `paper_trimmed/refs.bib` (itself the cited-keys subset of `paper/refs.bib`) — the bibliography is never retyped (GENERATED) |
| `src/figures/*` | Byte-identical copies of `paper_trimmed/figures/*` (GENERATED) |
| `main.pdf` | The built submission PDF (tectonic) |
| (repo) `tools/build_submission.py` | The generator — the ONLY place the submission is assembled |

## Invariants (battery-enforced)

`tests/test_submission_apparatus.py` (in the integrity battery):

1. regenerating the source tree (`--source-only` into a temp dir) is
   **byte-identical** to the committed `src/` — drift is impossible
   without a failing battery;
2. `src/refs.bib` is byte-identical to `paper_trimmed/refs.bib`, is a
   subset of `paper/refs.bib`, and every `\cite` key resolves;
3. the floor apparatus (TSS/HSS/inertia/persistence/Brier skill, the
   arms-clearing statement, the parity wording on the abstract pair)
   is present;
4. the full edition (paper/) carries the complete SCAFFOLD
   implementation-status disclosures, and the trimmed edition's
   limitations state the status honestly (sign-inverted departure,
   not the reference algorithm, implementation-case cells);
5. the six prior-art citations are cited; no GIC-framing keys; no
   untraceable constants; the 5e-3 fp16 bound present;
6. the **rendered bibliography** of both this PDF and the full
   edition is extracted and checked per edition: the entry count, a
   year on every entry, and anchor + title fragment + year for every
   pinned entry; the page count of both editions is pinned.

## Rebuild

```
python tools/build_submission.py          # regenerate src + build PDF
python tools/build_submission.py --source-only   # source tree only
```

Run after **any** change under `paper_trimmed/` — then re-run the
battery.
