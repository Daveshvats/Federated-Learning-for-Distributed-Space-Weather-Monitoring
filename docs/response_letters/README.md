# Response letters of the R-FS9 review cycle

This directory holds the authors' response letters to the four referee
dossiers of the R-FS9 cycle, committed to the repository as of v4.3
(Dossier R-FS9-R3, finding N1) so that every letter-level claim made in
the RUNLOG, the commit history, and the paper's appendix version table
is verifiable in-repo rather than out-of-band.

| File | Round | Manuscript version | Status |
|---|---|---|---|
| `RESPONSE_R-FS9-R1_v4.1_ERRATA.md` | R-FS9-R1 (re-review of v4.0) | v4.1 | Corrected in place at v4.2; inline erratum markers plus a closing erratum section list the statements Dossier R-FS9-R2 refuted |
| `RESPONSE_R-FS9-R2_v4.2.md` | R-FS9-R2 (re-review of v4.1) | v4.2 | As delivered; one erratum added at v4.4 (header page count — the v4.2 PDF was 52 pp, Dossier R-FS9-R4 P3) |
| `RESPONSE_R-FS9-R3_v4.3.md` | R-FS9-R3 (re-review of v4.2) | v4.3 | As delivered; closes the cycle's four clerical residue items (N1-N4); errata added at v4.4 (the letter's minimality claim understated its own Section 4 battery-sentence rewrites, Dossier R-FS9-R4 P1) |
| `RESPONSE_R-FS9-R5_v4.5.md` | R-FS9-R5 (full-body re-audit of v4.4/v4.5) | v4.5 | As delivered; discharges the sixth round's seven non-blocking findings (R7-1-R7-4, C1-C3) and the front-door refresh. The intervening R-FS9-R4 round (v4.4) was discharged via its commit message and RUNLOG entry — both verified against the diff by Dossier R-FS9-R5 — and carried no separate letter |
| `RESPONSE_R-FS9-R6_v4.6.md` | R-FS9-R6 (submission-manuscript audit of v4.5) | v4.6 | As delivered; discharges the seventh round's critical + five major findings and the full repair register, regenerates the journal submission from the record (`submission/`), and carries the first LETTER-MANIFEST block — the machine-checked diff enumeration that `tests/test_letters_manifest.py` verifies against git history (the R2-F1 / R3-N1 / R4-P1 drift species, ended) |
| `RESPONSE_R-FS9-R7_v4.7.md` | R-FS9-R7 (regenerated-submission audit of v4.6) | v4.7 | As delivered; discharges the eighth round's single blocking clerical condition (B1 — the two v4.5-era bibliography comments that silently truncated entries [7] and [22] in both PDFs) plus the six non-blocking hardening recommendations and the minor hygiene items; carries the LETTER-MANIFEST block; records the owner-directed full-paper cross-verification and proofread performed at the v4.7 build |

The first-round response to the original R-FS9 forensic audit predates
this convention and is documented in the repository history
(`docs/REVIEW2_RESPONSE.md` covers the second owner-side review of the
v3.x era). The v4.3 response letter is committed in the same revision
that delivers it, so no letter claim in this cycle remains
out-of-band. The v4.4 errata to the R-FS9-R2 and R-FS9-R3 letters
(Dossier R-FS9-R4, P1 and P3 — the cycle's closing round, executed
without re-review per its disposition) follow the same inline-marker
convention the R-FS9-R1 letter established: the wrong statement is
corrected in place with an `*[Erratum, v4.4: …]*` marker and a closing
erratum section records the correction, so an archived letter carries
its own audit trail. Since v4.6 every new letter additionally carries
a `LETTER-MANIFEST v1` block — a machine-checkable enumeration of the
revision's complete `git diff --name-status` — which the integrity
battery verifies via `tests/test_letters_manifest.py`, so a letter
that understates its own diff fails the build rather than surviving
to a later erratum.
