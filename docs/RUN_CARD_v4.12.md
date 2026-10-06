# RUN CARD (rev. v4.14) — the external-review compute items (2)
and (3): BOTH EXECUTED, BOTH INTEGRATED

The independent re-review of v4.11 asked for two small compute jobs
that this sandbox cannot run (no GPU, no dataset cache). Both were
prepared, guarded, and namespaced so they cannot disturb the frozen
v4.9.4 artefacts. This card is the exact recipe — and, as of v4.14,
the closed record of both executions.

**v4.14 status (2026-10-06, after the owner's third execution):**

- **Item 2 (region-disjoint) is EXECUTED and integrated at
  v4.14.** Both arms completed on the owner GPU (3,119 s; the
  artefact was pushed directly by the owner and committed at
  `outputs/raw_nobn_region_disjoint.json` — 742 lines, the
  Windows CRLF form, 25,722 bytes — the first full-fidelity
  owner push of the cycle). Verdict per the decision rules
  below: **the divergence disappears** — FedAvg's validation
  ROC-AUC declines (0.980→0.974) in parallel with the test
  decline (0.975→0.960) under the whole-region carve (390 of
  2,447 active regions held out, 1.49% validation prevalence),
  so the random carve's rising monitor (0.974→0.977; sharper at
  seed 43, 0.978→0.989) was region-sharing leakage and
  partition-5 recency is rejected. The shipped selections are
  unchanged within noise (FedAvg 0.9626→0.9617, FedProx
  0.9755→0.9743 — note the card's earlier "likely improve"
  guess did NOT materialise: the leak inflated the monitor's
  trend, not the validation-based selections); the shipped
  checkpoint is no longer the weakest test round of its own
  trajectory; and the validation-ROC sensitivity rule now
  selects round 5, simultaneously the best test round of the
  entire trajectory (0.975/0.450). FedProx shows no divergence
  in either carve, and its validation-F1 monitor recovers
  0.092→0.574 at the same seed — the fifth independent
  confirmation that the fixed-threshold monitor is the fragile
  layer. The paper (6.4 mechanisms paragraph resolved in place +
  the verdict/residuals paragraphs, the limitations' fifth
  owner-GPU batch row + queued-list removal, the conclusion
  clause, the appendix v4.14 row, the trimmed edition's
  divergence paragraph and discharged caveat) carries it, and
  `tests/test_region_disjoint_verdict.py` (80 checks) pins the
  artefact, the verdict, and every integration site. **No owner
  action remains on this card.**
- **Item 3 (seed 43): EXECUTED and integrated at v4.13/v4.13.1**
  (see below). Fully closed, nothing owner-side.
- **(Historical, v4.12.1:) Item 2 was blocked on two causes, both
  fixed in that revision:**
  1. *PowerShell reserves the `<` character.* The v4.12 card wrote
     its recipe with `<path>`-style placeholders; pasted literally
     (as the owner did, transcript lines preserved at the end of
     the committed seed-43 transcript), every invocation died as
     `ParserError: The '<' operator is reserved for future use`
     before Python ever ran. This revision contains **no angle
     brackets**: every command below is copy-paste-ready, and the
     one thing you may need to adjust is a single PowerShell
     variable at the top of the recipe.
  2. *The raw parse metadata is gone.* The region-disjoint split
     needs `p1..p4_meta.csv` (labels + NOAA AR column). The
     substrate build that created your `data.npz` recorded its raw
     dir as `/tmp/swansf_raw` — a POSIX path, i.e. almost certainly
     inside WSL, whose `/tmp` is wiped on reboot. The committed
     owner-box inventory (`logs/owner_dirlisting_2026-10-06.txt`)
     confirms: no `p*_meta.csv`, no partition directory, and no
     `.tar.gz` anywhere in the repo tree. The search + regeneration
     recipe below covers all three cases (files survive somewhere /
     only the Dataverse downloads survive / nothing survives).

The v4.12.1 errata (region source = the parse metadata `ar` column
with full pool coverage; pre-flighted `p1..p4_meta.csv` with a
guided exit instead of a POSIX-default traceback) still stands and
is battery-guarded by `tests/test_region_disjoint.py`.

## Context

- **Item 2 (validation story).** In `outputs/raw_nobn_eval.json`,
  FedAvg no-BN validation ROC-AUC *rises* (0.9736 → 0.9773) while
  test ROC-AUC *falls* (0.974 → 0.963). ROC-AUC is prevalence-
  invariant, so the 2.05%/1.31% prevalence difference cannot explain
  it. Two candidates remain: (a) the random validation carve shares
  active regions with the training shards (leakage), or (b) partition
  5 is the temporally latest fold (drift). The region-disjoint
  re-run separates them. **Note: the seed-43 run sharpened this
  question** — the divergence reproduces and widens at seed 43
  (val ROC 0.978 → 0.989 while test falls 0.976 → 0.959), so the
  item-2 run is now the no-BN control's one open compute item.
- **Item 3 (one seed).** Resolved at v4.13 (see above).

## Prerequisites

1. **The repo at v4.13** (branch `improvements`), working tree
   clean — `git pull` to pick up this revision.
2. **The substrate cache** `data/cache/rawsubstrate/data.npz` —
   present on the owner box (confirmed by the inventory); both runs
   resume it (the seed-43 run refuses to start without it, by
   design).
3. **The raw parse metadata** `p1..p4_meta.csv` — required by item 2
   only. See the recipe below.

## Item 3 follow-up — the JSON push — DONE (v4.13.1, 2026-10-06)

The owner pasted the full-fidelity JSON back (the whole file,
verbatim). It has been reconstructed byte-faithfully and COMMITTED
at `outputs/raw_nobn_eval_seed43.json`: the LF-committed form is
24,063 bytes (sha256 7c3d30fe9c35ca0e...bde63741, byte-pinned in
battery layer 6), and its Windows CRLF form is EXACTLY the
24,768 bytes the owner-disk inventory recorded, so the landing is
reconciled against two independent owner-side records (the pasted
bytes + the byte-pinned inventory). The sha256 freeze the v4.13
card promised is live: battery layer 6 now byte-pins the artefact,
re-verifies every transcript-consistency fact against it, and a
missing file is a FAILURE, not a SKIP. **No owner action remains
for item 3 — and since v4.14, none remains for item 2 either: the
item-2 artefact was owner-pushed (17af7e4) and the integration is
live. This card is now the closed record of both runs; the recipe
sections below are retained for the record and for anyone
re-running from scratch.**

## Item 2 — the region-disjoint re-run (PowerShell-safe recipe)

### Step 0: find out which case you are in

```powershell
# (a) search your user profile for surviving parse metadata or
#     Dataverse downloads (Documents, Downloads, Desktop, ...):
Get-ChildItem "$env:USERPROFILE" -Recurse -Depth 5 `
    -Include "p1_meta.csv","p*_raw.npz","partition*.tar.gz" `
    -ErrorAction SilentlyContinue |
    Select-Object -First 20 -ExpandProperty FullName

# (b) the substrate build recorded raw_dir = /tmp/swansf_raw (POSIX
#     -> almost certainly WSL; /tmp is wiped on reboot, but worth a look):
wsl -- ls /tmp/swansf_raw
wsl -- bash -lc "find /tmp ~ -maxdepth 4 -name p1_meta.csv 2>/dev/null"
```

- **Case A — a directory with `p1..p4_meta.csv` survived** (from
  step 0a or 0b): skip to "The run", passing that directory as
  `--raw-dir`.
- **Case B — only the Dataverse partition downloads survived**
  (folders named `partition1..4`, each containing `FL\` and `NF\`
  subfolders, or their `.tar.gz` archives): regenerate the metadata
  (Step 1) — minutes per partition, writes ~6 MB of CSV, no npz.
- **Case C — nothing survived**: re-download partitions 1–4 from
  the Harvard Dataverse dataset the repo's provenance record cites
  (`doi:10.7910/DVN/EBCFKM`, see `provenance/README.md`; the files
  are `partition1_instances.tar.gz` .. `partition4_instances.tar.gz`),
  extract them anywhere, then follow Case B.

### Step 1 (Cases B/C only; NOT NEEDED since v4.13.2 — the repo ships raw/p*_meta.csv): regenerate the parse metadata

```powershell
# from the repo root (C:\Users\deves\Documents\sf9).
# ONE variable to adjust: where you extracted / kept the partitions.
$parts = "$env:USERPROFILE\Downloads\swansf"

# this writes raw\p1_meta.csv .. raw\p4_meta.csv (flat, inside the
# repo; the runner resolves --raw-dir relative to the repo root,
# so no absolute paths are needed anywhere):
python provenance\swansf_parse_partition.py "$parts\partition1" raw\p1 --meta-only
python provenance\swansf_parse_partition.py "$parts\partition2" raw\p2 --meta-only
python provenance\swansf_parse_partition.py "$parts\partition3" raw\p3 --meta-only
python provenance\swansf_parse_partition.py "$parts\partition4" raw\p4 --meta-only
```

(If step 0a found a surviving archive, extract it first —
`Expand-Archive` or 7-Zip — and point `$parts` at the extract.)

### The run

```powershell
# ~1,700 s on the same GPU class as the original control; both
# arms; round-resumable (state namespace nobnrd_* never touches the
# frozen seed-42 checkpoints).
python experiments\run_raw_nobn.py --region-disjoint --raw-dir raw
```

(Case A: replace `--raw-dir raw` with the found directory, quoted if
it contains spaces — e.g. `--raw-dir "C:\tmp\swansf_raw"`.)

## What each run writes

- `outputs/raw_nobn_region_disjoint.json` — same schema as
  `raw_nobn_eval.json`, plus a `validation_split` block (region
  counts, val prevalence, disjointness statement, the v4.12.1
  `region_source` disclosure) and a `protocol.validation` marker.
- `outputs/raw_nobn_eval_seed43.json` — already written by the
  executed item-3 run; awaiting the push above.

## Built-in guards (the run is self-checking)

- The raw parse metadata files are pre-flighted; a missing file is a
  guided exit naming `--raw-dir` and the regeneration command.
- The region carve re-derives the frozen random split and verifies it
  by a label round-trip against the cache; any mismatch aborts
  loudly rather than mis-splitting.
- Region disjointness is asserted, not assumed; `ar = -1` sentinel
  rows abort.
- The whole-region assignment is deterministic (seed-42 region
  permutation); reruns are identical.
- All of the above are exercised on synthetic fixtures by the battery
  (tests/test_region_disjoint.py); the seed-43 verdict is pinned by
  tests/test_seed43_replication.py.

## How to read the outcomes (decision rules)

- **Item 2 — RESOLVED at v4.14: the divergence disappears**
  (val ROC no longer rises while test falls; the shipped
  checkpoint is no longer the weakest test round — measured:
  val ROC 0.980→0.974 declining in parallel with test
  0.975→0.960 under the region-disjoint carve; the random carve
  was region-leaky, and partition-5 recency is rejected). The
  paper's validation story updated at v4.14. One honest
  correction to this card's own earlier guess: the shipped
  numbers did NOT improve under the honest monitor (both
  deltas ≈ −0.001) — the leak inflated the monitor's trend, not
  the validation-based selections.
- **Item 2, divergence persists** (the alternative branch, NOT
  taken): partition-5 recency would have become the leading
  explanation; the FedAvg declining trajectory would have stayed
  a conservative lower bound. In the event, the test decline
  proved carve-invariant, so the lower-bound reading survives
  regardless.
- **Item 3 — RESOLVED at v4.13**: parity holds within ±0.005
  ROC-AUC (measured: −0.002/+0.001); the level claims stand with
  replication.

## Reporting back — CLOSED (v4.14)

The item-2 run completed, the owner pushed the artefact directly
(commit 17af7e4, `outputs/raw_nobn_region_disjoint.json`), the
analysis session validated it against the run's own console
output (40+ cross-checks; the pasted pre-push reconstruction had
four deep-decimal digit transpositions that the push corrected,
and the selected-vs-trajectory 4.6e-6 ROC gap at round 35 is a
property of the owner's own file — two separate post-hoc
evaluation passes), and the v4.14 integration landed per the
established convention: the artefact sha256-frozen by the new
battery module, the paper's Sections 6.4/9 and the limitations
re-run programme carrying the verdict, both editions recompiled.
Nothing remains owner-side on this card.
