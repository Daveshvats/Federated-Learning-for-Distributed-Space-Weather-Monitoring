# RUN CARD (rev. v4.13) — the external-review compute items (2) and (3)

The independent re-review of v4.11 asked for two small compute jobs
that this sandbox cannot run (no GPU, no dataset cache). Both are
prepared, guarded, and namespaced so they cannot disturb the frozen
v4.9.4 artefacts. This card is the exact recipe.

**v4.13 status (2026-10-06, after the owner's second execution):**

- **Item 3 (seed 43) is EXECUTED and integrated at v4.13.** Both
  arms completed on the owner GPU (2,899 s; transcript committed at
  `logs/run_raw_nobn_seed43_transcript.txt`). Verdict per the
  decision rules below: **parity holds** — shipped selections move
  −0.002/+0.001 ROC-AUC (FedAvg 0.9626→0.961, FedProx
  0.9755→0.976), every level claim survives, both dynamics
  signatures reproduce, and the thresholded validation-F1 monitor
  is the seed-sensitive layer (0.527→0.660 / 0.092→0.300). The
  paper's boundary (d), the re-run programme, the conclusion, the
  trimmed edition, and the new Table `tab:rawnobnrep` all carry it.
  One follow-up remains owner-side, cost ~10 seconds: **push the
  full-fidelity JSON** (see "Item 3 follow-up" below).
- **Item 2 (region-disjoint) is BLOCKED on a prerequisite, and the
  card's own v4.12 instructions were un-pasteable.** Two causes,
  both now fixed in this revision:
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

## Item 3 follow-up — push the JSON (10 seconds, do this first)

The run already wrote `outputs\raw_nobn_eval_seed43.json` (24,768
bytes — the box inventory confirms it exists). It has NOT been
committed yet. The v4.13 integration rides the committed console
transcript (every paper number battery-pinned to its parsed lines);
this push lands the full-fidelity record, which will be sha256-
frozen on arrival per the established convention:

```powershell
git pull    # picks up v4.13
git add outputs\raw_nobn_eval_seed43.json
git commit -m "artefact: raw_nobn_eval_seed43.json (owner-side, external-review item 3)"
git push
```

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

### Step 1 (Cases B/C only): regenerate the parse metadata

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

- **Item 2, divergence disappears** (val ROC no longer rises while
  test falls; the shipped checkpoint is no longer the weakest test
  round): the random carve was region-leaky — the paper's validation
  story updates at v4.14, and the shipped numbers likely *improve*
  under the honest monitor.
- **Item 2, divergence persists**: partition-5 recency (temporal
  drift) becomes the leading explanation; the FedAvg declining
  trajectory is real client drift and the row stays a conservative
  lower bound.
- **Item 3 — RESOLVED at v4.13**: parity holds within ±0.005
  ROC-AUC (measured: −0.002/+0.001); the level claims stand with
  replication.

## Reporting back

After the item-2 run completes:

```powershell
git add outputs\raw_nobn_region_disjoint.json
git commit -m "artefact: raw_nobn_region_disjoint.json (owner-side, external-review item 2)"
git push
```

Then paste (or leave in the pushed commit) the stdout tail — the
`[nobn]` summary lines and the REGION-DISJOINT carve line.
Integration at v4.14 then follows the established convention: the
artefact lands byte-faithfully, gets sha256-pinned, and the paper
integrates the verdict (Sections 6.4/9 and the limitations re-run
programme).
