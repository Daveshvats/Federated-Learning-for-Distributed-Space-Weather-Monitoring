# Execution environments of record

One table, every environment in the artefact record, what each ran, and
which committed artefact records it. Introduced at v4.3 in response to
Dossier R-FS9-R3's recorded observation (no finding number) that the
environment count in the record had drifted upward. Every row below
cites only what its own source states — the standard this cycle
converged on.

| # | Environment | What it ran | Recorded where |
|---|---|---|---|
| 1 | python 3.12.14; torch 2.14.1, numpy 2.2.4 (pinned) | The protocol run of record: training, frozen-threshold evaluation, and every result artefact of the v3.x–v4.0 era | `outputs/run_manifest.json` (`python` field; torch/numpy predicated by the `requirements.txt` pins — the v3.x manifest predates those fields) |
| 2 | python 3.13.5, numpy 2.2.4, torch 2.14.1+cpu | The v4.1 verification battery (219/219) and the v4.1 regeneration of `outputs/standard_metrics.json` | `logs/test_battery.log` (canonical v4.1 battery log, header line 2) |
| 3 | python 3.12.14, numpy 2.1.3, torch not importable | The v4.2–v4.5 verification batteries (195/0 at v4.2; 224/0 with 29 new sweep-coverage checks at v4.3, v4.4, and v4.5; 24 torch-dependent skips each) and the lag-definition sweep generation + re-execution | `logs/test_battery_v4.2.log`, `logs/test_battery_v4.3.log` (header line 2); the v4.4/v4.5 run counts in the RUNLOG rows (their logs are deliberately uncommitted — see the R7-1 note below); sweep determinism in the RUNLOG v4.2/v4.3 rows |

Notes, stated to the same standard:

* Environments 2 and 3 are verification environments. The only non-log
  artefacts they produced are `outputs/standard_metrics.json`
  (regenerated under environment 2 at v4.1) and
  `outputs/lag_definition_sweep.json` (environment 3; byte-identical on
  re-execution in the referee's independent torch-less environment).
* The v4.1 regeneration of the two partition figures ran in the venv
  interpreter that carries matplotlib; its exact library versions are
  not separately logged, and no printed claim in the paper rests on
  them. The canonical full-battery record remains the v4.1 log (219/219);
  the torch-less runs document the same skip delta the R-FS9-R3 panel's
  own fresh run showed.
* Since v4.4 (Dossier R-FS9-R4, P2) the battery's default write path is
  the versioned `logs/test_battery_<BATTERY_VERSION>.log`, and a run
  refuses to overwrite a log that already exists unless `--force-log`
  is passed. The canonical v4.1 record at `logs/test_battery.log` is
  therefore no longer a write target at all, and a documented
  verification re-run in a fresh clone can no longer shadow a committed
  log — the defect the panel reproduced in rounds 3 and 4.
* Since v4.5 (Dossier R-FS9-R5, R7-1) the default-path log is
  additionally UNCOMMITTED — `logs/test_battery_v*.log` is gitignored,
  and the v4.4 log was un-shipped — because a log shipped at the
  default write path made every fresh clone's FIRST battery run exit 1
  at the guard. Current-version run records therefore live in the
  RUNLOG; the frozen historical logs (canonical v4.1, v4.2, v4.3) stay
  committed and are never write targets.
* The v4.4 and v4.5 battery re-runs changed nothing in the apparatus:
  224/0 with the same 24 skips, per-file counts identical to the v4.3
  log, same environment as row 3. They exist to validate the guard
  fix (v4.4) and its R7-1 correction (v4.5) and to record the
  post-closure battery state; no paper number rests on them.
