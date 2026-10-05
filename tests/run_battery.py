"""
tests/run_battery.py  (v4.1 — Dossier R-FS9-R1 D1; v4.3 — R-FS9-R3
N2/N3: title made version-bearing, sweep-coverage module added;
v4.4 — R-FS9-R4 P2: default log path versioned + refuse-to-overwrite
guard; v4.5 — R-FS9-R5 R7-1: default-path log deliberately NOT
committed — the path is gitignored, so a fresh clone's first run
succeeds; v4.6 — R-FS9-R6: six new guards closing the round's
defect classes — test_import_graph (AST import resolution in every
environment, R8-1), test_scaffold_algebra (the shipped control-
variate algebra + cold-start record pinned, R8-2/R8-3),
test_central_criterion (B13 plumbing reaches every trainer, R8-5),
test_interpretability_artifact (artefact integrity + fallback
contract, R8-11), test_submission_apparatus (the regenerated
submission cannot drift, register item 4), and test_letters_manifest
(response letters' diff manifests machine-checked — the R2-F1 /
R3-N1 / R4-P1 drift species, ended); v4.7 — R-FS9-R7: crashed-guard
detection — a module that prints neither PASS/FAIL/RESULT lines nor
an explicit [SKIP] declaration is a battery FAILURE, not a dataset
skip (recommendation 2 / B3), and the apparatus test now pins the
RENDERED bibliography of both PDFs (register item 1 / B1); v4.8 —
R-FS9-R8 (C1, post-closure fold-in): a NON-ZERO EXIT is a battery
failure whatever its verdict lines say — the PASS-then-die and
SKIP-then-die escapes are closed, tracebacks are no longer
suppressed for crashed modules, and the dead second crash disjunct
is removed (T1) — plus the new module tests/test_raw_bn_diagnostic.py
(the v4.8 / E1 raw-substrate BN-diagnostic guard: static contract
pins everywhere, owner-side artefact validation once
outputs/raw_bn_diagnostic.json exists)
─────────────────────────────────────────
Single entry point for the integrity battery. Runs every test module,
aggregates PASS/FAIL across all of them, prints ONE total, and
writes a versioned log, logs/test_battery_<BATTERY_VERSION>.log
(since v4.4 / R-FS9-R4 P2 the battery never writes the un-versioned
canonical v4.1 filename logs/test_battery.log, and never silently
overwrites a log that already exists — pass --force-log to do so
deliberately). The paper cites exactly this number — the
reconciliation of the previous 78/58/66/142 four-way contradiction.
The log header records the interpreter and (when importable) the
torch/numpy versions, so the verification environment is an
artefact-backed claim (R-FS9-R1 C5).

Usage:
    python tests/run_battery.py
Exit code 0 iff every check passes. The default log is the versioned
logs/test_battery_<BATTERY_VERSION>.log — uncommitted by design
(R-FS9-R5 R7-1: a log at the default write path made every fresh
clone's first documented run exit 1 at the guard); the path is
gitignored, historical logs stay frozen, and an existing log is an
error, not a silent overwrite (--force-log overrides).
"""

import os
import re
import subprocess
import sys
import io
import contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Single source of truth for the log's title line. Dossier R-FS9-R3
# (N2) caught a v4.2-era log whose title still read "(v4.1)" because
# this string was hardcoded; bump BATTERY_VERSION with every release.
BATTERY_VERSION = "v4.8"

# Default write path, versioned off BATTERY_VERSION. Dossier R-FS9-R4
# (P2): the previous default was the un-versioned logs/test_battery.log
# — the canonical v4.1 record's own filename — so any documented
# verification run in a fresh clone silently shadowed the canonical
# 219/219 torch-equipped log in the working tree (reproduced by the
# panel in rounds 3 and 4). The guard in main() additionally refuses
# to overwrite a log that already exists unless --force-log is given.
# Dossier R-FS9-R5 (R7-1): the v4.4 batch SHIPPED a log at this path,
# so every fresh clone's first battery run hit the guard; the fix is
# to keep the default-path log out of git (.gitignore covers it) —
# the run record lives in the RUNLOG and the historical logs stay
# frozen where they are.
LOG = os.path.join(ROOT, "logs",
                   f"test_battery_{BATTERY_VERSION}.log")

# Ordered battery (torch-free synthetic tests first; runners that need
# the dataset/torch are guarded by availability and reported as SKIPPED
# with the reason — never silently).
MODULES = [
    "tests/test_pipeline_integrity.py",
    "tests/test_audit_artifact.py",
    "tests/test_lag_sweep_artifact.py",
    "tests/test_gpu_queue.py",
    "tests/test_gpu_queue_artefacts.py",
    "tests/test_leakage_gate.py",
    "tests/test_import_graph.py",
    "tests/test_scaffold_algebra.py",
    "tests/test_central_criterion.py",
    "tests/test_interpretability_artifact.py",
    "tests/test_submission_apparatus.py",
    "tests/test_letters_manifest.py",
    "tests/test_fl_smoke.py",
    "tests/test_event_level_lstm.py",
    "tests/test_raw_substrate.py",
    "tests/test_raw_lstm.py",
    "tests/test_raw_bn_diagnostic.py",
]

# unittest-style modules run with -v so each test case emits one line
UNITTEST_MODULES = {"tests/test_gpu_queue_artefacts.py",
                    "tests/test_leakage_gate.py"}


def run_module(mod):
    """Run one test module as a subprocess; capture output.

    Dossier R-FS9-R7 (B3) classification contract, amended at v4.8 by
    Dossier R-FS9-R8 (C1):
      * PASS/FAIL lines and RESULT lines are counted as before;
      * a module that skips MUST declare it itself with a line
        starting '[SKIP]' AND exit cleanly — a declared skip from a
        module that then dies no longer keeps the battery green;
      * a NON-ZERO EXIT is a battery failure whatever its verdict
        lines say (C1: a module that printed one [PASS] and then
        crashed used to report [ ERR] with zero failures counted
        while the battery stayed green, and the crash-after-skip
        branch suppressed the traceback — both escapes are closed).
        A module that already reported failures through its verdict
        lines is counted through those lines (status ERR, no double
        count) and its output tail is shown;
      * a module that produces NO verdict, NO declared skip and a
        ZERO exit — a silent guard — remains a battery FAILURE
        ('crashed', the v4.7 B3 rule).
    """
    if mod in UNITTEST_MODULES:
        cmd = [sys.executable, "-m", "unittest",
               mod.replace("/", ".").replace(".py", ""), "-v"]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=ROOT)
        out = proc.stdout + proc.stderr
        # unittest -v lines look like "test_x (module.Class) ... ok"
        passed = len(re.findall(r"\.\.\. ok\b", out))
        failed = (len(re.findall(r"\.\.\. (FAIL|ERROR)\b", out)) +
                  len(re.findall(r"^FAIL\b|^ERROR\b", out, re.M)))
        crashed = ((proc.returncode != 0 and failed == 0)
                   or (passed == 0 and failed == 0))
        return {"module": mod, "returncode": proc.returncode,
                "passed": passed, "failed": failed, "skipped": False,
                "crashed": crashed, "output": out}
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, mod)],
        capture_output=True, text=True, cwd=ROOT)
    out = proc.stdout + proc.stderr
    passed = failed = 0
    # count [PASS]/[FAIL] style lines
    passed = len(re.findall(r"\[PASS\]", out))
    failed = len(re.findall(r"\[FAIL\]", out))
    # also honour explicit RESULT lines some modules print
    m = re.search(r"RESULT:\s*(\d+)\s*passed,\s*(\d+)\s*failed", out)
    if m:
        passed, failed = int(m.group(1)), int(m.group(2))
    declared_skip = re.search(r"^\s*\[SKIP\]", out, re.M) is not None
    has_verdict = (passed > 0 or failed > 0 or m is not None)
    # R-FS9-R8 (C1): any non-zero exit with no reported failures is
    # a CRASH — the PASS-then-die and SKIP-then-die escapes are gone
    # (the previous second disjunct was dead: a non-zero exit with no
    # verdict and no skip was already covered by the first — R8 T1).
    crashed = ((proc.returncode != 0 and failed == 0)
               or (not has_verdict and not declared_skip))
    # a module-level skip requires a CLEAN exit and NO verdict — a
    # module that ran its checks AND printed an internal [SKIP] line
    # (a sub-check gated on torch/data) is a normal OK run whose
    # skip note is informational; a module that declared a skip and
    # then died is a crash (above), never a skip.
    skipped = (declared_skip and not has_verdict and failed == 0
               and proc.returncode == 0)
    return {"module": mod, "returncode": proc.returncode,
            "passed": passed, "failed": failed, "skipped": skipped,
            "crashed": crashed, "output": out}


def _env_header():
    """Interpreter + library versions for the log header (R-FS9-R1 C5)."""
    env = [f"python {sys.version.split()[0]}"]
    for lib in ("numpy", "torch"):
        try:
            mod = __import__(lib)
            env.append(f"{lib} {mod.__version__}")
        except Exception:
            env.append(f"{lib} not-importable-in-this-env")
    return ", ".join(env)


def main():
    if os.path.exists(LOG) and "--force-log" not in sys.argv:
        sys.exit(f"[battery] {LOG} already exists — refusing to silently "
                 "overwrite a log (R-FS9-R4 P2); move the file aside or "
                 "pass --force-log to overwrite deliberately.")
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    total_p = total_f = 0
    lines = []
    for mod in MODULES:
        if not os.path.exists(os.path.join(ROOT, mod)):
            lines.append(f"[SKIP] {mod} (file not present)")
            continue
        r = run_module(mod)
        total_p += r["passed"]
        total_f += r["failed"]
        if r["crashed"]:
            total_f += 1
            lines.append(f"[CRASH] {mod}: module exited "
                         f"rc={r['returncode']} reporting no failure "
                         f"verdicts, or produced no verdict and no "
                         f"declared [SKIP] — counted as a battery "
                         f"FAILURE (R-FS9-R7 B3 + R-FS9-R8 C1: a "
                         f"crashed or silent guard fails the "
                         f"battery whatever it printed)")
            tail = "\n".join(r["output"].splitlines()[-15:])
            lines.append("       last output:\n" +
                         "\n".join("         " + l
                                    for l in tail.splitlines()))
            continue
        status = "SKIP" if r["skipped"] else \
            ("OK" if r["returncode"] == 0 else "ERR")
        skip_note = ""
        m_skip = re.search(r"^\s*(\[SKIP\].*)$", r["output"], re.M)
        if r["skipped"]:
            reason = m_skip.group(1).strip() if m_skip else ""
            skip_note = f" — {reason}" if reason else \
                " (module declared a skip)"
        elif m_skip:
            # informational: the module ran its checks; one sub-check
            # is internally gated on torch/data
            skip_note = f" (internal gated check: " \
                        f"{m_skip.group(1).strip()})"
        lines.append(f"[{status:>4}] {mod}: "
                     f"{r['passed']} passed, {r['failed']} failed"
                     + skip_note)
        if r["returncode"] != 0 and not r["skipped"]:
            tail = "\n".join(r["output"].splitlines()[-15:])
            lines.append("       last output:\n" +
                         "\n".join("         " + l for l in tail.splitlines()))
    verdict = "PASS" if total_f == 0 else "FAIL"
    summary = (f"RESULT: {total_p} passed, {total_f} failed — "
               f"battery verdict {verdict}")

    with open(LOG, "w") as f:
        f.write(f"integrity battery — single authoritative count "
                f"({BATTERY_VERSION})\n")
        f.write("environment: " + _env_header() + "\n")
        f.write("=" * 60 + "\n")
        f.write("\n".join(lines) + "\n")
        f.write("=" * 60 + "\n")
        f.write(summary + "\n")
    print("\n".join(lines))
    print("=" * 60)
    print(summary)
    print(f"[battery] log -> {LOG}")
    sys.exit(0 if total_f == 0 else 1)


if __name__ == "__main__":
    main()
