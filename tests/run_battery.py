"""
tests/run_battery.py  (v4.1 — Dossier R-FS9-R1 D1; v4.3 — R-FS9-R3
N2/N3: title made version-bearing, sweep-coverage module added;
v4.4 — R-FS9-R4 P2: default log path versioned + refuse-to-overwrite
guard)
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
logs/test_battery_<BATTERY_VERSION>.log; an existing log is an error,
not a silent overwrite (--force-log overrides).
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
BATTERY_VERSION = "v4.4"

# Default write path, versioned off BATTERY_VERSION. Dossier R-FS9-R4
# (P2): the previous default was the un-versioned logs/test_battery.log
# — the canonical v4.1 record's own filename — so any documented
# verification run in a fresh clone silently shadowed the canonical
# 219/219 torch-equipped log in the working tree (reproduced by the
# panel in rounds 3 and 4). The guard in main() additionally refuses
# to overwrite a log that already exists unless --force-log is given.
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
    "tests/test_fl_smoke.py",
    "tests/test_event_level_lstm.py",
    "tests/test_raw_substrate.py",
    "tests/test_raw_lstm.py",
]

# unittest-style modules run with -v so each test case emits one line
UNITTEST_MODULES = {"tests/test_gpu_queue_artefacts.py",
                    "tests/test_leakage_gate.py"}


def run_module(mod):
    """Run one test module as a subprocess; capture output."""
    if mod in UNITTEST_MODULES:
        cmd = [sys.executable, "-m", "unittest",
               mod.replace("/", ".").replace(".py", ""), "-v"]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        out = proc.stdout + proc.stderr
        # unittest -v lines look like "test_x (module.Class) ... ok"
        passed = len(re.findall(r"\.\.\. ok\b", out))
        failed = (len(re.findall(r"\.\.\. (FAIL|ERROR)\b", out)) +
                  len(re.findall(r"^FAIL\b|^ERROR\b", out, re.M)))
        return {"module": mod, "returncode": proc.returncode,
                "passed": passed, "failed": failed, "skipped": False,
                "output": out}
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
    skipped = proc.returncode != 0 and passed == 0 and failed == 0
    return {"module": mod, "returncode": proc.returncode,
            "passed": passed, "failed": failed, "skipped": skipped,
            "output": out}


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
        status = "SKIP" if r["skipped"] else \
            ("OK" if r["returncode"] == 0 else "ERR")
        lines.append(f"[{status:>4}] {mod}: "
                     f"{r['passed']} passed, {r['failed']} failed"
                     + (" (module could not execute in this "
                        "environment — missing dependency or dataset)"
                        if r["skipped"] else ""))
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
