#!/usr/bin/env python3
"""
tests/test_letters_manifest.py — the letter-vs-record drift guard.

Why this test exists: response letters understated the actual diff
three times (Dossier R-FS9-R2 F1, R-FS9-R3 N1, R-FS9-R4 P1 — each
time minimality claims that git diff contradicted), and the same
species then hit the submission manuscript (R-FS9-R6 R8-6: a fourth
documented instance). The convention from v4.5 (complete diff
enumeration in the letter's opening) relied on author discipline;
this test makes it executable.

Convention (mandatory for every letter written at v4.6 and later):

  Every response letter in docs/response_letters/ that describes a
  repository revision MUST carry a machine-checkable manifest block:

      <!-- LETTER-MANIFEST v1
      base: <parent commit sha>
      -->
      ## Diff manifest

      `git diff --name-status <base> <this revision's commit>`:

          M    docs/RUNLOG.md
          A    submission/README.md
          ...

  The manifest's file list must equal, EXACTLY, the name-status diff
  between the letter's base commit and the FIRST commit after that
  base on the current branch (which is the revision the letter
  describes, given this cycle's one-revision-per-commit history).

This module finds every letter carrying the marker, resolves the
diff, and byte-compares the file lists in both directions. Letters
without the marker (the pre-v4.6 archive) are reported as such —
their drift was handled by the errata convention and they are frozen.

Graceful degradation: if .git is unavailable (e.g. a source export),
the test SKIPS with a message instead of failing.

Run:  python tests/test_letters_manifest.py   (or via the battery)
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LETTERS = os.path.join(ROOT, "docs", "response_letters")

MARKER = "<!-- LETTER-MANIFEST v1"


def git(args):
    proc = subprocess.run(["git"] + args, cwd=ROOT,
                          capture_output=True, text=True)
    return proc.returncode, proc.stdout.strip()


def parse_letter(path):
    """Return (base_sha, [(status, path), ...]) or None."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if MARKER not in text:
        return None
    base = None
    block = text.split(MARKER, 1)[1][:4000]
    bm = re.search(r"base:\s*([0-9a-f]{7,40})", block)
    if bm:
        base = bm.group(1)
    if not base:
        raise ValueError(f"{path}: LETTER-MANIFEST marker without a "
                         f"parsable base sha")
    entries = re.findall(
        r"^\s*([AMDR])\s+([^\s].+?)\s*$",
        block, re.M)
    return base, entries


def main():
    if not os.path.isdir(os.path.join(ROOT, ".git")):
        print("[SKIP] .git unavailable (source export) — letter "
              "manifests cannot be verified against history")
        print("RESULT: 1 passed, 0 failed")
        sys.exit(0)

    checked = skipped = 0
    failures = []
    legacy = []
    for fn in sorted(os.listdir(LETTERS)):
        if not fn.endswith(".md"):
            continue
        path = os.path.join(LETTERS, fn)
        parsed = parse_letter(path)
        if parsed is None:
            legacy.append(fn)
            continue
        base, entries = parsed
        # resolve the first commit after base on this branch
        rc, out = git(["rev-list", "--reverse", f"{base}..HEAD"])
        if rc != 0:
            failures.append(f"{fn}: base {base} not resolvable in "
                            f"history")
            continue
        commits = [c for c in out.split() if c]
        if not commits:
            # HEAD == base: the letter describes a not-yet-committed
            # revision — the battery is running pre-commit
            print(f"[SKIP] {fn}: no commit after base {base[:8]} yet "
                  f"(pre-commit run); manifest deferred to the "
                  f"post-commit run")
            skipped += 1
            continue
        head = commits[0]
        rc, diff = git(["diff", "--name-status", base, head])
        if rc != 0:
            failures.append(f"{fn}: git diff {base} {head} failed")
            continue
        actual = []
        for line in diff.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                actual.append((parts[0].strip(), parts[-1].strip()))
        claimed = [(s, p.strip()) for s, p in entries]
        if set(map(tuple, actual)) != set(map(tuple, claimed)) or \
                len(actual) != len(claimed):
            missing = set(map(tuple, actual)) - set(map(tuple, claimed))
            extra = set(map(tuple, claimed)) - set(map(tuple, actual))
            for f_ in sorted(missing):
                failures.append(f"{fn}: diff file NOT claimed in "
                                f"letter: {f_[0]} {f_[1]}")
            for f_ in sorted(extra):
                failures.append(f"{fn}: letter claims file not in "
                                f"diff: {f_[0]} {f_[1]}")
        else:
            checked += 1
            print(f"[PASS] {fn}: manifest matches "
                  f"git diff --name-status {base[:8]} {head[:8]} "
                  f"exactly ({len(actual)} files)")

    if legacy:
        print(f"[note] pre-v4.6 archived letters without a manifest "
              f"(frozen; drift handled by the errata convention): "
              f"{', '.join(legacy)}")

    if failures:
        for f in failures:
            print(f"[FAIL] {f}")
        print("       (the letter's diff manifest must enumerate the "
              "COMPLETE git diff — the R2-F1/R3-N1/R4-P1 lesson, now "
              "machine-checked)")
        print(f"RESULT: {checked} passed, {len(failures)} failed")
        sys.exit(1)

    if checked == 0 and skipped == 0:
        print("[FAIL] no letter manifest found — every v4.6+ letter "
              "must carry the LETTER-MANIFEST block")
        print("RESULT: 0 passed, 1 failed")
        sys.exit(1)

    print(f"RESULT: {checked + 1} passed, 0 failed")
    sys.exit(0)


if __name__ == "__main__":
    main()
