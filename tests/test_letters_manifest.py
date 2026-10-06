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
without the marker whose filename version predates v4.6 (the
pre-v4.6 archive) are reported as such — their drift was handled by
the errata convention and they are frozen.

Dossier R-FS9-R7 (B2) hardening, v4.7: a filename version of v4.6 or
later WITHOUT the manifest block is a FAILURE, not a silent pass as
frozen archive — the convention's entry point is no longer author
discipline. (A registry of manifest-bearing letters, realised as the
filename version threshold the panel suggested.)

Dossier R-FS9-R8 (C2) hardening, v4.8: the registry is now an
explicit PINNED MAP (LETTER_REGISTRY below), not the filename
version regex — the regex's blind spot was the unversioned, dotless
or below-threshold filename, silently classified as frozen legacy
and passed. A .md file that is not in the registry now FAILS however
its filename reads; a registry entry whose file is missing FAILS
(deletion is loud); the registry value declares whether the manifest
block is required.

Dossier R-FS9-R10 (B6-a) hardening, v4.9.2: the walk is now
RECURSIVE. The v4.8 registry closed the flat-directory escape
(unversioned/dotless filenames) but the walk itself was still
os.listdir — a letter placed in a SUBDIRECTORY of
docs/response_letters/ escaped the loop entirely and passed without
being classified at all (the loophole the Round-13 panel verified
live). Every .md file at ANY depth under the tree is now inventoried
by relative path and must appear in the registry exactly; a
subdirectory letter fails as out-of-registry however deeply it
hides.

Graceful degradation: if .git is unavailable (e.g. a source export),
the test SKIPS with a message instead of failing (and counts zero
passes — R-FS9-R7 B9: no vacuous passes).

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


def _letter_version(fn):
    """Filename version (major, minor), or None if unversioned.
    (Kept for the failure messages only — classification is the
    registry's job since v4.8 / R-FS9-R8 C2.)"""
    m = re.search(r"_v(\d+)\.(\d+)(?:[._-]|$)", fn)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _walk_letters():
    """Recursive inventory of every .md under docs/response_letters/
    (R-FS9-R10 B6-a, v4.9.2): relative POSIX paths, depth-agnostic.
    The v4.8 walk was os.listdir — flat — so a letter in a
    SUBDIRECTORY escaped the registry classification entirely (the
    live-verified loophole). Every .md at any depth is now seen."""
    found = []
    for dirpath, dirnames, filenames in os.walk(LETTERS):
        dirnames.sort()
        for fn in sorted(filenames):
            if not fn.endswith(".md"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), LETTERS)
            found.append(rel.replace(os.sep, "/"))
    return found


# Dossier R-FS9-R8 (C2, folded in at v4.8): the registry of letters is
# an explicit PINNED MAP, not a filename-version heuristic. The v4.7
# registry keyed on the filename version regex, so an unversioned,
# dotless or mis-below-threshold filename was silently classified as
# frozen legacy and passed. Now every .md file in
# docs/response_letters/ other than README.md must appear here
# EXACTLY; a letter that is not in the registry is a FAILURE however
# its filename reads, and the registry value declares whether the
# LETTER-MANIFEST block is required (True = the v4.6+ convention,
# False = pre-v4.6 frozen archive, drift handled by the errata
# convention). Update this map DELIBERATELY when a letter is added.
LETTER_REGISTRY = {
    "RESPONSE_R-FS9-R1_v4.1_ERRATA.md": False,
    "RESPONSE_R-FS9-R2_v4.2.md": False,
    "RESPONSE_R-FS9-R3_v4.3.md": False,
    "RESPONSE_R-FS9-R5_v4.5.md": False,
    "RESPONSE_R-FS9-R6_v4.6.md": True,
    "RESPONSE_R-FS9-R7_v4.7.md": True,
    "RESPONSE_v4.8_ERRATA.md": True,
    "RESPONSE_R-FS9-R10_v4.9.2.md": True,
    "RESPONSE_R-FS9-R10_v4.9.3.md": True,
    "RESPONSE_R-FS9-R10_v4.9.4.md": True,
    "RESPONSE_R-FS9-R10_v4.10.md": True,
    "RESPONSE_R-FS9-R11_v4.11.md": True,
    "RESPONSE_External_v4.12.md": True,
    "RESPONSE_External_v4.13.md": True,
    "RESPONSE_External_v4.14.md": True,
}


def main():
    if not os.path.isdir(os.path.join(ROOT, ".git")):
        print("[SKIP] .git unavailable (source export) — letter "
              "manifests cannot be verified against history (not "
              "counted as a pass — R-FS9-R7 B9)")
        print("RESULT: 0 passed, 0 failed")
        sys.exit(0)

    checked = skipped = 0
    failures = []
    legacy = []
    unregistered = []
    out_of_registry = []
    for rel in _walk_letters():          # R-FS9-R10 B6-a: recursive
        if rel == "README.md":            # the tree's own top-level readme
            continue
        fn = rel
        path = os.path.join(LETTERS, *rel.split("/"))
        # R-FS9-R8 (C2): classification is REGISTRY-first — a file not
        # in the pinned registry fails however its filename reads
        # (the unversioned/dotless/below-threshold escapes are gone).
        if fn not in LETTER_REGISTRY:
            out_of_registry.append(fn)
            continue
        manifest_required = LETTER_REGISTRY[fn]
        parsed = parse_letter(path)
        if parsed is None:
            if manifest_required:
                # Dossier R-FS9-R7 (B2): a v4.6+ letter without the
                # manifest block used to be silently classified as
                # frozen archive — that opt-out is a failure.
                unregistered.append(fn)
            else:
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

    # R-FS9-R8 (C2): a registry entry whose file is MISSING also
    # fails — deletion of a letter is loud, symmetrical with the
    # pinned module inventory of the import-graph guard.
    missing_registered = [fn for fn in sorted(LETTER_REGISTRY)
                          if not os.path.exists(
                              os.path.join(LETTERS, fn))]
    for fn in missing_registered:
        print(f"[FAIL] {fn}: registered in LETTER_REGISTRY but the "
              f"file does not exist — a letter was deleted or renamed "
              f"(R-FS9-R8 C2); update the registry DELIBERATELY")

    # the registry check (Dossier R-FS9-R7 B2 + R-FS9-R8 C2 +
    # R-FS9-R10 B6-a): every v4.6+ letter carries the manifest block,
    # and EVERY letter file at ANY DEPTH under the tree is classified
    # by the pinned registry — a file outside it fails however its
    # filename reads, wherever it hides
    for fn in out_of_registry:
        print(f"[FAIL] {fn}: letter not in the pinned LETTER_REGISTRY "
              f"(R-FS9-R8 C2 / R-FS9-R10 B6-a) — an unregistered .md "
              f"file anywhere under docs/response_letters/ cannot pass "
              f"as frozen archive by filename shape or by hiding in a "
              f"subdirectory; update the registry DELIBERATELY when "
              f"adding a letter")
    registry_ok = (not unregistered and not out_of_registry
                   and not missing_registered)
    if unregistered:
        for fn in unregistered:
            print(f"[FAIL] {fn}: registered as v4.6+ but no "
                  f"LETTER-MANIFEST block — the convention is not "
                  f"optional for new letters; a letter cannot opt out "
                  f"by omitting the block (R-FS9-R7 B2)")
    if registry_ok:
        print(f"[PASS] letter registry: all {len(LETTER_REGISTRY)} "
              f"registered letters accounted for (present on disk, "
              f"every v4.6+ letter carrying the LETTER-MANIFEST "
              f"block, no unregistered .md file anywhere under the "
              f"tree — recursive walk, R-FS9-R10 B6-a) — R-FS9-R7 B2 "
              f"+ R-FS9-R8 C2 closed: an explicit registry-of-letters, "
              f"not a filename regex, not a flat walk)")

    n_failed = (len(failures) + len(unregistered)
                + len(out_of_registry) + len(missing_registered))
    if n_failed:
        for f in failures:
            print(f"[FAIL] {f}")
        print("       (the letter's diff manifest must enumerate the "
              "COMPLETE git diff — the R2-F1/R3-N1/R4-P1 lesson, now "
              "machine-checked)")
        passed = checked + (1 if registry_ok else 0)
        print(f"RESULT: {passed} passed, {n_failed} failed")
        sys.exit(1)

    if checked == 0 and skipped == 0:
        print("[FAIL] no letter manifest found — every v4.6+ letter "
              "must carry the LETTER-MANIFEST block")
        print("RESULT: 0 passed, 1 failed")
        sys.exit(1)

    passed = checked + (1 if registry_ok else 0)
    print(f"RESULT: {passed} passed, 0 failed")
    sys.exit(0)


if __name__ == "__main__":
    main()
