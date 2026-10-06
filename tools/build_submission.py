#!/usr/bin/env python3
"""
tools/build_submission.py — regenerate the journal submission FROM
THE TRIMMED SUBMISSION EDITION (paper_trimmed/).

v4.12 (the independent external re-review's packaging item): the
submission edition is the TRIMMED paper — the re-reviewer's
ship-list structure (provenance and leakage, the leakage-free fold,
persistence baselines, the two BatchNorm findings, calibration
failure), 12 pages, no process language. The full 65-page edition
remains the extended record at paper/ (both editions are kept, by
owner decision). The submission is a VERBATIM COPY of paper_trimmed/
(main.tex, sections/, figures/, refs.bib) plus a tectonic compile:

  * there are NO declared transformations any more — the trimmed
    source IS the submission manuscript (the v4.6-v4.11 T1-T4
    transformations applied to the full record and are retired
    with it; their history lives in the full edition's appendix
    version table and the RUNLOG);
  * the version stamp is single-sourced from the battery's
    BATTERY_VERSION (the R-FS9-R11 P1 root-cause fix, kept): it is
    written to submission/README.md's build line at generation
    time, never into the manuscript itself (a publishable edition
    carries no version strings);
  * tests/test_submission_apparatus.py regenerates the source tree
    into a temp dir and byte-compares it with the committed
    submission/src, so drift between the record and the submission
    remains structurally impossible.

Usage: python tools/build_submission.py [--build|--source-only]
          [--outdir DIR]  (default: <repo>/submission; the battery's
          apparatus test regenerates into a temp dir and
          byte-compares)
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "paper_trimmed")     # the submission record
SUB = os.path.join(ROOT, "submission")
SUB_README = os.path.join(SUB, "README.md")


def record_version():
    """R-FS9-R11 P1 (root cause, the N2 lesson applied to the
    generator): the build stamp is single-sourced from the battery's
    BATTERY_VERSION constant — the repository's version register,
    bumped with every release — instead of being hardcoded anywhere.
    (At v4.12 the stamp moved from the manuscript's title page to
    the submission README's build line: a publishable edition
    carries no version strings.)"""
    path = os.path.join(ROOT, "tests", "run_battery.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    m = re.search(r'BATTERY_VERSION\s*=\s*"([^"]+)"', src)
    if not m:
        raise SystemExit(
            "BATTERY_VERSION not found in tests/run_battery.py — the "
            "build stamp is single-sourced from it (R-FS9-R11 P1); "
            "refusing to emit an unversioned build")
    return m.group(1)


RECORD_VERSION = record_version()


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def copy_bytes(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)


def stamp_readme():
    """Keep the submission README's build line current (the P1
    single-source lesson applied to the README stamp)."""
    m = re.compile(r"current build: \*\*v[0-9.]+\*\*")
    if not os.path.exists(SUB_README):
        return
    with open(SUB_README, encoding="utf-8") as f:
        text = f.read()
    new_line = f"current build: **{RECORD_VERSION}**"
    if f"current build: **{RECORD_VERSION}**" in text:
        return
    if m.search(text):
        text = m.sub(new_line, text, count=1)
        with open(SUB_README, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  - submission/README.md build line stamped {new_line}")
    else:
        print("  - WARNING: no 'current build' line found in "
              "submission/README.md to stamp")


def main():
    build_pdf = "--source-only" not in sys.argv
    outdir = SUB
    if "--outdir" in sys.argv:
        outdir = os.path.abspath(
            sys.argv[sys.argv.index("--outdir") + 1])
    SRC = os.path.join(outdir, "src")
    report = []

    # ── 0. clean slate ─────────────────────────────────────────────
    if os.path.isdir(SRC):
        shutil.rmtree(SRC)
    os.makedirs(os.path.join(SRC, "sections"))
    os.makedirs(os.path.join(SRC, "figures"))
    report.append("source: paper_trimmed/ (the trimmed submission "
                  "edition — v4.12, the external re-review's "
                  "ship-list structure)")

    # ── 1. verbatim copies, byte-verified ──────────────────────────
    copy_bytes(os.path.join(SOURCE, "main.tex"),
               os.path.join(SRC, "main.tex"))
    sec_dir = os.path.join(SOURCE, "sections")
    for fn in sorted(os.listdir(sec_dir)):
        if fn.endswith(".tex"):
            copy_bytes(os.path.join(sec_dir, fn),
                       os.path.join(SRC, "sections", fn))
    copy_bytes(os.path.join(SOURCE, "refs.bib"),
               os.path.join(SRC, "refs.bib"))
    fig_dir = os.path.join(SOURCE, "figures")
    n_figs = 0
    for fn in sorted(os.listdir(fig_dir)):
        copy_bytes(os.path.join(fig_dir, fn),
                   os.path.join(SRC, "figures", fn))
        n_figs += 1

    # byte-identity assertions (the drift discipline, kept)
    for base in (SOURCE, ):
        for rel in ("main.tex", "refs.bib"):
            if sha(os.path.join(base, rel)) != \
                    sha(os.path.join(SRC, rel)):
                raise AssertionError(f"copy drift: {rel}")
        for fn in sorted(os.listdir(os.path.join(base, "sections"))):
            if fn.endswith(".tex") and \
                    sha(os.path.join(base, "sections", fn)) != \
                    sha(os.path.join(SRC, "sections", fn)):
                raise AssertionError(f"copy drift: sections/{fn}")
    report.append(f"verbatim copies byte-verified: main.tex + "
                  f"{len([f for f in os.listdir(sec_dir) if f.endswith('.tex')])} "
                  f"section files + refs.bib + {n_figs} figures "
                  "(NO declared transformations — the trimmed source "
                  "IS the submission)")

    if outdir == SUB:
        stamp_readme()

    # ── 2. build PDF ──────────────────────────────────────────────
    if build_pdf:
        proc = subprocess.run(
            ["tectonic", "main.tex"], cwd=SRC,
            capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout[-3000:])
            print(proc.stderr[-3000:])
            raise AssertionError("tectonic build failed")
        pdf_src = os.path.join(SRC, "main.pdf")
        pdf_dst = os.path.join(outdir, "main.pdf")
        shutil.move(pdf_src, pdf_dst)
        report.append(f"main.pdf built (tectonic) -> submission/main.pdf "
                      f"(sha256 {sha(pdf_dst)[:16]}...)")

    print("submission regenerated from paper_trimmed/ "
          f"({RECORD_VERSION}):")
    for r in report:
        print("  -", r)
    print("OK")


if __name__ == "__main__":
    main()
