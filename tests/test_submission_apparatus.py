#!/usr/bin/env python3
"""
tests/test_submission_apparatus.py — R-FS9-R6 register item 4: the
submission manuscript is regenerated from the record and CANNOT drift.

Background (Dossier R-FS9-R6, findings R8-6/R8-7/R8-8/R8-9/R8-10):
the hand-prepared 48-page journal manuscript regressed to the
v3.9-era state — eleven defective bibliography entries (two dead
identifiers, one wrong-paper DOI), the entire skill-score floor
apparatus absent, both v4.5 SCAFFOLD disclosures missing, six
prior-art citations dropped, the retired GIC-framing citations back,
and two untraceable constants. Fourth documented instance of the
hand-prepared-artefact drift species.

v4.6 answer: the submission is GENERATED (tools/build_submission.py)
from the repository of record — sections, figures and refs.bib are
verbatim copies; deviations are declared transformations; this test
regenerates the source tree and byte-compares it with the committed
submission/src, so drift is impossible without the battery failing.

Checks:
  1. regeneration byte-compare: tools/build_submission.py
     --source-only into a temp dir == committed submission/src;
  2. refs.bib is a byte-identical copy of paper/refs.bib and every
     \\cite key resolves (the never-retype rule, executable);
  3. floor apparatus present (TSS, HSS, inertia, persistence, the
     arms-clearing statement, Brier skill);
  4. both SCAFFOLD disclosures present (five-departure enumeration,
     sign inversion, cold start, optimiser constants, prevalence
     asymmetry) + budget-table scheduler cell corrected;
  5. the six prior-art citations are cited; no GIC-framing keys
     (bolduc/baker are not even in refs.bib);
  6. no untraceable constants (74-hour CPU estimate, 4.4e-4), the
     5e-3 fp16 bound present, abstract pair carries metric names.

Run:  python tests/test_submission_apparatus.py  (or via the battery)
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SUB_SRC = os.path.join(ROOT, "submission", "src")
GEN = os.path.join(ROOT, "tools", "build_submission.py")
PAPER = os.path.join(ROOT, "paper")

SIX_KEYS = ["li2021fedbn", "wang2023bn", "guerraoui2024bn",
            "bnscaffold2024", "angryk2019", "ahmadzadeh2021"]


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def all_sources_text():
    parts = [read(os.path.join(SUB_SRC, "main.tex"))]
    secdir = os.path.join(SUB_SRC, "sections")
    for fn in sorted(os.listdir(secdir)):
        if fn.endswith(".tex"):
            parts.append(read(os.path.join(secdir, fn)))
    return "\n".join(parts)


def check_regeneration():
    tmp = tempfile.mkdtemp(prefix="subgen_")
    try:
        proc = subprocess.run(
            [sys.executable, GEN, "--source-only", "--outdir", tmp],
            capture_output=True, text=True, cwd=ROOT)
        if proc.returncode != 0:
            print("[FAIL] generator failed:\n" + proc.stderr[-1500:])
            return False
        gen_src = os.path.join(tmp, "src")
        diffs = []

        def walk(base, rel=""):
            for fn in sorted(os.listdir(os.path.join(base, rel))):
                r = os.path.join(rel, fn) if rel else fn
                p = os.path.join(base, r)
                if os.path.isdir(p):
                    walk(base, r)
                    continue
                if fn.endswith(".pdf"):
                    continue
                q = os.path.join(SUB_SRC, r)
                if not os.path.exists(q):
                    diffs.append(f"missing in committed tree: {r}")
                    continue
                with open(p, "rb") as a, open(q, "rb") as b:
                    if a.read() != b.read():
                        diffs.append(f"byte-differs: {r}")

        walk(gen_src)
        # committed files absent from a fresh generation = stale
        def walk_committed(base, rel=""):
            for fn in sorted(os.listdir(os.path.join(base, rel))):
                r = os.path.join(rel, fn) if rel else fn
                p = os.path.join(base, r)
                if os.path.isdir(p):
                    walk_committed(base, r)
                    continue
                if not fn.endswith(".pdf") and \
                        not os.path.exists(os.path.join(gen_src, r)):
                    diffs.append(f"stale committed file: {r}")
        walk_committed(SUB_SRC)

        if diffs:
            for d in diffs:
                print(f"[FAIL] regeneration mismatch: {d}")
            print("       (the record changed without regenerating the "
                  "submission — run tools/build_submission.py)")
            return False
        n = sum(len(files) for _, _, files in os.walk(gen_src))
        print(f"[PASS] submission/src regenerates byte-identically from "
              f"the record ({n} source files, PDF excluded)")
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_refs():
    ok = True
    bib = read(os.path.join(SUB_SRC, "refs.bib"))
    record_bib = read(os.path.join(PAPER, "refs.bib"))
    if bib != record_bib:
        print("[FAIL] submission refs.bib is NOT byte-identical to "
              "paper/refs.bib — the never-retype rule is broken")
        ok = False
    if "bolduc" in bib.lower() or "baker" in bib.lower():
        print("[FAIL] GIC-framing citations (bolduc/baker) present in "
              "the submission bibliography")
        ok = False
    # every \cite key must resolve
    src = all_sources_text()
    keys = set()
    for m in re.finditer(r"\\cite\{([^}]+)\}", src):
        for k in m.group(1).split(","):
            keys.add(k.strip())
    bib_keys = set(re.findall(r"@\w+\{([^,\s]+),", bib))
    unresolved = keys - bib_keys
    if unresolved:
        print(f"[FAIL] unresolved citation keys: {sorted(unresolved)}")
        ok = False
    if not ok:
        return False
    print(f"[PASS] refs.bib byte-identical to the record ({len(bib_keys)} "
          f"entries); all {len(keys)} cited keys resolve; no GIC keys")
    return True


def check_floor_apparatus():
    src = all_sources_text()
    markers = {
        "TSS": "TSS",
        "HSS": "HSS",
        "inertia": "inertia",
        "persistence floor": "persistence",
        "Brier skill": "Brier skill",
        "arms-clearing statement": "alone clear 24-hour-lagged "
                                   "persistence",
        "abstract metric names": "ROC-AUC 0.970 and TSS 0.404",
    }
    missing = [label for label, m in markers.items() if m not in src]
    if missing:
        for label in missing:
            print(f"[FAIL] floor-apparatus marker missing: {label}")
        return False
    print("[PASS] floor apparatus present: TSS/HSS/inertia/persistence "
          "floors, Brier skill, the v4.2-corrected arms-clearing "
          "statement, and metric names on the abstract pair")
    return True


def check_scaffold_disclosures():
    src = all_sources_text()
    markers = {
        "five-departure enumeration": "Five implementation details depart",
        "sign inversion": "sign-inverted",
        "cold start": "trained \\emph{cold}",
        "optimiser constants": "$0.9$ at a learning rate of "
                               "$2.5\\times10^{-4}$",
        "prevalence asymmetry": "$0.4887$",
        "implementation statement": "not the reference algorithm",
        "corrected scheduler cell": "LR schedule & constant (no "
                                    "scheduler) & cosine warm restarts",
        "central-side prevalence": "extended at v4.6 to the centralised "
                                   "comparators",
    }
    missing = [label for label, m in markers.items()
               if re.sub(r"\s+", " ", m) not in re.sub(r"\s+", " ", src)]
    if missing:
        for label in missing:
            print(f"[FAIL] SCAFFOLD-disclosure marker missing: {label}")
        return False
    print("[PASS] both SCAFFOLD disclosures + five-departure "
          "enumeration + corrected scheduler cell + central-side "
          "prevalence disclosure all present")
    return True


def check_citations_and_constants():
    ok = True
    src = all_sources_text()
    for k in SIX_KEYS:
        if f"\\cite{{{k}}}" not in src and \
                not re.search(r"\\cite\{[^}]*\b" + re.escape(k) +
                              r"\b[^}]*\}", src):
            print(f"[FAIL] prior-art citation not cited in submission: "
                  f"{k}")
            ok = False
    for bad in ["74\\,h", "estimated 74", "4.4\\times10",
                "4.4e-4", "4.4 $\\times$ 10"]:
        if bad in src:
            print(f"[FAIL] untraceable constant present in submission: "
                  f"{bad}")
            ok = False
    if "5\\times10^{-3}" not in src:
        print("[FAIL] the record's 5e-3 fp16 bound is missing")
        ok = False
    if not ok:
        return False
    print("[PASS] all six prior-art citations cited; no 74-hour CPU "
          "estimate, no 4.4e-4; the 5e-3 fp16 bound present")
    return True


def check_pdf():
    pdf = os.path.join(ROOT, "submission", "main.pdf")
    if not os.path.exists(pdf):
        print("[FAIL] submission/main.pdf is not built/committed")
        return False
    with open(pdf, "rb") as f:
        head = f.read(5)
    if head != b"%PDF-":
        print("[FAIL] submission/main.pdf is not a PDF")
        return False
    print("[PASS] submission/main.pdf present (built by the generator; "
          "rebuild with tools/build_submission.py after any paper/ "
          "change)")
    return True


def main():
    if not os.path.isdir(SUB_SRC):
        print("[FAIL] submission/src does not exist — run "
              "tools/build_submission.py")
        sys.exit(1)
    results = [
        check_regeneration(),
        check_refs(),
        check_floor_apparatus(),
        check_scaffold_disclosures(),
        check_citations_and_constants(),
        check_pdf(),
    ]
    passed = sum(1 for r in results if r)
    failed = sum(1 for r in results if not r)
    print(f"RESULT: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
