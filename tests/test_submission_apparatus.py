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
     5e-3 fp16 bound present, abstract pair carries metric names;
  7. RENDERED bibliography pinned (Dossier R-FS9-R7, register item 1 —
     B1 + B7): pdftotext extracts the compiled pages of BOTH the
     paper of record and the submission; the entry count must equal
     the refs.bib entry count, every rendered entry must carry a
     year (a truncated entry — the B1 species, a bare author list —
     carries none), the page count is pinned, and each of a pinned
     list of load-bearing entries ([7] Georgoulis, [22] Hassani, the
     six prior-art entries, and the Fu/Leka/Karimireddy anchors) must
     contain its title's first words and its year. Source
     byte-identity alone guaranteed fidelity to the record INCLUDING
     the record's own rendering defect — the rendered output is now
     checked directly. pdftotext (poppler-utils) is a battery
     dependency since v4.7; its absence fails this check loudly.

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

# Dossier R-FS9-R7 register item 1: pinned load-bearing entries of the
# RENDERED bibliography. Each tuple: (why load-bearing, anchor fragment
# unique to the entry, the title's first words, the year). Matching is
# case-insensitive and whitespace-normalised, so renumbering cannot
# break the pin — content is what is pinned, not position.
#
# Dossier R-FS9-R8 (C3, folded in at v4.8): ALL entries are pinned —
# the year-check + the eleven v4.7 pins could not catch an UNPINNED
# entry truncated AFTER its year field (the v4.7 guard left the last
# 20 entries' title/venue tails unguarded). The pin list length is
# asserted against the refs.bib entry count at check time (R8 T1),
# so a future entry cannot be added without a deliberate pin.
# Ligature rule (the v4.7 lesson, kept): fragments avoid ff/ffi/ffl
# words — pdftotext renders them as single non-ASCII glyphs which
# _norm strips (mcmahan2017's "Efficient" is pinned via
# "learning of deep networks from decentralized" instead).
PINNED_ENTRIES = [
    ("v4.8 full pin: mcmahan2017", "mcmahan",
     "learning of deep networks from decentralized", "2017"),
    ("v4.8 full pin: li2020", "tian li",
     "federated optimization in heterogeneous networks", "2020"),
    ("B1: georgoulis2021", "bloomfield",
     "flare likelihood and region eruption forecasting", "2021"),
    ("v4.8 full pin: angryk2020", "martens",
     "multivariate time series dataset for space weather", "2020"),
    ("v4.8 full pin: lin2017", "girshick",
     "focal loss for dense object detection", "2017"),
    ("v4.8 full pin: sarkar2020", "sarkar",
     "fed-focal loss for imbalanced", "2020"),
    ("v4.8 full pin: zhang2018", "dauphin",
     "beyond empirical risk minimization", "2018"),
    ("v4.8 full pin: boteler2019", "boteler",
     "a 21st century view of the march 1989", "2019"),
    ("anchor: leka2019", "k. d. leka",
     "comparison of flare forecasting methods", "2019"),
    ("B1: hassani2025", "hassani",
     "solar flare prediction using", "2025"),
    ("v4.8 full pin: bobra2015", "couvidat",
     "vector magnetic field data with a machine-learning", "2015"),
    ("v4.8 full pin: schrijver2007", "schrijver",
     "characteristic magnetic field pattern", "2007"),
    ("v4.8 full pin: hochreiter1997", "schmidhuber",
     "long short-term memory", "1997"),
    ("v4.8 full pin: chen2016", "guestrin",
     "a scalable tree boosting system", "2016"),
    ("v4.8 full pin: chawla2002", "kegelmeyer",
     "synthetic minority over-sampling technique", "2002"),
    ("v4.8 full pin: lundberg2017", "lundberg",
     "a unified approach to interpreting model", "2017"),
    ("v4.8 full pin: yoon2019", "van der schaar",
     "time-series generative adversarial networks", "2019"),
    ("v4.8 full pin: zhao2018", "civin",
     "federated learning with non-iid data", "2018"),
    ("v4.8 full pin: kairouz2021", "kairouz",
     "advances and open problems in federated", "2021"),
    ("v4.8 full pin: rieke2020", "hancox",
     "the future of digital health", "2020"),
    ("v4.8 full pin: eskandari2024", "eskandari",
     "enhancing multivariate time series-based", "2024"),
    ("anchor: fu2023", "junfeng fu",
     "federated transfer learning", "2023"),
    ("v4.8 full pin: loshchilov2019", "hutter",
     "decoupled weight decay regularization", "2019"),
    ("v4.8 full pin: zhu2019", "ligeng zhu",
     "deep leakage from gradients", "2019"),
    ("prior art: li2021fedbn", "fedbn",
     "federated learning on non-iid features", "2021"),
    ("prior art: wang2023bn", "yanmeng wang",
     "why batch normalization damage", "2025"),
    ("prior art: guerraoui2024bn", "guerraoui",
     "overcoming the challenges of batch normalization", "2024"),
    ("prior art: bnscaffold2024", "quintana",
     "controlling the drift of batch normalization", "2024"),
    ("prior art: angryk2019", "hostetter",
     "challenges with extreme", "2019"),
    ("prior art: ahmadzadeh2021", "how to train your flare prediction model",
     "robust sampling", "2021"),
    ("anchor: karimireddy2020", "karimireddy",
     "stochastic controlled averaging", "2020"),
]

# B7 (Dossier R-FS9-R7): tectonic PDFs are not byte-reproducible
# (same size, ~67 differing bytes across rebuilds) — the available pin
# for the compiled artefact is rendered content: page count + key
# strings. Deliberately bump these when the documents legitimately
# reflow (a disclosed recompile), never silently.
PINNED_PAGE_COUNTS = {
    os.path.join("paper", "main.pdf"): 59,
    os.path.join("submission", "main.pdf"): 59,
}
# (59 since v4.9: the executed E1 diagnostic's new subsection + Table 8
# + the appendix v4.9 row reflowed the document by three pages — a
# disclosed recompile, the pin bumped deliberately per the B7
# convention; 56 at v4.8, 55 at v4.5-v4.7)


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


def _norm(s):
    """Squash to lowercase alphanumerics — immune to line-wrap
    hyphenation (LaTeX splits 'Overcoming' as 'Over-' + 'coming'),
    spacing, punctuation, and case. Pinned fragments are chosen to
    avoid ff/ffi/ffl-ligature words, which pdftotext renders as
    single non-ASCII glyphs."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _pdf_text(path):
    proc = subprocess.run(["pdftotext", "-layout", path, "-"],
                          capture_output=True, text=True)
    return proc.returncode, proc.stdout


def _bibliography_entries(text):
    """Parse the rendered bibliography from pdftotext output.

    The bibliography is the document's last [N]-numbered run: its
    first entry is the LAST line starting with [1] (earlier ones are
    in-text citations that happen to start a line). Returns
    (entries, None) or (None, reason)."""
    lines = text.splitlines()
    starts = []
    for i, line in enumerate(lines):
        m = re.match(r"^\s{0,8}\[(\d{1,3})\]\s+\S", line)
        if m:
            starts.append((int(m.group(1)), i))
    if not starts:
        return None, "no [N]-numbered entry lines found in the PDF text"
    ones = [i for n, i in starts if n == 1]
    if not ones:
        return None, "no [1] bibliography start found"
    bib = sorted(x for x in starts if x[1] >= ones[-1])
    nums = [n for n, _ in bib]
    if nums != list(range(1, len(bib) + 1)):
        return None, ("entry numbering is not the consecutive run 1..N "
                      f"(found {nums[:10]}...)")
    entries = []
    for k, (n, i) in enumerate(bib):
        j = bib[k + 1][1] if k + 1 < len(bib) else len(lines)
        entries.append("\n".join(lines[i:j]))
    return entries, None


def check_rendered_bibliography():
    """Dossier R-FS9-R7, register item 1 (B1 + B7): pin the RENDERED
    bibliography of both compiled PDFs. Source byte-identity alone
    faithfully inherited the record's own rendering defect through
    two guarded compiles; the rendered output is now checked
    directly, so a bib entry that stops rendering fails the battery."""
    if shutil.which("pdftotext") is None:
        print("[FAIL] pdftotext (poppler-utils) is not available — it "
              "is a battery dependency since v4.7 (Dossier R-FS9-R7 "
              "B1): the rendered-bibliography check cannot run; "
              "install poppler-utils")
        return False
    bib_src = read(os.path.join(SUB_SRC, "refs.bib"))
    expected = len(re.findall(r"@\w+\{", bib_src))
    ok = True
    per_label_ok = {}
    for label, rel in (("paper of record", os.path.join("paper", "main.pdf")),
                       ("submission", os.path.join("submission", "main.pdf"))):
        pdf = os.path.join(ROOT, rel)
        rc, text = _pdf_text(pdf)
        if rc != 0 or not text:
            print(f"[FAIL] {label}: pdftotext failed on {rel}")
            ok = False
            continue
        pages = text.count("\f")
        if pages != PINNED_PAGE_COUNTS[rel]:
            print(f"[FAIL] {label}: rendered page count {pages} != pinned "
                  f"{PINNED_PAGE_COUNTS[rel]} — the document reflowed; "
                  f"bump the pin deliberately (Dossier R-FS9-R7 B7) "
                  f"with a disclosed recompile")
            ok = False
        entries, err = _bibliography_entries(text)
        if entries is None:
            print(f"[FAIL] {label}: {err}")
            ok = False
            continue
        if len(entries) != expected:
            print(f"[FAIL] {label}: {len(entries)} entries render, "
                  f"refs.bib carries {expected} — entries were lost or "
                  f"gained at render time")
            ok = False
            continue
        for i, e in enumerate(entries, 1):
            if not re.search(r"\b(19|20)\d{2}\b", e):
                print(f"[FAIL] {label}: entry [{i}] renders no year — "
                      f"truncated (the B1 species)? "
                      f"{e[:80].strip()!r}")
                ok = False
        for why, anchor, title_frag, year in PINNED_ENTRIES:
            hits = [e for e in entries
                    if _norm(anchor) in _norm(e)
                    and _norm(title_frag) in _norm(e)
                    and year in _norm(e)]
            if len(hits) != 1:
                print(f"[FAIL] {label}: pinned entry ({why}) matched "
                      f"{len(hits)} rendered entries — need exactly 1 "
                      f"(anchor {anchor!r}, title {title_frag!r}, "
                      f"year {year})")
                ok = False
        # R-FS9-R8 (C3/T1): the pin list is the COMPLETE bibliography —
        # its length is asserted against the refs.bib entry count, so
        # every entry's title/venue tail is guarded and a future entry
        # cannot be added without a deliberate pin.
        if len(PINNED_ENTRIES) != expected:
            print(f"[FAIL] {label}: PINNED_ENTRIES carries "
                  f"{len(PINNED_ENTRIES)} pins but refs.bib has "
                  f"{expected} entries — the full-pin contract (R8 "
                  f"C3) is broken; add the missing pin deliberately")
            ok = False
        if ok:
            per_label_ok[label] = (expected, pages)
    # R-FS9-R8 (T1): ONE consolidated [PASS] line for the whole check
    # (the v4.7 form printed one per PDF while the RESULT count treated
    # the check as one — the [PASS]-line count and the module's RESULT
    # line now agree).
    if ok and per_label_ok:
        parts = ", ".join(f"{k}: {v[0]} entries / {v[1]} pp"
                          for k, v in per_label_ok.items())
        print(f"[PASS] rendered bibliography ({parts}): every entry "
              f"carries a year, page counts pinned, and all "
              f"{len(PINNED_ENTRIES)} entries of refs.bib are pinned "
              f"by anchor + title fragment + year (the R8 C3 full-pin "
              f"contract) — the B1 truncation species is guarded at "
              f"render time for the whole bibliography, not just the "
              f"load-bearing subset")
    return ok


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
        check_rendered_bibliography(),
    ]
    passed = sum(1 for r in results if r)
    failed = sum(1 for r in results if not r)
    print(f"RESULT: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
