#!/usr/bin/env python3
"""
tests/test_submission_apparatus.py — R-FS9-R6 register item 4: the
submission manuscript is regenerated from the record and CANNOT drift.

v4.12 (the external re-review's packaging item): the submission is
now the TRIMMED edition (paper_trimmed/ — the ship-list structure,
12 pp, no process language), regenerated verbatim by
tools/build_submission.py; the full 65-page edition remains the
extended record at paper/ (page-pinned here as before). The
regeneration byte-compare, the refs never-retype rule, the floor
apparatus, the prior-art citations, and the rendered-bibliography
pins all remain; the SCAFFOLD disclosure checks now run against
the FULL edition's sources (the disclosures' record — the trimmed
edition carries SCAFFOLD only as the limitations qualifier), and
the rendered-bibliography pins are per-edition (the trimmed
refs.bib is the cited-keys subset: 20 entries).

Checks:
  1. regeneration byte-compare: tools/build_submission.py
     --source-only into a temp dir == committed submission/src;
  2. refs.bib is a byte-identical copy of paper_trimmed/refs.bib
     (itself the cited-keys subset of paper/refs.bib) and every
     \\cite key resolves (the never-retype rule, executable);
  3. floor apparatus present (TSS, HSS, inertia, persistence, the
     arms-clearing statement, Brier skill) + the v4.12 parity
     wording on the abstract pair;
  4. SCAFFOLD disclosures present in the FULL edition's sources
     (five-departure enumeration, sign inversion, cold start,
     optimiser constants, prevalence asymmetry) and the trimmed
     edition's limitations qualifier carries the status honestly;
  5. the six prior-art citations are cited; no GIC-framing keys
     (bolduc/baker are not even in refs.bib);
  6. no untraceable constants (74-hour CPU estimate, 4.4e-4), the
     5e-3 fp16 bound present;
  7. RENDERED bibliography pinned (Dossier R-FS9-R7, register item 1
     — B1 + B7): pdftotext extracts the compiled pages of BOTH the
     paper of record and the submission; per edition the entry count
     must equal that edition's refs.bib count, every rendered entry
     must carry a year, the page count is pinned, and each pinned
     entry must contain its title's first words and its year.

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
TRIMMED = os.path.join(ROOT, "paper_trimmed")

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
# v4.12: paper 65 pp (the v4.12 review-response edits — the SCAFFOLD
# consolidation appendix + the rewritten validation-story passage —
# reflowed the full edition from 63); submission 12 pp (the trimmed
# edition IS the submission since v4.12 — 63 while it mirrored the
# full record, v4.6-v4.11).
PINNED_PAGE_COUNTS = {
    os.path.join("paper", "main.pdf"): 65,
    os.path.join("submission", "main.pdf"): 12,
}


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


def paper_sources_text():
    """The FULL edition's sources (paper/) — the record for the
    disclosures the trimmed submission carries only in qualified
    summary form."""
    parts = [read(os.path.join(PAPER, "main.tex"))]
    secdir = os.path.join(PAPER, "sections")
    for fn in sorted(os.listdir(secdir)):
        if fn.endswith(".tex"):
            parts.append(read(os.path.join(PAPER, "sections", fn)))
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
    record_bib = read(os.path.join(TRIMMED, "refs.bib"))
    if bib != record_bib:
        print("[FAIL] submission refs.bib is NOT byte-identical to "
              "paper_trimmed/refs.bib — the never-retype rule is broken")
        ok = False
    full_bib = read(os.path.join(PAPER, "refs.bib"))
    # the trimmed bib must be a subset of the full record's entries
    for m in re.finditer(r"@\w+\{([^,\s]+),", bib):
        if f"{{{m.group(1)}," not in full_bib:
            print(f"[FAIL] trimmed-bib entry {m.group(1)} is not in "
                  "paper/refs.bib — the subset contract is broken")
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
    print(f"[PASS] refs.bib byte-identical to paper_trimmed/refs.bib "
          f"({len(bib_keys)} entries, all present in paper/refs.bib); "
          f"all {len(keys)} cited keys resolve; no GIC keys")
    return True


def check_floor_apparatus():
    src = all_sources_text()
    markers = {
        "TSS": "TSS",
        "HSS": "HSS",
        "inertia": "inertia",
        "persistence floor": "persistence",
        "Brier skill": "Brier skill",
        "arms-clearing statement": "clear 24-hour-lagged "
                                   "persistence",
        "abstract parity wording (v4.12)":
            "matching the centralised MLP",
    }
    missing = [label for label, m in markers.items() if m not in src]
    if missing:
        for label in missing:
            print(f"[FAIL] floor-apparatus marker missing: {label}")
        return False
    print("[PASS] floor apparatus present: TSS/HSS/inertia/persistence "
          "floors, Brier skill, the arms-clearing statement, and the "
          "v4.12 parity wording on the abstract pair (matching the "
          "centralised MLP within recorded retrain "
          "nondeterminism)")
    return True


def check_scaffold_disclosures():
    # v4.12: the full five-departure disclosures live in the FULL
    # edition (paper/) — the trimmed submission carries SCAFFOLD only
    # as the limitations qualifier, which must state the status
    # honestly (sign-inverted departure + not the reference algorithm
    # + implementation-case).
    src = paper_sources_text()
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
    sub = all_sources_text()
    sub_markers = {
        "trimmed: sign inversion named": "sign-inverted",
        "trimmed: not the reference algorithm":
            "departs from the reference algorithm",
        "trimmed: implementation-case qualifier": "implementation-case",
    }
    missing += [f"({label})" for label, m in sub_markers.items()
                if re.sub(r"\s+", " ", m) not in re.sub(r"\s+", " ", sub)]
    if missing:
        for label in missing:
            print(f"[FAIL] SCAFFOLD-disclosure marker missing: {label}")
        return False
    print("[PASS] full edition carries both SCAFFOLD disclosures + "
          "five-departure enumeration + corrected scheduler cell + "
          "central-side prevalence disclosure; the trimmed edition's "
          "limitations qualifier states the implementation status "
          "honestly")
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
    bibliography of both compiled PDFs. v4.12: the pins are
    PER-EDITION — the full edition pins its complete refs.bib; the
    trimmed submission pins its cited-keys subset (the pins whose
    entries the trimmed refs.bib carries), so the full-pin contract
    holds for both editions independently."""
    if shutil.which("pdftotext") is None:
        print("[FAIL] pdftotext (poppler-utils) is not available — it "
              "is a battery dependency since v4.7 (Dossier R-FS9-R7 "
              "B1): the rendered-bibliography check cannot run; "
              "install poppler-utils")
        return False

    def _norm_src(s):
        return re.sub(r"[^a-z0-9]", "", s.lower())

    # v4.12: match pins to bib KEYS via the full record's per-entry
    # text, by TITLE fragment (unique across the bibliography — the
    # citation-order anchors like "tian li" appear in neither the raw
    # entry nor the same order there), then keep the pins whose key
    # the trimmed refs.bib actually carries.
    full_bib_text = read(os.path.join(PAPER, "refs.bib"))
    full_entries = {}
    for chunk in re.split(r"(?=@\w+\{)", full_bib_text):
        m = re.match(r"@(\w+)\{([^,\s]+),", chunk)
        if m:
            full_entries[m.group(2)] = chunk
    sub_keys = set(re.findall(r"@\w+\{([^,\s]+),",
                              read(os.path.join(SUB_SRC, "refs.bib"))))
    sub_pins = []
    for p in PINNED_ENTRIES:
        matched = [k for k, txt in full_entries.items()
                   if _norm_src(p[2]) in _norm_src(txt)]
        if len(matched) == 1 and matched[0] in sub_keys:
            sub_pins.append(p)
    expected_sub = len(sub_keys)

    ok = True
    per_label_ok = {}
    for label, rel, bib_path, pins in (
            ("paper of record", os.path.join("paper", "main.pdf"),
             os.path.join(PAPER, "refs.bib"), PINNED_ENTRIES),
            ("submission", os.path.join("submission", "main.pdf"),
             os.path.join(SUB_SRC, "refs.bib"), sub_pins)):
        expected = len(re.findall(r"@\w+\{", read(bib_path)))
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
                  f"its refs.bib carries {expected} — entries were lost "
                  f"or gained at render time")
            ok = False
            continue
        for i, e in enumerate(entries, 1):
            if not re.search(r"\b(19|20)\d{2}\b", e):
                print(f"[FAIL] {label}: entry [{i}] renders no year — "
                      f"truncated (the B1 species)? "
                      f"{e[:80].strip()!r}")
                ok = False
        for why, anchor, title_frag, year in pins:
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
        # the full-pin contract, per edition: the pin list is the
        # COMPLETE bibliography of that edition
        if len(pins) != expected:
            print(f"[FAIL] {label}: pin list carries "
                  f"{len(pins)} pins but its refs.bib has "
                  f"{expected} entries — the full-pin contract (R8 "
                  f"C3) is broken; add the missing pin deliberately")
            ok = False
        if ok:
            per_label_ok[label] = (expected, pages)
    # R-FS9-R8 (T1): ONE consolidated [PASS] line for the whole check
    if ok and per_label_ok:
        parts = ", ".join(f"{k}: {v[0]} entries / {v[1]} pp"
                          for k, v in per_label_ok.items())
        print(f"[PASS] rendered bibliography ({parts}): every entry "
              f"carries a year, page counts pinned, and every entry "
              f"of each edition's refs.bib is pinned by anchor + "
              f"title fragment + year (the R8 C3 full-pin contract, "
              f"per edition) — the B1 truncation species is guarded "
              f"at render time for both editions")
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
