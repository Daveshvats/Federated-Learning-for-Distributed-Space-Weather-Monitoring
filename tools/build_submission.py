#!/usr/bin/env python3
"""
tools/build_submission.py — R-FS9-R6 register item 4: regenerate the
journal submission manuscript FROM THE RECORD, programmatically.

Provenance discipline (the lesson of Dossier R-FS9-R6, findings
R8-6/R8-7/R8-8/R8-9/R8-10): the previous submission manuscript was
hand-prepared outside the repository and regressed to the v3.9-era
state — eleven defective bibliography entries, the floor apparatus and
both SCAFFOLD disclosures dropped, six prior-art citations missing,
two untraceable constants. This generator makes that drift species
structurally impossible:

  * the section sources, figures, and bibliography are COPIED from
    the repository of record (paper/) — never retyped;
  * every deviation from the record is a DECLARED, machine-checked
    transformation applied in this file and asserted to match;
  * tests/test_submission_apparatus.py regenerates the source tree and
    byte-compares it with the committed submission/src, so the
    committed submission can never silently drift from the record.

Declared transformations (the complete list of deviations):
  T1  journal front matter: re-ordered title, submission-edition note,
      pdftitle metadata (title block composed here, not copied);
  T2  abstract pair carries metric names (made in the RECORD at v4.6:
      paper/main.tex now reads "ROC-AUC 0.970 and TSS 0.404"; this
      generator ASSERTS the names are present in the extracted
      abstract rather than re-editing it);
  T3  sec_experiments: the un-regenerable "estimated 74 h" CPU figure
      is dropped (R8-10) — the sentence keeps its record-backed
      content (the logged GPU hours remain in the same paragraph);
  T4  nothing else. Every other byte of every section, the entire
      refs.bib, and every figure are verbatim copies of the record.

Output: submission/src/{main.tex, sections/, figures/, refs.bib} and
submission/main.pdf (tectonic). Re-run after any paper/ change.

Usage: python tools/build_submission.py [--build|--source-only]
          [--outdir DIR]  (default: <repo>/submission; the battery's
          apparatus test regenerates into a temp dir and byte-compares)
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
SUB = os.path.join(ROOT, "submission")

JOURNAL_TITLE = ("Federated Solar-Flare Prediction on SWAN-SF: "
                 "A Provenance, Leakage, and Evaluation-Protocol Audit")
RECORD_TITLE = ("Federated Solar-Flare Prediction on SWAN-SF: A Benchmark "
                "Audit of Provenance, Leakage, and Evaluation Protocols")


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def copy_bytes(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)


def declared_replace(text, old, new, label, path):
    """Apply one declared transformation; assert it matched exactly once."""
    n = text.count(old)
    if n != 1:
        raise AssertionError(
            f"[{label}] declared transformation target appears {n}x "
            f"(expected 1x) in {path} — the record moved; update the "
            f"declared transformation list in tools/build_submission.py")
    return text.replace(old, new)


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

    record_main = read(os.path.join(PAPER, "main.tex"))

    # ── 1. preamble from the record + T1 metadata ─────────────────
    preamble = record_main.split("\\begin{document}")[0]
    preamble = declared_replace(
        preamble,
        "Federated Solar-Flare Prediction on SWAN-SF: A Benchmark\n"
        "              Audit of Provenance, Leakage, and Evaluation Protocols",
        "Federated Solar-Flare Prediction on SWAN-SF: A Provenance,\n"
        "              Leakage, and Evaluation-Protocol Audit",
        "T1-pdftitle", "paper/main.tex (pdftitle)")
    preamble = declared_replace(
        preamble,
        "\\title{Federated Solar-Flare Prediction on SWAN-SF: A Benchmark "
        "Audit\nof Provenance, Leakage, and Evaluation Protocols}",
        "\\title{" + JOURNAL_TITLE + "}",
        "T1-title", "paper/main.tex (\\title)")
    report.append("T1: journal title/metadata applied to preamble "
                  "(re-ordered submission title)")

    # ── 2. abstract from the record + T2 assertion ────────────────
    m = re.search(r"\\textbf\{Abstract ---\}\n(.*?)\n\\vspace\{0\.55em\}",
                  record_main, re.S)
    if not m:
        raise AssertionError("abstract block not found in paper/main.tex")
    abstract_body = m.group(1)
    if "ROC-AUC 0.970 and TSS 0.404" not in abstract_body:
        raise AssertionError(
            "T2: the record's abstract no longer carries metric names for "
            "the 0.970/0.404 pair — the submission must not regress the "
            "R-FS9-R6 R8-7 fix; restore them in paper/main.tex")
    report.append("T2: abstract extracted from the record with metric "
                  "names asserted present (ROC-AUC 0.970 and TSS 0.404)")

    keywords = re.search(
        r"\\noindent\\textbf\{Keywords:\}(.*?)\\end\{minipage\}",
        record_main, re.S)
    if not keywords:
        raise AssertionError("keywords block not found in paper/main.tex")
    tail = ("\\noindent\\textbf{Keywords:}" + keywords.group(1)).strip()

    # ── 3. section inputs from the record (order included) ────────
    inputs = re.findall(r"^.*\\input\{sections/[a-z_]+\}.*$|"
                        r"^\\appendix$",
                        record_main, re.M)
    if not inputs:
        raise AssertionError("no \\input lines found in paper/main.tex")
    input_block = "\n".join(inputs)

    # ── 4. compose submission/src/main.tex ────────────────────────
    header = (
        "% ════════════════════════════════════════════════════════════"
        "═══════\n"
        "%  JOURNAL SUBMISSION EDITION — GENERATED FILE, DO NOT EDIT\n"
        "%  " + JOURNAL_TITLE + "\n"
        "%  Regenerated programmatically from the repository record\n"
        "%  (paper/) by tools/build_submission.py at v4.9, per Dossier\n"
        "%  R-FS9-R6 register item 4. Every byte of the sections,\n"
        "%  figures, and refs.bib is a verbatim copy of the record;\n"
        "%  the only deviations are the declared transformations T1-T4\n"
        "%  in that script. Re-run the generator after any paper/ "
        "change.\n"
        "% ════════════════════════════════════════════════════════════"
        "═══════\n\n")
    title_block = (
        "\\begin{document}\n\n"
        "\\thispagestyle{plain}\n\n"
        "\\begin{center}\n"
        "    {\\color{black!85}\\rule{\\textwidth}{0.4pt}}\\\\[2.2em]\n"
        "    {\\LARGE\\bfseries Federated Solar-Flare Prediction on "
        "SWAN-SF:\\\\[0.35em]\n"
        "     A Provenance, Leakage, and\\\\[0.15em]\n"
        "     Evaluation-Protocol Audit\\par}\n"
        "    \\vspace{1.6em}\n"
        "    {\\large Deepanshu Balhara\\textsuperscript{1} \\quad and "
        "\\quad\n"
        "     Davesh Vats\\textsuperscript{2}\\par}\n"
        "    \\vspace{0.7em}\n"
        "    {\\small\\textit{Independent "
        "Researchers}\\textsuperscript{1,2}\\par}\n"
        "    % NOTE FOR AUTHORS: this file is GENERATED (tools/"
        "build_submission.py)\n"
        "    % and byte-compared by the integrity battery — never "
        "hand-edit it.\n"
        "    % To set institutional affiliations, edit this title block "
        "in the\n"
        "    % generator and re-run `python tools/build_submission.py` "
        "(Dossier\n"
        "    % R-FS9-R7, B13).\n"
        "    \\vspace{1.1em}\n"
        "    {\\small October 2026\\par}\n"
        "    \\vspace{0.9em}\n"
        "    {\\footnotesize\\textit{Journal submission edition — "
        "regenerated from the repository record (v4.9); source: "
        "\\texttt{submission/src}, generator: "
        "\\texttt{tools/build\\_submission.py}}\\par}\n"
        "    \\vspace{1.4em}\n"
        "    {\\color{black!85}\\rule{\\textwidth}{0.4pt}}\n"
        "\\end{center}\n\n"
        "\\vspace{0.4em}\n\n"
        "\\begin{center}\n"
        "\\begin{minipage}{0.92\\textwidth}\n"
        "\\footnotesize\n"
        "\\textbf{Abstract ---}\n" + abstract_body +
        "\n\n\\vspace{0.55em}\n" + tail + "\n\\end{minipage}\n"
        "\\end{center}\n\n\\vspace{0.8em}\n\n")

    write(os.path.join(SRC, "main.tex"),
          header + preamble + "\n" + title_block + "\n" + input_block +
          "\n\n\\FloatBarrier\n\\bibliography{refs}\n\n\\end{document}\n")
    report.append("main.tex composed: record preamble + journal title "
                  "block + record abstract/keywords + record section "
                  "order")

    # ── 5. sections: verbatim copies + T3 ─────────────────────────
    sec_dir = os.path.join(PAPER, "sections")
    t3_applied = False
    t3_limitations = False
    t3_results = False
    for fn in sorted(os.listdir(sec_dir)):
        if not fn.endswith(".tex"):
            continue
        text = read(os.path.join(sec_dir, fn))
        if fn == "sec_results.tex":
            # T3 (third occurrence, R8-10): the sim-74h CPU-sandbox
            # estimate in the LSTM-arm narrative
            text = declared_replace(
                text,
                "federated arms plus the pooled comparator, against an "
                "estimated\n"
                "$\\sim$74\\,h on the 2-core CPU sandbox---the one "
                "experiment where a\n"
                "GPU was a practical requirement rather than a "
                "convenience)",
                "federated arms plus the pooled comparator, against a "
                "CPU-only\n"
                "sandbox run that would have been impractically "
                "slow---the one experiment where a GPU was a\n"
                "practical requirement rather than a convenience)",
                "T3-74h-results", "paper/sections/sec_results.tex")
            t3_results = True
            report.append("T3: sec_results — the third occurrence (the "
                          "~74 h 2-core-sandbox estimate) dropped; the "
                          "logged 2.6 h wall-clock retained")
        if fn == "sec_experiments.tex":
            # T3: drop the un-regenerable CPU estimate (R8-10)
            text = declared_replace(
                text,
                "because the CPU-only protocol above would\n"
                "have needed an estimated 74\\,h---the one experiment "
                "where GPU was\n"
                "a practical requirement.",
                "because the CPU-only protocol above would\n"
                "have been impractically slow for the sequence arms"
                "---the one\n"
                "experiment where GPU was a practical requirement.",
                "T3-74h-experiments", "paper/sections/sec_experiments.tex")
            t3_applied = True
            report.append("T3: sec_experiments — the un-regenerable "
                          "'estimated 74 h' CPU figure dropped; logged "
                          "GPU hours (2.6 h + 8.7 h, in the same "
                          "paragraph) retained")
        if fn == "sec_limitations.tex":
            # T3 (second occurrence, R8-10): same un-regenerable CPU
            # estimate in the executed-programme narrative
            text = declared_replace(
                text,
                "0.970/0.404, at 2.6\\,h wall-clock against an "
                "estimated 74\\,h of CPU),",
                "0.970/0.404, at 2.6\\,h wall-clock on the owner GPU "
                "— the CPU-only protocol would have been impractically "
                "slow),",
                "T3-74h-limitations", "paper/sections/sec_limitations.tex")
            t3_limitations = True
            report.append("T3: sec_limitations — the second "
                          "occurrence of the un-regenerable '74 h' "
                          "CPU estimate re-derived from the logged "
                          "record (2.6 h wall-clock on the owner GPU)")
        write(os.path.join(SRC, "sections", fn), text)
    if not t3_applied or not t3_limitations or not t3_results:
        raise AssertionError("T3 target sections not found "
                             "(sec_experiments/sec_results/"
                             "sec_limitations)")
    for fn in sorted(os.listdir(sec_dir)):
        if fn.endswith(".tex"):
            a = sha(os.path.join(sec_dir, fn))
            b = sha(os.path.join(SRC, "sections", fn))
            if fn not in ("sec_experiments.tex", "sec_results.tex",
                          "sec_limitations.tex") and a != b:
                raise AssertionError(f"unexpected drift in copy: {fn}")

    # ── 6. refs.bib: byte-identical copy of the record ────────────
    copy_bytes(os.path.join(PAPER, "refs.bib"),
               os.path.join(SRC, "refs.bib"))
    if sha(os.path.join(PAPER, "refs.bib")) != \
            sha(os.path.join(SRC, "refs.bib")):
        raise AssertionError("refs.bib copy is not byte-identical")
    report.append("refs.bib: byte-identical copy of paper/refs.bib "
                  "(rebuild-the-bibliography rule — never retype; all "
                  "31 record entries, only cited keys print)")

    # ── 7. figures: byte-identical copies ─────────────────────────
    fig_dir = os.path.join(PAPER, "figures")
    for fn in sorted(os.listdir(fig_dir)):
        copy_bytes(os.path.join(fig_dir, fn),
                   os.path.join(SRC, "figures", fn))
        if sha(os.path.join(fig_dir, fn)) != \
                sha(os.path.join(SRC, "figures", fn)):
            raise AssertionError(f"figure copy drift: {fn}")
    report.append(f"figures: {len(os.listdir(fig_dir))} byte-identical "
                  "copies from paper/figures/")

    # ── 8. build PDF ──────────────────────────────────────────────
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

    print("submission regenerated from the record:")
    for r in report:
        print("  -", r)
    print("OK")


if __name__ == "__main__":
    main()
