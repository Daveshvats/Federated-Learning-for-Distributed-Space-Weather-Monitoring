#!/usr/bin/env python3
"""
tests/test_seed43_replication.py — the v4.12 external-review item-3
verdict guard (the seed-43 replication of the no-BatchNorm
raw-substrate arms).

The v4.12 response committed two owner-side compute items; item 3
(the no-BN control is single-seed) EXECUTED on the owner GPU on
2026-10-06 (2,899 s, RTX 4060) and integrated at v4.13 with the
verdict PARITY HOLDS. The full-fidelity artefact the run wrote
(outputs/raw_nobn_eval_seed43.json, 24,768 bytes) exists owner-side
(see logs/owner_dirlisting_2026-10-06.txt) but had not been pushed
at integration time, so the artefact of record for the paper's
seed-43 numbers is the owner console transcript, committed at
logs/run_raw_nobn_seed43_transcript.txt. This module pins the
transcript byte-identically, re-derives every number the paper
cites from its parsed lines, verifies the parity verdict against
the frozen seed-42 artefact (the run card's decision rule), pins
the paper's seed-43 text at every integration site (full edition,
trimmed edition, conclusion, appendix version row, run card), and
activates a consistency layer on the JSON the moment the owner
push lands (the sha256 freeze follows per the v4.9.4-onwards
convention).

Layers (all torch-free; pure file/text parsing):
  1. transcript byte-pin + parsed run facts (seed override, frozen
     substrate stats, shard draw, selections, trajectories,
     completion, device);
  2. cross-runner shard-draw identity vs the sequence arms' own
     seed-43 batch (logs/queue_seed43.log — the LSTM-era draw);
  3. owner-box inventory pins (the JSON exists unpushed; the raw
     parse metadata is absent — the item-2 blocker evidence);
  4. the parity verdict re-derived from the frozen seed-42
     artefact (decision rule: |delta ROC-AUC| <= 0.005 both arms,
     level claims standing);
  5. paper tex pins at every integration site (never hand-typed:
     the table rows are constructed from the parsed values);
  6. the JSON layer — LANDED 2026-10-06 (the owner paste, byte-
     reconciled against the inventory; sha256-frozen per the v4.9.4-
     onwards convention) + consistency with the transcript, protocol
     pins, monitor-pathology pins.

Run:  python tests/test_seed43_replication.py   (or via the battery)
"""
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRANSCRIPT = os.path.join(ROOT, "logs",
                          "run_raw_nobn_seed43_transcript.txt")
DIRLISTING = os.path.join(ROOT, "logs",
                          "owner_dirlisting_2026-10-06.txt")
QUEUE_SEED43 = os.path.join(ROOT, "logs", "queue_seed43.log")
ARTEFACT_S42 = os.path.join(ROOT, "outputs", "raw_nobn_eval.json")
ARTEFACT_S43 = os.path.join(ROOT, "outputs",
                            "raw_nobn_eval_seed43.json")

# v4.13.1: the push LANDED on 2026-10-06 (the owner pasted the full
# JSON; reconstructed byte-faithfully — the LF-committed form hashes
# to this value, and its CRLF form is exactly the 24,768 bytes the
# owner-disk inventory pinned, so the reconstruction is reconciled
# against TWO independent owner-side records)
JSON_SHA256 = (
    "7c3d30fe9c35ca0ef7da9049cdcb06d7c650cdeb63dea1bc747eb3b8bde63741"
)
JSON_BYTES_CRLF = 24768   # the owner-disk form (Windows text mode)
JSON_ELAPSED_S = 2898.813860177994   # the transcript's 2,899 s

TRANSCRIPT_SHA256 = (
    "a36355270368695339168d03db1aa83448dd4da59b6b874334ffc39dd"
    "4767c53")
DIRLISTING_SHA256 = (
    "f41046d463ad3c9710ef30ff31deda0aa4afff8b88561fdf7a1034c5c0"
    "c904cc")

PAPER_RESULTS = os.path.join(ROOT, "paper", "sections",
                             "sec_results.tex")
PAPER_LIMITS = os.path.join(ROOT, "paper", "sections",
                            "sec_limitations.tex")
PAPER_CONCL = os.path.join(ROOT, "paper", "sections",
                           "sec_conclusion.tex")
PAPER_APPX = os.path.join(ROOT, "paper", "sections",
                          "sec_appendix.tex")
TRIM_RESULTS = os.path.join(ROOT, "paper_trimmed", "sections",
                            "sec_results.tex")
TRIM_DISCUSS = os.path.join(ROOT, "paper_trimmed", "sections",
                            "sec_discussion.tex")
RUN_CARD = os.path.join(ROOT, "docs", "RUN_CARD_v4.12.md")

FAILED = []
N_CHECKS = 0


def check(name, cond, note=""):
    global N_CHECKS
    N_CHECKS += 1
    line = ("[PASS] " if cond else "[FAIL] ") + name
    if note and not cond:
        line += f"  {note}"
    print(line)
    if not cond:
        FAILED.append(name)


def _read(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


# ── layer 1: the transcript — byte pin + parsed run facts ────────────

def transcript_layer():
    if not os.path.exists(TRANSCRIPT):
        check("seed-43 owner transcript present (the item-3 "
              "artefact of record until the JSON lands)", False,
              "(file absent)")
        return None
    with open(TRANSCRIPT, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    check("transcript sha256 byte-pin (any edit to the owner record "
          "fails the battery)", digest == TRANSCRIPT_SHA256,
          f"(got {digest[:16]}…)")
    t = _read(TRANSCRIPT)

    check("seed-override banner: 42 -> 43 with the frozen carve "
          "and the nobns43_* namespace stated",
          "seed override: 42 -> 43" in t and "nobns43_*" in t)
    check("substrate stats line equals the frozen cache facts "
          "(train 214,888 / val 40,932 / test 75,365 at "
          "2.05%/2.05%/1.31%)",
          "substrate: train 214,888 (pos 2.05%) | val 40,932 "
          "(pos 2.05%) | test 75,365 (pos 1.31%)" in t)
    check("partition coverage line: 100% disjoint by construction",
          "coverage: 214,888/214,888 (100.0%)" in t)

    m = re.search(r"shards rebuilt \(Dirichlet alpha=1\.0, seed 43\):"
                  r" sizes \[([\d, ]+)\]", t)
    sizes = [int(x) for x in m.group(1).split(",")] if m else []
    check("shard sizes parsed: [7887, 51987, 8249, 15699, 22041, "
          "109025] summing to the 214,888 pool",
          sizes == [7887, 51987, 8249, 15699, 22041, 109025]
          and sum(sizes) == 214888)

    fa_blk = t.split("[nobn] fedavg (no-BN", 1)
    fa_blk = fa_blk[1].split("[nobn] fedprox (no-BN", 1)[0] \
        if len(fa_blk) > 1 else ""
    fp_blk = t.split("[nobn] fedprox (no-BN", 1)
    fp_blk = fp_blk[1].split("[nobn] report", 1)[0] \
        if len(fp_blk) > 1 else ""

    fa_mon = re.findall(r"Round\s+(\d+) \| val F1: ([\d.]+) \| "
                        r"val ROC: ([\d.]+)", fa_blk)
    fp_mon = re.findall(r"Round\s+(\d+) \| val F1: ([\d.]+) \| "
                        r"val ROC: ([\d.]+)", fp_blk)
    check("fedavg monitor: all 10 monitored rounds parsed "
          "(5..50)", [int(r) for r, _, _ in fa_mon] ==
          list(range(5, 51, 5)))
    check("fedprox monitor: all 10 monitored rounds parsed "
          "(5..50)", [int(r) for r, _, _ in fp_mon] ==
          list(range(5, 51, 5)))

    best = dict(re.findall(
        r"\[(fedavg|fedprox)-noBN\] best_val_f1=([\d.]+) at round "
        r"(\d+)", t) and
        [(a, (float(b), int(c))) for a, b, c in
         re.findall(r"\[(fedavg|fedprox)-noBN\] best_val_f1="
                    r"([\d.]+) at round (\d+)", t)])
    check("fedavg best_val_f1=0.6596 at round 45 (the shipped "
          "selection rule)", best.get("fedavg") == (0.6596, 45))
    check("fedprox best_val_f1=0.2998 at round 50",
          best.get("fedprox") == (0.2998, 50))

    sel = {a: (int(r), float(ro), float(pr)) for a, r, ro, pr in
           re.findall(r"\[nobn\] (fedavg|fedprox): selected r(\d+) "
                      r"ROC ([\d.]+) PR ([\d.]+)", t)}
    check("fedavg selected checkpoint: r45 ROC 0.961 PR 0.349",
          sel.get("fedavg") == (45, 0.961, 0.349))
    check("fedprox selected checkpoint: r50 ROC 0.976 PR 0.321",
          sel.get("fedprox") == (50, 0.976, 0.321))

    fa_tr = {int(r): (float(ro), float(pr)) for r, ro, pr in
             re.findall(r"\[fedavg-noBN\] r(\d+): test "
                        r"ROC=([\d.]+) PR=([\d.]+)", t)}
    fp_tr = {int(r): (float(ro), float(pr)) for r, ro, pr in
             re.findall(r"\[fedprox-noBN\] r(\d+): test "
                        r"ROC=([\d.]+) PR=([\d.]+)", t)}
    check("fedavg test trajectory: 10 rounds parsed, "
          "DECLINING 0.976 -> 0.959 (the val-up/test-down "
          "divergence signature replicates at seed 43)",
          len(fa_tr) == 10 and
          fa_tr[5][0] > fa_tr[50][0] and
          abs(fa_tr[5][0] - 0.976) < 5e-4 and
          abs(fa_tr[50][0] - 0.959) < 5e-4)
    check("fedavg selection lands below its own trajectory peak "
          "(the weakest-round selection pathology reproduces)",
          fa_tr[45][0] < max(v[0] for v in fa_tr.values()))
    check("fedavg val ROC rising 0.978 -> 0.989 while test falls "
          "(the divergence, both sides)",
          float(fa_mon[0][2]) < float(fa_mon[-1][2]) and
          abs(float(fa_mon[0][2]) - 0.978) < 5e-4 and
          abs(float(fa_mon[-1][2]) - 0.989) < 5e-4)
    check("fedprox monitor collapse: val F1 exactly 0.0000 at "
          "rounds 5,10,15,20 before the late climb (the compressed "
          "0.35-threshold pathology reproduces)",
          all(float(f1) == 0.0 for r, f1, _ in fp_mon
              if int(r) <= 20) and
          float(fp_mon[-1][1]) == 0.2998)
    check("fedprox test trajectory: 10 rounds parsed, CLIMBING "
          "0.939 -> 0.976 (the proximal stabilisation signature "
          "reproduces)",
          len(fp_tr) == 10 and fp_tr[5][0] < fp_tr[50][0])

    check("run completion: report line (2899 s) + done marker",
          "raw_nobn_eval_seed43.json (2899s total)" in t and
          "[nobn] done." in t)
    check("device provenance: the RTX 4060 owner GPU on record",
          "RTX 4060" in t)
    check("the two failed angle-bracket invocations preserved "
          "verbatim (the run card's v4.13 rewrite is motivated by "
          "this record)", t.count("operator is reserved for future "
                                  "use.") >= 4)
    return sel


# ── layer 2: cross-runner shard-draw identity ────────────────────────

def cross_runner_layer(sizes_expected):
    if not os.path.exists(QUEUE_SEED43):
        check("logs/queue_seed43.log present (the sequence arms' "
              "seed-43 batch)", False, "(file absent)")
        return
    q = _read(QUEUE_SEED43)
    q_sizes = [int(x.replace(",", "")) for x in
               re.findall(r"n=\s*([\d,]+)", q)]
    check("cross-runner reproducibility: the no-BN seed-43 Dirichlet "
          "draw equals the sequence arms' seed-43 draw in "
          "logs/queue_seed43.log (same six client sizes)",
          q_sizes[:6] == sizes_expected)


# ── layer 3: the owner-box inventory pins ────────────────────────────

def inventory_layer():
    if not os.path.exists(DIRLISTING):
        check("owner-box inventory present (the item-2 blocker "
              "evidence)", False, "(file absent)")
        return
    with open(DIRLISTING, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    check("owner-box inventory sha256 byte-pin", digest ==
          DIRLISTING_SHA256, f"(got {digest[:16]}…)")
    d = _read(DIRLISTING)
    body = [l for l in d.splitlines() if not l.lstrip().startswith("#")]
    body_txt = "\n".join(body)
    check("inventory records outputs/raw_nobn_eval_seed43.json "
          "(24,768 bytes) present owner-side — the full-fidelity "
          "artefact awaiting the owner push",
          "raw_nobn_eval_seed43.json" in body_txt and
          "24768" in body_txt)
    check("inventory records the frozen substrate cache "
          "(data.npz, 76,679,279 bytes)",
          "data.npz" in body_txt and "76679279" in body_txt)
    check("item-2 blocker evidence: no p1..p4_meta.csv, no "
          "partition directory, and no .tar.gz anywhere in the "
          "owner repo tree",
          not re.search(r"p[1-4]_meta\.csv", body_txt) and
          not re.search(r"\.tar\.gz", body_txt) and
          not re.search(r"partition[1-9]\s*$", body_txt, re.M))


# ── layer 4: the parity verdict vs the frozen seed-42 artefact ───────

def parity_layer(sel):
    if not os.path.exists(ARTEFACT_S42):
        check("frozen seed-42 artefact present (raw_nobn_eval.json)",
              False, "(file absent)")
        return
    with open(ARTEFACT_S42, encoding="utf-8") as f:
        s42 = json.load(f)
    ref = s42.get("reference_columns", {})
    ok_rule = True
    for algo in ("fedavg", "fedprox"):
        t42 = s42[algo]["selected_checkpoint_test"]["test"]
        _, roc43, _ = sel[algo]
        delta = abs(float(t42["roc_auc"]) - roc43)
        ok_rule = ok_rule and delta <= 0.005
    check("the run card's decision rule: |delta ROC-AUC| <= 0.005 "
          "on both arms (measured: fedavg 0.9626->0.961 = 0.0016, "
          "fedprox 0.9755->0.976 = 0.0005) — parity holds",
          ok_rule)
    fa43 = sel["fedavg"][1]
    fp43 = sel["fedprox"][1]
    check("level claims replicated: no-BN FedAvg 0.961 within "
          "0.005 of the FedAvg-LSTM sequence arm (0.958) and "
          "no-BN FedProx 0.976 above the centralised MLP (0.971)",
          abs(fa43 - float(ref["fedavg_lstm"]["roc_auc"])) <= 0.005
          and fp43 >= float(ref["centralized_mlp"]["roc_auc"]))


# ── layer 5: paper tex pins (every integration site) ─────────────────

def paper_layer(sel):
    if not os.path.exists(PAPER_RESULTS):
        check("paper/sections/sec_results.tex present", False,
              "(file absent)")
        return
    with open(ARTEFACT_S42, encoding="utf-8") as f:
        s42 = json.load(f)

    tex = _read(PAPER_RESULTS)
    rows = (
        ("FedAvg", s42["fedavg"]["selected_checkpoint_test"]["test"],
         sel["fedavg"]),
        ("FedProx",
         s42["fedprox"]["selected_checkpoint_test"]["test"],
         sel["fedprox"]),
    )
    for label, t42, s43 in rows:
        expected = (f"{label} & {float(t42['roc_auc']):.3f} & "
                    f"{float(t42['pr_auc']):.3f} & {s43[1]:.3f} & "
                    f"{s43[2]:.3f} \\\\")
        check(f"tab:rawnobnrep row content-pinned to the seed-42 "
              f"artefact + the parsed seed-43 transcript (live-"
              f"derived, never hand-typed): {label}",
              expected in tex, f"(expected {expected!r})")
    for phrase in ("The single-seed boundary is",
                   r"0.963$\to$0.961", r"0.975$\to$0.976",
                   r"0.527$\to$0.660", r"0.092$\to$0.300",
                   "2{,}899", r"raw\_nobn\_seed43\_transcript"):
        check(f"sec_results.tex carries the boundary-(d) discharge "
              f"text: {phrase!r}", phrase in tex)

    lim = _read(PAPER_LIMITS)
    check("sec_limitations.tex: the fourth owner-GPU batch row in "
          "the executed re-run programme (2,899 s, verdict parity "
          "holds)", "fourth owner-GPU batch" in lim and
          "2{,}899" in lim)
    check("sec_limitations.tex: the seed-43 replication REMOVED "
          "from the queued list (an executed item cannot stay "
          "queued)",
          "the seed-43 replication of the no-BN MLP arms;" not in lim
          and "blocked only on regenerating the raw parse metadata"
          in lim)

    con = _read(PAPER_CONCL)
    check("sec_conclusion.tex: the parity claim now carries the "
          "replication clause", "the seed-43 replication confirms"
          in con and "within 0.002 ROC-AUC" in con)

    app = _read(PAPER_APPX)
    check("sec_appendix.tex: the v4.13 version-history row",
          "v4.13 &" in app and "parity holds" in app)

    tr = _read(TRIM_RESULTS)
    check("trimmed edition results: the single-seed boundary "
          "replaced by the replication verdict (zero process "
          "language — no owner, no versions, no queue)",
          "now carry a second seed" in tr and
          r"0.963$\to$0.961" in tr and
          r"0.53$\to$0.66" in tr)

    td = _read(TRIM_DISCUSS)
    check("trimmed edition discussion: the limitations sentence "
          "confirms the second-seed replication",
          "its second-seed replication confirms" in td and
          "within 0.002 ROC-AUC" in td)

    rc = _read(RUN_CARD)
    # v4.13.1: the card's item-3 follow-up flipped from "push the
    # JSON" (the recipe) to DONE (landed + sha256-frozen) — the pin
    # follows the card state, never a stale recipe
    check("run card rev. v4.13.1: item 3 executed with the verdict, "
          "the JSON push marked DONE (landed, sha256-frozen), and "
          "the PowerShell-safe item-2 recipe ($parts) still intact",
          "rev. v4.13.1" in rc and "parity holds" in rc and
          "JSON push" in rc and "DONE (v4.13.1" in rc and
          "No owner action remains" in rc and
          "git add outputs\\raw_nobn_eval_seed43.json" not in rc and
          "$parts" in rc)


# ── layer 6: the JSON layer (LANDED v4.13.1; sha256-frozen) ───────

def json_layer(sel):
    # v4.13.1: the artefact is COMMITTED now — a missing file is a
    # FAILURE, not a SKIP (the pre-landing gate is retired)
    if not os.path.exists(ARTEFACT_S43):
        check("the seed-43 JSON is committed (landed 2026-10-06)",
              False)
        return
    raw = open(ARTEFACT_S43, "rb").read()
    lf = raw.replace(b"\r\n", b"\n")
    digest = hashlib.sha256(lf).hexdigest()
    check("the seed-43 JSON sha256 byte-pin (CRLF/LF-stable; the "
          "landed owner paste)", digest == JSON_SHA256,
          f"(got {digest[:16]}…)")
    crlf_len = (len(raw) if raw.count(b"\r\n")
                else len(raw) + lf.count(b"\n"))
    check("owner-disk reconciliation: the CRLF form is exactly the "
          f"inventory's {JSON_BYTES_CRLF:,} bytes",
          crlf_len == JSON_BYTES_CRLF)
    with open(ARTEFACT_S43, encoding="utf-8") as f:
        d = json.load(f)
    check("elapsed_s equals the transcript's 2,899 s",
          abs(float(d.get("elapsed_s", -1)) - JSON_ELAPSED_S) < 1e-9)
    check("the JSON's runner and purpose register the v4.12 item-3 "
          "protocol (seed-43 re-init and shard draw)",
          d.get("runner") == "experiments/run_raw_nobn.py" and
          "seed 43 re-init and shard draw" in str(d.get("purpose", "")) +
          str(d.get("fedavg", {}).get("protocol", {}).get(
              "validation", "")))
    for algo in ("fedavg", "fedprox"):
        blk = d.get(algo)
        if blk is None:
            check(f"seed-43 JSON carries arm {algo}", False)
            continue
        check(f"{algo}: protocol.seed == 43",
              blk.get("protocol", {}).get("seed") == 43)
        t = blk.get("selected_checkpoint_test", {}).get("test", {})
        _, roc43, pr43 = sel[algo]
        check(f"{algo}: JSON selected test equals the transcript "
              f"summary at 3dp ({roc43:.3f}/{pr43:.3f})",
              abs(float(t.get("roc_auc", -1)) - roc43) <= 5e-4 and
              abs(float(t.get("pr_auc", -1)) - pr43) <= 5e-4)
        best_round = blk.get("best_round")
        check(f"{algo}: JSON best_round == the transcript's "
              f"({sel[algo][0]})", best_round == sel[algo][0])
        hist = {h["round"]: float(h["f1"])
                for h in blk.get("history", [])}
        if algo == "fedprox":
            check("fedprox: JSON history reproduces the early-round "
                  "monitor collapse (val F1 0.0 through round 20)",
                  all(hist.get(r, -1) == 0.0
                      for r in (5, 10, 15, 20)))


def main():
    print("layer 1 — the seed-43 owner transcript (byte pin + "
          "parsed run facts):")
    sel = transcript_layer()
    print("layer 2 — cross-runner shard-draw identity (the "
          "sequence arms' seed-43 batch):")
    if sel is not None:
        cross_runner_layer([7887, 51987, 8249, 15699, 22041,
                            109025])
    print("layer 3 — the owner-box inventory pins:")
    inventory_layer()
    print("layer 4 — the parity verdict vs the frozen seed-42 "
          "artefact (the run card's decision rule):")
    if sel is not None:
        parity_layer(sel)
    print("layer 5 — paper integration pins (full + trimmed "
          "editions, conclusion, appendix, run card):")
    if sel is not None:
        paper_layer(sel)
    print("layer 6 — the landed JSON layer (sha256-frozen):")
    if sel is not None:
        json_layer(sel)
    print(f"RESULT: {N_CHECKS - len(FAILED)} passed, "
          f"{len(FAILED)} failed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
