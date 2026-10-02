"""
experiments/run_gpu_queue.py  (v3.8 — the ONE-COMMAND owner-GPU queue)
────────────────────────────────────────────────────────────────────────
Executes the ENTIRE remaining owner-side GPU programme as a single
resumable batch.  No further commands will be requested from the owner
after this queue unless its results reveal a defect.

    python experiments/run_gpu_queue.py              # everything
    python experiments/run_gpu_queue.py --dry-run    # plan + preflight only
    python experiments/run_gpu_queue.py --only scaffold
    python experiments/run_gpu_queue.py --only seed43,smote
    python experiments/run_gpu_queue.py --force-cpu  # discouraged (days)

STEPS (each writes its own report + event-level JSON; all artefacts are
ADDITIVE — no frozen seed-42 file is ever overwritten):

 1. scaffold  SCAFFOLD-LSTM arm, seed 42, 50 rounds (the only arm not
              yet executed on the raw substrate)  -> raw_lstm_scaffold.json
 2. seed43    full replication at seed 43 (central + FedAvg + FedProx +
              SCAFFOLD — same frozen protocol, reseeded init/shards)
              -> raw_lstm_seed43.json
 3. smote     natural-prevalence per-client SMOTE ablation (FedAvg +
              FedProx, 3D-aware flatten/reshape)      -> raw_lstm_smote.json

PROPERTIES
  - Skip-if-done: a step whose outputs already exist is SKIPPED, so the
    queue is safely re-runnable and crash-resumable (the runner itself
    also resumes per-round from data/cache/rawsubstrate/*_<tag>.pt).
  - Live logs: every step is tee'd to logs/queue_<step>.log.
  - RUNLOG.md: each step appends a timestamped machine-written line to
    the repo-root RUNLOG.md ledger (owner asks vs executions).
  - Cross-platform: pure Python, no bash; child processes are forced to
    UTF-8 so the runner's output survives Windows console codepages.

GPU:  expected on the owner's RTX 4060 Laptop, from the measured seed-42
run (9,290 s for central+fedavg+fedprox):  scaffold ~1.2 h, seed43
~3.5-4 h, smote ~2.5-3 h  =>  ~7-8 h total, run it overnight (plugged
in).  On CPU the same queue would take days — the driver aborts without
CUDA unless --force-cpu is given.
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join("experiments", "run_federated_lstm.py")
RUNLOG = os.path.join(ROOT, "RUNLOG.md")
LOGDIR = os.path.join(ROOT, "logs")
CACHE = os.path.join(ROOT, "data", "cache", "rawsubstrate")

STEPS = [
    {
        "name": "scaffold",
        "eta": "~1.2 h (RTX 4060 Laptop)",
        "argv": ["--tag", "scaffold", "--scaffold-only"],
        "outputs": ["outputs/raw_lstm_scaffold.json",
                    "outputs/event_level_raw_scaffold.json"],
        "why": "SCAFFOLD-LSTM arm, seed 42, 50 rounds (frozen protocol)",
    },
    {
        "name": "seed43",
        "eta": "~3.5-4 h (RTX 4060 Laptop)",
        "argv": ["--seed", "43", "--scaffold"],
        "outputs": ["outputs/raw_lstm_seed43.json",
                    "outputs/event_level_raw_seed43.json"],
        "why": "seed replication: central + FedAvg + FedProx + SCAFFOLD, "
               "reseeded init/shards (frozen val carve kept)",
    },
    {
        "name": "smote",
        "eta": "~2.5-3 h (RTX 4060 Laptop)",
        "argv": ["--tag", "smote", "--smote", "--no-central-lstm"],
        "outputs": ["outputs/raw_lstm_smote.json",
                    "outputs/event_level_raw_smote.json"],
        "why": "natural-prevalence per-client SMOTE ablation "
               "(FedAvg + FedProx, ratio 0.25)",
    },
]

SEND_BACK = [
    "outputs/raw_lstm_scaffold.json", "outputs/event_level_raw_scaffold.json",
    "outputs/raw_lstm_seed43.json", "outputs/event_level_raw_seed43.json",
    "outputs/raw_lstm_smote.json", "outputs/event_level_raw_smote.json",
]


def _ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _artefact_hash(path):
    """sha256[:16] of an output artefact, embedded in the ledger line so
    post-hoc edits to the artefact are detectable (v4.0, review M7)."""
    import hashlib
    try:
        with open(os.path.join(ROOT, path), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        return "MISSING"


def runlog_append(line):
    """Machine-written ledger line (kept append-only, never rewritten)."""
    with open(RUNLOG, "a", encoding="utf-8") as f:
        f.write(f"{_ts()} | {line}\n")


def outputs_exist(step):
    return all(os.path.exists(os.path.join(ROOT, o))
               for o in step["outputs"])


def preflight(force_cpu=False):
    """Cheap environment checks BEFORE burning any GPU hours."""
    problems = []
    print(f"[queue] repo root : {ROOT}")
    print(f"[queue] python    : {sys.executable} "
          f"({sys.version.split()[0]})")
    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"[queue] torch     : {torch.__version__} | "
              f"CUDA available: {cuda}")
        if not cuda and not force_cpu:
            problems.append(
                "no CUDA visible to this python — the queue takes DAYS on "
                "CPU.  Run on the GPU machine, or pass --force-cpu.")
    except Exception as e:
        problems.append(f"torch import failed: {e}")
    for mod in ("numpy", "pandas", "sklearn", "imblearn"):
        try:
            __import__(mod)
        except Exception as e:
            problems.append(f"{mod} import failed: {e} "
                            "(pip install -r requirements.txt)")
    for f in ("X_train_3d.npy", "X_val_3d.npy", "processed_p5_3d.npy",
              "y3d.npz"):
        p = os.path.join(CACHE, f)
        if not os.path.exists(p):
            problems.append(f"missing 3D cache: {p}")
    aux = None
    for cand in (os.path.join(CACHE, "aux"), os.path.join(CACHE, "_aux")):
        if os.path.exists(os.path.join(cand, "match_test_p5.csv")):
            aux = cand
            break
    if aux is None:
        problems.append(
            "event-level aux metadata not found (data/cache/rawsubstrate/"
            "aux or _aux) — event-level JSONs would be skipped")
    else:
        print(f"[queue] aux meta  : {aux}")
    return problems


def run_step(step, python_exe):
    cmd = [python_exe, RUNNER] + step["argv"] + ["--force"]
    log_path = os.path.join(LOGDIR, f"queue_{step['name']}.log")
    os.makedirs(LOGDIR, exist_ok=True)
    print(f"[queue] START {step['name']}: {' '.join(cmd)}")
    print(f"[queue]        eta {step['eta']} | log {log_path}")
    t0 = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
        rc = proc.wait()
    return rc, time.time() - t0, log_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of steps to run "
                         f"({', '.join(s['name'] for s in STEPS)})")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan + preflight and exit")
    ap.add_argument("--force-cpu", action="store_true",
                    help="proceed even without CUDA (NOT recommended)")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter used for the child runs")
    args = ap.parse_args()

    chosen = STEPS
    if args.only:
        want = {w.strip() for w in args.only.split(",") if w.strip()}
        unknown = want - {s["name"] for s in STEPS}
        if unknown:
            raise SystemExit(f"[queue] unknown step(s): {sorted(unknown)}")
        chosen = [s for s in STEPS if s["name"] in want]

    print("=" * 70)
    print(" SF9 owner-GPU queue — SCAFFOLD-LSTM / seed-43 / SMOTE (one batch)")
    print("=" * 70)
    problems = preflight(force_cpu=args.force_cpu)
    if problems:
        print("\n[queue] PREFLIGHT PROBLEMS:")
        for p in problems:
            print(f"  !! {p}")
        # dry-run is a plan preview on ANY machine; only a real execution
        # aborts without CUDA (a --force-cpu override exists for that)
        if (not args.force_cpu and not args.dry_run
                and any("CUDA" in p for p in problems)):
            raise SystemExit("[queue] aborting (no CUDA; --force-cpu to "
                             "override)")
    else:
        print("[queue] preflight OK")

    print(f"\n[queue] plan ({len(chosen)} step(s), "
          f"{'DRY RUN' if args.dry_run else 'EXECUTE'}):")
    for s in chosen:
        status = "SKIP (done)" if outputs_exist(s) else "RUN"
        print(f"  {s['name']:<9s} {status:<12s} {s['why']}")
        print(f"  {'':9s} eta {s['eta']}")
        for o in s["outputs"]:
            print(f"  {'':9s} -> {o}")
    if args.dry_run:
        print("\n[queue] dry run — nothing executed")
        return

    if not os.path.exists(RUNLOG):
        with open(RUNLOG, "w", encoding="utf-8") as f:
            f.write("# RUNLOG — owner-side execution ledger "
                    "(auto-created by run_gpu_queue.py)\n")

    t_all = time.time()
    statuses = []
    for s in chosen:
        if outputs_exist(s):
            print(f"\n[queue] SKIP {s['name']} — all outputs exist")
            runlog_append(f"{s['name']} | SKIP | 0s | already done")
            statuses.append((s["name"], "SKIP", 0.0))
            continue
        rc, dur, log = run_step(s, args.python)
        ok = (rc == 0) and outputs_exist(s)
        status = "DONE" if ok else f"FAILED (rc={rc})"
        hashes = " ".join(f"{o}#{_artefact_hash(o)}"
                          for o in s["outputs"])
        runlog_append(f"{s['name']} | {status} | {dur:.0f}s | "
                      f"{hashes} | log {log}")
        statuses.append((s["name"], status, dur))
        if not ok:
            print(f"[queue] step {s['name']} FAILED — continuing with the "
                  "next step (re-run this queue to resume)")

    print("\n" + "=" * 70)
    print(f" QUEUE SUMMARY  ({time.time() - t_all:.0f}s total)")
    print("=" * 70)
    for name, status, dur in statuses:
        print(f"  {name:<9s} {status:<16s} {dur/60:7.1f} min")
    if all(st == "DONE" or st == "SKIP" for _, st, _ in statuses):
        print("\n[queue] all steps complete — send these files back "
              "(a few KB each):")
        for f in SEND_BACK:
            print(f"  {f}")
        print("\n[queue] after that, NO further GPU work is planned; the "
              "paper integration (v3.8) runs on CPU here.")
    else:
        print("\n[queue] some steps incomplete — just re-run "
              "'python experiments/run_gpu_queue.py' to resume")


if __name__ == "__main__":
    main()
