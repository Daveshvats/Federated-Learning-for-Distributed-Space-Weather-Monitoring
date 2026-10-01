"""
tests/test_gpu_queue.py  (v3.8)
──────────────────────────────
Unit tests for the owner-GPU queue package:

  1. apply_smote 3D path (data_preparation): (n, T, F) float16 shard is
     flattened, resampled, reshaped; minority reaches the SMOTE_RATIO
     target; determinism under a fixed seed; 2D path unchanged; tiny
     minorities (< 2) fall through untouched
  2. find_aux_dir (run_federated_lstm): raw_dir wins, then cache aux,
     then the Windows-renamed _aux; None when nothing matches
  3. runner contract: --scaffold/--scaffold-only/--smote/--force flags
     exist; MLP_COUNTERPART_MAP covers the scaffold arm
  4. queue plan safety: every step's outputs are ADDITIVE (no frozen
     seed-42 artefact is ever targeted); argv wiring matches the
     documented tags; RUNLOG helper appends
  5. driver --dry-run exits 0 on a clean plan

Tests 1-4 run without torch; 5 runs the real driver in dry-run mode.
Run:  python tests/test_gpu_queue.py
"""

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


# ── 1. apply_smote 3D ────────────────────────────────────────────────────

def test_apply_smote_3d():
    import config as cfg
    from data_preparation import apply_smote
    rng = np.random.default_rng(7)
    n, T, F = 400, 6, 4
    X = rng.normal(size=(n, T, F)).astype(np.float16)
    y = np.zeros(n, dtype=np.float32)
    pos = rng.choice(n, 12, replace=False)          # 12 positives, 3.6%
    y[pos] = 1.0

    X2, y2 = apply_smote(X, y, seed=42)
    n_maj = int((y == 0).sum())
    target = int(round(cfg.SMOTE_RATIO * n_maj))
    check("smote3d: shape (n+, T, F)",
          X2.ndim == 3 and X2.shape[1:] == (T, F) and X2.shape[0] >= n,
          f"got {X2.shape}")
    check("smote3d: minority reached the ratio target",
          int(y2.sum()) == target, f"pos {int(y2.sum())} vs {target}")
    check("smote3d: original rows preserved (first n unchanged)",
          np.allclose(X2[:n].astype(np.float32),
                      X.astype(np.float32), atol=1e-6))
    check("smote3d: labels of original rows preserved",
          np.array_equal(y2[:n], y))
    check("smote3d: no NaN/Inf in synthetic windows",
          bool(np.isfinite(X2).all()))
    check("smote3d: synthetic windows are convex interpolations "
          "(within per-feature min/max of the minority)",
          _within_hull(X2[n:].astype(np.float32), X[pos].astype(np.float32)))

    X3, y3 = apply_smote(X, y, seed=42)
    check("smote3d: deterministic under the same seed",
          np.array_equal(X2.astype(np.float32), X3.astype(np.float32)))
    X4, _ = apply_smote(X, y, seed=43)
    check("smote3d: different seed -> different synthetics",
          not np.array_equal(X2[n:].astype(np.float32),
                             X4[n:].astype(np.float32)))


def _within_hull(synth, minority):
    lo = minority.reshape(len(minority), -1).min(axis=0)
    hi = minority.reshape(len(minority), -1).max(axis=0)
    s = synth.reshape(len(synth), -1)
    return bool(((s >= lo - 1e-5) & (s <= hi + 1e-5)).all())


def test_apply_smote_guards():
    from data_preparation import apply_smote
    rng = np.random.default_rng(8)
    X = rng.normal(size=(60, 6, 4)).astype(np.float16)
    y0 = np.zeros(60, dtype=np.float32)            # 0 positives
    Xa, ya = apply_smote(X, y0, seed=1)
    check("smote: minority < 2 -> shard untouched",
          Xa is X and ya is y0)
    y1 = np.ones(60, dtype=np.float32)             # majority 0 -> at target
    Xb, yb = apply_smote(X, y1, seed=1)
    check("smote: already at/above ratio -> shard untouched",
          Xb is X and yb is y1)
    # 2D path still works (MLP shards)
    X2d = rng.normal(size=(200, 10))
    y2d = np.zeros(200, dtype=np.float32)
    y2d[:6] = 1.0
    X2r, y2r = apply_smote(X2d, y2d, seed=42)
    check("smote: 2D path unchanged (n grows, pos at target)",
          X2r.ndim == 2 and X2r.shape[1] == 10 and
          int(y2r.sum()) == int(round(0.25 * 194)))


# ── 2. find_aux_dir ──────────────────────────────────────────────────────

def test_find_aux_dir():
    from experiments.run_federated_lstm import find_aux_dir
    tmp = tempfile.mkdtemp()
    try:
        raw = os.path.join(tmp, "raw")
        cache = os.path.join(tmp, "cache")
        os.makedirs(raw)
        os.makedirs(os.path.join(cache, "_aux"))
        check("aux: None when nothing matches",
              find_aux_dir(raw, cache) is None)
        with open(os.path.join(cache, "_aux", "match_test_p5.csv"), "w") as f:
            f.write("x")
        check("aux: _aux (Windows rename) resolves",
              find_aux_dir(raw, cache) ==
              os.path.join(cache, "_aux"))
        os.makedirs(os.path.join(cache, "aux"))
        with open(os.path.join(cache, "aux", "match_test_p5.csv"), "w") as f:
            f.write("x")
        check("aux: plain aux preferred over _aux",
              find_aux_dir(raw, cache) == os.path.join(cache, "aux"))
        with open(os.path.join(raw, "match_test_p5.csv"), "w") as f:
            f.write("x")
        check("aux: raw_dir wins over both cache candidates",
              find_aux_dir(raw, cache) == raw)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── 3. runner contract ───────────────────────────────────────────────────

def test_runner_contract():
    from experiments import run_federated_lstm as R
    check("runner: MLP counterpart map covers scaffold",
          R.MLP_COUNTERPART_MAP.get("scaffold_lstm") == "scaffold_mlp")
    help_out = subprocess.run(
        [sys.executable, os.path.join("experiments",
                                      "run_federated_lstm.py"), "--help"],
        capture_output=True, text=True, cwd=os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
    txt = help_out.stdout
    check("runner: --help exits 0", help_out.returncode == 0)
    for flag in ("--scaffold", "--scaffold-only", "--smote", "--force"):
        check(f"runner: {flag} documented", flag in txt)


# ── 4. queue plan safety ─────────────────────────────────────────────────

def test_queue_plan():
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "experiments"))
    import run_gpu_queue as Q
    frozen = {"outputs/raw_lstm_eval.json",
              "outputs/event_level_raw_lstm_p5.json"}
    touched = set()
    for s in Q.STEPS:
        touched.update(s["outputs"])
        check(f"queue: step {s['name']} has 2 additive outputs",
              len(s["outputs"]) == 2 and all(
                  o.startswith("outputs/") for o in s["outputs"]))
    check("queue: NO step touches a frozen seed-42 artefact",
          not (touched & frozen), str(touched & frozen))
    check("queue: step names unique",
          len({s["name"] for s in Q.STEPS}) == len(Q.STEPS))
    scaff = next(s for s in Q.STEPS if s["name"] == "scaffold")
    check("queue: scaffold step uses --scaffold-only + tag",
          "--scaffold-only" in scaff["argv"] and "scaffold" in scaff["argv"])
    seed43 = next(s for s in Q.STEPS if s["name"] == "seed43")
    check("queue: seed43 step reseeds AND adds scaffold arm",
          "43" in seed43["argv"] and "--scaffold" in seed43["argv"])
    smote = next(s for s in Q.STEPS if s["name"] == "smote")
    check("queue: smote step enables --smote and skips central",
          "--smote" in smote["argv"] and "--no-central-lstm" in smote["argv"])
    check("queue: send-back list = all step outputs",
          set(Q.SEND_BACK) == touched)

    # RUNLOG append (isolated copy)
    tmp = tempfile.mkdtemp()
    try:
        old = Q.RUNLOG
        Q.RUNLOG = os.path.join(tmp, "RUNLOG.md")
        open(Q.RUNLOG, "w").write("# head\n")
        Q.runlog_append("unit | DONE | 1s | fixture")
        content = open(Q.RUNLOG).read()
        check("queue: runlog_append appends a timestamped line",
              content.startswith("# head\n") and "unit | DONE" in content
              and len(content.strip().splitlines()) == 2)
        Q.RUNLOG = old
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── 5. driver dry-run ────────────────────────────────────────────────────

def test_driver_dry_run():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "experiments/run_gpu_queue.py",
                        "--dry-run"], capture_output=True, text=True,
                       cwd=repo, timeout=300)
    check("driver: --dry-run exits 0", r.returncode == 0,
          r.stderr[-300:] if r.returncode else "")
    check("driver: dry-run prints the 3 steps",
          all(s in r.stdout for s in ("scaffold", "seed43", "smote")))
    check("driver: dry-run executes nothing",
          "DRY RUN" in r.stdout and "START" not in r.stdout)


def main():
    print("── tests/test_gpu_queue.py ──")
    test_apply_smote_3d()
    test_apply_smote_guards()
    test_find_aux_dir()
    test_runner_contract()
    test_queue_plan()
    test_driver_dry_run()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
