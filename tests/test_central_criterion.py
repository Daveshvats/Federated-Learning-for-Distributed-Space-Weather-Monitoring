#!/usr/bin/env python3
"""
tests/test_central_criterion.py — R-FS9-R6 R8-5: every get_criterion
call site must pass global_pos_rate (the B13 computed-prevalence
plumbing), so the hardcoded 0.4887 fallback in losses.py can no longer
be reached silently by any trainer.

Background (Dossier R-FS9-R6, R8-5): the founding-audit B13 fix
(computed global prevalence) was extended to the SCAFFOLD local path
at v4.5 (R-FS9-R5 R7-4) but stopped at the federated boundary — all
three centralised trainers (centralized_baseline.py,
experiments/run_budget_matched.py, experiments/run_federated_lstm.py)
constructed their focal criterion WITHOUT global_pos_rate and
therefore fell back to the hardcoded 0.4887 cleaned-substrate
prevalence on the raw substrate, undisclosed. v4.6 plumbs all three
call sites and discloses that the published central operating points
predate the alignment.

Method: AST-parse every repository Python source, find every call of
get_criterion(...), and assert the keyword global_pos_rate is
supplied. The definition itself (federated_learning.get_criterion)
legitimately defaults the parameter to None; calls are what matter.
torch-free — the check is static, so it holds in every environment.

Run:  python tests/test_central_criterion.py   (or via the battery)
"""
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SCAN_DIRS = ["", "experiments", "tools", "leakage_audit", "tests"]


def _sources():
    for d in SCAN_DIRS:
        base = os.path.join(ROOT, d) if d else ROOT
        if not os.path.isdir(base):
            continue
        for fn in sorted(os.listdir(base)):
            if fn.endswith(".py"):
                yield os.path.join(base, fn)


def main():
    calls_total = 0
    offenders = []
    for src in _sources():
        rel = os.path.relpath(src, ROOT)
        with open(src, "r", encoding="utf-8") as f:
            try:
                tree = ast.parse(f.read(), filename=src)
            except SyntaxError as e:
                offenders.append(f"{rel}: syntax error {e}")
                continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "id", None) or \
                getattr(func, "attr", None)
            if name != "get_criterion":
                continue
            calls_total += 1
            kw = {k.arg for k in node.keywords}
            if "global_pos_rate" not in kw:
                offenders.append(
                    f"{rel}: get_criterion(...) called WITHOUT "
                    f"global_pos_rate — the B13 computed-prevalence "
                    f"plumbing does not reach this trainer (R8-5 class)")

    if offenders:
        for o in offenders:
            print(f"[FAIL] {o}")
        print(f"RESULT: 0 passed, {len(offenders)} failed")
        sys.exit(1)

    print(f"[PASS] all {calls_total} get_criterion call sites pass "
          f"global_pos_rate — the B13 plumbing reaches every trainer "
          f"(federated arms AND centralised comparators); the 0.4887 "
          f"fallback is unreachable by omission")
    print("RESULT: 1 passed, 0 failed")
    sys.exit(0)


if __name__ == "__main__":
    main()
