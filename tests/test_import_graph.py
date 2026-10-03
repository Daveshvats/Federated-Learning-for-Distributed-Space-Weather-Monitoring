#!/usr/bin/env python3
"""
tests/test_import_graph.py — R-FS9-R6 R8-1: the import-graph guard.

Why this test exists (Dossier R-FS9-R6, finding R8-1, critical):
v4.5 deleted the stale constant cfg.BATCH_SIZE (R-FS9-R5 R7-2) while
model.py still executed `from config import (..., BATCH_SIZE, ...)` —
so `import model` raised ImportError in every torch-equipped
environment, i.e. precisely the paper's own environment class, while
the torch-less verification battery was structurally blind to it
(the torch-dependent modules skip there). The documented quick-start
command failed at HEAD while the battery read 224/0.

This module closes the CLASS of defect, not just the instance, in
three layers:

  1. STATIC — runs in EVERY environment, torch or not. For every
     Python source in the repository that imports names from another
     LOCAL module, each imported name is resolved by AST against the
     target module's module-level bindings. Target modules are parsed,
     never executed, so a missing torch can never mask a missing
     symbol again. Deletions of any still-imported constant fail here
     regardless of environment.
  2. DYNAMIC — torch-gated. When torch IS importable, the library
     core (model, federated_learning, losses, ...) is smoke-imported
     exactly as the quick-start would, proving the graph executable
     in the paper's environment class. (Entry-point scripts are
     covered statically by layer 1 without executing them.)
  3. INVENTORY — Dossier R-FS9-R7 (B4/B8 hardening): the set of local
     modules is PINNED, plain `import X` statements of local modules
     are counted, and the descriptive counts are asserted. The panel
     demonstrated that deleting an entire module reclassified its
     importers as external and skipped them; the pinned inventory
     makes deletion a loud failure, and the count assertions end
     descriptive drift (the v4.6 letter said 294 while the guard
     printed 295 — R-FS9-R7 B8). Since v4.7 the scan also walks
     provenance/ and data_manifest/ (previously unwalked).

Known conservative direction: names bound only under
`if TYPE_CHECKING:` are counted as bindings (they may not exist at
runtime), and star-imports of local modules are rejected outright
(none exist; explicit names are the house style).

Run:  python tests/test_import_graph.py   (or via the battery)
"""
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Directories whose .py sources participate in the local import graph.
# Dossier R-FS9-R7 (B4): provenance/ and data_manifest/ joined the
# scan at v4.7 (nine previously-unwalked tracked sources).
SCAN_DIRS = ["", "experiments", "tests", "tools", "leakage_audit",
             "provenance", "data_manifest"]

# Dossier R-FS9-R7 (B4): the pinned local-module inventory. Deleting
# or moving any module fails the battery until this pin is updated
# DELIBERATELY (the panel's demonstrated escape: a deleted module's
# importers were reclassified external and silently skipped).
EXPECTED_MODULES = frozenset({
    "centralized_baseline", "communication_cost", "config",
    "data_preparation", "evaluate_clients", "evaluation",
    "experiments", "experiments.analyze_smote_validity",
    "experiments.raw_substrate", "experiments.run_ablations",
    "experiments.run_alpha_promotion", "experiments.run_bn_diagnostic",
    "experiments.run_budget_matched",
    "experiments.run_calibration_comparison",
    "experiments.run_client_holdout", "experiments.run_event_level",
    "experiments.run_event_level_lstm",
    "experiments.run_event_level_raw", "experiments.run_federated_lstm",
    "experiments.run_gpu_queue", "experiments.run_interpretability",
    "experiments.run_lag_definition_sweep",
    "experiments.run_multiseed", "experiments.run_nobn_control",
    "experiments.run_partition_disjoint",
    "experiments.run_raw_substrate", "experiments.run_standard_metrics",
    "experiments.run_sweep", "federated_learning", "fix_encoding",
    "interpretability", "leakage_audit", "leakage_audit.audit_leakage",
    "load_cleaned_data", "losses", "main", "model",
    "partition_clients", "patch_reference_fields", "secure_aggregation",
    "tests", "tests.run_battery", "tests.test_audit_artifact",
    "tests.test_central_criterion", "tests.test_event_level_lstm",
    "tests.test_fl_smoke", "tests.test_gpu_queue",
    "tests.test_gpu_queue_artefacts", "tests.test_import_graph",
    "tests.test_interpretability_artifact",
    "tests.test_lag_sweep_artifact", "tests.test_leakage_gate",
    "tests.test_letters_manifest", "tests.test_pipeline_integrity",
    "tests.test_raw_lstm", "tests.test_raw_substrate",
    "tests.test_scaffold_algebra", "tests.test_submission_apparatus",
    "tools.build_submission", "tools.make_fig_clients",
    "tools.make_fig_partition", "visualize_results",
    "provenance.rebuild_audit_totals", "provenance.swansf_aggregate_meta",
    "provenance.swansf_audit_artifact", "provenance.swansf_event_level",
    "provenance.swansf_match_v4", "provenance.swansf_parse_partition",
    "provenance.swansf_verify_leakage", "data_manifest.generate_manifest",
    "data_manifest.verify_manifest",
})

# Dossier R-FS9-R7 (B8): asserted descriptive counts — drift fails
# loudly. Update DELIBERATELY when imports change.
EXPECTED_FROM_IMPORTS = 297   # `from <local module> import NAME` names
EXPECTED_PLAIN_IMPORTS = 38   # plain `import <local module>` statements
# (38 since v4.7: the interpretability guard's new argmax check adds a
# function-local `import config as cfg` — the count pin caught it on
# the first battery run, exactly as designed)

# Library core imported dynamically whenever torch is present — the
# transitive closure R8-1 broke. Entry scripts are NOT executed (they
# are, however, fully covered by the static layer as importers).
TORCH_SMOKE_MODULES = [
    "model", "losses", "federated_learning", "centralized_baseline",
    "evaluation", "evaluate_clients", "interpretability",
    "data_preparation", "partition_clients", "secure_aggregation",
    "communication_cost", "config",
]


def _module_map():
    """Map importable local module dotted-names to source files."""
    mods = {}
    for d in SCAN_DIRS:
        base = os.path.join(ROOT, d) if d else ROOT
        if not os.path.isdir(base):
            continue
        for fn in sorted(os.listdir(base)):
            if not fn.endswith(".py") or fn.startswith("__"):
                continue
            stem = fn[:-3]
            dotted = f"{d}.{stem}" if d else stem
            mods[dotted] = os.path.join(base, fn)
        # a directory with __init__.py is itself an importable package
        if os.path.exists(os.path.join(base, "__init__.py")):
            mods[d if d else "."] = os.path.join(base, "__init__.py")
    return mods


def _bindings(path):
    """Module-level names bound by the target source (AST, no exec).

    Walks the module body and the control-flow nodes that execute at
    import time (If / Try / With / For / While), collecting def/class
    names, assignment targets, and re-imported aliases. Interiors of
    functions and classes are excluded — those are not importable.
    """
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    names = set()

    def add_target(t):
        if isinstance(t, ast.Name):
            names.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                add_target(e)
        # Subscript/Attribute targets bind nothing new at module level.

    def walk_body(body):
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                names.add(stmt.name)
            elif isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    add_target(t)
            elif isinstance(stmt, ast.AnnAssign):
                add_target(stmt.target)
            elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                for a in stmt.names:
                    names.add(a.asname if a.asname else a.name.split(".")[0])
            elif isinstance(stmt, ast.If):
                walk_body(stmt.body)
                walk_body(stmt.orelse)
            elif isinstance(stmt, ast.Try):
                walk_body(stmt.body)
                for h in stmt.handlers:
                    if h.name:
                        names.add(h.name)
                    walk_body(h.body)
                walk_body(stmt.orelse)
                walk_body(stmt.finalbody)
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                for item in stmt.items:
                    if item.optional_vars is not None:
                        add_target(item.optional_vars)
                walk_body(stmt.body)
            elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
                if isinstance(stmt, (ast.For, ast.AsyncFor)):
                    add_target(stmt.target)
                walk_body(stmt.body)
                if getattr(stmt, "orelse", None):
                    walk_body(stmt.orelse)

    walk_body(tree.body)
    return names


def _sources():
    for d in SCAN_DIRS:
        base = os.path.join(ROOT, d) if d else ROOT
        if not os.path.isdir(base):
            continue
        for fn in sorted(os.listdir(base)):
            if fn.endswith(".py"):
                yield os.path.join(base, fn)


def static_resolution():
    """Layer 1: every local-name import resolves.

    Returns (from_checked, plain_checked, errors)."""
    mods = _module_map()
    errors = []
    checked = 0
    plain_checked = 0
    for src in _sources():
        with open(src, "r", encoding="utf-8") as f:
            try:
                tree = ast.parse(f.read(), filename=src)
            except SyntaxError as e:
                errors.append(f"{src}: syntax error {e}")
                continue
        rel = os.path.relpath(src, ROOT)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                # R-FS9-R7 (B4): plain imports of LOCAL modules are
                # counted (existence is implied by the module map); a
                # deleted module additionally fails the inventory pin.
                for a in node.names:
                    if a.name.split(".")[0] in mods:
                        plain_checked += 1
                continue
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level and node.level > 0:
                # relative import: resolve against the file's package
                pkg = os.path.dirname(rel).replace(os.sep, ".")
                modname = f"{pkg}.{node.module}" if node.module else pkg
            else:
                modname = node.module or ""
            # only local targets; everything else is external
            target = mods.get(modname)
            if target is None and modname:
                head = modname.split(".")[0]
                if head in mods or modname.split(".")[:-1] and \
                        ".".join(modname.split(".")[:-1]) in mods:
                    target = mods.get(modname)
            if target is None:
                continue  # external module (os, torch, sklearn, ...)
            # a package __init__.py or a package dir behaves the same:
            # imported names may be submodules OR __init__ bindings
            if os.path.isdir(target) or target.endswith("__init__.py"):
                pkgdir = target if os.path.isdir(target) \
                    else os.path.dirname(target)
                for a in node.names:
                    if a.name == "*":
                        errors.append(
                            f"{rel}: star-import of local package "
                            f"'{modname}' — explicit names required")
                        continue
                    sub = os.path.join(pkgdir, a.name + ".py")
                    if os.path.exists(sub):
                        checked += 1
                        continue
                    init = os.path.join(pkgdir, "__init__.py")
                    if a.name in _bindings(init):
                        checked += 1
                    else:
                        errors.append(
                            f"{rel}: 'from {modname} import {a.name}' — "
                            f"neither a submodule nor an __init__ binding")
                continue
            tgt_names = _bindings(target)
            for a in node.names:
                if a.name == "*":
                    errors.append(
                        f"{rel}: star-import of local module "
                        f"'{modname}' — explicit names required")
                    continue
                if a.name not in tgt_names:
                    errors.append(
                        f"{rel}: 'from {modname} import {a.name}' — "
                        f"'{a.name}' is not defined at module level in "
                        f"{os.path.relpath(target, ROOT)}")
                else:
                    checked += 1
    return checked, plain_checked, errors


def torch_smoke():
    """Layer 2: import the library core when torch is available."""
    try:
        import importlib.util
        if importlib.util.find_spec("torch") is None:
            return None, []  # torch absent -> static layer stands alone
    except Exception:
        return None, []
    import importlib
    errors = []
    for m in TORCH_SMOKE_MODULES:
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001 — report any import failure
            errors.append(f"torch-present smoke import of '{m}' failed: {e}")
    return len(TORCH_SMOKE_MODULES), errors


def main():
    failures = 0

    checked, plain_checked, errors = static_resolution()

    # layer 3 — pinned inventory (R-FS9-R7 B4): module deletion is loud
    mods = _module_map()
    missing = EXPECTED_MODULES - set(mods)
    extra = set(mods) - EXPECTED_MODULES

    static_ok = (not errors and not missing and not extra
                 and checked == EXPECTED_FROM_IMPORTS
                 and plain_checked == EXPECTED_PLAIN_IMPORTS)
    if static_ok:
        print(f"[PASS] static import-graph resolution: all {checked} "
              f"local-name imports and {plain_checked} plain local "
              f"imports resolve by AST; the {len(EXPECTED_MODULES)}-"
              f"module inventory is pinned (counts asserted — B8; "
              f"environment-independent, torch absence cannot mask "
              f"a symbol)")
    else:
        failures += 1
        print("[FAIL] static import-graph resolution:")
        for e in errors:
            print(f"       {e}")
        if missing:
            print(f"       modules missing vs the pinned inventory: "
                  f"{sorted(missing)} — a module was deleted or moved "
                  f"(R-FS9-R7 B4); update EXPECTED_MODULES deliberately")
        if extra:
            print(f"       modules unknown to the pinned inventory: "
                  f"{sorted(extra)} — update EXPECTED_MODULES "
                  f"deliberately when adding modules")
        if checked != EXPECTED_FROM_IMPORTS:
            print(f"       local-name import count: resolved {checked}, "
                  f"pinned {EXPECTED_FROM_IMPORTS} — descriptive drift "
                  f"(R-FS9-R7 B8); update the pin deliberately")
        if plain_checked != EXPECTED_PLAIN_IMPORTS:
            print(f"       plain local import count: resolved "
                  f"{plain_checked}, pinned {EXPECTED_PLAIN_IMPORTS} "
                  f"— update the pin deliberately")

    n, smoke_errors = torch_smoke()
    if n is None:
        # R-FS9-R7 (B9): a skipped check is NOT counted as passed —
        # the RESULT below excludes it in torch-less environments.
        print("[SKIP] torch-present smoke import (torch not importable "
              "here — the static layer above still enforces the class)")
        smoke_ok = None
    elif smoke_errors:
        failures += 1
        print(f"[FAIL] torch-present smoke import: "
              f"{len(smoke_errors)} module(s) failed")
        for e in smoke_errors:
            print(f"       {e}")
        smoke_ok = False
    else:
        print(f"[PASS] torch-present smoke import: {n} library modules "
              f"import cleanly in the paper's environment class")
        smoke_ok = True

    passed = (1 if static_ok else 0) + (1 if smoke_ok else 0)
    skipped = 1 if smoke_ok is None else 0
    failed = failures
    print(f"RESULT: {passed} passed, {failed} failed"
          + (f" ({skipped} skipped — torch-gated, not counted)"
             if skipped else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
