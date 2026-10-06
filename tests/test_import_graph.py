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

  4. SCOPE (Dossier R-FS9-R10 B6-c, v4.9.2): external imports are
     classified by WHERE THEY COME FROM, not by a hand-pinned name
     list. The v4.8 C6 whitelist was name-based: a name on the list
     passed regardless of source, a name off it failed regardless of
     scope, and every stdlib import the cycle ever added ("gc" at
     v4.9, "ctypes" at v4.9.1) required a DELIBERATE whitelist edit
     for a module that was never a dependency at all. Now:
       - stdlib scope: `sys.stdlib_module_names` (the interpreter's
         own authoritative set — environment-viral, never stale)
       - third-party scope: requirements.txt, parsed at run time
         (never hand-typed; a NEW dependency still fails DELIBERATELY
         until requirements.txt is updated — the C6 discipline,
         kept), plus a site-packages origin check for whatever is
         importable in the running environment (a name that resolves
         OUTSIDE site-packages/prefix and outside the repository is
         an error, not an implicit external)
       - anything resolving INSIDE the repository but outside the
         pinned module inventory is an error (a SCAN_DIRS gap is
         loud, not silently external)

Known conservative direction: names bound only under
`if TYPE_CHECKING:` are counted as bindings (they may not exist at
runtime), and star-imports of local modules are rejected outright
(none exist; explicit names are the house style).

Run:  python tests/test_import_graph.py   (or via the battery)
"""
import ast
import importlib.util
import os
import re
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
    "experiments.run_arm_b_central_sanity",
    "experiments.run_partition_disjoint",
    "experiments.run_raw_bn_diagnostic",
    "experiments.run_raw_nobn",
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
    "tests.test_raw_bn_diagnostic", "tests.test_raw_lstm",
    "tests.test_raw_substrate", "tests.test_region_disjoint",
    "tests.test_region_meta",
    "tests.test_seed43_replication",
    "tests.test_region_disjoint_verdict",
    "tests.test_scaffold_algebra",
    "tests.test_submission_apparatus",
    "tools.build_submission", "tools.make_fig_clients",
    "tools.make_fig_partition", "visualize_results",
    "provenance.rebuild_audit_totals", "provenance.swansf_aggregate_meta",
    "provenance.swansf_audit_artifact", "provenance.swansf_event_level",
    "provenance.swansf_match_v4", "provenance.swansf_parse_partition",
    "provenance.swansf_verify_leakage", "data_manifest.generate_manifest",
    "data_manifest.verify_manifest",
})

# Dossier R-FS9-R8 (C6, folded in at v4.8) — REWORKED at v4.9.2 into
# the SCOPE-BASED classification (R-FS9-R10 B6-c): external imports
# are classified by where they come from, not by a hand-pinned name
# list. The name-based whitelist's two demonstrated failures: every
# stdlib import the cycle added ("gc" v4.9, "ctypes" v4.9.1) needed a
# deliberate pin for a non-dependency, and a listed name passed
# regardless of whether it was still a declared dependency. Now the
# stdlib scope is the interpreter's own sys.stdlib_module_names, the
# third-party scope is requirements.txt parsed at run time (never
# hand-typed), and anything importable is additionally verified to
# actually LIVE in the environment (site-packages / interpreter
# prefix) — a name that resolves somewhere else, or inside this
# repository but outside the pinned inventory, is an error.
#
# An import whose head is neither a local module, a local module's
# STEM (the sys.path-inserted script style — run_gpu_queue,
# run_multiseed, swansf_audit_artifact), nor in a declared scope is
# an ERROR — the unresolved-name escape (disclosed at v4.6, closed at
# v4.8) stays closed.
#
# Distribution-name -> import-name aliases for requirements.txt
# entries whose PyPI name differs from the importable module.
DIST_TO_IMPORT = {"scikit-learn": "sklearn",
                   "imbalanced-learn": "imblearn"}


def _stdlib_scope():
    """The interpreter's authoritative stdlib set (Python >= 3.10).
    '__future__' is a stdlib module but is listed separately for
    older interpreters' benefit."""
    names = set(getattr(sys, "stdlib_module_names", ()))
    names.add("__future__")
    names.add("builtins")
    return frozenset(names)


def _third_party_scope():
    """The project's DECLARED third-party scope: requirements.txt,
    parsed at run time — never hand-typed (the v4.0 convention)."""
    path = os.path.join(ROOT, "requirements.txt")
    out = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                dist = re.split(r"[<>=!~\s;\[]", line, 1)[0].strip()
                if dist:
                    out.add(DIST_TO_IMPORT.get(dist, dist))
    return frozenset(out)


def _resolve_origin(head):
    """Where does an importable head actually live? Returns one of
    'stdlib', 'third_party', 'repo', 'other', or None (not importable
    in this environment)."""
    try:
        spec = importlib.util.find_spec(head)
    except Exception:
        return None
    if spec is None:
        return None
    origin = os.path.realpath(spec.origin) if spec.origin else ""
    if not origin:
        # built-in / frozen modules have no file origin
        return "stdlib"
    if "site-packages" in origin or "dist-packages" in origin:
        return "third_party"
    for base in (os.path.realpath(sys.prefix),
                 os.path.realpath(getattr(sys, "base_prefix", ""))):
        if base and origin.startswith(base):
            return "stdlib"
    root_real = os.path.realpath(ROOT)
    if origin.startswith(root_real):
        return "repo"
    return "other"


def _classify_external(head, rel, full):
    """R-FS9-R10 B6-c scope classification for a non-local import
    head. Returns an error message or None (pass).

    Order: stdlib scope first (never a dependency, never a pin — the
    'gc'/'ctypes' churn ends); then origin verification for whatever
    is importable (inside the repo but unpinned = a SCAN_DIRS gap;
    outside every declared scope = an error, not an implicit
    external); then the declared-dependency scope (requirements.txt,
    parsed at run time) for everything else, including heads not
    installed in THIS environment (the torch-less battery: a
    declared dependency passes without being importable — scope,
    not availability, is what the static layer checks)."""
    if head in STDLIB_SCOPE:
        return None
    origin = _resolve_origin(head)
    if origin == "stdlib":
        return None
    if origin == "repo":
        return (f"{rel}: import of unresolved module '{full}' — it "
                f"resolves INSIDE the repository but outside the "
                f"pinned module inventory (a SCAN_DIRS gap is loud, "
                f"R-FS9-R10 B6-c)")
    if origin == "other":
        return (f"{rel}: import of unresolved module '{full}' — it "
                f"resolves outside every declared scope (neither "
                f"stdlib, nor site-packages, nor this repository) — "
                f"R-FS9-R10 B6-c")
    # origin None (not installed here) or 'third_party': the
    # declared-dependency scope decides
    if head in THIRD_PARTY_SCOPE:
        return None
    return (f"{rel}: import of unresolved module '{full}' — not a "
            f"local module or stem, not stdlib scope, and not a "
            f"declared dependency in requirements.txt (R-FS9-R8 C6 / "
            f"R-FS9-R10 B6-c)")


STDLIB_SCOPE = _stdlib_scope()
THIRD_PARTY_SCOPE = _third_party_scope()

# Dossier R-FS9-R7 (B8): asserted descriptive counts — drift fails
# loudly. Update DELIBERATELY when imports change.
EXPECTED_FROM_IMPORTS = 340  # `from <local module> import NAME` names
# (340 since v4.12.1: +2 — tests/test_region_disjoint.py's `from
# experiments.run_raw_nobn import (TRAIN_PARTS,
# region_disjoint_split)` — the region-disjoint kit-errata guard;
# 338 since v4.12: +2 — experiments/run_raw_nobn.py's run kit grew
# its raw-substrate from-import from (build, CACHE) to
# (build, CACHE, TRAIN_PARTS, load_labels) for the region-disjoint
# validation carve (the external re-review's item 2);
# 336 since v4.9.3: +1 — tests/test_raw_bn_diagnostic.py layer 3b's
# dynamic guard does `from experiments.run_raw_nobn import
# select_round_by_val_roc` (the ask-#9 crash regression guard, the
# runner's helper exercised for real); 335 was the v4.9.2 figure:
# +26 from the two R-FS9-R10 register runners —
# run_arm_b_central_sanity.py [config/evaluation/experiments.raw_
# substrate/partition_clients imports + the function-local model
# imports] and run_raw_nobn.py [the same families + federated_
# learning/model via _load_torch_stack]; 309 was the v4.8-v4.9.1
# figure: +10 from run_raw_bn_diagnostic.py and +2 bare-stem
# from-imports that the C6 stem-map resolution counts)
EXPECTED_PLAIN_IMPORTS = 45  # plain `import <local module>` statements
# (45 since v4.13.2: +1 — tests/test_region_meta.py's function-local
# `import experiments.run_raw_nobn as runner` — the repo-shipped parse
# metadata guard, exercising the runner's own pre-flight/load paths;
# 44 since v4.12.1: +1 — tests/test_region_disjoint.py's `import
# config as cfg`; 43 since v4.9.2: +3 from the two register runners — each runner's
# `import config as cfg` plus run_raw_nobn.py's function-local
# `import model as _model_mod` inside _load_torch_stack; 40 was the
# v4.8-v4.9.1 figure; external plain imports such as torch.nn are
# counted only when their head is a LOCAL module)
# ("gc" joined the v4.9 whitelist: the raw-lstm win32 teardown guard's
# function-local `import gc` — stdlib, previously unpinned)
# ("ctypes" joined the v4.9.1 whitelist the same way, with NO count
# change: ctypes is an external plain import, and plain imports are
# counted only when their head is a LOCAL module)
# (v4.9.2 / R-FS9-R10 B6-c: the name-based EXTERNAL_MODULES
# whitelist is REPLACED by the scope-based classification — stdlib
# scope from sys.stdlib_module_names, third-party scope parsed from
# requirements.txt at run time, origin verification for whatever is
# importable; the counts above are unaffected by the rework)

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
    # R-FS9-R8 (C6): bare-stem view of the local inventory — scripts
    # that sys.path-insert their own directory import siblings by
    # stem (run_gpu_queue, run_multiseed, swansf_audit_artifact).
    stem_map = {}
    for dotted, path in mods.items():
        stem_map.setdefault(dotted.split(".")[-1], path)
    local_heads = set(mods) | set(stem_map)
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
                # R-FS9-R7 (B4): plain imports of LOCAL modules (by
                # dotted name OR by stem) are counted; a deleted
                # module additionally fails the inventory pin.
                # R-FS9-R10 (B6-c): anything else is classified by
                # SCOPE — stdlib (the interpreter's own set), the
                # declared requirements.txt dependency scope, or an
                # origin check — an unresolved name is an ERROR, not
                # an implicit external.
                for a in node.names:
                    head = a.name.split(".")[0]
                    if head in local_heads:
                        plain_checked += 1
                    else:
                        err = _classify_external(head, rel, a.name)
                        if err:
                            errors.append(err)
                continue
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level and node.level > 0:
                # relative import: resolve against the file's package
                pkg = os.path.dirname(rel).replace(os.sep, ".")
                modname = f"{pkg}.{node.module}" if node.module else pkg
            else:
                modname = node.module or ""
            # only local targets; everything else must be whitelisted
            target = mods.get(modname)
            if target is None and modname:
                head = modname.split(".")[0]
                if head in mods or modname.split(".")[:-1] and \
                        ".".join(modname.split(".")[:-1]) in mods:
                    target = mods.get(modname)
                # R-FS9-R8 (C6): bare-stem from-imports (script-dir
                # style) resolve through the stem map
                if target is None and modname in stem_map:
                    target = stem_map[modname]
            if target is None:
                # external module (os, torch, sklearn, ...) — or an
                # unresolved name, which is an ERROR (R-FS9-R8 C6:
                # external-by-default was the escape hatch; R-FS9-R10
                # B6-c: the classification is now scope-based)
                head = (modname or "?").split(".")[0]
                if modname:
                    if head in local_heads:
                        # a LOCAL head whose deeper path did not
                        # resolve (e.g. from experiments.<missing>
                        # import X) — an unresolved local import, the
                        # original C6 error path
                        errors.append(
                            f"{rel}: from-import of unresolved module "
                            f"'{modname}' — the local head resolves but "
                            f"the full module path does not")
                    else:
                        err = _classify_external(head, rel, modname)
                        if err:
                            errors.append(err)
                continue
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
