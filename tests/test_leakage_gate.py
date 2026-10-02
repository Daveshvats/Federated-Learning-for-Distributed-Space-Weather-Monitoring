#!/usr/bin/env python3
"""
tests/test_leakage_gate.py — exercise the leakage-gate failure path
(Dossier R-FS9-R1 item D1).

v4.0 wired region ids into the runtime audit so the gate FAILS loudly
on the shipped in-partition protocol (2,999 shared ARs) instead of
silently skipping the region check — but no test in the v4.0 battery
proved the gate actually fires. This module exercises, on synthetic
indices (no dataset, no torch):

  1. the failure path:  shared regions across train/test ->
     region_leakage fails, overall_pass False, and the main.py gate
     decision rule (refuse unless --allow-in-partition) evaluates
     exactly as the pipeline applies it;
  2. the clean path:    disjoint regions -> region check passes;
  3. the skip path:     region_ids=None -> skipped with pass=None and
     the gate does NOT refuse (documented v3.x semantics);
  4. window overlap:    identical sample keys on both sides -> fail.

Run:  python -m unittest tests.test_leakage_gate -v   (or via the battery)
"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from leakage_audit.audit_leakage import run_audit  # noqa: E402


def _gate_decision(audit, region_ids_available, allow_in_partition):
    """Replicates the main.py gate branch exactly (main.py:270-284):
    refuse (raise) iff the audit failed AND region ids were loaded AND
    the operator did not pass --allow-in-partition."""
    if not audit.get("overall_pass"):
        if region_ids_available and not allow_in_partition:
            return "REFUSE"
        return "ALLOW-QUANTIFIED"
    return "ALLOW-CLEAN"


class LeakageGateFailurePath(unittest.TestCase):

    def test_shared_regions_fail_and_gate_refuses(self):
        # 12 regions; regions 0-4 appear on BOTH sides of the boundary
        rng = np.random.default_rng(0)
        region_ids = rng.integers(0, 12, size=400)
        train_idx = np.arange(0, 200)
        val_idx = np.arange(200, 240)
        test_idx = np.arange(240, 400)   # shares regions with train
        audit = run_audit(train_idx=train_idx, val_idx=val_idx,
                          test_idx=test_idx, region_ids=region_ids)
        self.assertFalse(audit["region_leakage"]["pass"])
        self.assertFalse(audit["overall_pass"])
        self.assertEqual(_gate_decision(audit, True, False), "REFUSE")
        # the operator override exists and is honoured
        self.assertEqual(_gate_decision(audit, True, True),
                         "ALLOW-QUANTIFIED")

    def test_disjoint_regions_pass(self):
        region_ids = np.concatenate([np.repeat(np.arange(0, 50), 4),
                                     np.repeat(np.arange(50, 100), 4),
                                     np.repeat(np.arange(100, 150), 4)])
        train_idx = np.arange(0, 200)
        val_idx = np.arange(200, 400)
        test_idx = np.arange(400, 600)
        audit = run_audit(train_idx=train_idx, val_idx=val_idx,
                          test_idx=test_idx, region_ids=region_ids)
        self.assertTrue(audit["region_leakage"]["pass"])
        self.assertTrue(audit["overall_pass"])
        self.assertEqual(_gate_decision(audit, True, False), "ALLOW-CLEAN")

    def test_missing_region_ids_skips_but_gate_stays_open(self):
        # v3.x semantics: no region table -> the region check is not
        # run at all (the report omits the key), and the gate stays
        # open on the strength of the index-level checks
        region_ids = np.repeat(np.arange(0, 10), 10)
        train_idx = np.arange(0, 50)
        val_idx = np.arange(50, 70)
        test_idx = np.arange(70, 100)
        audit = run_audit(train_idx=train_idx, val_idx=val_idx,
                          test_idx=test_idx, region_ids=None)
        self.assertNotIn("region_leakage", audit)
        self.assertTrue(audit["split_exclusivity"]["pass"])
        self.assertTrue(audit["overall_pass"])
        self.assertEqual(_gate_decision(audit, True, False), "ALLOW-CLEAN")

    def test_window_overlap_detects_same_series_split(self):
        # numeric window origins from ONE series at stride 1, window 60:
        # splitting at 60/80 puts train and test windows inside each
        # other's span — the detector must flag the boundary
        origins = np.arange(0, 100)
        audit = run_audit(train_idx=np.arange(0, 60),
                          val_idx=np.arange(60, 80),
                          test_idx=np.arange(80, 100),
                          sample_keys=origins, stride=1, window=60)
        self.assertFalse(audit["window_overlap"]["pass"])
        self.assertGreater(audit["window_overlap"]["boundary_violations"], 0)

    def test_window_overlap_clean_when_series_separated(self):
        # origins spaced a full window apart never overlap
        origins = np.arange(0, 100) * 60
        audit = run_audit(train_idx=np.arange(0, 60),
                          val_idx=np.arange(60, 80),
                          test_idx=np.arange(80, 100),
                          sample_keys=origins, stride=1, window=60)
        self.assertTrue(audit["window_overlap"]["pass"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
