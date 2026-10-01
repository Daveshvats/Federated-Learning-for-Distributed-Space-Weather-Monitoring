#!/usr/bin/env python3
"""
tests/test_gpu_queue_artefacts.py — guard the v3.8 GPU-queue artefacts.

Pure-JSON checks (no torch, no data): the six committed artefacts of
the second owner-GPU batch must keep their protocol fingerprints,
frozen substrate identity, embedded frozen-counterpart bit-matches,
and the headline event-level counts that the paper (v3.8) reports.

Run:  python -m tests.test_gpu_queue_artefacts   (or via the battery)
"""
import json
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")

SUBSTRATE = {"train": 214888, "val": 40932, "test": 75365,
             "test_pos": 0.013136071152985096}


def _load(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return json.load(f)


def _test(res):
    return res["test"]


class GPUScaffoldArtefact(unittest.TestCase):
    """outputs/raw_lstm_scaffold.json + event_level_raw_scaffold.json"""

    @classmethod
    def setUpClass(cls):
        cls.w = _load("raw_lstm_scaffold.json")
        cls.e = _load("event_level_raw_scaffold.json")

    def test_protocol(self):
        p = self.w["protocol"]
        self.assertEqual(
            (p["seed"], p["rounds"], p["clients"], p["alpha"], p["mu"],
             p["tag"], p["smoke"], p["scaffold"], p["smote"]),
            (42, 50, 6, 1.0, 0.01, "scaffold", False, True, False))
        self.assertTrue(str(p["device"]).startswith("cuda"))

    def test_substrate_identity(self):
        s = self.w["substrate"]
        for k, v in SUBSTRATE.items():
            self.assertAlmostEqual(s[k], v, places=12, msg=k)

    def test_window_metrics(self):
        t = _test(self.w["results"]["scaffold_lstm"])
        self.assertAlmostEqual(t["roc_auc"], 0.970, delta=5e-4)
        self.assertAlmostEqual(t["pr_auc"], 0.294, delta=5e-4)
        self.assertAlmostEqual(t["prevalence"], 990 / 75365, places=9)

    def test_mlp_counterpart_bitmatch(self):
        ref = _load("raw_substrate_eval.json")["results"]["scaffold_mlp"]
        cp = self.w["results"]["scaffold_lstm"]["rawsubstr_mlp_counterpart"]
        self.assertAlmostEqual(
            cp["roc_auc"], ref["test"]["roc_auc"], places=12)
        self.assertAlmostEqual(
            cp["pr_auc"], ref["test"]["pr_auc"], places=12)

    def test_event_counts(self):
        m = self.e["models"]["scaffold_lstm"]
        self.assertEqual(self.e["n_true_events"], 65)
        self.assertEqual(self.e["flare_peak_match_rate"], 1.0)
        self.assertEqual(m["n_detected_events"], 55)
        self.assertAlmostEqual(
            m["false_alarm_windows_per_day"], 0.41, delta=0.02)
        self.assertAlmostEqual(
            m["lead_to_flare_peak_minutes"]["median"] / 60, 23.2, delta=0.5)

    def test_thresholds_consistent(self):
        self.assertAlmostEqual(
            self.e["models"]["scaffold_lstm"]["threshold"],
            self.w["results"]["scaffold_lstm"]["threshold"], places=9)


class GPUSeed43Artefact(unittest.TestCase):
    """outputs/raw_lstm_seed43.json + event_level_raw_seed43.json"""

    @classmethod
    def setUpClass(cls):
        cls.w = _load("raw_lstm_seed43.json")
        cls.e = _load("event_level_raw_seed43.json")
        cls.ref42 = _load("raw_lstm_eval.json")["results"]

    def test_protocol(self):
        p = self.w["protocol"]
        self.assertEqual((p["seed"], p["tag"], p["smoke"]), (43, "seed43",
                                                             False))
        self.assertEqual(
            set(self.w["results"]),
            {"central_lstm", "fedavg_lstm", "fedprox_lstm", "scaffold_lstm"})

    def test_substrate_identity(self):
        s = self.w["substrate"]
        for k, v in SUBSTRATE.items():
            self.assertAlmostEqual(s[k], v, places=12, msg=k)

    def test_window_metrics(self):
        exp = {"central_lstm": (0.962, 0.357),
               "fedavg_lstm": (0.961, 0.271),
               "fedprox_lstm": (0.965, 0.381),
               "scaffold_lstm": (0.968, 0.368)}
        for arm, (ro, pr) in exp.items():
            t = _test(self.w["results"][arm])
            self.assertAlmostEqual(t["roc_auc"], ro, delta=5e-4, msg=arm)
            self.assertAlmostEqual(t["pr_auc"], pr, delta=5e-4, msg=arm)

    def test_mlp_counterparts_bitmatch(self):
        ref = _load("raw_substrate_eval.json")["results"]
        m = {"central_lstm": "centralized_mlp", "fedavg_lstm": "fedavg_mlp",
             "fedprox_lstm": "fedprox_mlp", "scaffold_lstm": "scaffold_mlp"}
        for arm, mk in m.items():
            cp = self.w["results"][arm]["rawsubstr_mlp_counterpart"]
            self.assertAlmostEqual(cp["roc_auc"], ref[mk]["test"]["roc_auc"],
                                   places=12, msg=arm)
            self.assertAlmostEqual(cp["pr_auc"], ref[mk]["test"]["pr_auc"],
                                   places=12, msg=arm)

    def test_lstm_seed42_counterparts_bitmatch(self):
        for arm in ("central_lstm", "fedavg_lstm", "fedprox_lstm"):
            cp = self.w["results"][arm]["lstm_seed42_counterpart"]
            self.assertAlmostEqual(
                cp["roc_auc"], _test(self.ref42[arm])["roc_auc"],
                places=12, msg=arm)

    def test_cross_seed_roc_stability(self):
        for arm in ("central_lstm", "fedavg_lstm", "fedprox_lstm"):
            self.assertLessEqual(
                abs(_test(self.w["results"][arm])["roc_auc"] -
                    _test(self.ref42[arm])["roc_auc"]), 0.005, msg=arm)

    def test_event_counts(self):
        exp = {"central_lstm": 43, "fedavg_lstm": 36,
               "fedprox_lstm": 45, "scaffold_lstm": 50}
        self.assertEqual(self.e["flare_peak_match_rate"], 1.0)
        for arm, n in exp.items():
            self.assertEqual(
                self.e["models"][arm]["n_detected_events"], n, msg=arm)


class GPUSmoteArtefact(unittest.TestCase):
    """outputs/raw_lstm_smote.json + event_level_raw_smote.json"""

    @classmethod
    def setUpClass(cls):
        cls.w = _load("raw_lstm_smote.json")
        cls.e = _load("event_level_raw_smote.json")
        cls.ref42 = _load("raw_lstm_eval.json")["results"]

    def test_protocol(self):
        p = self.w["protocol"]
        self.assertEqual((p["seed"], p["tag"], p["smote"],
                          p["smote_ratio"], p["scaffold"]),
                         (42, "smote", True, 0.25, False))
        self.assertEqual(set(self.w["results"]),
                         {"fedavg_lstm", "fedprox_lstm"})

    def test_substrate_identity(self):
        s = self.w["substrate"]
        for k, v in SUBSTRATE.items():
            self.assertAlmostEqual(s[k], v, places=12, msg=k)

    def test_window_metrics_and_deltas(self):
        t = _test(self.w["results"]["fedavg_lstm"])
        self.assertAlmostEqual(t["roc_auc"], 0.961, delta=5e-4)
        self.assertAlmostEqual(t["pr_auc"], 0.307, delta=5e-4)
        t = _test(self.w["results"]["fedprox_lstm"])
        self.assertAlmostEqual(t["roc_auc"], 0.946, delta=5e-4)
        self.assertAlmostEqual(t["pr_auc"], 0.288, delta=5e-4)
        # the ablation's headline negative: SMOTE costs FedProx ~0.116 PR
        d = (_test(self.w["results"]["fedprox_lstm"])["pr_auc"] -
             _test(self.ref42["fedprox_lstm"])["pr_auc"])
        self.assertTrue(-0.13 <= d <= -0.10, f"delta {d:+.4f}")

    def test_event_counts(self):
        self.assertEqual(self.e["flare_peak_match_rate"], 1.0)
        self.assertEqual(
            self.e["models"]["fedavg_lstm"]["n_detected_events"], 40)
        self.assertEqual(
            self.e["models"]["fedprox_lstm"]["n_detected_events"], 37)


if __name__ == "__main__":
    unittest.main(verbosity=2)
