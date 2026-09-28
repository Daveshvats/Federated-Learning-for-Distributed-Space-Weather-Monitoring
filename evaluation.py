"""
evaluation.py  (v3.0 — improvements branch)
────────────────────────────────────────────
Threshold selection, operational metrics, and probability calibration.

WHY THIS MODULE EXISTS (audit findings B1, B4, B8, B10):
  - The old pipeline searched F-beta-optimal thresholds directly on the
    TEST set. All thresholded metrics were optimistically biased.
  - The old pipeline never recomputed FL accuracy after thresholding
    (stale-accuracy bug) and never reported PR-AUC / Brier / recall@FPR
    despite a 1.9%-positive test set, where ROC-AUC is misleadingly
    optimistic and precision is the binding constraint.
  - Training prevalence (~49% after RUS balancing) and test prevalence
    (~1.9%) differ by ~25x: raw probabilities are unusable for
    operational decisions without calibration.

PROTOCOL (enforced by main.py, verified by tests):
  1. fit calibration on VALIDATION probabilities
  2. search the F-beta-optimal threshold on (calibrated) VALIDATION probs
  3. freeze calibration + threshold
  4. apply exactly once to TEST probabilities
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve, roc_curve,
)


# ─────────────────────────────────────────────────────────────────────────────
# Operational metrics
# ─────────────────────────────────────────────────────────────────────────────

def recall_at_fpr(y_true, y_probs, fpr_targets=(0.005, 0.01, 0.02, 0.05)):
    """
    Recall achievable at fixed false-positive-rate budgets.
    On a 1.9%-positive test set these are the metrics an operator
    actually cares about (alarm budget), not ROC-AUC.
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    out = {}
    for target in fpr_targets:
        idx = np.searchsorted(fpr, target, side="left")
        idx = min(idx, len(tpr) - 1)
        out[f"recall@{target*100:g}%FPR"] = float(tpr[idx])
    return out


def expected_calibration_error(y_true, y_probs, n_bins=15):
    """ECE: weighted bin-wise |accuracy - confidence|."""
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (y_probs > lo) & (y_probs <= hi)
        if in_bin.sum() == 0:
            continue
        acc = y_true[in_bin].mean()
        conf = y_probs[in_bin].mean()
        ece += (in_bin.sum() / len(y_true)) * abs(acc - conf)
    return float(ece)


def compute_all_metrics(y_true, y_probs, threshold, beta=2.0):
    """
    Complete metric set at a given threshold. Recomputes EVERYTHING
    (fixes stale-accuracy bug B4).
    """
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs, dtype=float)
    preds = (y_probs >= threshold).astype(int)

    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    f2 = fbeta_score(y_true, preds, beta=beta, zero_division=0)

    metrics = {
        "threshold": float(threshold),
        "accuracy":  float(accuracy_score(y_true, preds)),
        "precision": float(prec),
        "recall":    float(rec),
        "f1":        float(f1),
        f"f{beta:.0f}": float(f2),
        "roc_auc":   float(roc_auc_score(y_true, y_probs)),
        "pr_auc":    float(average_precision_score(y_true, y_probs)),
        "brier":     float(brier_score_loss(y_true, y_probs)),
        "ece":       expected_calibration_error(y_true, y_probs),
        "prevalence": float(y_true.mean()),
    }
    metrics.update(recall_at_fpr(y_true, y_probs))
    metrics["preds"] = preds
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Threshold selection — VALIDATION ONLY
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_threshold_fbeta(y_true, y_probs, beta=2.0,
                                 grid=(0.05, 0.95, 0.005)):
    """
    Pure threshold search (no printing, no side effects). Caller is
    responsible for passing VALIDATION data — main.py enforces this and
    tests/test_pipeline_integrity.py verifies the call contract.
    """
    lo, hi, step = grid
    thresholds = np.arange(lo, hi + 1e-9, step)
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs, dtype=float)

    best_fb, best_thresh = -1.0, 0.5
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        fb = fbeta_score(y_true, preds, beta=beta, zero_division=0)
        if fb > best_fb:
            best_fb, best_thresh = fb, float(t)
    return best_thresh, float(best_fb)


def select_operating_point(y_true, y_probs, beta=2.0, grid=(0.05, 0.95, 0.005)):
    """Threshold search + full validation metrics at the chosen point."""
    t, fb = find_optimal_threshold_fbeta(y_true, y_probs, beta=beta, grid=grid)
    m = compute_all_metrics(y_true, y_probs, t, beta=beta)
    return t, m


# ─────────────────────────────────────────────────────────────────────────────
# Calibration (fit on VALIDATION, apply to TEST)
# ─────────────────────────────────────────────────────────────────────────────

class PriorShiftCorrection:
    """
    Corrects the train->test prior shift by logit-space reweighting:

        p_adj = p * (pi_test/pi_train) / [p*(pi_test/pi_train) + (1-p)*(1-pi_test)/(1-pi_train)]

    Uses VALIDATION prevalence as the operational-prevalence estimate.
    This is the theoretically-correct correction for a fixed classifier
    under prior shift (Saerens et al., 2002).
    """
    name = "prior_shift"

    def fit(self, y_val, p_val):
        self.train_rate_ = float(np.mean(y_val))  # training-pool prevalence
        return self

    def set_prevalences(self, train_rate, test_rate):
        self.train_rate_ = float(train_rate)
        self.test_rate_ = float(test_rate)
        return self

    def transform(self, p):
        p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
        tr = getattr(self, "train_rate_", 0.5)
        te = getattr(self, "test_rate_", 0.5)
        odds = (p / (1 - p)) * (te * (1 - tr)) / (tr * (1 - te) + 1e-12)
        return np.clip(odds / (1 + odds), 1e-7, 1 - 1e-7)


class PlattScaling:
    """Logistic regression on log-odds (Platt, 1999)."""
    name = "platt"

    def fit(self, y_val, p_val):
        from sklearn.linear_model import LogisticRegression
        p_val = np.clip(np.asarray(p_val, dtype=float), 1e-7, 1 - 1e-7)
        logit = np.log(p_val / (1 - p_val)).reshape(-1, 1)
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000)
        self.lr_.fit(logit, np.asarray(y_val).astype(int))
        return self

    def transform(self, p):
        p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
        logit = np.log(p / (1 - p)).reshape(-1, 1)
        return self.lr_.predict_proba(logit)[:, 1]


class IsotonicCalibration:
    """Isotonic regression (non-parametric monotone map)."""
    name = "isotonic"

    def fit(self, y_val, p_val):
        from sklearn.isotonic import IsotonicRegression
        self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        self.iso_.fit(np.asarray(p_val, dtype=float),
                      np.asarray(y_val).astype(int))
        return self

    def transform(self, p):
        return np.asarray(self.iso_.predict(np.asarray(p, dtype=float)))


class TemperatureScaling:
    """
    Single-parameter logit temperature scaling (Guo et al., 2017).
    NLL-optimal T found by Golden-section search on validation.
    """
    name = "temperature"

    def fit(self, y_val, p_val):
        p_val = np.clip(np.asarray(p_val, dtype=float), 1e-7, 1 - 1e-7)
        z = np.log(p_val / (1 - p_val))
        y = np.asarray(y_val).astype(int)

        def nll(T):
            zz = z / max(T, 1e-6)
            # numerically stable BCE with logits
            nll_v = np.maximum(zz, 0) - zz * y + np.log1p(np.exp(-np.abs(zz)))
            return float(nll_v.mean())

        lo, hi = 1e-3, 100.0
        for _ in range(200):  # golden-section
            m1 = lo + 0.382 * (hi - lo)
            m2 = lo + 0.618 * (hi - lo)
            if nll(m1) < nll(m2):
                hi = m2
            else:
                lo = m1
        self.T_ = float((lo + hi) / 2)
        return self

    def transform(self, p):
        p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
        z = np.log(p / (1 - p)) / self.T_
        return 1.0 / (1.0 + np.exp(-z))


class IdentityCalibration:
    name = "none"

    def fit(self, y_val, p_val):
        return self

    def transform(self, p):
        return np.asarray(p, dtype=float)


_CALIBRATORS = {
    "none": IdentityCalibration,
    "prior_shift": PriorShiftCorrection,
    "platt": PlattScaling,
    "isotonic": IsotonicCalibration,
    "temperature": TemperatureScaling,
}


def make_calibrator(method):
    if method not in _CALIBRATORS:
        raise ValueError(f"unknown calibration method: {method}")
    return _CALIBRATORS[method]()


def select_calibration(y_val, p_val, methods=("none", "prior_shift", "platt",
                                              "isotonic", "temperature")):
    """
    Select the calibration method by VALIDATION Brier score
    (threshold-independent proper scoring rule).
    """
    results = {}
    best_name, best_brier = None, np.inf
    for m in methods:
        try:
            cal = make_calibrator(m).fit(y_val, p_val)
            p_cal = cal.transform(p_val)
            b = brier_score_loss(np.asarray(y_val).astype(int), p_cal)
            results[m] = b
            if b < best_brier:
                best_name, best_brier = m, b
        except Exception as e:
            results[m] = f"failed: {e}"
    return best_name, results
