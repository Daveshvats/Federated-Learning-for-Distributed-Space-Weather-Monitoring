"""
centralized_baseline.py  (v3.0 — improvements branch)
─────────────────────────────────────────────────────
Pooled (centralized) reference baselines.

v3.0 FIXES (audit findings B9, B14):
  - XGBoost scale_pos_weight is COMPUTED from the training labels
    (n_neg / n_pos) instead of the hardcoded 10.0 that was chosen
    "to handle test imbalance" (test-set knowledge in training).
  - ADDED centralized MLP baseline: the crucial comparison that
    isolates the cost of FEDERATION (centralized MLP vs federated MLP,
    same architecture) — the old pipeline only compared federated MLP
    vs XGBoost, which confounds architecture with federation.
  - ADDED climatology baseline (historical event rate) — the floor any
    learned model must beat.
  - Evaluation uses evaluation.compute_all_metrics (full metric set).

Naming note (CLAIMS.md): XGBoost is the "centralized XGBoost reference
baseline", NOT a "centralized upper bound".
"""

import numpy as np
from typing import Dict, List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

import config as cfg
from evaluation import compute_all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Climatology baseline (predict training prevalence everywhere)
# ─────────────────────────────────────────────────────────────────────────────

class ClimatologyModel:
    """Constant-probability predictor: outputs the training flare rate."""

    def __init__(self):
        self.rate_ = 0.5

    def fit(self, X_train, y_train):
        self.rate_ = float(np.mean(y_train))
        return self

    def predict_proba(self, X):
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        return np.full((n, 2), [1.0 - self.rate_, self.rate_])


# ─────────────────────────────────────────────────────────────────────────────
# Centralized MLP (torch; lazy import so the module works without torch)
# ─────────────────────────────────────────────────────────────────────────────

def train_centralized_mlp(X_train, y_train, X_val, y_val, seed=None,
                          epochs=30, batch_size=256, verbose=True):
    """
    Train the SAME SolarMLP architecture on the POOLED training data.

    This is the reference that isolates the cost of federation:
        federation cost = centralized MLP - federated MLP
    (same architecture, same loss, same data volume).

    Early stopping on VALIDATION loss (patience 5).
    """
    try:
        import torch
        from model import SolarMLP, get_device
    except ImportError as e:
        print(f"[Centralized MLP] torch unavailable ({e}) — skipped.")
        return None

    seed = cfg.SEED if seed is None else seed
    torch.manual_seed(seed)
    device = get_device()

    input_dim = X_train.shape[1]
    model = SolarMLP(input_dim=input_dim).to(device)
    from federated_learning import get_criterion

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_v = torch.tensor(y_val, dtype=torch.float32).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR,
                                  weight_decay=1e-4)
    criterion = get_criterion(device, 0, 1, focal_alpha=0.25)

    ds = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                         shuffle=True, drop_last=False)

    best_val_loss, best_state, patience, wait = np.inf, None, 5, 0
    for epoch in range(epochs):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb,
                             client_pos_rate=float(y_train.mean()))
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_v)
            val_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                val_logits, y_v).item()
        if val_loss < best_val_loss - 1e-4:
            best_val_loss, wait = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
        if verbose and (epoch + 1) % 5 == 0:
            print(f"    [Central MLP] epoch {epoch+1}: val loss {val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def mlp_probs(model, X, batch_size=4096):
    """Batched probability extraction from a torch MLP."""
    import torch
    device = next(model.parameters()).device
    model.eval()
    out = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            e = min(s + batch_size, len(X))
            Xb = torch.tensor(X[s:e], dtype=torch.float32).to(device)
            out[s:e] = torch.sigmoid(model(Xb)).cpu().numpy().flatten()
    return np.clip(out, 1e-7, 1 - 1e-7)


# ─────────────────────────────────────────────────────────────────────────────
# Training-budget audit (review R15)
# ─────────────────────────────────────────────────────────────────────────────

def training_budget_report(n_train, n_val=None, central_epochs=30,
                           central_batch=256, central_lr=None,
                           central_early_stopping="patience 5 on validation BCE loss",
                           fl_rounds=50, fl_local_epochs=10, fl_batch=512,
                           fl_lr=None):
    """
    Review finding R15: the centralized-MLP vs FedProx comparison is only
    interpretable if the optimisation budgets are stated explicitly.
    Effective passes over data:

      centralized MLP : <= central_epochs full passes over the pooled
                        training set (early stopping may reduce this;
                        the restored checkpoint is the best-validation
                        epoch, and the number of epochs actually executed
                        is recorded in the run log)
      federated model : fl_rounds x fl_local_epochs passes over EACH
                        client's own shard (i.e. the same shard sees
                        500 epochs at the default settings — but each
                        pass covers only that client's ~1/6 of the pool)

    Both are reported so the comparison direction is auditable; a
    budget-matched rerun (centralized epochs set to fl_rounds *
    fl_local_epochs, or FL rounds reduced) is a queued action.

    Returns a dict meant to be embedded verbatim in results.json.
    """
    return {
        "centralized_mlp": {
            "max_epochs": int(central_epochs),
            "batch_size": int(central_batch),
            "lr": (float(central_lr) if central_lr is not None
                    else (float(fl_lr) if fl_lr is not None else None)),
            "effective_max_passes_over_pool": int(central_epochs),
            "early_stopping": central_early_stopping,
            "checkpoint_rule": "best validation BCE loss",
        },
        "federated": {
            "rounds": int(fl_rounds),
            "local_epochs_per_round": int(fl_local_epochs),
            "batch_size": int(fl_batch),
            "lr": float(fl_lr) if fl_lr is not None else None,
            "effective_passes_per_client_shard": int(fl_rounds * fl_local_epochs),
            "checkpoint_rule": "best validation F1 round",
        },
        "budgets_matched": bool(central_epochs == fl_rounds * fl_local_epochs),
        "note": ("Optimisation budgets are NOT matched at the default "
                 "settings; the centralized-vs-federated comparison "
                 "confounds federation with budget. A budget-matched "
                 "rerun is queued (review R15)."),
        "n_train": int(n_train),
        "n_val": None if n_val is None else int(n_val),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Train all centralized baselines
# ─────────────────────────────────────────────────────────────────────────────

def train_centralized(X_train, y_train, X_val=None, y_val=None,
                      seed=None) -> Dict:
    """
    Train climatology, LR, centralized MLP, and XGBoost on pooled data.

    XGBoost scale_pos_weight is computed from TRAIN prevalence (B9).
    """
    seed = cfg.SEED if seed is None else seed
    models = {}

    # ── Climatology ──────────────────────────────
    print("[Centralised] Fitting climatology baseline (train prevalence) ...")
    clim = ClimatologyModel().fit(X_train, y_train)
    models["climatology"] = clim

    # ── Logistic Regression ──────────────────────
    print("[Centralised] Training Logistic Regression ...")
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
    )
    lr.fit(X_train, y_train)
    models["logistic_regression"] = lr

    # ── Centralized MLP (federation-cost reference, B14) ─────
    if X_val is not None and y_val is not None:
        print("[Centralised] Training centralized MLP (federation-cost reference) ...")
        mlp = train_centralized_mlp(X_train, y_train, X_val, y_val, seed=seed)
        if mlp is not None:
            models["centralized_mlp"] = mlp

    # ── XGBoost (reference baseline) ─────────────
    print("[Centralised] Training XGBoost (scale_pos_weight from TRAIN, B9) ...")
    n_pos = max(int(np.sum(y_train == 1)), 1)
    n_neg = max(int(np.sum(y_train == 0)), 1)
    spw = float(n_neg / n_pos)
    print(f"[Centralised] XGBoost scale_pos_weight = {spw:.2f} "
          f"(train pos rate {n_pos/ (n_pos+n_neg):.3f})")

    xgb_params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=spw,       # B9: computed from train, not test
        random_state=seed,
        verbosity=0,
        eval_metric="logloss",
    )

    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:
        cuda = False

    if cuda:
        try:
            xgb = XGBClassifier(**xgb_params, device="cuda", tree_method="hist")
            xgb.fit(X_train, y_train)
        except Exception:
            xgb = XGBClassifier(**xgb_params)
            xgb.fit(X_train, y_train)
    else:
        xgb = XGBClassifier(**xgb_params)
        xgb.fit(X_train, y_train)
    models["xgboost"] = xgb

    return models


def model_probs(model, X):
    """Probability extraction for any centralized model."""
    if isinstance(model, ClimatologyModel):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # centralized MLP
    return mlp_probs(model, X)


def evaluate_centralized(models: Dict, X, y, threshold) -> Dict[str, Dict]:
    """
    Evaluate each centralized model at the given (frozen) threshold.
    Full metric set via compute_all_metrics.
    """
    results = {}
    for name, model in models.items():
        probs = model_probs(model, X)
        res = compute_all_metrics(y, probs, threshold, beta=cfg.FBETA_BETA)
        res["probs"] = probs
        res.pop("preds", None)
        results[name] = res
        print(f"[Centralised] {name:<22} | F1: {res['f1']:.3f} | "
              f"R: {res['recall']:.3f} | PR-AUC: {res['pr_auc']:.3f} | "
              f"ROC-AUC: {res['roc_auc']:.3f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SHAP interpretability
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap(xgb_model, X_sample, feature_names: List[str],
                 sample_size=500):
    """
    SHAP values for the XGBoost reference baseline.
    v3.0: feature_names are the REAL (stat-prefixed) names (B12 fixed
    upstream), so the plot shows physically interpretable labels.
    """
    import shap

    print("[SHAP] Computing SHAP values (XGBoost reference) ...")
    try:
        sample_size = min(sample_size, len(X_sample))
        sample = X_sample[:sample_size]
        explainer = shap.TreeExplainer(xgb_model)
        shap_vals = explainer.shap_values(sample)
        if isinstance(shap_vals, list):  # some shap versions return list
            shap_vals = shap_vals[1] if len(shap_vals) == 2 else shap_vals[0]
        mean_shap = np.abs(shap_vals).mean(axis=0)
        print("[SHAP] Done.")
        return shap_vals, mean_shap
    except Exception as e:
        print(f"[SHAP] Failed: {e}")
        return None, None
