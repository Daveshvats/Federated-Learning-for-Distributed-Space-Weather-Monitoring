"""
experiments/run_budget_matched.py  (review R15 / queued item 6)
────────────────────────────────────────────────────────────────
Budget-matched centralised reference.

The shipped centralised MLP trains with a 30-epoch cap and early
stopping (patience 5 on validation BCE), while each federated client's
shard receives 50 x 10 = 500 local passes. This runner retrains the
SAME SolarMLP on the SAME pooled data with the optimisation budget
raised to 500 epochs, in two variants:

    (a) cap=500, early stopping retained   (shipped protocol, larger cap)
    (b) cap=500, early-stop break disabled (FL-style fixed budget; the
        best-validation-BCE checkpoint rule is kept)

Both are evaluated with the identical frozen protocol (prior_shift
calibration + F-beta threshold on validation, single test pass) and
compared against the federated arms. Epoch-level resumable (state +
optimizer checkpointed every 5 epochs), and the main.py phase cache is
reused for the data matrices (identical frozen-protocol identity; the
pooled training data is not shard-dependent, so no contamination is
possible).

Usage:
    python experiments/run_budget_matched.py
    python experiments/run_budget_matched.py --epochs 500 --batch-size 256
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg

FPR_TARGETS = (0.005, 0.01, 0.02, 0.05)


def _atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def _train_budget_matched(X_train, y_train, X_val, y_val, seed,
                          epochs, batch_size, early_stop, tag, cache):
    """train_centralized_mlp clone with a 500-epoch cap + epoch-level
    resume. Returns (model, stopped_early_at|False)."""
    import torch
    from model import SolarMLP, get_device
    from federated_learning import get_criterion

    ckpt = os.path.join(cache, f"mlp_{tag}.pt")
    device = get_device()
    torch.manual_seed(seed)
    model = SolarMLP(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR,
                                  weight_decay=1e-4)
    criterion = get_criterion(device, 0, 1, focal_alpha=0.25)
    start_epoch, best_val_loss, best_state, wait = 0, np.inf, None, 0
    stopped_early = None

    if os.path.exists(ckpt):
        d = torch.load(ckpt, weights_only=False)
        model.load_state_dict(d["model"])
        optimizer.load_state_dict(d["optim"])
        start_epoch = d["epoch"]
        best_val_loss = d["best_val_loss"]
        best_state = d["best_state"]
        wait = d.get("wait", 0)
        stopped_early = d.get("stopped_early")
        if d.get("final_state") is not None:
            model.load_state_dict(d["final_state"])
        print(f"[{tag}] resumed from epoch {start_epoch}")

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_v = torch.tensor(y_val, dtype=torch.float32).to(device)
    ds = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                         shuffle=True, drop_last=False)

    if stopped_early is None:
        for epoch in range(start_epoch, epochs):
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
                vl = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(X_v), y_v).item()
            if vl < best_val_loss - 1e-4:
                best_val_loss, wait = vl, 0
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            else:
                wait += 1
                if early_stop and wait >= 5:
                    stopped_early = epoch + 1
                    break
            if (epoch + 1) % 5 == 0:
                print(f"    [{tag}] epoch {epoch+1}: val loss {vl:.4f}",
                      flush=True)
            tmp = ckpt + ".tmp"
            torch.save({"model": model.state_dict(),
                        "optim": optimizer.state_dict(),
                        "epoch": epoch + 1, "best_val_loss": best_val_loss,
                        "best_state": best_state, "wait": wait,
                        "stopped_early": stopped_early}, tmp)
            os.replace(tmp, ckpt)
        if stopped_early is None:
            stopped_early = False

    if best_state is not None:
        model.load_state_dict(best_state)
    tmp = ckpt + ".tmp"
    torch.save({"model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "epoch": max(start_epoch, epochs),
                "best_val_loss": best_val_loss,
                "best_state": best_state, "wait": wait,
                "stopped_early": stopped_early,
                "final_state": model.state_dict()}, tmp)
    os.replace(tmp, ckpt)
    return model, stopped_early


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    t0 = time.time()
    from evaluation import (make_calibrator, find_optimal_threshold_fbeta,
                            compute_all_metrics,
                            select_fpr_thresholds_on_validation,
                            frozen_operating_point_metrics)
    from centralized_baseline import mlp_probs

    cfg.USE_LSTM = False
    cache = os.path.join("data", "cache", "budget")
    os.makedirs(cache, exist_ok=True)

    # ── data: reuse main.py phase cache (pooled matrices) ────────────────
    data_cache = os.path.join("data", "cache", "phase_data.npz")
    if os.path.exists(data_cache):
        z = np.load(data_cache, allow_pickle=False)
        X_train, y_train = z["X_train"], z["y_train"]
        X_val, y_val = z["X_val"], z["y_val"]
        X_test, y_test = z["X_test"], z["y_test"]
        print("[BudgetMatch] data reused from main.py phase cache")
    else:
        from data_preparation import load_or_generate_data, preprocess
        df = load_or_generate_data()
        splits = preprocess(df, val_fraction=cfg.VAL_SPLIT)
        X_train, y_train = splits.X_train, splits.y_train
        X_val, y_val = splits.X_val, splits.y_val
        X_test, y_test = splits.X_test, splits.y_test

    def evaluate(model):
        p_val = mlp_probs(model, X_val)
        p_test = mlp_probs(model, X_test)
        cal = make_calibrator(cfg.CALIBRATION_METHOD).fit(y_val, p_val)
        if cfg.CALIBRATION_METHOD == "prior_shift":
            cal.set_prevalences(train_rate=float(y_train.mean()),
                                test_rate=float(y_val.mean()))
        p_val_c = cal.transform(p_val)
        p_test_c = cal.transform(p_test)
        t, fb = find_optimal_threshold_fbeta(
            y_val, p_val_c, beta=cfg.FBETA_BETA, grid=cfg.THRESHOLD_GRID)
        metrics = compute_all_metrics(y_test, p_test_c, t, beta=cfg.FBETA_BETA)
        frozen = select_fpr_thresholds_on_validation(y_val, p_val_c,
                                                     FPR_TARGETS)
        ops = frozen_operating_point_metrics(y_test, p_test_c, frozen)
        return {"threshold": float(t), "val_fbeta": float(fb),
                "test": metrics, "frozen_operating_points": ops}

    print(f"\n=== variant (a): {args.epochs}-epoch cap, early stopping kept ===")
    ma, sa = _train_budget_matched(
        X_train, y_train, X_val, y_val, cfg.SEED, args.epochs,
        args.batch_size, early_stop=True, tag="a", cache=cache)
    print(f"[a] stopped at: {sa if sa else 'ran to cap'}")

    print(f"\n=== variant (b): {args.epochs}-epoch cap, no early stop "
          "(FL-style fixed budget) ===")
    mb, sb = _train_budget_matched(
        X_train, y_train, X_val, y_val, cfg.SEED, args.epochs,
        args.batch_size, early_stop=False, tag="b", cache=cache)

    report = {
        "purpose": ("budget-matched centralised reference (review R15): "
                    "same pooled data, same SolarMLP, optimisation budget "
                    "raised to the federated arm's 500 effective passes"),
        "seed": cfg.SEED, "epochs_cap": args.epochs,
        "batch_size": args.batch_size,
        "calibration": cfg.CALIBRATION_METHOD,
        "variants": {
            "a_cap500_early_stop": {"stopped_early_at": sa, **evaluate(ma)},
            "b_cap500_no_early_stop": {"stopped_early_at": sb, **evaluate(mb)},
        },
        "references": {
            "headline_centralized_mlp": {
                "epochs": "<=30 (early stop patience 5)",
                "roc_auc": 0.888, "pr_auc": 0.153, "f1": 0.059},
            "fedprox_mlp": {
                "budget": "50 rounds x 10 local epochs per client shard",
                "roc_auc": 0.954, "pr_auc": 0.307, "f1": 0.042},
            "source": "outputs/results.json (reproduced frozen protocol)"},
        "elapsed_s": round(time.time() - t0, 1),
    }

    path = args.output or os.path.join(cfg.OUTPUT_DIR, "budget_matched.json")
    _atomic_json(report, path)
    print(f"\n[BudgetMatch] report -> {path}")
    for v, m in report["variants"].items():
        t = m["test"]
        print(f"  {v:<28} ROC-AUC {t['roc_auc']:.3f} | PR-AUC "
              f"{t['pr_auc']:.3f} | F1 {t['f1']:.3f} | "
              f"stopped: {m['stopped_early_at']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
