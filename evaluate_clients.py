"""
evaluate_clients.py  (v3.0 — improvements branch)
────────────────────────────────────────────────────
Client-level evaluation (revision Stage 10).

Question answered: does federation actually help EACH participant,
or only the global average?

For every client shard k:
    * local-only  : a fresh model trained ONLY on shard k (the
                    "what would this observatory achieve alone" baseline)
    * fedavg      : global FedAvg model evaluated on shard k
    * fedprox     : global FedProx model evaluated on shard k

Reports ROC-AUC, PR-AUC, recall, precision, F1 at the frozen thresholds.
Evaluations run on the client's VALIDATION-held-out portion when
available; otherwise on the shard itself (documented).
"""

import numpy as np
import torch

import config as cfg
from model import make_fresh_model, clone_model
from federated_learning import (local_train_fedavg, get_model_probs,
                                evaluate_model)
from evaluation import compute_all_metrics


def split_shards_for_holdout(shards, holdout_fraction=0.2, seed=42):
    """
    Review R14 corrected experiment: partition each client shard into a
    FEDERATION-TRAIN portion and a completely UNTOUCHED holdout BEFORE
    any federated training happens. The global models must then be
    trained only on the returned fed_shards, and evaluated on the
    holdouts — giving clean per-client generalisation estimates rather
    than the within-distribution scores of the v3.0 client evaluation.

    Returns (fed_shards, holdouts) as lists of (X, y) tuples.
    """
    fed_shards, holdouts = [], []
    for k, (X_c, y_c) in enumerate(shards):
        n = len(y_c)
        rng = np.random.RandomState(seed + k)
        perm = rng.permutation(n)
        cut = int((1.0 - holdout_fraction) * n)
        tr, ev = perm[:cut], perm[cut:]
        fed_shards.append((X_c[tr], y_c[tr]))
        holdouts.append((X_c[ev], y_c[ev]))
    return fed_shards, holdouts


def evaluate_client_level(shards, global_models, model_name,
                          threshold=None, batch_size=2048, seed=None,
                          local_epochs=None, n_local_runs=1,
                          holdouts=None):
    # v4.0 (review m9): X_val/y_val removed — they were accepted and
    # never used (dead parameters).
    """
    Parameters
    ----------
    shards : list of (X, y) tuples (training shards)
    global_models : tuple of trained global models (fedavg, fedprox, ...)
    model_name : 'MLP' | 'LSTM' (display)
    threshold : frozen threshold from the main protocol (None -> 0.35)
    holdouts : optional list of (X, y) untouched client holdouts
        (from split_shards_for_holdout). When provided, global models
        are evaluated on the untouched holdouts instead of the
        within-shard 20% slices, and the shards argument must be the
        FEDERATION-TRAIN portions (review R14 protocol).

    Returns a list of per-client dicts + a summary block.
    """
    seed = cfg.SEED if seed is None else seed
    local_epochs = local_epochs or max(1, cfg.LOCAL_EPOCHS // 2)
    threshold = threshold or cfg.DEFAULT_THRESHOLD

    use_lstm = cfg.USE_LSTM
    device = next(global_models[0].parameters()).device

    rows = []
    for k, (X_c, y_c) in enumerate(shards):
        if len(y_c) < 20:
            print(f"[Client Eval] shard {k} too small ({len(y_c)}) — skipped")
            continue

        # evaluation set: untouched holdout when provided (R14), else the
        # legacy within-shard 20% slice (documented optimistic bias)
        if holdouts is not None and k < len(holdouts):
            X_ev, y_ev = holdouts[k]
            ev_protocol = "untouched_holdout"
        else:
            # split the shard: train local model on 80%, evaluate on 20%
            n = len(y_c)
            rng = np.random.RandomState(seed + k)
            perm = rng.permutation(n)
            cut = int(0.8 * n)
            tr, ev = perm[:cut], perm[cut:]
            X_ev, y_ev = X_c[ev], y_c[ev]
            ev_protocol = "within_shard_slice"

        # local-only model trains on the (federation-train) shard
        X_tr, y_tr = X_c, y_c

        if (y_ev == 1).sum() == 0 or (y_ev == 0).sum() == 0:
            # degenerate held-out split; evaluate on the full shard
            X_ev, y_ev = X_c, y_c

        # local-only model
        input_dim = X_tr.shape[2] if X_tr.ndim == 3 else X_tr.shape[1]
        local_model, _ = make_fresh_model(input_dim, use_lstm=use_lstm,
                                          seed=seed + k)
        local_model = local_train_fedavg(local_model, X_tr, y_tr,
                                         epochs=local_epochs,
                                         total_rounds=1, seed=seed + k)
        p_local = get_model_probs(local_model, X_ev, device, batch_size)

        row = {
            "client": cfg.CLIENT_NAMES[k] if k < len(cfg.CLIENT_NAMES) else f"Client {k}",
            "evaluation_protocol": ev_protocol,
            "n_train": int(len(y_tr)),
            "pos_rate_train": float(y_tr.mean()),
            "n_eval": int(len(y_ev)),
            "pos_rate_eval": float(y_ev.mean()),
            "local": compute_all_metrics(y_ev, p_local, threshold,
                                         beta=cfg.FBETA_BETA),
        }

        for gi, g in enumerate(global_models):
            if g is None:
                continue
            label = ["fedavg", "fedprox", "scaffold"][gi] if gi < 3 \
                else f"model{gi}"
            p_g = get_model_probs(g, X_ev, device, batch_size)
            row[label] = compute_all_metrics(y_ev, p_g, threshold,
                                             beta=cfg.FBETA_BETA)

        # federated benefit summary
        for label in ("fedavg", "fedprox", "scaffold"):
            if label in row:
                row[f"delta_pr_auc_{label}_vs_local"] = \
                    row[label]["pr_auc"] - row["local"]["pr_auc"]
                row[f"delta_recall_{label}_vs_local"] = \
                    row[label]["recall"] - row["local"]["recall"]

        rows.append({k2: v for k2, v in row.items() if isinstance(v, (int, float, str))
                     or k2 in ("local", "fedavg", "fedprox", "scaffold")})

        print(f"[Client Eval] {row['client']}: n={row['n_train']:,} "
              f"(pos {row['pos_rate_train']*100:.1f}%) | "
              f"local PR-AUC {row['local']['pr_auc']:.3f} | "
              f"fed PR-AUC {row.get('fedavg', {}).get('pr_auc', float('nan')):.3f}")

    summary = {
        "n_clients_evaluated": len(rows),
        "evaluation_protocol": rows[0].get("evaluation_protocol") if rows else None,
        "mean_local_pr_auc": float(np.mean([r["local"]["pr_auc"] for r in rows])) if rows else None,
        "mean_fedavg_pr_auc": float(np.mean([r["fedavg"]["pr_auc"] for r in rows])) if rows and "fedavg" in rows[0] else None,
        "mean_fedprox_pr_auc": float(np.mean([r["fedprox"]["pr_auc"] for r in rows])) if rows and "fedprox" in rows[0] else None,
        "clients_helped_by_federation": int(sum(
            1 for r in rows if r.get("delta_pr_auc_fedprox_vs_local", 0) > 0)),
    }
    return {"per_client": rows, "summary": summary}
