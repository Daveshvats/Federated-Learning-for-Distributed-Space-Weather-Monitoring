"""
experiments/run_federated_lstm.py  (review item 8, LSTM half — raw-substrate edition)
────────────────────────────────────────────────────────
Train the FROZEN protocol's LSTM arms (FedAvg-LSTM, FedProx-LSTM — the
same arms `main.py` runs in LSTM mode) on the SAME leakage-free raw
SWAN-SF substrate as experiments/run_raw_substrate.py:

    identical windows, identical FPCKNN/LSBZM preprocessing, identical
    frozen validation carve (seed 42, stratified 84/16), identical
    Dirichlet shards (alpha=1.0, 6 clients, seed 42), identical 50-round
    protocol (10 local epochs, Fed-Focal, prior-shift calibration,
    validation-frozen F-beta thresholds + frozen-FPR operating points),
    single-pass test on raw partition 5 (natural 1.31% prevalence).

The centralized references (LR / XGBoost / central MLP) are NOT
retrained here — they are the frozen numbers of
run_raw_substrate.py (outputs/raw_substrate_eval.json), reported
alongside as counterparts.  A pooled CENTRALIZED SolarLSTM comparator
(default ON, --no-central-lstm to skip) isolates
architecture-vs-federation on the same substrate.

DEVICE
    Auto: CUDA if available, else CPU.  --device cpu|cuda to force.
    Round-resumable on any device via data/cache/rawsubstrate/*.pt
    checkpoints (states validate the protocol fingerprint — seed,
    rounds, mu, use_lstm — and stale states are ignored, so smoke runs
    never poison the real run).

3D SUBSTRATE (data/cache/rawsubstrate/)
    X_train_3d.npy (214,888, 60, 24) float16 + X_val_3d.npy (40,932)
    + processed_p5_3d.npy (75,365) + y3d.npz.  Built deterministically
    from the cached stage-A/B arrays (interp_p*.npy) and the fitted
    LSBZM parameters by re-applying the SAME stage-D transform, and
    split with the EXACT frozen call (train_test_split, seed 42,
    stratified) — asserted equal to the 2D substrate's labels and
    cross-checked against its 144-stat features on 512-window samples
    (float16 quantisation bound ~5e-3).

USAGE
    python experiments/run_federated_lstm.py --raw-dir /tmp/swansf_raw
    python experiments/run_federated_lstm.py --build-only
    python experiments/run_federated_lstm.py --tag smoke --rounds 1 \
        --limit-train 20000 --limit-eval 4000      # wiring + timing check
    python experiments/run_federated_lstm.py --seed 43
        # multi-seed: keeps the frozen val carve; reseeds init/shards

    python experiments/run_federated_lstm.py --tag scaffold --scaffold-only
        # SCAFFOLD-LSTM arm only (seed 42) -> outputs/raw_lstm_scaffold.json
    python experiments/run_federated_lstm.py --seed 43 --scaffold
        # full 4-arm replication -> outputs/raw_lstm_seed43.json
    python experiments/run_federated_lstm.py --tag smote --smote \
        --no-central-lstm
        # natural-prevalence SMOTE ablation -> outputs/raw_lstm_smote.json

    python experiments/run_gpu_queue.py
        # the whole remaining GPU queue in ONE resumable command
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from experiments.raw_substrate import (CACHE, TRAIN_PARTS, TEST_PART,
                                       N_FEATS, load_labels, apply_lsbzm)

FROZEN_SPLIT_SEED = cfg.SEED      # captured BEFORE any --seed override
FPR_TARGETS = (0.005, 0.01, 0.02, 0.05)
CHUNK = 4096                      # windows per LSBZM re-application chunk
MLP_COUNTERPART_MAP = {"fedavg_lstm": "fedavg_mlp",
                       "fedprox_lstm": "fedprox_mlp",
                       "scaffold_lstm": "scaffold_mlp",
                       "central_lstm": "centralized_mlp"}


# ── small helpers ───────────────────────────────────────────────────────────

def _atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def _atomic_torch(obj, path):
    import torch
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ── 3D substrate (identical frozen split; cross-checked vs 2D) ─────────────

def build_3d(raw_dir, out_dir=CACHE):
    """Return {X_train, y_train, X_val, y_val, X_test, y_test} where the
    X arrays are (n, 60, 24) float16 LSBZM-normalised series and the
    split is EXACTLY the frozen 2D substrate's (same windows, same
    train/val membership, same row order).  Cached as X_train_3d.npy /
    X_val_3d.npy / y3d.npz; test = processed_p5_3d.npy from build()."""
    tr_path = os.path.join(out_dir, "X_train_3d.npy")
    va_path = os.path.join(out_dir, "X_val_3d.npy")
    te_path = os.path.join(out_dir, "processed_p5_3d.npy")
    y_path = os.path.join(out_dir, "y3d.npz")

    if all(os.path.exists(p) for p in (tr_path, va_path, te_path, y_path)):
        z = np.load(y_path)
        print("[3D] resumed from cache (X_train_3d/X_val_3d/y3d)",
              flush=True)
        d = {"X_train": np.load(tr_path), "y_train": z["y_train"],
             "X_val": np.load(va_path), "y_val": z["y_val"],
             "X_test": np.load(te_path), "y_test": z["y_test"]}
        crosscheck_2d(d, out_dir)
        return d

    # prerequisites from the 2D pipeline's caches
    params_path = os.path.join(out_dir, "lsbzm_params.json")
    scales_path = os.path.join(out_dir, "scales.npy")
    for p in (params_path, scales_path, te_path):
        if not os.path.exists(p):
            raise SystemExit(
                f"[3D] missing {p} — run experiments/raw_substrate.py "
                "first (it builds interp_p*.npy, lsbzm_params.json, "
                "scales.npy, processed_p5_3d.npy)")
    params = json.load(open(params_path))
    scales = np.load(scales_path).astype(np.float32)

    ys = [load_labels(raw_dir, p) for p in range(1, 6)]
    y_pool = np.concatenate([ys[p - 1] for p in TRAIN_PARTS])

    # EXACT frozen split call (raw_substrate.build, stage D)
    from sklearn.model_selection import train_test_split
    val_idx = train_test_split(
        np.arange(len(y_pool)), test_size=cfg.VAL_SPLIT,
        random_state=FROZEN_SPLIT_SEED, stratify=y_pool)[1]
    is_val = np.zeros(len(y_pool), bool)
    is_val[val_idx] = True
    n_train = int((~is_val).sum())
    n_val = int(is_val.sum())

    # destination ranks: train rows in ASCENDING pooled order (matches
    # X_pool[train_idx]); val rows in val_idx order (matches
    # X_pool[val_idx]) — both exactly as the frozen 2D substrate.
    train_pos = np.full(len(y_pool), -1, np.int64)
    train_pos[~is_val] = np.arange(n_train)
    val_pos = np.full(len(y_pool), -1, np.int64)
    val_pos[val_idx] = np.arange(n_val)

    t0 = time.time()
    off = 0
    X_train = X_val = None
    for p in TRAIN_PARTS:
        a = np.load(os.path.join(out_dir, f"interp_p{p}.npy"))
        n, T, F_ = a.shape
        if X_train is None:      # allocate once from the data's own shape
            X_train = np.empty((n_train, T, F_), np.float16)
            X_val = np.empty((n_val, T, F_), np.float16)
        for s in range(0, n, CHUNK):
            e = min(s + CHUNK, n)
            Xc = a[s:e].astype(np.float32) * scales   # RAW space
            apply_lsbzm(Xc, params)                   # -> [0, 1] in place
            loc = np.arange(off + s, off + e)
            m_tr = ~is_val[loc]
            X_train[train_pos[loc[m_tr]]] = Xc[m_tr].astype(np.float16)
            X_val[val_pos[loc[~m_tr]]] = Xc[~m_tr].astype(np.float16)
        off += n
        del a
        print(f"  [3D] P{p} re-transformed ({off:,}/{len(y_pool):,} "
              f"pooled) [{time.time()-t0:.0f}s]", flush=True)

    y_train = y_pool[~is_val].astype(np.float32)
    y_val = y_pool[val_idx].astype(np.float32)
    y_test = ys[TEST_PART - 1].astype(np.float32)

    np.save(tr_path, X_train)
    np.save(va_path, X_val)
    np.savez(y_path, y_train=y_train, y_val=y_val, y_test=y_test)
    print(f"[3D] built: train {X_train.shape} | val {X_val.shape} | "
          f"test {np.load(te_path).shape}  [{time.time()-t0:.0f}s]",
          flush=True)
    d = {"X_train": X_train, "y_train": y_train, "X_val": X_val,
         "y_val": y_val, "X_test": np.load(te_path), "y_test": y_test}
    crosscheck_2d(d, out_dir)
    return d


def crosscheck_2d(d, out_dir=CACHE):
    """Assert the 3D rebuild carries the EXACT frozen split (labels
    equal, row order equal) and reproduces the 2D substrate's 144-stat
    features on 512-window samples per split (float16 storage
    quantisation keeps max|diff| < 5e-3).  Skipped when the 2D cache
    is absent (e.g. a GPU box that only shipped data3d)."""
    p2 = os.path.join(out_dir, "data.npz")
    if not os.path.exists(p2):
        print("[3D] 2D cache absent — cross-check skipped", flush=True)
        return None
    z = np.load(p2)
    ok = (np.array_equal(d["y_train"], z["y_train"]) and
          np.array_equal(d["y_val"], z["y_val"]) and
          np.array_equal(d["y_test"], z["y_test"]) and
          d["X_train"].shape[0] == z["X_train"].shape[0] and
          d["X_val"].shape[0] == z["X_val"].shape[0])
    assert ok, "3D split does not match the frozen 2D substrate"
    from load_cleaned_data import flatten_3d_to_2d
    rng = np.random.default_rng(0)
    worst = 0.0
    for split in ("train", "val", "test"):
        n = len(d[f"y_{split}"])
        idx = rng.choice(n, min(512, n), replace=False)
        A = flatten_3d_to_2d(d[f"X_{split}"][idx].astype(np.float32),
                             method=cfg.FLATTEN_METHOD)
        B = np.asarray(z[f"X_{split}"][idx], dtype=np.float64)
        diff = float(np.abs(np.asarray(A, np.float64) - B).max())
        worst = max(worst, diff)
    print(f"[3D] cross-check vs 2D substrate: labels EQUAL "
          f"(train/val/test), 144-stat max|diff| = {worst:.2e} "
          f"(f16 bound ~5e-3)", flush=True)
    assert worst < 5e-3, f"3D/2D feature mismatch: max|diff|={worst}"
    return worst

# ── optional pooled centralized SolarLSTM comparator ────────────────────────

def train_centralized_lstm(X_train, y_train, X_val, y_val, seed=None,
                           cache_path=None, epochs=30, batch_size=256,
                           verbose=True):
    """Pooled SolarLSTM on the raw substrate — same architecture, loss
    and optimiser family as the federated LSTM arms (AdamW cfg.LR,
    focal criterion, gradient clip 5.0, early stopping patience 5 on
    validation BCE).  Epoch-resumable via cache_path."""
    import torch
    from model import SolarLSTM, get_device
    from federated_learning import get_criterion
    device = get_device()
    seed = cfg.SEED if seed is None else seed
    model = SolarLSTM(input_size=N_FEATS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR,
                            weight_decay=1e-4)
    crit = get_criterion(device, 0, 1, focal_alpha=0.25)
    start_epoch, best_vl, best_state, wait = 0, np.inf, None, 0

    if cache_path and os.path.exists(cache_path):
        st = torch.load(cache_path, weights_only=False)
        if st.get("done"):
            m = SolarLSTM(input_size=N_FEATS).to(device)
            m.load_state_dict(st["model"])
            m.eval()
            print("[Central LSTM] identical completed run found — "
                  "reusing cached model", flush=True)
            return m
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_epoch = st["epoch"]
        best_vl, best_state, wait = st["best_vl"], st["best_state"], st["wait"]
        print(f"[Central LSTM] resuming from epoch {start_epoch}", flush=True)
    else:
        torch.manual_seed(seed)

    Xt = torch.as_tensor(X_train, dtype=torch.float16)   # f16 master
    yt = torch.as_tensor(np.asarray(y_train, np.float32))
    pos_rate = float(np.mean(y_train))
    n = len(yt)

    for epoch in range(start_epoch, epochs):
        model.train()
        perm = torch.randperm(n)
        for s in range(0, n, batch_size):
            j = perm[s:s + batch_size]
            Xb = Xt[j].to(device).float()
            yb = yt[j].to(device)
            opt.zero_grad()
            loss = crit(model(Xb), yb, client_pos_rate=pos_rate)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
        model.eval()
        tot = cnt = 0
        with torch.no_grad():
            for s in range(0, len(y_val), 4096):
                Xb = torch.tensor(X_val[s:s + 4096],
                                  dtype=torch.float32).to(device)
                yb = torch.tensor(np.asarray(y_val[s:s + 4096],
                                             np.float32)).to(device)
                tot += torch.nn.functional \
                    .binary_cross_entropy_with_logits(
                        model(Xb), yb, reduction="sum").item()
                cnt += len(yb)
        vl = tot / max(cnt, 1)
        if vl < best_vl - 1e-4:
            best_vl, wait = vl, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            wait += 1
        if verbose:
            print(f"    [Central LSTM] epoch {epoch + 1}: val BCE "
                  f"{vl:.4f} (best {best_vl:.4f}, wait {wait})", flush=True)
        if cache_path:
            _atomic_torch({"epoch": epoch + 1, "model": model.state_dict(),
                           "opt": opt.state_dict(), "best_vl": best_vl,
                           "best_state": best_state, "wait": wait,
                           "seed": seed, "epochs": epochs,
                           "done": False}, cache_path)
        if wait >= 5:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    if cache_path:
        _atomic_torch({"done": True, "model": model.state_dict(),
                       "seed": seed, "epochs": epochs,
                       "best_vl": best_vl}, cache_path)
    return model


# ── event-level evaluation (same recipe as run_event_level_raw.py) ─────────

def find_aux_dir(raw_dir, cache_dir=None):
    """Resolve the event-level aux metadata directory: raw_dir first,
    then data/cache/rawsubstrate/aux, then _aux (Windows unzippers
    rename aux -> _aux because AUX is a reserved device name — this
    exact rename silently skipped the owner's in-run event-level pass
    in v3.6).  Returns None when no match_test_p5.csv is found."""
    cd = CACHE if cache_dir is None else cache_dir
    for cand in (raw_dir, os.path.join(cd, "aux"), os.path.join(cd, "_aux")):
        if os.path.exists(os.path.join(cand, "match_test_p5.csv")):
            return cand
    return None


def event_level_eval(arms_probs, y_test, thresholds, raw_dir, out_json,
                     purpose, cooldown=5):
    """Event-level metrics on raw P5 for the LSTM arms.  Aux metadata is
    looked up via find_aux_dir (raw_dir, then cache aux, then the
    Windows-renamed _aux).  Returns the report dict or None if absent."""
    import pandas as pd
    from experiments.run_event_level import event_level_metrics
    aux = find_aux_dir(raw_dir)
    if aux is None:
        print("[event] aux metadata not found (raw_dir / cache aux / _aux)"
              " — event-level skipped", flush=True)
        return None

    y = np.asarray(y_test).astype(int)
    match = pd.read_csv(os.path.join(aux, "match_test_p5.csv"),
                        low_memory=False)
    match = match[match["verified"] == True]                # noqa: E712
    match = match.drop_duplicates("raw_idx")
    r2p = dict(zip(match["raw_idx"], match["pkl_row"]))
    pooled = pd.read_csv(os.path.join(aux, "test_meta_pooled.csv"),
                         low_memory=False)
    pooled = pooled[pooled["partition"] == 5]
    p2meta = {r.pkl_row: r for r in pooled.itertuples()}

    rows = []
    for i in range(len(y)):
        p = r2p.get(i)
        if p is None or p not in p2meta:
            continue
        m = p2meta[p]
        rows.append((i, m.event_id, m.ts_end_min, str(m.flare_class),
                     int(m.label)))
    meta = pd.DataFrame(rows, columns=["raw_idx", "event_id", "ts_end_min",
                                       "flare_class", "label"])
    assert (meta["label"].values == y[meta["raw_idx"].values]).all()

    idx = meta["raw_idx"].values
    ev = meta["event_id"].values
    ts = meta["ts_end_min"].values.astype(float)
    yy = y[idx]
    true_groups = meta.loc[meta["label"] == 1, "event_id"].unique()
    print(f"[event] {len(y):,} raw windows | {len(meta):,} matched "
          f"({len(meta)/len(y):.2%}) | {meta['event_id'].nunique()} groups "
          f"| {len(true_groups)} true events", flush=True)

    fl = pd.read_csv(os.path.join(aux, "integrated_flare_data",
                                  "goes_flares_integrated.csv"),
                     low_memory=False)
    fl["peak_dt"] = pd.to_datetime(fl["peak_time"])
    fl["cls"] = fl["goes_class"].astype(str)
    peak_map, matched, ambiguous = {}, 0, 0
    for g in true_groups:
        sub = meta[meta["event_id"] == g]
        last_end = sub["ts_end_min"].max()
        cls = str(sub["flare_class"].iloc[0])
        lo = pd.Timestamp(last_end, unit="m")
        cand = fl[(fl["cls"] == cls) & (fl["peak_dt"] >= lo)
                  & (fl["peak_dt"] <= lo + pd.Timedelta(hours=48))]
        if len(cand):
            peak_map[g] = (cand["peak_dt"].iloc[0]
                           - pd.Timestamp(0)).total_seconds() / 60.0
            matched += 1
            ambiguous += int(len(cand) > 1)
    print(f"[event] flare peaks matched {matched}/{len(true_groups)} "
          f"(ambiguous {ambiguous})", flush=True)

    out = {"purpose": purpose, "cooldown_windows": cooldown,
           "n_windows": int(len(yy)), "n_matched_by_audit": int(len(meta)),
           "n_event_groups": int(meta["event_id"].nunique()),
           "n_true_events": int(len(true_groups)),
           "flare_peak_match_rate": matched / max(len(true_groups), 1),
           "models": {}}
    for name, p in arms_probs.items():
        thr = thresholds[name]
        r = event_level_metrics(yy, p[idx], ev, threshold=thr,
                                timestamps=ts, cooldown_windows=cooldown)
        alerts = (p[idx] >= thr).astype(int)
        leads = []
        for g, peak_min in peak_map.items():
            gi = np.where(ev == g)[0]
            g_alerts = alerts[gi]
            if g_alerts.sum() > 0:
                first_alert = ts[gi][g_alerts == 1].min()
                leads.append(peak_min - first_alert)
        r["lead_to_flare_peak_minutes"] = {
            "median": float(np.median(leads)) if leads else None,
            "p10": float(np.percentile(leads, 10)) if leads else None,
            "p90": float(np.percentile(leads, 90)) if leads else None,
            "n": len(leads),
            "negative_means_alert_after_peak": True,
        }
        out["models"][name] = r
        lt = r["lead_to_flare_peak_minutes"]
        print(f"[event {name:<14s}] thr {thr:.3f} | events "
              f"{r['n_detected_events']}/{r['n_true_events']} "
              f"({r['event_detection_rate']*100:.1f}%) | FA/day "
              f"{r['false_alarm_windows_per_day']:.2f} | alerts/day "
              f"{r['alerts_per_day']:.2f} | lead "
              f"{(lt['median']/60 if lt['median'] is not None else float('nan')):.1f}h",
              flush=True)
    _atomic_json(out, out_json)
    print(f"[event] -> {out_json}", flush=True)
    return out


# ── driver ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="/tmp/swansf_raw")
    ap.add_argument("--tag", default=None,
                    help="cache/output tag (default 'lstm', or 'seed<k>')")
    ap.add_argument("--seed", type=int, default=None,
                    help="override training seed (keeps frozen val carve)")
    ap.add_argument("--rounds", type=int, default=None,
                    help="override N_ROUNDS (smoke only; resume states "
                         "fingerprint rounds, so real runs stay at 50)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu",
                                                          "cuda"])
    ap.add_argument("--build-only", action="store_true",
                    help="build/cross-check the 3D substrate and exit")
    ap.add_argument("--central-lstm", dest="central_lstm",
                    action="store_true", default=True,
                    help="train the pooled SolarLSTM comparator (default)")
    ap.add_argument("--no-central-lstm", dest="central_lstm",
                    action="store_false")
    ap.add_argument("--no-event-level", dest="event_level",
                    action="store_false", default=True)
    ap.add_argument("--limit-train", type=int, default=0,
                    help="smoke: subsample train windows")
    ap.add_argument("--limit-eval", type=int, default=0,
                    help="smoke: subsample val/test windows")
    ap.add_argument("--output", default=None)
    ap.add_argument("--scaffold", action="store_true",
                    help="add the SCAFFOLD-LSTM arm (same 50-round budget "
                         "and SGD control-variate recipe as the MLP "
                         "SCAFFOLD arm of run_raw_substrate.py)")
    ap.add_argument("--scaffold-only", action="store_true",
                    help="train ONLY the SCAFFOLD arm (no other arm is "
                         "retrained; combine with --tag so no frozen "
                         "artefact is touched)")
    ap.add_argument("--smote", action="store_true",
                    help="enable per-client SMOTE (natural-prevalence "
                         "ablation; 3D-aware flatten/reshape since v3.8)")
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting an existing --output file")
    args = ap.parse_args()

    import torch
    if args.device != "auto":
        import model as _model
        _model.DEVICE = torch.device(args.device)
        print(f"[LSTM] device forced to {args.device}", flush=True)
    try:
        torch.backends.cudnn.benchmark = False   # reproducibility first
    except Exception:
        pass
    cfg.USE_LSTM = True
    if args.scaffold or args.scaffold_only:
        cfg.USE_SCAFFOLD = True       # run_scaffold no-ops without it
    else:
        cfg.USE_SCAFFOLD = False
    if args.smote:
        cfg.USE_SMOTE = True          # per-client SMOTE ablation arm
    if args.scaffold_only:
        args.scaffold = True
        args.central_lstm = False     # ONLY the scaffold arm runs

    seed = FROZEN_SPLIT_SEED if args.seed is None else args.seed
    if seed != FROZEN_SPLIT_SEED:
        cfg.SEED = seed        # downstream: model init, shards, criterion
    tag = args.tag or ("lstm" if seed == FROZEN_SPLIT_SEED
                       else f"seed{seed}")
    n_rounds = cfg.N_ROUNDS if args.rounds is None else args.rounds
    out_path = args.output or ("outputs/raw_lstm_eval.json"
                               if tag == "lstm"
                               else f"outputs/raw_lstm_{tag}.json")
    if os.path.exists(out_path) and not args.force:
        raise SystemExit(
            f"[LSTM] {out_path} already exists — refusing to overwrite a "
            f"frozen artefact.  Pass --force or use a different "
            f"--tag/--output.")

    t0 = time.time()
    d = build_3d(args.raw_dir)
    X_train, y_train = d["X_train"], d["y_train"]
    X_val, y_val = d["X_val"], d["y_val"]
    X_test, y_test = d["X_test"], d["y_test"]
    n_full_train = len(y_train)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    print(f"[LSTM] device={dev} | seed={seed} | tag={tag} | train "
          f"{len(y_train):,} (pos {y_train.mean():.2%}) | val "
          f"{len(y_val):,} | test {len(y_test):,} "
          f"(pos {y_test.mean():.2%})", flush=True)
    if args.build_only:
        print("[LSTM] --build-only: 3D substrate ready, exiting", flush=True)
        return

    smoke = (args.limit_train > 0 or args.limit_eval > 0
             or n_rounds != cfg.N_ROUNDS)
    if smoke:
        print(f"[LSTM] SMOKE MODE (tag={tag}, rounds={n_rounds}, "
              f"limit_train={args.limit_train}, "
              f"limit_eval={args.limit_eval})", flush=True)
    if args.limit_train > 0:
        rng = np.random.default_rng(123)
        idx = rng.choice(len(y_train), min(args.limit_train, len(y_train)),
                         replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
    if args.limit_eval > 0:
        rng = np.random.default_rng(124)
        vi = rng.choice(len(y_val), min(args.limit_eval, len(y_val)),
                        replace=False)
        X_val, y_val = X_val[vi], y_val[vi]
        ti = rng.choice(len(y_test), min(args.limit_eval, len(y_test)),
                        replace=False)
        X_test, y_test = X_test[ti], y_test[ti]

    # ── shards: identical frozen recipe ─────────────────────────────────
    from partition_clients import partition_data_dirichlet
    shards = partition_data_dirichlet(
        X_train, y_train, alpha=1.0, n_clients=cfg.N_CLIENTS,
        seed=cfg.SEED, min_samples=cfg.MIN_SAMPLES_PER_CLIENT)

    # ── arms ────────────────────────────────────────────────────────────
    from federated_learning import (run_fedavg, run_fedprox, run_scaffold,
                                    get_model_probs)
    models, walls = {}, {}
    if args.central_lstm:
        ts = time.time()
        models["central_lstm"] = train_centralized_lstm(
            X_train, y_train, X_val, y_val, seed=cfg.SEED,
            cache_path=os.path.join(CACHE, f"central_{tag}.pt"))
        walls["central_lstm"] = time.time() - ts
    if not args.scaffold_only:
        ts = time.time()
        models["fedavg_lstm"], _ = run_fedavg(
            shards, X_val, y_val, n_rounds=n_rounds, use_lstm=True,
            eval_batch_size=cfg.EVAL_BATCH_SIZE, seed=cfg.SEED,
            resume_path=os.path.join(CACHE, f"fedavg_{tag}.pt"))
        walls["fedavg_lstm"] = time.time() - ts
        ts = time.time()
        models["fedprox_lstm"], _ = run_fedprox(
            shards, X_val, y_val, n_rounds=n_rounds, mu=cfg.MU, use_lstm=True,
            eval_batch_size=cfg.EVAL_BATCH_SIZE, seed=cfg.SEED,
            resume_path=os.path.join(CACHE, f"fedprox_{tag}.pt"))
        walls["fedprox_lstm"] = time.time() - ts
    if args.scaffold:
        ts = time.time()
        scaf_model, _ = run_scaffold(
            shards, X_val, y_val, n_rounds=n_rounds, use_lstm=True,
            eval_batch_size=cfg.EVAL_BATCH_SIZE, seed=cfg.SEED,
            resume_path=os.path.join(CACHE, f"scaffold_{tag}.pt"))
        if scaf_model is not None:
            models["scaffold_lstm"] = scaf_model
            walls["scaffold_lstm"] = time.time() - ts

    # ── evaluation: identical frozen protocol ───────────────────────────
    from evaluation import (make_calibrator, find_optimal_threshold_fbeta,
                            compute_all_metrics,
                            select_fpr_thresholds_on_validation,
                            frozen_operating_point_metrics)
    # any trained arm fixes the eval device (fedprox absent in
    # --scaffold-only / partial-arm runs)
    device = next(next(iter(models.values())).parameters()).device \
        if models else "cpu"

    REF = {}
    rp = "outputs/raw_substrate_eval.json"
    if os.path.exists(rp):
        r = json.load(open(rp))["results"]
        REF = {k: {"roc_auc": v["test"]["roc_auc"],
                   "pr_auc": v["test"]["pr_auc"]}
               for k, v in r.items()}
    # frozen seed-42 LSTM arms — cross-tag runs (scaffold / seed43 /
    # smote) inline their seed-42 counterpart for immediate comparison
    LSTM_REF = {}
    if tag != "lstm" and os.path.exists("outputs/raw_lstm_eval.json"):
        r0 = json.load(open("outputs/raw_lstm_eval.json"))["results"]
        LSTM_REF = {k: {"roc_auc": v["test"]["roc_auc"],
                        "pr_auc": v["test"]["pr_auc"]}
                   for k, v in r0.items()}

    results, probs_dump = {}, {}
    for name, model in models.items():
        p_val = np.asarray(get_model_probs(model, X_val, device,
                                           cfg.EVAL_BATCH_SIZE), dtype=float)
        p_test = np.asarray(get_model_probs(model, X_test, device,
                                            cfg.EVAL_BATCH_SIZE),
                            dtype=float)
        cal = make_calibrator(cfg.CALIBRATION_METHOD).fit(y_val, p_val)
        if cfg.CALIBRATION_METHOD == "prior_shift":
            cal.set_prevalences(train_rate=float(np.mean(y_train)),
                                test_rate=float(np.mean(y_val)))
        p_val_c = cal.transform(p_val)
        p_test_c = cal.transform(p_test)
        t, fb = find_optimal_threshold_fbeta(
            y_val, p_val_c, beta=cfg.FBETA_BETA, grid=cfg.THRESHOLD_GRID)
        metrics = compute_all_metrics(y_test, p_test_c, t,
                                      beta=cfg.FBETA_BETA)
        frozen = select_fpr_thresholds_on_validation(y_val, p_val_c,
                                                     FPR_TARGETS)
        ops = frozen_operating_point_metrics(y_test, p_test_c, frozen)
        ref = REF.get(MLP_COUNTERPART_MAP.get(name))
        results[name] = {
            "threshold": float(t), "val_fbeta": float(fb),
            "test": metrics, "frozen_operating_points": ops,
            "rawsubstr_mlp_counterpart": ref,
            "lstm_seed42_counterpart": LSTM_REF.get(name),
            "wall_s": round(walls.get(name, 0.0), 1),
        }
        probs_dump[name] = np.asarray(p_test_c, dtype=np.float32)
        print(f"  {name:<16s} ROC-AUC {metrics['roc_auc']:.3f} "
              f"PR-AUC {metrics['pr_auc']:.3f}"
              + (f"  (MLP counterpart {ref['roc_auc']:.3f}/"
                 f"{ref['pr_auc']:.3f})" if ref else ""), flush=True)
        lref = LSTM_REF.get(name)
        if lref:
            print(f"  {'':16s} seed-42 LSTM counterpart "
                  f"{lref['roc_auc']:.3f}/{lref['pr_auc']:.3f}",
                  flush=True)

    report = {
        "purpose": ("review item 8 (LSTM half): frozen-protocol LSTM arms "
                    "(FedAvg/FedProx SolarLSTM) trained on the RAW "
                    "unbalanced SWAN-SF benchmark with in-pipeline "
                    "FPCKNN/LSBZM — identical substrate, frozen split, "
                    "and Dirichlet shards as run_raw_substrate.py; "
                    "single-pass test on raw P5. Optional pooled "
                    "centralized SolarLSTM comparator isolates "
                    "architecture vs federation. Optional arms: "
                    "SCAFFOLD-LSTM (--scaffold / --scaffold-only) and "
                    "per-client SMOTE (--smote)."),
        "protocol": {"rounds": n_rounds, "mu": cfg.MU, "alpha": 1.0,
                     "seed": cfg.SEED, "clients": cfg.N_CLIENTS,
                     "val_split": cfg.VAL_SPLIT,
                     "input": "3D (60, 24) LSBZM-normalised series",
                     "lstm": {"hidden": cfg.LSTM_HIDDEN_SIZE,
                              "layers": cfg.LSTM_NUM_LAYERS,
                              "dropout": cfg.LSTM_DROPOUT,
                              "bidirectional": cfg.LSTM_BIDIRECTIONAL},
                     "calibration": cfg.CALIBRATION_METHOD,
                     "loss": cfg.LOSS_VARIANT, "device": str(device),
                     "tag": tag, "smoke": bool(smoke),
                     "arms": sorted(models.keys()),
                     "scaffold": bool(cfg.USE_SCAFFOLD),
                     "smote": bool(getattr(cfg, "USE_SMOTE", False)),
                     "smote_ratio": (cfg.SMOTE_RATIO
                                      if getattr(cfg, "USE_SMOTE", False)
                                      else None)},
        "substrate": {"train": int(len(y_train)), "val": int(len(y_val)),
                      "test": int(len(y_test)),
                      "test_pos": float(np.mean(y_test)),
                      "source": ("data/cache/rawsubstrate (frozen split, "
                                 "cross-checked vs the 2D substrate)")},
        "results": results,
        "elapsed_s": round(time.time() - t0, 1),
    }
    _atomic_json(report, out_path)
    np.savez_compressed(os.path.join(CACHE, f"test_probs_{tag}.npz"),
                        y_test=y_test.astype(np.int8), **probs_dump)

    if args.event_level and not smoke:
        thresholds = {k: v["threshold"] for k, v in results.items()}
        event_level_eval(
            probs_dump, y_test, thresholds, args.raw_dir,
            ("outputs/event_level_raw_lstm_p5.json" if tag == "lstm"
             else f"outputs/event_level_raw_{tag}.json"),
            "event-level evaluation of the raw-substrate LSTM arms "
            "on raw P5, natural prevalence")

    print(f"\n[LSTM] report -> {out_path} ({time.time()-t0:.0f}s)",
          flush=True)

    if smoke:
        arm = "fedavg_lstm" if walls.get("fedavg_lstm") else \
            ("scaffold_lstm" if walls.get("scaffold_lstm") else None)
        if arm:
            r = walls[arm]
            est = r * (n_full_train / max(len(y_train), 1)) \
                * cfg.N_ROUNDS / max(n_rounds, 1)
            print(f"[smoke] {arm} {n_rounds} round(s) on "
                  f"{len(y_train):,} windows took {r:.0f}s -> full arm "
                  f"({n_full_train:,} windows x {cfg.N_ROUNDS} rounds) "
                  f"estimated {est/3600:.1f} h on this device", flush=True)


if __name__ == "__main__":
    main()

