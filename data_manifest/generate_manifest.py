"""
data_manifest/generate_manifest.py  (v3.0 — Stage 1: dataset provenance)
────────────────────────────────────────────────────────────────────────
Freezes the dataset contract in a machine-readable manifest BEFORE any
experiment runs. The manifest is the audit artefact NASA review asks
for: dataset version, partitions found, shapes, prevalence, feature
order, normalization, imputation, label definition, forecast horizon,
and split protocol.

Usage:
    python data_manifest/generate_manifest.py           # scan data/cleaned
    python data_manifest/generate_manifest.py --json data_manifest/manifest.json
"""

import argparse
import glob
import hashlib
import json
import os
import pickle
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg

BASE_FEATURES = cfg.FEATURE_COLS  # the canonical 24, correct order
STATS = ["mean", "std", "max", "min", "trend", "slope"]


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def scan_cleaned_dir(data_dir):
    """Inventory of the cleaned SWAN-SF export (files, hashes, shapes)."""
    inventory = {"train": [], "test": []}
    for split in ("train", "test"):
        d = os.path.join(data_dir, split)
        if not os.path.isdir(d):
            continue
        for pkl in sorted(glob.glob(os.path.join(d, "Partition*.pkl"))):
            entry = {"file": os.path.basename(pkl),
                     "sha256": file_sha256(pkl),
                     "bytes": os.path.getsize(pkl)}
            try:
                with open(pkl, "rb") as f:
                    arr = pickle.load(f)
                entry["shape"] = list(arr.shape) if hasattr(arr, "shape") \
                    else [len(arr)]
                entry["dtype"] = str(arr.dtype) if hasattr(arr, "dtype") else "?"
            except Exception as e:
                entry["load_error"] = str(e)
            inventory[split].append(entry)
    return inventory


def label_stats_from_inventory(inventory):
    """Prevalence from label pickles (X files carry shapes)."""
    out = {"train": None, "test": None}
    for split in ("train", "test"):
        y_files = [e for e in inventory[split]
                   if "Labels" in e["file"] and "shape" in e]
        if not y_files:
            continue
        try:
            path = os.path.join(cfg.CLEANED_DATA_DIR, split,
                                y_files[0]["file"])
            with open(path, "rb") as f:
                y = pickle.load(f)
            out[split] = {"n": int(len(y)),
                          "n_positive": int((y == 1).sum()),
                          "prevalence": float((y == 1).mean())}
        except Exception:
            pass
    return out


def build_manifest(data_dir=None, include_hashes=True):
    data_dir = data_dir or cfg.CLEANED_DATA_DIR
    inventory = scan_cleaned_dir(data_dir)

    n_base = len(BASE_FEATURES)
    flat_names = [f"{s}_{b}" for s in STATS for b in BASE_FEATURES]

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator_version": cfg.VERSION,
        "dataset": {
            "name": "SWAN-SF benchmark (Angryk et al., 2020, Sci Data)",
            "variant": "Cleaned SWAN-SF (RUS-Tomek-TimeGAN, LSBZM-Norm, "
                       "FPCKNN-impute, WithoutC)",
            "source_repo": "https://github.com/samresume/Cleaned-SWANSF-Dataset",
            "source_doi": "10.7910/DVN/EBCFKM",
            "data_dir": data_dir,
            "partitions_found": sorted({
                e["file"].split("_")[0].replace("Partition", "")
                for split in inventory.values() for e in split
            }),
        },
        "feature_contract": {
            "n_base_features": n_base,
            "base_features": BASE_FEATURES,
            "flattening": cfg.FLATTEN_METHOD,
            "flat_feature_count": len(flat_names),
            "flat_feature_order": flat_names,
            "v25_correction": "AREA_ACR/HARPNUM_MOD/TIME_SINCE_LAST_FLARE "
                              "replaced by TOTFZ/TOTFY/TOTFX Lorentz components",
        },
        "normalization": {
            "applied_by_dataset": "LSBZM (L2-BoxCox-Z-MinMax)",
            "pipeline_reapplies_scaler": not cfg.CLEANED_ALREADY_NORMALIZED,
        },
        "imputation": "FPCKNN (dataset-side); NaN->0 guard in loader",
        "label_definition": {
            "positive": "M/X-class flare within forecast window",
            "forecast_horizon": "24h (SWAN-SF default; C=WithoutC keeps "
                                "class-imbalance labels)",
            "note": "exact horizon fixed by the cleaned export, not this repo",
        },
        "split_protocol": {
            "source": "dataset-defined train/test partitions (time-based)",
            "validation": f"carved from TRAIN only ({cfg.VAL_SPLIT:.0%}), "
                          f"stratified, seed {cfg.SEED}",
            "test_policy": "touched exactly once at final evaluation",
        },
        "files": inventory if include_hashes else
                 {k: [{"file": e["file"], "shape": e.get("shape")}
                      for e in v] for k, v in inventory.items()},
    }

    stats = label_stats_from_inventory(inventory)
    if stats["train"] or stats["test"]:
        manifest["prevalence"] = stats
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=cfg.CLEANED_DATA_DIR)
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-hash", action="store_true",
                    help="skip SHA-256 for large files")
    args = ap.parse_args()

    manifest = build_manifest(args.data_dir, include_hashes=not args.no_hash)

    out = args.json or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "manifest.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Manifest] written -> {out}")

    # human summary
    print(f"[Manifest] partitions found: "
          f"{manifest['dataset']['partitions_found'] or 'NONE (data missing)'}")
    print(f"[Manifest] base features: {manifest['feature_contract']['n_base_features']}")
    print(f"[Manifest] flat features: {manifest['feature_contract']['flat_feature_count']}")
    if manifest.get("prevalence", {}).get("train"):
        print(f"[Manifest] train prevalence: "
              f"{manifest['prevalence']['train']['prevalence']:.4f}")
    if manifest.get("prevalence", {}).get("test"):
        print(f"[Manifest] test prevalence:  "
              f"{manifest['prevalence']['test']['prevalence']:.4f}")


if __name__ == "__main__":
    main()
