"""
data_manifest/verify_manifest.py  (v4.0 — Stage 1: dataset verification)
────────────────────────────────────────────────────────────────────────
Verifies every partition file on disk against the frozen manifest
(SHA-256, byte size, shape). This is the executable backing for the
paper's "byte-verified" claim: the word is now a command, not an
assertion.

Usage:
    python data_manifest/verify_manifest.py                # verify + log
    python data_manifest/verify_manifest.py --quiet        # exit code only

Exit codes: 0 = all files verified, 1 = any mismatch/missing file.
Writes logs/verify_manifest.log (one line per file + verdict) and
prints the manifest hash to embed in outputs/run_manifest.json.

Importable:  verify(data_dir=None, manifest_path=None, log_path=None)
             returns a dict {ok, n_files, n_match, mismatches, manifest_sha256}.
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg

DEFAULT_MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "manifest.json")
DEFAULT_LOG = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "logs", "verify_manifest.log")


def file_sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify(data_dir=None, manifest_path=DEFAULT_MANIFEST,
           log_path=DEFAULT_LOG):
    """Verify every manifest file against disk. Returns summary dict."""
    data_dir = data_dir or cfg.CLEANED_DATA_DIR
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    results = []
    n_match = 0
    mismatches = []
    for split in ("train", "test"):
        for entry in manifest.get("files", {}).get(split, []):
            fname = entry["file"]
            path = os.path.join(data_dir, split, fname)
            rec = {"split": split, "file": fname,
                   "expected_sha256": entry["sha256"]}
            if not os.path.exists(path):
                rec["status"] = "MISSING"
                mismatches.append(rec)
            else:
                digest = file_sha256(path)
                rec["actual_sha256"] = digest
                size_ok = (entry.get("bytes") is None or
                           os.path.getsize(path) == entry["bytes"])
                if digest == entry["sha256"] and size_ok:
                    rec["status"] = "VERIFIED"
                    n_match += 1
                else:
                    rec["status"] = "MISMATCH"
                    if not size_ok:
                        rec["bytes_expected"] = entry.get("bytes")
                        rec["bytes_actual"] = os.path.getsize(path)
                    mismatches.append(rec)
            results.append(rec)

    # shape check where cheap (label pickles are small; feature pickles
    # are hashed byte-wise — shape re-derivation would re-load 4.7 GB,
    # so byte identity subsumes it)
    manifest_sha = file_sha256(manifest_path)

    ok = (len(mismatches) == 0) and (len(results) == 20)
    summary = {
        "ok": ok,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha,
        "n_files": len(results),
        "n_verified": n_match,
        "mismatches": mismatches,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
    }

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            f.write(f"dataset manifest verification "
                    f"{summary['verified_utc']}\n")
            f.write(f"manifest: {manifest_path}\n")
            f.write(f"manifest sha256: {manifest_sha}\n")
            f.write(f"data dir: {data_dir}\n")
            f.write("-" * 72 + "\n")
            for rec in results:
                f.write(f"[{rec['status']:>8}] {rec['split']:>5} "
                        f"{rec['file']}\n")
                if rec["status"] == "MISMATCH":
                    f.write(f"           expected {rec['expected_sha256']}\n")
                    f.write(f"           actual   {rec.get('actual_sha256')}\n")
            f.write("-" * 72 + "\n")
            verdict = "PASS" if ok else "FAIL"
            f.write(f"VERDICT: {verdict} — {n_match}/{len(results)} files "
                    f"byte-identical to the frozen manifest\n")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=cfg.CLEANED_DATA_DIR)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    summary = verify(args.data_dir, args.manifest, args.log)

    if not args.quiet:
        print(f"[Verify] manifest sha256: {summary['manifest_sha256']}")
        print(f"[Verify] {summary['n_verified']}/{summary['n_files']} "
              f"files byte-identical")
        for rec in summary["mismatches"]:
            print(f"[Verify] {rec['status']}: {rec['file']}")
        print(f"[Verify] log -> {args.log}")
        print(f"[Verify] VERDICT: "
              f"{'PASS' if summary['ok'] else 'FAIL'}")
    sys.exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
