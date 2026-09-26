"""F6-b: fetch a small, deterministic subset of the Kaggle competition
"PhysioNet - Digitization of ECG Images" (physionet-ecg-image-digitization).

Only what the bench needs is downloaded, file by file: train.csv, then for
each selected record its ground-truth CSV and its images. Nothing is
versioned; sha256 of every file goes to <dest>/fetch_manifest.json.

Credentials come from the environment, never from arguments or files in the
repo: KAGGLE_USERNAME + KAGGLE_KEY (legacy key) or KAGGLE_API_TOKEN. The
competition rules must have been accepted on kaggle.com (otherwise 403).

Expected layout (verified on first run with --list; the script stops if the
truth CSV of a record is missing):
    train.csv                       id, fs, sig_len
    train/<id>/<id>.csv             one column per lead (mV)
    train/<id>/<id>-<NNNN>.png      image variants of the same record
"""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

COMPETITION = "physionet-ecg-image-digitization"
IMAGE_TYPES = [f"{i:04d}" for i in range(1, 13)]


def _api():
    # kaggle >= 2 prefers KAGGLE_API_TOKEN; new-style keys (KGAT_...) given as
    # KAGGLE_KEY are forwarded so either variable name works.
    key = os.environ.get("KAGGLE_KEY", "")
    if key.startswith("KGAT_") and not os.environ.get("KAGGLE_API_TOKEN"):
        os.environ["KAGGLE_API_TOKEN"] = key
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(api, remote: str, dest_root: Path) -> Path | None:
    """Download `remote` (competition-relative path) to dest_root/remote.
    Returns None if the file does not exist remotely (404)."""
    target = dest_root / remote
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        api.competition_download_file(COMPETITION, remote, path=str(target.parent), quiet=True)
    except Exception as e:  # kaggle raises various HTTP error types
        msg = str(e)
        if "404" in msg or "Not Found" in msg:
            return None
        raise
    name = Path(remote).name
    zipped = target.parent / (name + ".zip")
    if not target.exists() and zipped.exists():
        with zipfile.ZipFile(zipped) as zf:
            zf.extractall(target.parent)
        zipped.unlink()
    if not target.exists():
        raise RuntimeError(f"download of {remote} produced no file in {target.parent}")
    return target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=Path("runs/f6b/kaggle"))
    ap.add_argument("--n-records", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260925, help="record selection seed")
    ap.add_argument("--ids", nargs="*", default=None, help="explicit record ids")
    ap.add_argument("--types", nargs="*", default=IMAGE_TYPES, help="image variants NNNN")
    ap.add_argument("--list", action="store_true", help="list the first remote files and exit")
    args = ap.parse_args()

    api = _api()
    if args.list:
        resp = api.competition_list_files(COMPETITION, page_size=200)
        files = getattr(resp, "files", None) or []
        for f in files[:200]:
            print(getattr(f, "name", f), getattr(f, "total_bytes", ""))
        print(f"next_page_token={getattr(resp, 'next_page_token', None)!r}")
        return 0

    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    train_csv = fetch(api, "train.csv", dest)
    if train_csv is None:
        print("train.csv not found in competition files", file=sys.stderr)
        return 2
    with open(train_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    by_id = {r["id"]: r for r in rows}
    if args.ids:
        ids = args.ids
    else:
        all_ids = sorted(by_id)
        ids = sorted(random.Random(args.seed).sample(all_ids, min(args.n_records, len(all_ids))))

    manifest: dict = {
        "competition": COMPETITION,
        "fetched_at": datetime.now(UTC).isoformat(),
        "selection": {"seed": args.seed, "ids": ids, "types": args.types},
        "files": {},
        "missing": [],
    }
    manifest["files"]["train.csv"] = {"sha256": sha256_file(train_csv)}
    for rid in ids:
        if rid not in by_id:
            print(f"id {rid} not in train.csv", file=sys.stderr)
            return 2
        truth = fetch(api, f"train/{rid}/{rid}.csv", dest)
        if truth is None:
            print(f"ground truth train/{rid}/{rid}.csv missing: layout differs", file=sys.stderr)
            return 2
        manifest["files"][f"train/{rid}/{rid}.csv"] = {"sha256": sha256_file(truth)}
        for t in args.types:
            remote = f"train/{rid}/{rid}-{t}.png"
            p = fetch(api, remote, dest)
            if p is None:
                manifest["missing"].append(remote)
                continue
            manifest["files"][remote] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        print(f"{rid}: fs={by_id[rid].get('fs')} sig_len={by_id[rid].get('sig_len')}", flush=True)
    (dest / "fetch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{len(manifest['files'])} files, {len(manifest['missing'])} missing -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
