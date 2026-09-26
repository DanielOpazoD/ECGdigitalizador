"""Re-score F6-b cases from the confirmed runs already on disk (no engine re-run).

Reads <work>/cases.jsonl (never modified: a running bench may still append to
it) and writes the rescored rows to --out. Per-image failure rows
(engine_error, time_scale_unknown, ...) are copied unchanged. The confirmed run
of each arm is the row's `run_dirs[arm]` when recorded, otherwise the newest
run under <work>/<id>/<type>/<engine>/runs whose segments carry signals,
classified as `evidence` if any segment has the image-evidence time-axis step.

Same scope as the bench: real Kaggle images, not clinical validation.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f6b_kaggle_bench import EVIDENCE_STAGE, read_truth, score_run

from ecg_photo.contracts import load_manifest


def classify_run(run_dir: Path) -> str | None:
    """'evidence' / 'engine' for a confirmed run, None for a raw engine run."""
    rm = load_manifest(run_dir / "manifest.json")
    if not any(s.signal_path for s in rm.segments):
        return None
    stages = {p.stage for s in rm.segments for p in s.processing}
    return "evidence" if EVIDENCE_STAGE in stages else "engine"


def find_confirmed_runs(case_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    runs = case_dir / "runs"
    if not runs.exists():
        return found
    for rd in sorted(runs.iterdir(), key=lambda p: p.stat().st_mtime):
        if (rd / "manifest.json").exists() and (arm := classify_run(rd)):
            found[arm] = rd  # newest wins
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=Path("runs/f6b/bench"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rhythm-lead", default="II")
    args = ap.parse_args()

    with open(args.data / "train.csv", newline="") as fh:
        meta = {r["id"]: r for r in csv.DictReader(fh)}
    truths: dict[str, dict] = {}
    rows = [json.loads(line) for line in (args.work / "cases.jsonl").read_text().splitlines()]
    n_rescored = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            rid = r["record"]
            case_dir = args.work / rid / r["image_type"] / r["engine"]
            found = find_confirmed_runs(case_dir)
            for arm, rel in (r.get("run_dirs") or {}).items():
                found[arm] = case_dir / rel
            for arm, cases in r["arms"].items():
                if not all("lead" in c for c in cases) or arm not in found:
                    continue  # per-image failure row: nothing to rescore
                if rid not in truths:
                    truths[rid] = read_truth(args.data / "train" / rid / f"{rid}.csv")
                fs = float(meta[rid]["fs"])
                r["arms"][arm] = score_run(found[arm], truths[rid], fs, args.rhythm_lead)
                n_rescored += 1
            fh.write(json.dumps(r) + "\n")
    print(f"rescored {n_rescored} arm results from {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
