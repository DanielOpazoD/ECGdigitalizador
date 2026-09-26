"""F7: does the truth-free quality report (ecg_photo.qc) separate good from bad
digitizations? Runs run_qc on the engine-axis confirmed runs of the F6-b
Kaggle bench and compares each label with the per-lead r@lag against the
truth (from the rescored cases). Real Kaggle images; not clinical validation.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f6b_kaggle_bench import git_rev
from f6b_rescore import find_confirmed_runs

from ecg_photo.qc import run_qc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("runs/f6b/bench"))
    ap.add_argument("--cases", type=Path, default=Path("runs/f6b/cases.rescored.jsonl"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for line in args.cases.read_text().splitlines():
        r = json.loads(line)
        case_dir = args.work / r["record"] / r["image_type"] / r["engine"]
        runs = find_confirmed_runs(case_dir)
        if "engine" not in runs:
            continue  # engine failed on this image
        q = run_qc(runs["engine"])
        ok = [c["r_at_lag"] for c in r["arms"]["engine"] if c.get("status") == "ok"]
        rows.append(
            {
                "record": r["record"],
                "image_type": r["image_type"],
                "engine": r["engine"],
                "label": q.label.value,
                "flags": q.flags,
                "einthoven_residual": q.einthoven_residual,
                "goldberger_residual": q.goldberger_residual,
                "median_r_at_lag": float(np.median(ok)) if ok else None,
            }
        )
    groups: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["median_r_at_lag"] is not None:
            groups[f"{r['engine']}|{r['label']}"].append(r["median_r_at_lag"])
    summary = {
        k: {
            "n": len(v),
            "median_r_at_lag": float(np.median(v)),
            "p10_r_at_lag": float(np.percentile(v, 10)),
        }
        for k, v in sorted(groups.items())
    }
    flags = Counter(f for r in rows for f in r["flags"])
    res = {
        "generated": datetime.now(UTC).isoformat(),
        "scope": "F6-b Kaggle engine-axis runs; QC label vs truth r@lag; not clinical validation",
        "commit": git_rev(Path(__file__).resolve().parent.parent),
        "by_label": summary,
        "flag_counts": dict(flags),
        "images": rows,
    }
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("| motor | etiqueta | n | r@lag med | p10 |")
    print("|---|---|---|---|---|")
    for k, s in summary.items():
        e, lab = k.split("|")
        print(f"| {e} | {lab} | {s['n']} | {s['median_r_at_lag']:.3f} | {s['p10_r_at_lag']:.3f} |")
    print(dict(flags))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
