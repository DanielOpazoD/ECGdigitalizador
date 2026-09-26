"""F7: does the truth-free quality report (ecg_photo.qc) separate good from bad
digitizations? Runs run_qc on the engine-axis confirmed runs of the F6-b
Kaggle bench and compares each label with the per-lead r@lag against the
truth (from the rescored cases).

Also, per image:
- the RR contrast: the "printed" RR is the mean RR that detect_qrs finds on
  the true lead II (full 10 s), compared with the RR of the digitized strip;
- the limb-lead identity check under several absolute rms floors
  (--identity-floors, mV), against the limb leads' own r@lag.
Real Kaggle images; not clinical validation.
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f6b_kaggle_bench import git_rev, read_truth
from f6b_rescore import find_confirmed_runs

from ecg_photo.qc import detect_qrs, run_qc

LIMB = ("I", "II", "III", "aVR", "aVL", "aVF")
IDENTITY_FLAGS = {"LIMB_LEADS_INCONSISTENT", "LIMB_LEADS_DOUBTFUL"}


def truth_rr_ms(sig: np.ndarray, fs: float) -> float | None:
    beats = detect_qrs(sig, fs)
    return float(np.mean(np.diff(beats)) / fs * 1000.0) if len(beats) >= 3 else None


def label_summary(rows: list[dict], key: str) -> dict:
    groups: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["median_r_at_lag"] is not None:
            groups[f"{r['engine']}|{r[key]}"].append(r["median_r_at_lag"])
    return {
        k: {
            "n": len(v),
            "median_r_at_lag": float(np.median(v)),
            "p10_r_at_lag": float(np.percentile(v, 10)),
        }
        for k, v in sorted(groups.items())
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("runs/f6b/kaggle"))
    ap.add_argument("--work", type=Path, default=Path("runs/f6b/bench"))
    ap.add_argument("--cases", type=Path, default=Path("runs/f6b/cases.rescored.jsonl"))
    ap.add_argument("--identity-floors", type=float, nargs="+", default=[0.0, 0.1, 0.15, 0.2])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    with open(args.data / "train.csv", newline="") as fh:
        meta = {r["id"]: r for r in csv.DictReader(fh)}
    truth_rr: dict[str, float | None] = {}
    rows = []
    for line in args.cases.read_text().splitlines():
        r = json.loads(line)
        rid = r["record"]
        case_dir = args.work / rid / r["image_type"] / r["engine"]
        runs = find_confirmed_runs(case_dir)
        if "engine" not in runs:
            continue  # engine failed on this image
        if rid not in truth_rr:
            fs = float(meta[rid]["fs"])
            truth_rr[rid] = truth_rr_ms(
                read_truth(args.data / "train" / rid / f"{rid}.csv")["II"], fs
            )
        ok = [c["r_at_lag"] for c in r["arms"]["engine"] if c.get("status") == "ok"]
        limb = [
            c["r_at_lag"]
            for c in r["arms"]["engine"]
            if c.get("status") == "ok" and c.get("lead") in LIMB
        ]
        row: dict = {
            "record": rid,
            "image_type": r["image_type"],
            "engine": r["engine"],
            "median_r_at_lag": float(np.median(ok)) if ok else None,
            "limb_median_r_at_lag": float(np.median(limb)) if limb else None,
            "by_floor": {},
        }
        for floor in args.identity_floors:
            q = run_qc(runs["engine"], identity_floor_mV=floor)
            row["by_floor"][str(floor)] = {
                "label": q.label.value,
                "flags": q.flags,
                "einthoven_residual": q.einthoven_residual,
                "goldberger_residual": q.goldberger_residual,
            }
        if truth_rr[rid] is not None:
            q = run_qc(runs["engine"], printed_rr_ms=truth_rr[rid])
            row.update(
                truth_rr_ms=truth_rr[rid],
                rr_measured_ms=q.rr_measured_ms,
                rr_error_pct=q.rr_error_pct,
                rr_flags=[f for f in q.flags if f.startswith("RR_")],
            )
        base = row["by_floor"][str(args.identity_floors[0])]
        row["label"], row["flags"] = base["label"], base["flags"]
        rows.append(row)

    floors = {}
    for floor in args.identity_floors:
        f = str(floor)
        for r in rows:
            r["_label"] = r["by_floor"][f]["label"]
        conf: Counter[str] = Counter()
        for r in rows:
            lr = r["limb_median_r_at_lag"]
            if lr is None:
                continue
            band = "limb_r>=0.9" if lr >= 0.9 else ("limb_r<0.7" if lr < 0.7 else "limb_r_0.7-0.9")
            ident = set(r["by_floor"][f]["flags"]) & IDENTITY_FLAGS
            tag = (
                "inconsistent"
                if "LIMB_LEADS_INCONSISTENT" in ident
                else ("doubtful" if ident else "consistent")
            )
            conf[f"{band}|{tag}"] += 1
        floors[f] = {"by_label": label_summary(rows, "_label"), "identity_vs_limb_r": dict(conf)}
    for r in rows:
        r.pop("_label", None)

    errs = [abs(r["rr_error_pct"]) for r in rows if r.get("rr_error_pct") is not None]
    rr = {
        "n": len(errs),
        "median_abs_error_pct": float(np.median(errs)) if errs else None,
        "p90_abs_error_pct": float(np.percentile(errs, 90)) if errs else None,
        "n_over_5pct": int(sum(e > 5 for e in errs)),
        "n_not_measurable": int(sum("RR_NOT_MEASURABLE" in r.get("rr_flags", []) for r in rows)),
    }
    res = {
        "generated": datetime.now(UTC).isoformat(),
        "scope": "F6-b Kaggle engine-axis runs; QC label vs truth r@lag; not clinical validation",
        "commit": git_rev(Path(__file__).resolve().parent.parent),
        "by_label": floors[str(args.identity_floors[0])]["by_label"],
        "by_identity_floor_mV": floors,
        "rr_vs_truth": rr,
        "flag_counts": dict(Counter(f for r in rows for f in r["flags"])),
        "images": rows,
    }
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    for f, s in floors.items():
        print(f"\n### suelo {f} mV")
        print("| motor | etiqueta | n | r@lag med | p10 |")
        print("|---|---|---|---|---|")
        for k, v in s["by_label"].items():
            e, lab = k.split("|")
            print(
                f"| {e} | {lab} | {v['n']} | {v['median_r_at_lag']:.3f} | {v['p10_r_at_lag']:.3f} |"
            )
        print(dict(sorted(s["identity_vs_limb_r"].items())))
    print("\nRR vs verdad:", rr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
