"""F10 (objective O7 on real images): do PR / QRS / QT survive digitization
of real scans and phone photos?

Kaggle PhysioNet ECG Image Digitization images (F6-b data: 10 records x 9
image types; 0001 is a clean rendering, the others real scans and photos of
printouts). No cardiologist annotations exist for these records, so the
reference is the same algorithm (ecg_photo.intervals) applied to the true
signal, cut exactly as printed (NaN outside each lead's slot). The difference
isolates what digitization adds; the algorithm's own error against
cardiologists is F9 (LUDB).

Uses the Ahus runs confirmed on the image-evidence time axis saved by
F7 step 5 (runs/f7/aligned); no engine is re-run. Also reports whether the
quality label (qc.run_qc, no truth) and the per-interval status predict the
error.
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f6b_kaggle_bench import git_rev, read_truth

from ecg_photo.contracts import load_manifest
from ecg_photo.intervals import intervals_json, measure_run, measure_signals
from ecg_photo.qc import STANDARD_LEADS, run_qc

KEYS = ("pr_ms", "qrs_ms", "qt_ms", "qrs_axis_deg")  # axis in degrees (F12)


def confirmed_run(case_dir: Path) -> Path | None:
    """Newest run of the case with a calibrated signal (evidence axis)."""
    runs = []
    for r in (case_dir / "runs").glob("run-*"):
        if not (r / "manifest.json").exists():
            continue
        m = load_manifest(r / "manifest.json")
        if any(s.signal_path for s in m.segments):
            runs.append((r.stat().st_mtime, r))
    return max(runs)[1] if runs else None


def stats(d: list[float]) -> dict:
    if not d:
        return {"n": 0}
    a = np.array(d)
    return {
        "n": len(a),
        "mean_diff_ms": float(a.mean()),
        "sd_diff_ms": float(a.std(ddof=1)) if len(a) > 1 else None,
        "median_abs_ms": float(np.median(np.abs(a))),
        "p90_abs_ms": float(np.percentile(np.abs(a), 90)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("runs/f6b/kaggle/train"))
    ap.add_argument("--runs", type=Path, default=Path("runs/f7/aligned"))
    ap.add_argument("--engine", default="ahus")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for rec_dir in sorted(p for p in args.data.iterdir() if p.is_dir()):
        rid = rec_dir.name
        truth = read_truth(rec_dir / f"{rid}.csv")
        fs_truth = float(
            next(
                line.split(",")[1]
                for line in (args.data.parent / "train.csv").read_text().splitlines()
                if line.startswith(rid + ",")
            )
        )
        ref = measure_signals({n: (truth[n], fs_truth) for n in STANDARD_LEADS if n in truth})
        for case in sorted((args.runs / rid).glob(f"*/{args.engine}")):
            itype = case.parent.name
            run = confirmed_run(case)
            row: dict = {"record": rid, "image_type": itype, "truth": intervals_json(ref)}
            if run is None:
                row["digitized"] = None
            else:
                qc = run_qc(run)
                row["qc_label"] = qc.label
                row["digitized"] = intervals_json(measure_run(run, rr_ms=qc.rr_measured_ms))
            rows.append(row)

    by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        dig, tr = r["digitized"], r["truth"]
        if dig is None:
            continue
        for k in KEYS:
            if dig[k] is None or tr[k] is None:
                continue
            d = dig[k] - tr[k]
            if k == "qrs_axis_deg":  # angles wrap at +-180
                d = (d + 180.0) % 360.0 - 180.0
            by_type[r["image_type"]][k].append(d)
            groups[f"status={dig['status'][k]}"][k].append(d)
            groups[f"qc={r['qc_label']}"][k].append(d)
            if dig["status"][k] == "ok" and r["qc_label"] != "insufficient":
                groups["ok & qc!=insufficient"][k].append(d)
            groups["all"][k].append(d)
    found = {
        k: sum(1 for r in rows if r["digitized"] and r["digitized"][k] is not None) for k in KEYS
    }
    res = {
        "generated": datetime.now(UTC).isoformat(),
        "scope": "Kaggle F6-b images, Ahus, evidence time axis; reference = same algorithm on the "
        "printed truth signal; not clinical validation",
        "commit": git_rev(Path(__file__).resolve().parent.parent),
        "n_images": len(rows),
        "n_confirmed": sum(1 for r in rows if r["digitized"] is not None),
        "found": found,
        "by_type": {t: {k: stats(v[k]) for k in KEYS} for t, v in sorted(by_type.items())},
        "groups": {g: {k: stats(v[k]) for k in KEYS} for g, v in sorted(groups.items())},
        "cases": [
            {
                "record": r["record"],
                "image_type": r["image_type"],
                "qc_label": r.get("qc_label"),
                "truth": {k: r["truth"][k] for k in (*KEYS, "status")},
                "digitized": {k: r["digitized"][k] for k in (*KEYS, "status")}
                if r["digitized"]
                else None,
            }
            for r in rows
        ],
    }
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")

    def line(label: str, s: dict) -> str:
        cells = []
        for k in KEYS:
            v = s[k]
            cells.append(
                "—"
                if not v["n"]
                else f"{v['n']} · {v['mean_diff_ms']:+.0f} ± {v['sd_diff_ms'] or 0:.0f} · "
                f"{v['median_abs_ms']:.0f}"
            )
        return f"| {label} | " + " | ".join(cells) + " |"

    print(f"imágenes {res['n_images']}, confirmadas {res['n_confirmed']}, halladas {found}")
    print("| grupo | PR n · media ± DE · |dif.| med | QRS | QT | eje (°) |")
    print("|---|---|---|---|---|")
    for t, s in res["by_type"].items():
        print(line(t, s))
    for g, s in res["groups"].items():
        print(line(g, s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
