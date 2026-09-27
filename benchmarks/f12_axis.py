"""F12: frontal QRS axis (ecg_photo.intervals.qrs_axis) against two
references on the F11 PTB-XL sample (300 records, signals cut as a 3x4 + II
printout and full 10 s):

- GE 12SL frontal R axis (PTB-XL+, what a GE electrocardiograph prints), in
  degrees;
- the cardiologists' axis label of PTB-XL (heart_axis), as a category: MID
  and LAD (German "Linkstyp", a normal variant) normal; ALAD left deviation;
  RAD, ARAD, AXR right.

Category of a measured axis: left < -30 deg, normal -30..+90, right > +90.
No parameter is tuned here; not clinical validation.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f6b_kaggle_bench import git_rev
from f9_intervals_eval import printed
from f11_intervals_12sl import LEADS

from ecg_photo.intervals import measure_signals

# PTB-XL uses the German axis types: LAD ("Linkstyp", about -30..+30 deg,
# median -13 deg here both by us and by 12SL) is a normal variant; ALAD
# ("ueberdrehter Linkstyp") is the left axis deviation (< -30 deg).
CARDIO = {
    "MID": "normal",
    "LAD": "normal",
    "ALAD": "left",
    "RAD": "right",
    "ARAD": "right",
    "AXR": "right",
}


def category(axis: float | None) -> str | None:
    """left: -90..-30; normal: -30..+90; right: +90..180 and the extreme
    (northwest) quadrant -180..-90."""
    if axis is None:
        return None
    if -30.0 <= axis <= 90.0:
        return "normal"
    if -90.0 <= axis < -30.0:
        return "left"
    return "right"


def angdiff(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def read_12sl_axis(path: Path, ids: set[int]) -> dict[int, float]:
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            eid = int(float(row["ecg_id"]))
            v = row.get("R_AxisFrontal_Global")
            if eid in ids and v not in (None, "", "nan"):
                out[eid] = float(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("runs/f11"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ids = {int(x) for x in (args.data / "sample_ids.txt").read_text().split()}
    ge = read_12sl_axis(args.data / "12sl_features.csv", ids)
    cardio = {}
    with open(args.data / "ptbxl_database.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            eid = int(float(row["ecg_id"]))
            if eid in ids and row["heart_axis"] in CARDIO:
                cardio[eid] = CARDIO[row["heart_axis"]]

    rows = []
    for eid in sorted(ids):
        r = wfdb.rdrecord(str(args.data / "ptbxl" / f"{eid:05d}_hr"))
        fs = float(r.fs)
        sig = {LEADS[n.upper()]: r.p_signal[:, i].astype(float) for i, n in enumerate(r.sig_name)}
        row = {"ecg_id": eid, "ge_deg": ge.get(eid), "cardio": cardio.get(eid)}
        for arm, s in (("printed", printed(sig, fs)), ("full", sig)):
            iv = measure_signals({k: (v, fs) for k, v in s.items()})
            row[arm] = {"deg": iv.qrs_axis_deg, "status": iv.status["qrs_axis_deg"]}
        rows.append(row)

    summary: dict = {}
    for arm in ("printed", "full"):
        s: dict = {}
        for only_ok in (False, True):
            sel = [r for r in rows if r[arm]["deg"] is not None]
            if only_ok:
                sel = [r for r in sel if r[arm]["status"] == "ok"]
            d = np.array(
                [angdiff(r[arm]["deg"], r["ge_deg"]) for r in sel if r["ge_deg"] is not None]
            )
            ge_cat = [
                (category(r[arm]["deg"]), category(r["ge_deg"]))
                for r in sel
                if r["ge_deg"] is not None
            ]
            ca_cat = [(category(r[arm]["deg"]), r["cardio"]) for r in sel if r["cardio"]]
            ge_vs_ca = [
                (category(r["ge_deg"]), r["cardio"])
                for r in sel
                if r["cardio"] and r["ge_deg"] is not None
            ]
            s["ok" if only_ok else "all"] = {
                "n": len(sel),
                "vs_12sl_deg": {
                    "n": len(d),
                    "mean": float(d.mean()),
                    "median_abs": float(np.median(np.abs(d))),
                    "p90_abs": float(np.percentile(np.abs(d), 90)),
                    "within_15": float(np.mean(np.abs(d) <= 15)),
                },
                "category_agree_12sl": float(np.mean([a == b for a, b in ge_cat])),
                "category_agree_cardiologists": float(np.mean([a == b for a, b in ca_cat])),
                "n_cardiologists": len(ca_cat),
                "12sl_category_agree_cardiologists": float(np.mean([a == b for a, b in ge_vs_ca])),
                "confusion_vs_cardiologists": dict(Counter(f"{b}->{a}" for a, b in ca_cat)),
            }
        summary[arm] = s
    res = {
        "generated": datetime.now(UTC).isoformat(),
        "scope": "PTB-XL (F11 sample) axis vs GE 12SL and cardiologist labels; not clinical "
        "validation",
        "commit": git_rev(Path(__file__).resolve().parent.parent),
        "n_records": len(rows),
        "summary": summary,
        "records": rows,
    }
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    for arm, s in summary.items():
        for k, v in s.items():
            g = v["vs_12sl_deg"]
            print(
                f"{arm} {k}: n={v['n']} vs 12SL: media {g['mean']:+.1f}°, |dif.| mediana "
                f"{g['median_abs']:.1f}°, p90 {g['p90_abs']:.0f}°, ≤15° {100 * g['within_15']:.0f} % | "
                f"categoría = 12SL {100 * v['category_agree_12sl']:.0f} %, = cardiólogos "
                f"{100 * v['category_agree_cardiologists']:.0f} % (n={v['n_cardiologists']}; "
                f"12SL = cardiólogos {100 * v['12sl_category_agree_cardiologists']:.0f} %)"
            )
            print("   ", v["confusion_vs_cardiologists"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
