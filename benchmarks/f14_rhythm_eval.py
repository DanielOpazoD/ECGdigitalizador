"""F14: irregular-rhythm alert (ecg_photo.rhythm) against the rhythm labels of
PTB-XL (cardiologist SCP codes).

Records: the F11 random sample (300) plus a targeted one (200 AFIB, 100 other
irregular rhythms: SARRH, SVARR, BIGU, TRIGU). Classes: AF (AFIB/AFLT),
regular sinus (SR, STACH, SBRAD and no irregular code), other irregular.
Signals cut as a 3x4 + II printout (the alert reads the 10 s II strip; the
P-wave count comes from the printed short leads). The thresholds are chosen
on even ecg_ids and reported on odd ones. Not clinical validation.
"""

import argparse
import ast
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f6b_kaggle_bench import git_rev
from f9_intervals_eval import printed
from f11_intervals_12sl import LEADS

import ecg_photo.rhythm as rh
from ecg_photo.intervals import measure_signals

AF = {"AFIB", "AFLT"}
OTHER_IRREGULAR = {"SARRH", "SVARR", "BIGU", "TRIGU"}
REGULAR = {"SR", "STACH", "SBRAD"}


def label(codes: set[str]) -> str | None:
    if codes & AF:
        return "af"
    if codes & OTHER_IRREGULAR:
        return "other_irregular"
    if codes & REGULAR:
        return "regular"
    return None  # paced, SVT without rhythm label...: left out


def features(data: Path, eid: int) -> dict | None:
    r = wfdb.rdrecord(str(data / "ptbxl" / f"{eid:05d}_hr"))
    fs = float(r.fs)
    sig = {LEADS[n.upper()]: r.p_signal[:, i].astype(float) for i, n in enumerate(r.sig_name)}
    pr = printed(sig, fs)
    iv = measure_signals({k: (v, fs) for k, v in pr.items()})
    return rh.rhythm_features(pr["II"], fs, p_leads=iv.n_leads.get("pr_ms", 0))


def rates(rows: list[dict], flag: str) -> dict:
    out = {}
    for cls in ("af", "other_irregular", "regular"):
        sel = [r for r in rows if r["class"] == cls and r[flag] is not None]
        out[cls] = {
            "n": len(sel),
            "flagged": float(np.mean([r[flag] for r in sel])) if sel else None,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("runs/f11"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    codes = {}
    with open(args.data / "ptbxl_database.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            codes[int(float(row["ecg_id"]))] = set(ast.literal_eval(row["scp_codes"]))
    ids = {int(x) for x in (args.data / "sample_ids.txt").read_text().split()}
    ids |= {int(x) for x in (args.data / "rhythm_ids.txt").read_text().split()}
    rows = []
    for eid in sorted(ids):
        if not (args.data / "ptbxl" / f"{eid:05d}_hr.dat").exists():
            continue
        cls = label(codes.get(eid, set()))
        if cls is None:
            continue
        f = features(args.data, eid)
        rows.append({"ecg_id": eid, "class": cls, "tune": eid % 2 == 0, **(f or {})})

    tune = [r for r in rows if r["tune"] and r.get("irregularity") is not None]
    # threshold: the lowest that keeps >= 95 % of regular sinus records unflagged
    reg = sorted(r["irregularity"] for r in tune if r["class"] == "regular")
    thr = float(np.percentile(reg, 95))
    for r in rows:
        irr = r.get("irregularity")
        r["irregular"] = None if irr is None else bool(irr > thr)
        r["irregular_no_p"] = (
            None if irr is None else bool(irr > thr and r.get("p_leads", 0) < rh.P_LEADS_MIN)
        )
    check = [r for r in rows if not r["tune"]]
    res = {
        "generated": datetime.now(UTC).isoformat(),
        "scope": "PTB-XL rhythm labels (cardiologists); printed 3x4 + II; not clinical validation",
        "commit": git_rev(Path(__file__).resolve().parent.parent),
        "threshold_irregularity_from_even_ids": thr,
        "n": {"tune": len(tune), "check": len(check)},
        "check_irregular": rates(check, "irregular"),
        "check_irregular_no_p": rates(check, "irregular_no_p"),
        "tune_irregular": rates(tune, "irregular"),
        "records": rows,
    }
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"umbral de irregularidad (p95 de sinusales, ids pares): {thr:.3f}")
    for name in ("check_irregular", "check_irregular_no_p"):
        s = res[name]
        print(
            name,
            " · ".join(
                f"{k}: {100 * v['flagged']:.0f} % de {v['n']}" for k, v in s.items() if v["n"]
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
