"""F15: QRS and ST amplitudes of ecg_photo.intervals against GE 12SL
(PTB-XL+), on the PTB-XL records downloaded for F11/F14, cut as a 3x4 + II
printout. Per lead: R amplitude, S amplitude, ST at the J point; and the
Sokolow-Lyon index S(V1) + max(R(V5), R(V6)) computed the same way from both.
Nothing is tuned here; odd ecg_ids are reported apart as a held-out check.
ST is only added to the outputs if its error is small (criterion fixed
beforehand: |bias| <= 0.02 mV and SD <= 0.05 mV). Not clinical validation.
"""

import argparse
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

import ecg_photo.intervals as ivm
from ecg_photo.intervals import measure_signals
from ecg_photo.qc import detect_qrs, observed_span

GE_LEAD = {v: k for k, v in {"I": "I", "II": "II", "III": "III", "AVR": "aVR", "AVL": "aVL",
           "AVF": "aVF", "V1": "V1", "V2": "V2", "V3": "V3", "V4": "V4", "V5": "V5", "V6": "V6"}.items()}  # fmt: skip
FEATURES = {"r_amp_mV": "R_Amp", "s_amp_mV": "S_Amp", "st_j_mV": "ST_Amp"}
ST_CRITERION = {"abs_bias_max": 0.02, "sd_max": 0.05}


def st_j(x: np.ndarray, fs: float) -> float | None:
    """ST at the J point of the median beat (evaluated here only; not in the
    product unless it meets ST_CRITERION)."""
    span, _ = observed_span(x, fs)
    beats = detect_qrs(span, fs) if len(span) else np.array([], int)
    if len(beats) < 2:
        return None
    rr_s = float(np.median(np.diff(beats))) / fs
    b = ivm.median_beat(span, fs, beats, rr_s)
    if b is None:
        return None
    f = ivm.beat_fiducials(b, fs, rr_s)
    if f["qrs_on"] is None or f["qrs_off"] is None:
        return None
    q_on, q_off = int(f["qrs_on"]), int(f["qrs_off"])
    iso = float(np.median(b[max(0, q_on - round(0.02 * fs)) : q_on + 1]))
    return float(b[min(q_off, len(b) - 1)] - iso)


def sokolow(r_v5: float | None, r_v6: float | None, s_v1: float | None) -> float | None:
    if s_v1 is None or (r_v5 is None and r_v6 is None):
        return None
    return abs(s_v1) + max(v for v in (r_v5, r_v6) if v is not None)


def stats(d: list[float]) -> dict:
    if not d:
        return {"n": 0}
    a = np.array(d)
    return {
        "n": len(a),
        "bias": float(a.mean()),
        "sd": float(a.std(ddof=1)) if len(a) > 1 else None,
        "median_abs": float(np.median(np.abs(a))),
        "p90_abs": float(np.percentile(np.abs(a), 90)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("runs/f11"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ids = {
        int(x)
        for f in ("sample_ids.txt", "rhythm_ids.txt")
        for x in (args.data / f).read_text().split()
    }
    ge: dict[int, dict] = {}
    with open(args.data / "12sl_features.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            eid = int(float(row["ecg_id"]))
            if eid in ids:
                ge[eid] = row
    diffs: dict[str, dict[str, list[float]]] = {"all": {}, "odd": {}}
    rows = []
    for eid in sorted(ge):
        rec = args.data / "ptbxl" / f"{eid:05d}_hr"
        if not rec.with_suffix(".dat").exists():
            continue
        r = wfdb.rdrecord(str(rec))
        fs = float(r.fs)
        sig = {LEADS[n.upper()]: r.p_signal[:, i].astype(float) for i, n in enumerate(r.sig_name)}
        pr = printed(sig, fs)
        iv = measure_signals({k: (v, fs) for k, v in pr.items()})
        ours = {li.lead: li for li in iv.leads}
        row: dict = {"ecg_id": eid}
        for li in iv.leads:
            for feat, col in FEATURES.items():
                g = ge[eid].get(f"{col}_{GE_LEAD[li.lead]}")
                v = st_j(pr[li.lead], fs) if feat == "st_j_mV" else getattr(li, feat)
                if v is None or g in (None, "", "nan"):
                    continue
                key = f"{feat}:{li.lead}"
                d = v - float(g)
                for part in ("all",) + (("odd",) if eid % 2 else ()):
                    diffs[part].setdefault(key, []).append(d)
                    diffs[part].setdefault(f"{feat}:all_leads", []).append(d)

        def geval(col: str, lead: str, g: dict = ge[eid]) -> float | None:
            v = g.get(f"{col}_{lead}")
            return None if v in (None, "", "nan") else float(v)

        so = sokolow(
            getattr(ours.get("V5"), "r_amp_mV", None),
            getattr(ours.get("V6"), "r_amp_mV", None),
            getattr(ours.get("V1"), "s_amp_mV", None),
        )
        sg = sokolow(geval("R_Amp", "V5"), geval("R_Amp", "V6"), geval("S_Amp", "V1"))
        if so is not None and sg is not None:
            for part in ("all",) + (("odd",) if eid % 2 else ()):
                diffs[part].setdefault("sokolow_lyon", []).append(so - sg)
            row["sokolow"] = {"ours": round(so, 3), "12sl": round(sg, 3)}
        rows.append(row)
    summary = {part: {k: stats(v) for k, v in sorted(d.items())} for part, d in diffs.items()}
    st = summary["odd"].get("st_j_mV:all_leads", {"n": 0})
    st_ok = bool(
        st["n"]
        and abs(st["bias"]) <= ST_CRITERION["abs_bias_max"]
        and st["sd"] <= ST_CRITERION["sd_max"]
    )
    res = {
        "generated": datetime.now(UTC).isoformat(),
        "scope": "PTB-XL printed 3x4 + II vs GE 12SL amplitudes (mV); not clinical validation",
        "commit": git_rev(Path(__file__).resolve().parent.parent),
        "n_records": len(rows),
        "st_criterion": ST_CRITERION,
        "st_meets_criterion_on_odd_ids": st_ok,
        "summary": summary,
        "records": rows,
    }
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"registros {len(rows)}; ST cumple el criterio en impares: {st_ok}")
    for key in ("r_amp_mV:all_leads", "s_amp_mV:all_leads", "st_j_mV:all_leads",
                "r_amp_mV:V5", "s_amp_mV:V1", "st_j_mV:V2", "st_j_mV:II", "sokolow_lyon"):  # fmt: skip
        s = summary["odd"].get(key, {"n": 0})
        if s["n"]:
            print(
                f"{key:22s} n={s['n']:5d} sesgo {s['bias']:+.3f} DE {s['sd']:.3f} "
                f"|dif.| mediana {s['median_abs']:.3f} p90 {s['p90_abs']:.3f} mV"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
