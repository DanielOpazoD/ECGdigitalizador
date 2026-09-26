"""F11: do our intervals agree with what a GE electrocardiograph prints?

The user's GE MAC2000 prints heart rate, PR, QRS, QT and QTc computed by the
GE Marquette 12SL program. PTB-XL+ (PhysioNet) publishes the 12SL global
measurements for every PTB-XL record; this compares them with
ecg_photo.intervals on the same PTB-XL signals, cut as a 3x4 + II printout
shows them (the most a perfect digitization gives) and on the full 10 s.

The first half of the records (by position in the sorted sample) sets the
agreement tolerance per interval (95th percentile of |ours - 12SL| over
values marked ok); the second half checks what fraction falls inside it.
Those tolerances are what `ecg-photo process --printed-...` uses to flag a
disagreement with the printed header. Not clinical validation.
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

from ecg_photo.intervals import intervals_json, measure_signals
from ecg_photo.qc import rhythm_rr

GE = {
    "pr_ms": "PR_Int_Global",
    "qrs_ms": "QRS_Dur_Global",
    "qt_ms": "QT_Int_Global",
    "qtc_bazett_ms": "QT_IntBazett_Global",
    "rr_ms": "RR_Mean_Global",
}
LEADS = {
    "I": "I", "II": "II", "III": "III", "AVR": "aVR", "AVL": "aVL", "AVF": "aVF",
    "V1": "V1", "V2": "V2", "V3": "V3", "V4": "V4", "V5": "V5", "V6": "V6",
}  # fmt: skip


def read_ge(path: Path) -> dict[int, dict[str, float | None]]:
    out: dict[int, dict[str, float | None]] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out[int(float(row["ecg_id"]))] = {
                k: (float(row[c]) if row.get(c) not in (None, "", "nan") else None)
                for k, c in GE.items()
            }
    return out


def measure(sig: dict[str, np.ndarray], fs: float) -> dict:
    _n, rr, _med = rhythm_rr(sig["II"], fs)
    d = intervals_json(measure_signals({k: (v, fs) for k, v in sig.items()}, rr_ms=rr))
    return {k: d[k] for k in ("rr_ms", "pr_ms", "qrs_ms", "qt_ms", "qtc_bazett_ms", "status")}


def diffs(rows: list[dict], arm: str, key: str, only_ok: bool) -> np.ndarray:
    st_key = "qt_ms" if key == "qtc_bazett_ms" else key
    out = []
    for r in rows:
        m, g = r[arm], r["ge"]
        if m[key] is None or g[key] is None:
            continue
        if only_ok and key != "rr_ms" and m["status"].get(st_key) != "ok":
            continue
        out.append(m[key] - g[key])
    return np.array(out)


def stats(d: np.ndarray, tol: float | None = None) -> dict:
    if len(d) == 0:
        return {"n": 0}
    s = {
        "n": len(d),
        "mean_diff_ms": float(d.mean()),
        "sd_diff_ms": float(d.std(ddof=1)) if len(d) > 1 else None,
        "median_abs_ms": float(np.median(np.abs(d))),
        "p95_abs_ms": float(np.percentile(np.abs(d), 95)),
    }
    if tol is not None:
        s["within_tol"] = float(np.mean(np.abs(d) <= tol))
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("runs/f11"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ge = read_ge(args.data / "12sl_sample.csv")
    rows = []
    for eid in sorted(ge):
        rec = args.data / "ptbxl" / f"{eid:05d}_hr"
        if not rec.with_suffix(".dat").exists():
            continue
        r = wfdb.rdrecord(str(rec))
        fs = float(r.fs)
        sig = {LEADS[n.upper()]: r.p_signal[:, i].astype(float) for i, n in enumerate(r.sig_name)}
        rows.append(
            {
                "ecg_id": eid,
                "ge": ge[eid],
                "printed": measure(printed(sig, fs), fs),
                "full": measure(sig, fs),
            }
        )
    tune, check = rows[0::2], rows[1::2]
    keys = ("rr_ms", "pr_ms", "qrs_ms", "qt_ms", "qtc_bazett_ms")
    tol = {k: float(np.percentile(np.abs(diffs(tune, "printed", k, True)), 95)) for k in keys}
    summary = {
        arm: {
            k: {
                "all": stats(diffs(check, arm, k, False)),
                "ok": stats(diffs(check, arm, k, True), tol[k]),
            }
            for k in keys
        }
        for arm in ("printed", "full")
    }
    res = {
        "generated": datetime.now(UTC).isoformat(),
        "scope": "PTB-XL signals vs GE 12SL global measurements (PTB-XL+); not clinical validation",
        "commit": git_rev(Path(__file__).resolve().parent.parent),
        "n_records": len(rows),
        "tolerance_ms_from_first_half_p95": tol,
        "second_half": summary,
        "records": rows,
    }
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"registros {len(rows)} (ajuste {len(tune)}, comprobación {len(check)})")
    print("tolerancias (p95 |dif.| `ok`, primera mitad):", {k: round(v) for k, v in tol.items()})
    print(
        "| brazo | medida | todos: n · media ± DE | `ok`: n · media ± DE · |dif.| med | `ok` dentro de tol. |"
    )
    print("|---|---|---|---|---|")
    for arm, s in summary.items():
        for k, v in s.items():
            a, o = v["all"], v["ok"]
            if not a["n"]:
                continue
            print(
                f"| {arm} | {k} | {a['n']} · {a['mean_diff_ms']:+.1f} ± {a['sd_diff_ms']:.1f} | "
                f"{o['n']} · {o['mean_diff_ms']:+.1f} ± {o['sd_diff_ms']:.1f} · "
                f"{o['median_abs_ms']:.0f} | {100 * o['within_tol']:.0f} % |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
