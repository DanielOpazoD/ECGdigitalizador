"""F9 (objective O7): how accurate are PR, QRS and QT from ecg_photo.intervals?

Ground truth: LUDB (PhysioNet, 200 records, 12 leads, 10 s, 500 Hz) with P,
QRS and T onsets/offsets annotated by cardiologists in every lead. The truth
interval of a record uses the same aggregation as the measurement: per lead,
the median over annotated beats; per record, the median over leads.

Arms:
- full: the 12 true signals, 10 s each (the algorithm alone);
- printed: the true signals cut to what a 3x4 + II printout shows (2.5 s per
  lead, II 10 s), i.e. the most a perfect digitization could give;
- digitized (--digitize N): the printout rendered as an image (F8 renderer),
  Ahus, confirm_scale on the image-evidence time axis, then measured.

Parameters were tuned on the odd records (--split odd); report on the even
ones (--split even). Reference tolerances: the CSE criteria for measurement
programs (mean difference / SD against the reference: PR and QRS 10 / 10 ms,
QT 25 / 30 ms). Not clinical validation.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f6b_kaggle_bench import git_rev

from ecg_photo.intervals import intervals_json, measure_signals
from ecg_photo.qc import STANDARD_LEADS

LUDB_LEADS = dict(
    zip(
        ("i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"),
        STANDARD_LEADS,
        strict=True,
    )
)
KEYS = ("pr_ms", "qrs_ms", "qt_ms")
CSE = {"pr_ms": (10, 10), "qrs_ms": (10, 10), "qt_ms": (25, 30)}
SLOTS = {  # 3x4 + II printout: start of each short lead's 2.5 s slot
    "I": 0.0, "II": 0.0, "III": 0.0,
    "aVR": 2.5, "aVL": 2.5, "aVF": 2.5,
    "V1": 5.0, "V2": 5.0, "V3": 5.0,
    "V4": 7.5, "V5": 7.5, "V6": 7.5,
}  # fmt: skip


def waves(rec: Path, ext: str) -> list[tuple[str, int, int]]:
    """(symbol, onset, offset) of each annotated wave of one lead."""
    a = wfdb.rdann(str(rec), ext)
    out = []
    sym, smp = a.symbol, a.sample
    for i in range(1, len(sym) - 1):
        if sym[i] not in ("(", ")") and sym[i - 1] == "(" and sym[i + 1] == ")":
            out.append((sym[i], int(smp[i - 1]), int(smp[i + 1])))
    return out


def truth_intervals(rec: Path, fs: float) -> dict[str, float | None]:
    per_lead: dict[str, list[float]] = {k: [] for k in KEYS}
    for ext in LUDB_LEADS:
        w = waves(rec, ext)
        qrs = [x for x in w if x[0] == "N"]
        vals: dict[str, list[float]] = {k: [] for k in KEYS}
        for j, (_s, on, off) in enumerate(qrs):
            vals["qrs_ms"].append((off - on) * 1000 / fs)
            nxt = qrs[j + 1][1] if j + 1 < len(qrs) else 10**9
            prv = qrs[j - 1][2] if j > 0 else -(10**9)
            ps = [x for x in w if x[0] == "p" and prv <= x[1] < on and x[2] <= on]
            if ps:
                vals["pr_ms"].append((on - ps[-1][1]) * 1000 / fs)
            ts = [x for x in w if x[0] == "t" and off <= x[1] < nxt]
            if ts:
                vals["qt_ms"].append((ts[0][2] - on) * 1000 / fs)
        for k in KEYS:
            if vals[k]:
                per_lead[k].append(float(np.median(vals[k])))
    return {k: float(np.median(v)) if v else None for k, v in per_lead.items()}


def read_signals(rec: Path) -> tuple[dict[str, np.ndarray], float]:
    r = wfdb.rdrecord(str(rec))
    sig = {
        LUDB_LEADS[n.lower()]: r.p_signal[:, i].astype(np.float64) for i, n in enumerate(r.sig_name)
    }
    return sig, float(r.fs)


def printed(sig: dict[str, np.ndarray], fs: float) -> dict[str, np.ndarray]:
    out = {}
    for name, x in sig.items():
        if name == "II":
            out[name] = x.copy()
            continue
        a, b = round(SLOTS[name] * fs), round((SLOTS[name] + 2.5) * fs)
        out[name] = x[a:b].copy()
    return out


def summarize(rows: list[dict], arm: str, only_ok: bool = False) -> dict:
    out = {}
    for k in KEYS:
        d = np.array(
            [
                r[arm][k] - r["truth"][k]
                for r in rows
                if r.get(arm)
                and r[arm].get(k) is not None
                and r["truth"][k] is not None
                and (not only_ok or r[arm]["status"][k] == "ok")
            ]
        )
        n_truth = sum(r["truth"][k] is not None for r in rows if r.get(arm))
        if len(d) == 0:
            out[k] = {"n": 0, "n_truth": n_truth}
            continue
        mt, st = CSE[k]
        out[k] = {
            "n": len(d),
            "n_truth": int(n_truth),
            "mean_diff_ms": float(d.mean()),
            "sd_diff_ms": float(d.std(ddof=1)) if len(d) > 1 else None,
            "median_abs_ms": float(np.median(np.abs(d))),
            "p90_abs_ms": float(np.percentile(np.abs(d), 90)),
            "cse_mean_sd_ms": [mt, st],
            "within_cse": bool(abs(d.mean()) <= mt and (len(d) < 2 or d.std(ddof=1) <= st)),
        }
    return out


def digitize(
    sig: dict[str, np.ndarray], fs: float, work: Path, dig
) -> dict[str, tuple[np.ndarray, float]] | None:
    from f8_layout_eval import render_page

    from ecg_photo.ingest import estimate_page_grids, ingest
    from ecg_photo.intervals import run_signals
    from ecg_photo.pipeline import confirm_scale, digitize_page

    img = work / "page.png"
    render_page(sig, fs, "3x4+1R", img)
    m = ingest(img, work / "rev")
    estimate_page_grids(work / "rev", m)
    run = digitize_page(
        work / "rev", "page-1", dig, engine_duration_s=10.0, runs_root=work / "runs"
    )
    try:
        done = confirm_scale(
            run.run_dir,
            gain_mm_mV=10.0,
            speed_mm_s=25.0,
            author="f9",
            reason="rendered printout",
            time_source="evidence",
        )
    except ValueError:
        return None
    return run_signals(done.run_dir)


def saved_confirmed_run(work: Path) -> Path | None:
    """Newest confirmed run (calibrated signal) saved by a previous --digitize."""
    from ecg_photo.contracts import load_manifest

    runs = [
        r
        for r in sorted((work / "runs").glob("run-*"), key=lambda p: p.stat().st_mtime)
        if (r / "manifest.json").exists()
        and any(s.signal_path for s in load_manifest(r / "manifest.json").segments)
    ]
    return runs[-1] if runs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ludb", type=Path, default=Path("runs/f9/ludb"))
    ap.add_argument("--split", choices=["odd", "even", "all"], default="even")
    ap.add_argument("--digitize", type=int, default=0, help="also run Ahus on the first N records")
    ap.add_argument("--work", type=Path, default=Path("runs/f9/digitized"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--remeasure",
        action="store_true",
        help="re-measure the digitized arm from the saved confirmed runs (no engine run)",
    )
    args = ap.parse_args()

    ids = [int(x.split("/")[-1]) for x in (args.ludb / "RECORDS").read_text().split()]
    if args.split != "all":
        ids = [i for i in ids if (i % 2 == 1) == (args.split == "odd")]
    dig = None
    if args.digitize:
        from ecg_photo.digitizers.ahus import AhusDigitizer

        root = Path("external/ahus").resolve()
        dig = AhusDigitizer(
            ahus_root=root,
            python_exe=Path("external/.venv-ahus/bin/python").absolute(),
            base_config=root / "src/config/inference_wrapper_george-moody-2024.yml",
            duration_s=10.0,
            device="cpu",
        )
    rows = []
    cache = args.work / "cases.jsonl"
    done_dig = {}
    if cache.exists() and not args.remeasure:
        for line in cache.read_text().splitlines():
            r = json.loads(line)
            done_dig[r["record"]] = r["digitized"]
    for n, rid in enumerate(ids):
        rec = args.ludb / "data" / str(rid)
        sig, fs = read_signals(rec)
        row: dict = {"record": rid, "truth": truth_intervals(rec, fs)}
        full = measure_signals({k: (v, fs) for k, v in sig.items()})
        pr = printed(sig, fs)
        row["full"] = intervals_json(full)
        row["printed"] = intervals_json(measure_signals({k: (v, fs) for k, v in pr.items()}))
        if dig is not None and n < args.digitize:
            saved = saved_confirmed_run(args.work / str(rid))
            if args.remeasure and saved is not None:
                from ecg_photo.intervals import run_signals

                done_dig[rid] = intervals_json(measure_signals(run_signals(saved)))
            if rid not in done_dig:
                s = digitize(sig, fs, args.work / str(rid), dig)
                res = intervals_json(measure_signals(s)) if s else None
                args.work.mkdir(parents=True, exist_ok=True)
                with open(cache, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"record": rid, "digitized": res}) + "\n")
                done_dig[rid] = res
            row["digitized"] = done_dig[rid]
        rows.append(row)
    arms = ["full", "printed"] + (["digitized"] if dig is not None else [])
    res = {
        "generated": datetime.now(UTC).isoformat(),
        "scope": "LUDB records (cardiologist annotations); not clinical validation",
        "commit": git_rev(Path(__file__).resolve().parent.parent),
        "split": args.split,
        "n_records": len(rows),
        "summary": {arm: summarize(rows, arm) for arm in arms}
        | {f"{arm}|ok": summarize(rows, arm, only_ok=True) for arm in arms},
        "records": [
            {
                "record": r["record"],
                "truth": r["truth"],
                **{
                    a: {k: r[a][k] for k in (*KEYS, "status", "spread_ms")} if r.get(a) else None
                    for a in arms
                },
            }
            for r in rows
        ],
    }
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("| brazo | intervalo | n/n verdad | dif. media | DE | |dif.| mediana | p90 | CSE |")
    print("|---|---|---|---|---|---|---|---|")
    for arm, s in res["summary"].items():
        for k, v in s.items():
            if not v.get("n"):
                print(f"| {arm} | {k} | 0/{v['n_truth']} | | | | | |")
                continue
            print(
                f"| {arm} | {k} | {v['n']}/{v['n_truth']} | {v['mean_diff_ms']:+.1f} | "
                f"{v['sd_diff_ms']:.1f} | {v['median_abs_ms']:.1f} | {v['p90_abs_ms']:.1f} | "
                f"{'sí' if v['within_cse'] else 'no'} |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
