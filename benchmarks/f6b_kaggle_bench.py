"""F6-b bench: real printouts/scans/photos (Kaggle "PhysioNet - Digitization of
ECG Images", train split) -> OUR pipeline -> metrics against the provided signal.

Per record and image variant (<id>-<NNNN>.png) and engine:
    ingest -> estimate_page_grids -> digitize_page(engine_duration_s=10)
    -> confirm_scale(gain=10, speed=25, time_source="evidence")   [arm evidence]
    -> confirm_scale(gain=10, speed=25, time_source="engine")     [arm engine]
Both arms reuse the same engine pass. 25 mm/s and 10 mm/mV are entered as
manual evidence (competition images are standard-calibrated printouts); the
record's own fs/sig_len are used only for scoring, never fed to the pipeline.

Metrics per lead on observed samples only (metrics_common.segment_metrics):
r@lag, rmse, amplitude ratio, duration error, competition-style SNR (not the
official metric). Failures are cases, never dropped: ingest_rejected,
engine_error, time_scale_unknown (no grid evidence -> evidence arm refuses),
no_signal. Each finished (record, variant, engine) is appended to
<work>/cases.jsonl, so an interrupted bench resumes where it stopped.

Scope: real printed/scanned/photographed images of real signals from the
competition's training set; NOT clinical validation, NOT our local devices.
"""

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_common import segment_metrics, trim_to_observed

from ecg_photo.contracts import load_manifest
from ecg_photo.digitizers.base import Digitizer
from ecg_photo.ingest import IngestRejected, estimate_page_grids, ingest
from ecg_photo.pipeline import confirm_scale, digitize_page

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
ARMS = ("evidence", "engine")
ENGINE_DURATION_S = 10.0  # standard 10 s page; the same assumption a user would configure
SCOPE = (
    "Kaggle PhysioNet ECG image digitization train split: real printouts, scans and "
    "photographs of real signals; not clinical validation; not local devices"
)


def read_truth(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        cols: list[list[float]] = [[] for _ in header]
        for row in reader:
            for i, v in enumerate(row):
                cols[i].append(float(v) if v.strip() not in ("", "nan", "NaN") else math.nan)
    return {h.strip(): np.asarray(c, dtype=float) for h, c in zip(header, cols, strict=True)}


def truth_window(sig: np.ndarray, fs: float) -> tuple[np.ndarray, float]:
    """Finite span of a truth column and its start (s). Full column if no NaN."""
    idx = np.where(np.isfinite(sig))[0]
    if len(idx) == 0:
        return sig[:0], 0.0
    a, b = int(idx[0]), int(idx[-1]) + 1
    return sig[a:b], a / fs


def percentile(a: list, q: float) -> float | None:
    v = [x for x in a if x is not None and isinstance(x, float) and math.isfinite(x)]
    return float(np.percentile(v, q)) if v else None


def git_rev(path: Path) -> str | None:
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=False
    )
    return r.stdout.strip() or None


def build_digitizers(args) -> dict[str, tuple[Digitizer, Path]]:
    out: dict[str, tuple[Digitizer, Path]] = {}
    if "ahus" in args.engines:
        from ecg_photo.digitizers.ahus import AhusDigitizer

        out["ahus"] = (
            AhusDigitizer(
                ahus_root=args.ahus_root.resolve(),
                python_exe=args.ahus_python.absolute(),
                base_config=args.base_config.resolve(),
                duration_s=ENGINE_DURATION_S,
                device="cpu",
            ),
            args.ahus_root.resolve(),
        )
    if "ecg-digitiser" in args.engines:
        from ecg_photo.digitizers.ecg_digitiser import EcgDigitiserDigitizer

        out["ecg-digitiser"] = (
            EcgDigitiserDigitizer(
                root=args.digitiser_root.resolve(),
                python_exe=args.digitiser_python.absolute(),
                model_dir=args.digitiser_model,
            ),
            args.digitiser_root.resolve(),
        )
    return out


def score_run(run_dir: Path, truth: dict[str, np.ndarray], fs: float, rhythm: str) -> list[dict]:
    rm = load_manifest(run_dir / "manifest.json")
    cases: list[dict] = []
    for seg in rm.segments:
        lead = seg.lead_label
        base = {"lead": lead, "segment_id": seg.segment_id}
        if seg.signal_path is None:
            cases.append({**base, "status": "no_signal"})
            continue
        if lead not in truth:
            cases.append({**base, "status": "lead_not_in_truth"})
            continue
        sig = np.load(run_dir / seg.signal_path, allow_pickle=False)
        observed = np.isfinite(sig)
        if seg.observed_mask_path and (run_dir / seg.observed_mask_path).exists():
            observed = np.load(run_dir / seg.observed_mask_path, allow_pickle=False)
        t_sig, t_start = truth_window(truth[lead], fs)
        full = len(t_sig) == len(truth[lead])
        expected = (ENGINE_DURATION_S if lead == rhythm else 2.5) if full else len(t_sig) / fs
        est_fs = float(seg.working_fs_hz or fs)
        if not full and len(sig) / est_fs > expected + 0.5:
            # canvas estimate (engine time axis) vs slot-cropped truth: engines
            # place short leads differently, so compare the observed span only
            sig, observed, placed_s = trim_to_observed(sig, observed, est_fs)
            base["trimmed_to_observed"] = True
            base["engine_placement_s"] = placed_s
        # measured extent of the observed samples, same definition in both arms
        # (a segment's own duration on the engine arm is the assumed page canvas)
        obs_idx = np.nonzero(observed)[0]
        extent_s = (obs_idx[-1] - obs_idx[0] + 1) / est_fs if len(obs_idx) else None
        m = segment_metrics(
            t_sig,
            sig,
            observed,
            est_fs=est_fs,
            truth_fs=fs,
            observed_duration_s=extent_s,
            expected_window_s=expected,
        )
        if not full and m.get("best_lag_s") is not None:
            m["offset_err_s"] = m["best_lag_s"]  # truth already cropped to its slot
        cases.append({**base, "truth_start_s": t_start, **m})
    return cases


def run_one(
    img: Path,
    work: Path,
    engine: str,
    digitizer: Digitizer,
    truth: dict[str, np.ndarray],
    fs: float,
    rhythm: str,
) -> dict:
    """One (image, engine): returns {'grid':..., 'wall_s':..., 'arms': {arm: [cases]}}."""
    rev_dir = work / "rev"
    out: dict = {"grid": None, "wall_s": None, "arms": {}}
    try:
        if not (rev_dir / "manifest.json").exists():
            manifest = ingest(img, rev_dir)
            manifest, grids = estimate_page_grids(rev_dir, manifest)
        else:
            manifest = load_manifest(rev_dir / "manifest.json")
            grids = []
    except IngestRejected as e:
        for arm in ARMS:
            out["arms"][arm] = [{"status": "ingest_rejected", "reason": str(e.reason_code)}]
        return out
    gpath = rev_dir / "pages" / f"{manifest.pages[0].page_id}.grid.json"
    if gpath.exists():
        g = json.loads(gpath.read_text())
        out["grid"] = {k: g.get(k) for k in ("px_per_mm_x", "px_per_mm_y", "refine_method")}
    elif grids:
        out["grid"] = {"px_per_mm_x": grids[0].px_per_mm_x, "px_per_mm_y": grids[0].px_per_mm_y}
    t0 = time.monotonic()
    try:
        run = digitize_page(
            rev_dir,
            manifest.pages[0].page_id,
            digitizer,
            engine_duration_s=ENGINE_DURATION_S,
            runs_root=work / "runs",
        )
    except Exception as e:  # noqa: BLE001 - recorded as a case
        out["wall_s"] = time.monotonic() - t0
        for arm in ARMS:
            out["arms"][arm] = [
                {"status": "engine_error", "error": f"{type(e).__name__}: {e}"[:400]}
            ]
        return out
    out["wall_s"] = time.monotonic() - t0
    for arm in ARMS:
        try:
            done = confirm_scale(
                run.run_dir,
                gain_mm_mV=10.0,
                speed_mm_s=25.0,
                author="f6b-bench",
                reason="competition printouts at 25 mm/s, 10 mm/mV",
                time_source=arm,  # type: ignore[arg-type]
            )
        except ValueError as e:
            msg = str(e)
            code = next(
                (c for c in ("TIME_SCALE_UNKNOWN", "CALIBRATION_MISSING") if c in msg), "ERROR"
            )
            status = "time_scale_unknown" if code != "ERROR" else "confirm_error"
            out["arms"][arm] = [{"status": status, "reason": code, "error": msg[:400]}]
            continue
        out["arms"][arm] = score_run(done.run_dir, truth, fs, rhythm)
    return out


def aggregate(rows: list[dict]) -> dict:
    """rows: per-lead cases (plus per-image failure rows without a lead)."""
    ok = [r for r in rows if r.get("status") == "ok"]
    statuses: dict[str, int] = {}
    for r in rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    return {
        "n_rows": len(rows),
        "n_ok": len(ok),
        "statuses": statuses,
        "median_r_at_lag": percentile([r["r_at_lag"] for r in ok], 50),
        "iqr_r_at_lag": [
            percentile([r["r_at_lag"] for r in ok], 25),
            percentile([r["r_at_lag"] for r in ok], 75),
        ],
        "median_snr_db": percentile([r.get("snr_db") for r in ok], 50),
        "median_rmse_mV": percentile([r["rmse_mV"] for r in ok], 50),
        "median_amp_ratio": percentile([r["amp_ratio"] for r in ok], 50),
        "median_duration_err_pct": percentile([r["duration_err_pct"] for r in ok], 50),
        "frac_r_ge_0.9": (
            sum(1 for r in ok if math.isfinite(r["r_at_lag"]) and r["r_at_lag"] >= 0.9) / len(ok)
            if ok
            else None
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="dir from f6b_fetch_kaggle.py")
    ap.add_argument("--work", type=Path, default=Path("runs/f6b/bench"))
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--types", nargs="*", default=None, help="image variants NNNN (default all)")
    ap.add_argument("--rhythm-lead", default="II")
    ap.add_argument(
        "--engines", nargs="+", default=["ahus", "ecg-digitiser"], choices=["ahus", "ecg-digitiser"]
    )
    ap.add_argument("--ahus-root", type=Path, default=Path("external/ahus"))
    ap.add_argument("--ahus-python", type=Path, default=Path("external/.venv-ahus/bin/python"))
    ap.add_argument(
        "--base-config",
        type=Path,
        default=Path("external/ahus/src/config/inference_wrapper_george-moody-2024.yml"),
    )
    ap.add_argument("--digitiser-root", type=Path, default=Path("external/ecg-digitiser"))
    ap.add_argument(
        "--digitiser-python", type=Path, default=Path("external/.venv-digitiser/bin/python")
    )
    ap.add_argument("--digitiser-model", type=Path, default=Path("models/M3"))
    ap.add_argument("--out", type=Path, default=None, help="results json (default work/...)")
    args = ap.parse_args()

    data = args.data.resolve()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    with open(data / "train.csv", newline="") as fh:
        meta = {r["id"]: r for r in csv.DictReader(fh)}
    ids = args.ids or sorted(p.name for p in (data / "train").iterdir() if p.is_dir())
    digitizers = build_digitizers(args)

    cases_path = work / "cases.jsonl"
    done: set[tuple[str, str, str]] = set()
    if cases_path.exists():
        for line in cases_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["record"], r["image_type"], r["engine"]))

    for rid in ids:
        fs = float(meta[rid]["fs"])
        truth = read_truth(data / "train" / rid / f"{rid}.csv")
        imgs = sorted((data / "train" / rid).glob(f"{rid}-*.png"))
        for img in imgs:
            itype = img.stem.rsplit("-", 1)[1]
            if args.types and itype not in args.types:
                continue
            for engine, (dig, _root) in digitizers.items():
                if (rid, itype, engine) in done:
                    continue
                res = run_one(
                    img, work / rid / itype / engine, engine, dig, truth, fs, args.rhythm_lead
                )
                row = {"record": rid, "image_type": itype, "engine": engine, "fs": fs, **res}
                with open(cases_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
                n_ok = {
                    a: sum(1 for c in cs if c.get("status") == "ok")
                    for a, cs in res["arms"].items()
                }
                print(f"{rid} {itype} {engine}: wall={res['wall_s']} ok={n_ok}", flush=True)

    # aggregate everything recorded so far (also earlier sessions)
    records = [json.loads(line) for line in cases_path.read_text().splitlines()]
    groups: dict[str, list[dict]] = {}
    for r in records:
        for arm, cs in r["arms"].items():
            for c in cs:
                groups.setdefault(f"{r['engine']}|{arm}|ALL", []).append(c)
                groups.setdefault(f"{r['engine']}|{arm}|{r['image_type']}", []).append(c)
    aggregates = {k: aggregate(v) for k, v in sorted(groups.items())}
    fetch_manifest = data / "fetch_manifest.json"
    results = {
        "timestamp": datetime.now(UTC).isoformat(),
        "scope": SCOPE,
        "calibration_note": "25 mm/s and 10 mm/mV entered as manual evidence for every image",
        "engine_duration_s_assumed": ENGINE_DURATION_S,
        "rhythm_lead": args.rhythm_lead,
        "engines": {
            e: {
                "engine_id": d.spec.engine_id,
                "commit": d.spec.commit,
                "commit_observed": git_rev(root),
                "weights_sha256": [w.sha256 for w in d.spec.weights],
            }
            for e, (d, root) in digitizers.items()
        },
        "our_commit": git_rev(Path(__file__).resolve().parent.parent),
        "python": sys.version.split()[0],
        "machine": {"platform": platform.platform(), "machine": platform.machine()},
        "data": json.loads(fetch_manifest.read_text()) if fetch_manifest.exists() else None,
        "wall_time_s": {
            f"{r['record']}-{r['image_type']}-{r['engine']}": r["wall_s"] for r in records
        },
        "grids": {f"{r['record']}-{r['image_type']}": r["grid"] for r in records},
        "aggregates": aggregates,
        "cases": records,
    }
    out = args.out or work / "f6b_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    def fmt(v, d=3):
        return f"{v:.{d}f}" if isinstance(v, float) else "-"

    print("\n| motor | eje | tipo | ok/filas | r@lag med | SNR dB med | err dur % | r>=0.9 |")
    print("|---|---|---|---|---|---|---|---|")
    for k, a in aggregates.items():
        e, arm, t = k.split("|")
        print(
            f"| {e} | {arm} | {t} | {a['n_ok']}/{a['n_rows']} | {fmt(a['median_r_at_lag'])} | "
            f"{fmt(a['median_snr_db'], 1)} | {fmt(a['median_duration_err_pct'], 2)} | "
            f"{fmt(a['frac_r_ge_0.9'], 2)} |"
        )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
