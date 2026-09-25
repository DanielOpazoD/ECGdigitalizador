"""F2 synthetic bench: fixture -> ecg-image-kit PNG -> engine digitize -> metrics.

All 12 WFDB leads carry the same fixture signal (smoke test only, not a
physiological 12-lead). Results are keyed by engine.
"""

import argparse
import json
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import wfdb  # type: ignore[import-untyped]
from scipy.signal import find_peaks

from ecg_photo.digitizers.ahus import AhusDigitizer
from ecg_photo.digitizers.base import Digitizer
from ecg_photo.digitizers.ecg_digitiser import EcgDigitiserDigitizer
from ecg_photo.fixtures import write_fixture_revision

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
TRUTH_FS = 500.0
TRUTH_DURATION_S = 3.0
RECORD_FS = 500.0
RECORD_DURATION_S = 10.0
MAX_LAG_S = 0.2


def build_12lead_record(truth: np.ndarray, records_dir: Path) -> Path:
    records_dir.mkdir(parents=True, exist_ok=True)
    n10 = int(RECORD_DURATION_S * RECORD_FS)
    tile = np.tile(truth, int(np.ceil(n10 / len(truth))))[:n10]
    sig = np.tile(tile.reshape(-1, 1), (1, 12))
    wfdb.wrsamp(
        "tiled12",
        fs=RECORD_FS,
        units=["mV"] * 12,
        sig_name=LEADS,
        p_signal=sig,
        fmt=["16"] * 12,
        adc_gain=[1000.0] * 12,
        baseline=[0] * 12,
        write_dir=str(records_dir),
    )
    return records_dir / "tiled12"


def run_imgkit(
    imgkit_root: Path, imgkit_python: Path, record_base: Path, out_dir: Path, seed: int
) -> float:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(imgkit_python),
        "gen_ecg_image_from_data.py",
        "-i",
        str(record_base.with_suffix(".dat")),
        "-hea",
        str(record_base.with_suffix(".hea")),
        "-o",
        str(out_dir),
        "-st",
        "0",
        "-se",
        str(seed),
        "--lead_bbox",
        "--store_config",
    ]
    t0 = time.monotonic()
    subprocess.run(cmd, cwd=imgkit_root, check=True)
    return time.monotonic() - t0


def git_rev(path: Path) -> str:
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True
    )
    return r.stdout.strip()


def torch_version(python_exe: Path) -> str:
    r = subprocess.run(
        [str(python_exe), "-c", "import torch;print(torch.__version__)"],
        capture_output=True,
        text=True,
        check=True,
    )
    return r.stdout.strip()


def median_rr_s(signal: np.ndarray, fs: float) -> float | None:
    filled = np.nan_to_num(signal)
    peaks, _ = find_peaks(filled, height=0.4, distance=int(0.4 * fs))
    if len(peaks) < 2:
        return None
    return float(np.median(np.diff(peaks)) / fs)


def metrics_for_lead(truth: np.ndarray, est: np.ndarray, est_fs: float) -> dict[str, float | None]:
    t_est = np.arange(len(est)) / est_fs
    t_truth = np.arange(len(truth)) / TRUTH_FS
    finite = np.isfinite(est)
    coverage = float(np.mean(finite))
    exact_zero_fraction = float(np.mean(np.nan_to_num(est) == 0))
    rr_truth = median_rr_s(truth, TRUTH_FS)
    rr_est = median_rr_s(est, est_fs)
    empty: dict[str, float | None] = {
        "coverage": coverage,
        "exact_zero_fraction": exact_zero_fraction,
        "snr_db": float("nan"),
        "rmse_mV": float("nan"),
        "pearson_r": float("nan"),
        "amplitude_ratio": float("nan"),
        "best_lag_ms": None,
        "r_at_best_lag": float("nan"),
        "rr_est_s": rr_est,
        "rr_truth_s": rr_truth,
        "r_peak_amp_ratio": float("nan"),
    }
    if finite.sum() < 2:
        return empty

    def pearson_at(lag_samples: int) -> float:
        shifted = t_est[finite] + lag_samples / est_fs
        in_range = (shifted >= 0.0) & (shifted <= t_truth[-1])
        if in_range.sum() < 2:
            return float("nan")
        e = est[finite][in_range]
        tt = np.interp(shifted[in_range], t_truth, truth)
        if np.std(e) == 0 or np.std(tt) == 0:
            return float("nan")
        return float(np.corrcoef(e, tt)[0, 1])

    max_lag = round(MAX_LAG_S * est_fs)
    best_lag, best_r = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        r = pearson_at(lag)
        if np.isfinite(r) and r > best_r:
            best_r, best_lag = r, lag

    truth_on_est = np.interp(t_est[finite], t_truth, truth)
    e = est[finite]
    resid = truth_on_est - e
    sig_pow = float(np.sum(truth_on_est**2))
    err_pow = float(np.sum(resid**2))
    snr = 10.0 * np.log10(sig_pow / err_pow) if err_pow > 0 else float("inf")
    amp_e = float(np.nanmax(e) - np.nanmin(e))
    amp_t = float(np.nanmax(truth) - np.nanmin(truth))

    def median_peak_height(signal: np.ndarray, fs: float) -> float | None:
        filled = np.nan_to_num(signal)
        peaks, props = find_peaks(filled, height=0.4, distance=int(0.4 * fs))
        if len(peaks) < 1:
            return None
        return float(np.median(props["peak_heights"]))

    h_est = median_peak_height(est, est_fs)
    h_truth = median_peak_height(truth, TRUTH_FS)
    r_peak_amp_ratio = (
        h_est / h_truth
        if (h_est is not None and h_truth is not None and h_truth != 0)
        else float("nan")
    )
    return {
        "coverage": coverage,
        "exact_zero_fraction": exact_zero_fraction,
        "snr_db": float(snr),
        "rmse_mV": float(np.sqrt(np.mean(resid**2))),
        "pearson_r": pearson_at(0),
        "amplitude_ratio": amp_e / amp_t if amp_t > 0 else float("nan"),
        "best_lag_ms": float(best_lag * 1000.0 / est_fs),
        "r_at_best_lag": float(best_r) if np.isfinite(best_r) else float("nan"),
        "rr_est_s": rr_est,
        "rr_truth_s": rr_truth,
        "r_peak_amp_ratio": r_peak_amp_ratio,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ahus-root", type=Path)
    ap.add_argument("--ahus-python", type=Path)
    ap.add_argument("--imgkit-root", type=Path, required=True)
    ap.add_argument("--imgkit-python", type=Path, required=True)
    ap.add_argument("--base-config", type=Path)
    ap.add_argument("--digitiser-root", type=Path)
    ap.add_argument("--digitiser-python", type=Path)
    ap.add_argument("--digitiser-model", type=Path, default=Path("models/M3"))
    ap.add_argument(
        "--engines", nargs="+", default=["ahus", "ecg-digitiser"], choices=["ahus", "ecg-digitiser"]
    )
    ap.add_argument("--work", type=Path, default=Path("runs/f2"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9])
    args = ap.parse_args()

    work: Path = args.work.resolve()
    imgkit_root = args.imgkit_root.resolve()
    imgkit_python = args.imgkit_python.absolute()
    work.mkdir(parents=True, exist_ok=True)

    rev_dir = work / "fixture_revision"
    manifest = write_fixture_revision(rev_dir, "calibrated")
    seg = manifest.segments[0]
    assert seg.signal_path is not None
    truth = np.load(rev_dir / seg.signal_path)
    record_base = build_12lead_record(truth, work / "records")

    digitizers: dict[str, tuple[Digitizer, Path]] = {}
    if "ahus" in args.engines:
        assert args.ahus_root and args.ahus_python and args.base_config
        digitizers["ahus"] = (
            AhusDigitizer(
                ahus_root=args.ahus_root.resolve(),
                python_exe=args.ahus_python.absolute(),
                base_config=args.base_config.resolve(),
                duration_s=RECORD_DURATION_S,
                device="cpu",
            ),
            args.ahus_root.resolve(),
        )
    if "ecg-digitiser" in args.engines:
        assert args.digitiser_root and args.digitiser_python
        digitizers["ecg-digitiser"] = (
            EcgDigitiserDigitizer(
                root=args.digitiser_root.resolve(),
                python_exe=args.digitiser_python.absolute(),
                model_dir=args.digitiser_model,
            ),
            args.digitiser_root.resolve(),
        )

    cases: dict[str, list[dict[str, object]]] = {e: [] for e in digitizers}
    wall_times: dict[str, float] = {}
    gt_summary: dict[str, object] = {}
    for seed in args.seeds:
        img_dir = work / f"imgkit_seed{seed}"
        t_img = run_imgkit(imgkit_root, imgkit_python, record_base, img_dir, seed)
        pngs = sorted(img_dir.glob("*.png"))
        jsons = sorted(img_dir.glob("*.json"))
        if not pngs:
            raise RuntimeError(f"imgkit produced no PNG for seed {seed}")
        if jsons:
            gt = json.loads(jsons[0].read_text())
            gt_summary[str(seed)] = {
                "width": gt.get("width"),
                "height": gt.get("height"),
                "resolution": gt.get("resolution"),
                "x_grid": gt.get("x_grid"),
                "y_grid": gt.get("y_grid"),
            }
        wall_times[f"{seed}_imgkit"] = t_img
        for engine, (digitizer, _root) in digitizers.items():
            out = digitizer.run(pngs[0], work / f"{engine}_seed{seed}")
            wall_times[f"{seed}_{engine}"] = out.wall_time_s
            for lead in LEADS:
                m = metrics_for_lead(truth, out.leads[lead], out.fs_hz)
                cases[engine].append(
                    {
                        "seed": seed,
                        "lead": lead,
                        "layout": out.layout_detected,
                        "config_hash": out.config_hash,
                        "fs_hz": out.fs_hz,
                        **m,
                    }
                )
            print(f"seed {seed} {engine}: layout={out.layout_detected} wall={out.wall_time_s:.1f}s")

    results = {
        "timestamp": datetime.now(UTC).isoformat(),
        "engines": {
            engine: {
                "engine_id": d[0].spec.engine_id,
                "repo": d[0].spec.repo,
                "commit": d[0].spec.commit,
                "license": d[0].spec.license,
                "weights_sha256": [w.sha256 for w in d[0].spec.weights],
                "commit_observed": git_rev(d[1]),
            }
            for engine, d in digitizers.items()
        },
        "imgkit_commit": git_rev(imgkit_root),
        "imgkit_ground_truth": gt_summary,
        "truth": {
            "fs_hz": TRUTH_FS,
            "duration_s": TRUTH_DURATION_S,
            "record_fs_hz": RECORD_FS,
            "record_duration_s": RECORD_DURATION_S,
            "note": "all 12 WFDB leads identical fixture signal; smoke test only",
        },
        "wall_time_s": wall_times,
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "metrics": cases,
    }
    out_path = work / "f2_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
