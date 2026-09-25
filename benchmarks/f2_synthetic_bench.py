"""F2 synthetic bench: fixture -> ecg-image-kit PNG -> Ahus digitize -> metrics.

All 12 WFDB leads carry the same fixture signal (smoke test only, not a
physiological 12-lead).
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

from ecg_photo.digitizers.ahus import AhusDigitizer
from ecg_photo.fixtures import write_fixture_revision

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
TRUTH_FS = 500.0
TRUTH_DURATION_S = 3.0
RECORD_FS = 500.0
RECORD_DURATION_S = 10.0


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


def metrics_for_lead(truth: np.ndarray, est: np.ndarray, est_fs: float) -> dict[str, float]:
    t_est = np.arange(len(est)) / est_fs
    t_truth = np.arange(len(truth)) / TRUTH_FS
    finite = np.isfinite(est)
    coverage = float(np.mean(finite))
    if finite.sum() < 2:
        return {
            "coverage": coverage,
            "snr_db": float("nan"),
            "rmse_mV": float("nan"),
            "pearson_r": float("nan"),
            "amplitude_ratio": float("nan"),
        }
    truth_on_est = np.interp(t_est[finite], t_truth, truth)
    e = est[finite]
    resid = truth_on_est - e
    sig_pow = float(np.sum(truth_on_est**2))
    err_pow = float(np.sum(resid**2))
    snr = 10.0 * np.log10(sig_pow / err_pow) if err_pow > 0 else float("inf")
    rmse = float(np.sqrt(np.mean(resid**2)))
    if np.std(e) > 0 and np.std(truth_on_est) > 0:
        r = float(np.corrcoef(e, truth_on_est)[0, 1])
    else:
        r = float("nan")
    amp_e = float(np.nanmax(e) - np.nanmin(e))
    amp_t = float(np.nanmax(truth) - np.nanmin(truth))
    return {
        "coverage": coverage,
        "snr_db": float(snr),
        "rmse_mV": rmse,
        "pearson_r": r,
        "amplitude_ratio": amp_e / amp_t if amp_t > 0 else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ahus-root", type=Path, required=True)
    ap.add_argument("--ahus-python", type=Path, required=True)
    ap.add_argument("--imgkit-root", type=Path, required=True)
    ap.add_argument("--imgkit-python", type=Path, required=True)
    ap.add_argument("--base-config", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=Path("runs/f2"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9])
    args = ap.parse_args()

    work: Path = args.work.resolve()
    ahus_root = args.ahus_root.resolve()
    ahus_python = args.ahus_python.absolute()
    imgkit_root = args.imgkit_root.resolve()
    imgkit_python = args.imgkit_python.absolute()
    base_config = args.base_config.resolve()
    work.mkdir(parents=True, exist_ok=True)

    rev_dir = work / "fixture_revision"
    manifest = write_fixture_revision(rev_dir, "calibrated")
    seg = manifest.segments[0]
    assert seg.signal_path is not None
    truth = np.load(rev_dir / seg.signal_path)
    record_base = build_12lead_record(truth, work / "records")

    digitizer = AhusDigitizer(
        ahus_root=ahus_root,
        python_exe=ahus_python,
        base_config=base_config,
        duration_s=RECORD_DURATION_S,
        device="cpu",
    )

    cases: list[dict[str, object]] = []
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
        out = digitizer.run(pngs[0], work / f"ahus_seed{seed}")
        wall_times[str(seed)] = out.wall_time_s
        wall_times[f"{seed}_imgkit"] = t_img
        for lead in LEADS:
            m = metrics_for_lead(truth, out.leads[lead], out.fs_hz)
            cases.append(
                {
                    "seed": seed,
                    "lead": lead,
                    "layout": out.layout_detected,
                    "config_hash": out.config_hash,
                    "fs_hz": out.fs_hz,
                    **m,
                }
            )
        print(f"seed {seed}: layout={out.layout_detected} ahus={out.wall_time_s:.1f}s")

    results = {
        "timestamp": datetime.now(UTC).isoformat(),
        "engine": {
            "engine_id": digitizer.spec.engine_id,
            "repo": digitizer.spec.repo,
            "commit": digitizer.spec.commit,
            "license": digitizer.spec.license,
            "weights_sha256": [w.sha256 for w in digitizer.spec.weights],
        },
        "imgkit_commit": git_rev(imgkit_root),
        "ahus_commit": git_rev(ahus_root),
        "config_hash": cases[0]["config_hash"] if cases else None,
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
            "torch_version": torch_version(ahus_python),
        },
        "metrics": cases,
    }
    out_path = work / "f2_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
