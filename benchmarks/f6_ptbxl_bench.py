"""F6-a bench: PTB-XL records -> ECG-Image-Kit PNG -> OUR pipeline -> metrics.

Real 12-lead physiological signals (PTB-XL, CC BY 4.0) rendered as synthetic
ECG-Image-Kit printouts, then driven through ingest -> estimate_page_grids ->
digitize_page -> confirm_scale(time_source="evidence") for each engine.

Scope: real signals, synthetic images — NOT real photographs, NOT clinical
validation.
"""

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import wfdb  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_common import segment_metrics

from ecg_photo.digitizers.ahus import AhusDigitizer
from ecg_photo.digitizers.base import Digitizer
from ecg_photo.digitizers.ecg_digitiser import EcgDigitiserDigitizer
from ecg_photo.ingest import estimate_page_grids, ingest
from ecg_photo.pipeline import confirm_scale, digitize_page

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
PTBXL_BASE = "https://physionet.org/files/ptb-xl/1.0.3/records500/00000"
PTBXL_LICENSE_URL = "https://physionet.org/files/ptb-xl/1.0.3/LICENSE.txt"
SCOPE = (
    "real 12-lead signals (PTB-XL), synthetic ECG-Image-Kit renders; "
    "not real photographs; not clinical validation"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def download_ptbxl(records: list[int], dest: Path) -> dict[str, dict]:
    dest.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, dict] = {}
    files = [("LICENSE", PTBXL_LICENSE_URL)]
    for n in records:
        base = f"{n:05d}_hr"
        files += [
            (f"{base}.hea", f"{PTBXL_BASE}/{base}.hea"),
            (f"{base}.dat", f"{PTBXL_BASE}/{base}.dat"),
        ]
    for name, url in files:
        out = dest / name
        if not out.exists():
            try:
                urllib.request.urlretrieve(url, out)
            except Exception as e:
                raise RuntimeError(f"PTB-XL download failed: {url}: {e}") from e
        hashes[name] = {"sha256": sha256_file(out), "url": url, "bytes": out.stat().st_size}
    return hashes


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
    subprocess.run(cmd, cwd=imgkit_root, check=True, capture_output=True)
    return time.monotonic() - t0


def git_rev(path: Path) -> str:
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True
    )
    return r.stdout.strip()


def rhythm_lead_of(imgkit_json: dict, sig_len: int) -> str:
    """Lead whose imgkit entry spans the full record (full-width rhythm strip)."""
    names = [
        l["lead_name"]
        for l in imgkit_json.get("leads", [])
        if l.get("end_sample") == sig_len and l.get("start_sample") == 0
    ]
    return names[-1] if names else "II"


def percentile(a: list[float], q: float) -> float | None:
    a = [x for x in a if x is not None and math.isfinite(x)]
    return float(np.percentile(a, q)) if a else None


def aggregate(cases: list[dict]) -> dict:
    ok = [c for c in cases if c["status"] == "ok"]

    def vals(k):
        return [c[k] for c in ok]

    agg = {
        "n_leads": len(cases),
        "n_ok": len(ok),
        "n_no_signal": sum(1 for c in cases if c["status"] == "no_signal"),
        "n_engine_error": sum(1 for c in cases if c["status"] == "engine_error"),
        "median_r_at_lag": percentile(vals("r_at_lag"), 50),
        "iqr_r_at_lag": [
            percentile(vals("r_at_lag"), 25),
            percentile(vals("r_at_lag"), 75),
        ],
        "median_rmse_mV": percentile(vals("rmse_mV"), 50),
        "iqr_rmse_mV": [
            percentile(vals("rmse_mV"), 25),
            percentile(vals("rmse_mV"), 75),
        ],
        "median_amp_ratio": percentile(vals("amp_ratio"), 50),
        "median_duration_err_pct": percentile(vals("duration_err_pct"), 50),
        "median_abs_offset_err_s": percentile(
            [abs(c["offset_err_s"]) for c in ok if c["offset_err_s"] is not None], 50
        ),
        "frac_r_ge_0.9": (
            sum(1 for c in ok if math.isfinite(c["r_at_lag"]) and c["r_at_lag"] >= 0.9) / len(ok)
            if ok
            else None
        ),
        "per_lead": {},
    }
    for lead in LEADS:
        sub = [c for c in cases if c["lead"] == lead and c["status"] == "ok"]
        if sub:
            agg["per_lead"][lead] = {
                "n": len(sub),
                "median_r_at_lag": percentile([c["r_at_lag"] for c in sub], 50),
                "median_rmse_mV": percentile([c["rmse_mV"] for c in sub], 50),
                "median_duration_err_pct": percentile([c["duration_err_pct"] for c in sub], 50),
            }
    return agg


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
    ap.add_argument("--work", type=Path, default=Path("runs/f6"))
    ap.add_argument("--records", type=int, nargs="+", required=True)
    args = ap.parse_args()

    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    imgkit_root = args.imgkit_root.resolve()
    imgkit_python = args.imgkit_python.absolute()

    file_hashes = download_ptbxl(args.records, work / "ptbxl")

    digitizers: dict[str, tuple[Digitizer, Path]] = {}
    if "ahus" in args.engines:
        assert args.ahus_root and args.ahus_python and args.base_config
        digitizers["ahus"] = (
            AhusDigitizer(
                ahus_root=args.ahus_root.resolve(),
                python_exe=args.ahus_python.absolute(),
                base_config=args.base_config.resolve(),
                duration_s=10.0,
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

    cases: dict[str, list[dict]] = {e: [] for e in digitizers}
    wall: dict[str, float] = {}
    grids_out: dict[str, dict] = {}
    failures: list[dict] = []

    for n in args.records:
        rec_base = work / "ptbxl" / f"{n:05d}_hr"
        t_img = run_imgkit(imgkit_root, imgkit_python, rec_base, work / f"imgkit_{n:05d}", n)
        wall[f"{n:05d}_imgkit"] = t_img
        img_dir = work / f"imgkit_{n:05d}"
        pngs = sorted(img_dir.glob("*.png"))
        jsons = sorted(img_dir.glob("*.json"))
        if not pngs or not jsons:
            raise RuntimeError(f"imgkit produced no output for record {n}")
        gt = json.loads(jsons[0].read_text())

        rec = wfdb.rdrecord(str(rec_base))
        truth = rec.p_signal
        sig_names = [s.upper() for s in rec.sig_name]
        truth_fs = float(rec.fs)
        sig_len = int(truth.shape[0])
        rhythm = rhythm_lead_of(gt, sig_len)

        # imgkit renders at 25 mm/s / 10 mm/mV (ecg_plot grid calibration;
        # config.json has no paper_speed/gain keys -> recorded, not asserted)
        speed_cfg = gt.get("paper_speed")
        gain_cfg = gt.get("gain_mm_mV")
        if speed_cfg is not None and float(speed_cfg) != 25.0:
            failures.append({"record": n, "error": f"paper_speed={speed_cfg}"})
            continue
        if gain_cfg is not None and float(gain_cfg) != 10.0:
            failures.append({"record": n, "error": f"gain={gain_cfg}"})
            continue

        rev_dir = work / f"rec{n:05d}" / "rev"
        manifest = ingest(pngs[0], rev_dir)
        manifest, grids = estimate_page_grids(rev_dir, manifest)
        grids_out[f"{n:05d}"] = {
            "px_per_mm_x": grids[0].px_per_mm_x,
            "px_per_mm_y": grids[0].px_per_mm_y,
            "rhythm_lead": rhythm,
        }
        page_id = manifest.pages[0].page_id

        for engine, (digitizer, _root) in digitizers.items():
            t0 = time.monotonic()
            try:
                run = digitize_page(rev_dir, page_id, digitizer, engine_duration_s=10.0)
                run2 = confirm_scale(
                    run.run_dir,
                    gain_mm_mV=10.0,
                    speed_mm_s=25.0,
                    author="f6-bench",
                    reason="imgkit renders at 25 mm/s, 10 mm/mV (imgkit defaults)",
                    time_source="evidence",
                )
                wall[f"{n:05d}_{engine}"] = time.monotonic() - t0
            except Exception as e:  # noqa: BLE001 - recorded per lead
                wall[f"{n:05d}_{engine}"] = time.monotonic() - t0
                for lead in LEADS:
                    cases[engine].append(
                        {"record": n, "lead": lead, "status": "engine_error", "error": str(e)[:300]}
                    )
                failures.append({"record": n, "engine": engine, "error": str(e)[:300]})
                print(f"rec {n:05d} {engine}: ERROR {e}", flush=True)
                continue

            from ecg_photo.contracts import load_manifest

            rm = load_manifest(run2.manifest)
            for seg in rm.segments:
                lead = seg.lead_label
                if lead not in LEADS or seg.signal_path is None:
                    cases[engine].append(
                        {
                            "record": n,
                            "lead": lead,
                            "status": "no_signal",
                            "segment_id": seg.segment_id,
                        }
                    )
                    continue
                sig = np.load(run2.run_dir / seg.signal_path, allow_pickle=False)
                observed = np.isfinite(sig)
                if seg.observed_mask_path and (run2.run_dir / seg.observed_mask_path).exists():
                    observed = np.load(run2.run_dir / seg.observed_mask_path, allow_pickle=False)
                tlead = (
                    truth[:, sig_names.index(lead.upper())] if lead.upper() in sig_names else None
                )
                if tlead is None:
                    cases[engine].append(
                        {
                            "record": n,
                            "lead": lead,
                            "status": "no_signal",
                            "error": "lead not in truth",
                        }
                    )
                    continue
                m = segment_metrics(
                    tlead,
                    sig,
                    observed,
                    est_fs=float(seg.working_fs_hz or truth_fs),
                    truth_fs=truth_fs,
                    observed_duration_s=seg.observed_duration_s,
                    expected_window_s=10.0 if lead == rhythm else 2.5,
                )
                cases[engine].append({"record": n, "lead": lead, "segment_id": seg.segment_id, **m})
            print(f"rec {n:05d} {engine}: wall={wall[f'{n:05d}_{engine}']:.1f}s", flush=True)

    aggregates = {e: aggregate(c) for e, c in cases.items()}
    results = {
        "timestamp": datetime.now(UTC).isoformat(),
        "scope": SCOPE,
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
        "our_commit": git_rev(Path(__file__).parent.parent),
        "python": sys.version.split()[0],
        "machine": {"platform": platform.platform(), "machine": platform.machine()},
        "ptbxl": {
            "version": "1.0.3",
            "license": "CC BY 4.0",
            "citation": (
                "Wagner P. et al., PTB-XL, a large publicly available "
                "electrocardiography dataset 1.0.3, PhysioNet, doi 10.13026/kfzx-aw45"
            ),
            "records": [f"{n:05d}_hr" for n in args.records],
            "files": file_hashes,
        },
        "calibration_note": (
            "imgkit renders at 25 mm/s, 10 mm/mV (ecg_plot grid calibration); "
            "store_config json exposes no paper_speed/gain keys"
        ),
        "grid_estimates": grids_out,
        "wall_time_s": wall,
        "cases": cases,
        "aggregates": aggregates,
        "failures": failures,
    }
    out_path = work / "f6_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # compact markdown table
    print(
        "\n| engine | n_ok/n | median r@lag | median rmse_mV | median amp | med dur err% | r>=0.9 |"
    )
    print("|---|---|---|---|---|---|---|")

    def fmt(v, d=3):
        return f"{v:.{d}f}" if isinstance(v, float) else "-"

    for e, a in aggregates.items():
        print(
            f"| {e} | {a['n_ok']}/{a['n_leads']} | {fmt(a['median_r_at_lag'])} | "
            f"{fmt(a['median_rmse_mV'])} | {fmt(a['median_amp_ratio'])} | "
            f"{fmt(a['median_duration_err_pct'], 2)} | {fmt(a['frac_r_ge_0.9'], 2)} |"
        )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
