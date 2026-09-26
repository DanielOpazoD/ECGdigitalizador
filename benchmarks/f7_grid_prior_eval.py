"""F7: re-evaluate the `evidence` time axis of the F6-b Kaggle bench with the
current grid estimator (incl. the page-size prior), without re-running engines.

For every (record, image type, engine) of <work>/cases.jsonl: copy the raw
engine run, recompute the page grid from its own page raster, confirm scale
with time_source=evidence (25 mm/s, 10 mm/mV as manual evidence) and score it
against the Kaggle truth like the bench does. Writes cases JSONL + a results
JSON with aggregates per engine x image type. Real Kaggle images; not clinical
validation.
"""

import argparse
import csv
import json
import shutil
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f6b_kaggle_bench import (
    aggregate,
    git_rev,
    read_truth,
    score_run,
    short_error,
    with_missing_leads,
)
from f6b_rescore import classify_run

from ecg_photo.contracts import load_manifest
from ecg_photo.grid import estimate_grid, resolve_ambiguous_period
from ecg_photo.pipeline import confirm_scale


def raw_engine_run(case_dir: Path) -> Path | None:
    runs = case_dir / "runs"
    if not runs.exists():
        return None
    raws = [
        rd for rd in runs.iterdir() if (rd / "manifest.json").exists() and classify_run(rd) is None
    ]
    return max(raws, key=lambda p: p.stat().st_mtime) if raws else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("runs/f6b/kaggle"))
    ap.add_argument("--work", type=Path, default=Path("runs/f6b/bench"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs/f7/grid_prior"))
    ap.add_argument("--rhythm-lead", default="II")
    args = ap.parse_args()

    with open(args.data / "train.csv", newline="") as fh:
        meta = {r["id"]: r for r in csv.DictReader(fh)}
    rows = [json.loads(line) for line in (args.work / "cases.jsonl").read_text().splitlines()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_cases = args.out_dir / "cases.jsonl"
    truths: dict[str, dict] = {}
    records = []
    with open(out_cases, "w", encoding="utf-8") as fh:
        for r in rows:
            rid, itype, engine = r["record"], r["image_type"], r["engine"]
            raw = raw_engine_run(args.work / rid / itype / engine)
            if raw is None:
                continue  # engine failed on this image: nothing to confirm
            dst = args.out_dir / rid / itype / engine / "runs" / raw.name
            if dst.parent.exists():
                shutil.rmtree(dst.parent)
            shutil.copytree(raw, dst)
            page_id = load_manifest(dst / "manifest.json").segments[0].page_id
            img = np.asarray(Image.open(dst / "pages" / f"{page_id}.png").convert("RGB"))
            grid = resolve_ambiguous_period(estimate_grid(img), img)
            (dst / "pages" / f"{page_id}.grid.json").write_text(
                json.dumps(asdict(grid), indent=2), encoding="utf-8"
            )
            row = {
                "record": rid,
                "image_type": itype,
                "engine": engine,
                "grid": {
                    "px_per_mm_x": grid.px_per_mm_x,
                    "ambiguous_period_px_x": grid.ambiguous_period_px_x,
                    "period_step_mm": grid.period_step_mm,
                },
            }
            try:
                done = confirm_scale(
                    dst,
                    gain_mm_mV=10.0,
                    speed_mm_s=25.0,
                    author="f7-eval",
                    reason="competition printouts at 25 mm/s, 10 mm/mV",
                    time_source="evidence",
                )
            except ValueError as e:
                row["evidence"] = [{"status": "time_scale_unknown", "error": short_error(e)}]
            else:
                if rid not in truths:
                    truths[rid] = read_truth(args.data / "train" / rid / f"{rid}.csv")
                fs = float(meta[rid]["fs"])
                row["evidence"] = score_run(done.run_dir, truths[rid], fs, args.rhythm_lead)
            fh.write(json.dumps(row) + "\n")
            records.append(row)
            n_ok = sum(c.get("status") == "ok" for c in row["evidence"])
            print(f"{rid} {itype} {engine}: step={grid.period_step_mm} ok={n_ok}", flush=True)

    groups: dict[str, list[dict]] = {}
    rhythm: dict[str, list[float]] = {}
    for r in records:
        for key in (f"{r['engine']}|ALL", f"{r['engine']}|{r['image_type']}"):
            groups.setdefault(key, []).extend(with_missing_leads(r["evidence"]))
            for c in r["evidence"]:
                if c.get("lead") == args.rhythm_lead and c.get("duration_err_pct") is not None:
                    rhythm.setdefault(key, []).append(c["duration_err_pct"])
    results = {
        "generated": datetime.now(UTC).isoformat(),
        "scope": "F6-b Kaggle images, evidence time axis re-confirmed with the current "
        "grid estimator (page-size prior); not clinical validation",
        "commit": git_rev(Path(__file__).resolve().parent.parent),
        "aggregates": {k: aggregate(v) for k, v in sorted(groups.items())},
        "rhythm_lead_duration_err_pct": {
            k: {
                "n": len(v),
                "median": float(np.median(v)),
                "max_abs": float(np.max(np.abs(v))),
            }
            for k, v in sorted(rhythm.items())
        },
        "cases": records,
    }
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    def fmt(v: object, d: int = 3) -> str:
        return f"{v:.{d}f}" if isinstance(v, float) else "–"

    print("\n| motor | tipo | ok/filas | r@lag med | SNR dB | err dur. II % (med / máx) | no-ok |")
    print("|---|---|---|---|---|---|---|")
    for k, a in results["aggregates"].items():
        e, t = k.split("|")
        rd = results["rhythm_lead_duration_err_pct"].get(k)
        bad = ", ".join(f"{s} {n}" for s, n in sorted(a["statuses"].items()) if s != "ok")
        dur = f"{fmt(rd['median'], 2)} / {fmt(rd['max_abs'], 2)}" if rd else "–"
        print(
            f"| {e} | {t} | {a['n_ok']}/{a['n_rows']} | {fmt(a['median_r_at_lag'])} | "
            f"{fmt(a['median_snr_db'], 1)} | {dur} | {bad} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
