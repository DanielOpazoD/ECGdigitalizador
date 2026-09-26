"""F7: how much does image resolution cost, and does upsampling before the
engine recover it? Phone photos shared through messaging apps arrive at
~1000-1600 px wide (a GE MAC2000 photo received at 1036 px lost aVL and
drifted +4 % in RR).

For each Kaggle record of the chosen image type (default 0005, phone photo of
a printout, 4032 px): resize to each target width (Lanczos), optionally
upsample back by a factor, run Ahus through the normal pipeline (engine time
axis, 25 mm/s, 10 mm/mV) and score against the truth like the F6-b bench.
Resumable (cases.jsonl). Real Kaggle images; not clinical validation.
"""

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f6b_kaggle_bench import (
    ENGINE_DURATION_S,
    aggregate,
    git_rev,
    read_truth,
    run_one,
    with_missing_leads,
)
from f6b_rescore import find_confirmed_runs

from ecg_photo.digitizers.ahus import AhusDigitizer
from ecg_photo.qc import run_qc


def variant_image(src: Path, dst: Path, width: int | None, upsample: float) -> tuple[int, int]:
    """Write src resized to `width` (None: original) then upsampled by
    `upsample`; returns the final size."""
    im = Image.open(src).convert("RGB")
    if width is not None and width < im.width:
        h = round(im.height * width / im.width)
        im = im.resize((width, h), Image.Resampling.LANCZOS)
    if upsample != 1.0:
        im = im.resize(
            (round(im.width * upsample), round(im.height * upsample)), Image.Resampling.LANCZOS
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst)
    return im.width, im.height


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("runs/f6b/kaggle"))
    ap.add_argument("--work", type=Path, default=Path("runs/f7/resolution"))
    ap.add_argument("--type", default="0005")
    ap.add_argument(
        "--variants",
        nargs="+",
        default=["fullx1", "1600x1", "1000x1", "1000x2"],
        help="WIDTHxUPSAMPLE (width 'full' keeps the original)",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    with open(args.data / "train.csv", newline="") as fh:
        meta = {r["id"]: r for r in csv.DictReader(fh)}
    root = Path("external/ahus").resolve()
    dig = AhusDigitizer(
        ahus_root=root,
        python_exe=Path("external/.venv-ahus/bin/python").absolute(),
        base_config=(root / "src/config/inference_wrapper_george-moody-2024.yml"),
        duration_s=ENGINE_DURATION_S,
        device="cpu",
    )
    args.work.mkdir(parents=True, exist_ok=True)
    cases_path = args.work / "cases.jsonl"
    done = set()
    if cases_path.exists():
        for line in cases_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["record"], r["variant"]))
    ids = sorted(p.name for p in (args.data / "train").iterdir() if p.is_dir())
    for rid in ids:
        src = args.data / "train" / rid / f"{rid}-{args.type}.png"
        if not src.exists():
            continue
        truth = read_truth(args.data / "train" / rid / f"{rid}.csv")
        fs = float(meta[rid]["fs"])
        for v in args.variants:
            if (rid, v) in done:
                continue
            w_s, up_s = v.split("x")
            width = None if w_s == "full" else int(w_s)
            case_dir = args.work / rid / v
            img = case_dir / "input.png"
            size = variant_image(src, img, width, float(up_s))
            res = run_one(img, case_dir, "ahus", dig, truth, fs, "II")
            runs = find_confirmed_runs(case_dir)
            qc = run_qc(runs["engine"]) if "engine" in runs else None
            row = {
                "record": rid,
                "variant": v,
                "size": size,
                "wall_s": res["wall_s"],
                "engine": res["arms"].get("engine", []),
                "qc_label": qc.label.value if qc else None,
                "qc_flags": qc.flags if qc else None,
            }
            with open(cases_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            ok = [c["r_at_lag"] for c in row["engine"] if c.get("status") == "ok"]
            med = f"{np.median(ok):.3f}" if ok else "-"
            print(f"{rid} {v} {size}: r_med={med} qc={row['qc_label']}", flush=True)

    rows = [json.loads(line) for line in cases_path.read_text().splitlines()]
    agg = {}
    for v in args.variants:
        cases = [c for r in rows if r["variant"] == v for c in with_missing_leads(r["engine"])]
        a = aggregate(cases)
        labels: dict[str, int] = {}
        for r in rows:
            if r["variant"] == v:
                labels[str(r["qc_label"])] = labels.get(str(r["qc_label"]), 0) + 1
        a["qc_labels"] = labels
        agg[v] = a
    res_json = {
        "generated": datetime.now(UTC).isoformat(),
        "scope": f"Kaggle image type {args.type}, Ahus, engine axis; not clinical validation",
        "commit": git_rev(Path(__file__).resolve().parent.parent),
        "aggregates": agg,
        "cases": rows,
    }
    args.out.write_text(json.dumps(res_json, indent=2), encoding="utf-8")
    print("| variante | ok/filas | r@lag med [IQR] | SNR dB | etiquetas QC |")
    print("|---|---|---|---|---|")
    for v, a in agg.items():
        lo, hi = a["iqr_r_at_lag"]
        print(
            f"| {v} | {a['n_ok']}/{a['n_rows']} | {a['median_r_at_lag']:.3f} "
            f"[{lo:.2f}-{hi:.2f}] | {a['median_snr_db']:.1f} | {a['qc_labels']} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
