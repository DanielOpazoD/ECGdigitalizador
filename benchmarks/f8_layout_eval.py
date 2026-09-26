"""F8 (objective O8): does Ahus digitize printouts that are not 3x4 + II?

The engine config used so far (inference_wrapper_george-moody-2024.yml) lets
the layout identifier choose only among 3x4-type layouts, so a 6x2 printout
(six rows of two 5 s columns, common in other electrocardiographs) cannot be
recognised. Ahus ships a wider layout set (lead_layouts_all.yml).

For each PTB-XL record (real signal, 10 s, 500 Hz, the F6-a records): render
a clean printout in each layout (25 mm/s, 10 mm/mV, red mm grid, 200 dpi),
run Ahus with each layout set through the normal pipeline and score every
lead against the printed part of the truth (same scorer as F6-b; truth NaN
outside the lead's printed slot). Synthetic rendering (not photos); real
signals; not clinical validation. Resumable (cases.jsonl).
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f6_ptbxl_bench import download_ptbxl
from f6b_kaggle_bench import ENGINE_DURATION_S, aggregate, git_rev, run_one, with_missing_leads

from ecg_photo.digitizers.ahus import AhusDigitizer

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
# rows of (lead, t0_s, t1_s); the last row of 3x4+1R is the II rhythm strip
LAYOUTS: dict[str, list[list[tuple[str, float, float]]]] = {
    "3x4+1R": [
        [(n, 2.5 * c, 2.5 * (c + 1)) for c, n in enumerate(row)]
        for row in (("I", "aVR", "V1", "V4"), ("II", "aVL", "V2", "V5"), ("III", "aVF", "V3", "V6"))
    ]
    + [[("II", 0.0, 10.0)]],
    "6x2": [[(a, 0.0, 5.0), (b, 5.0, 10.0)] for a, b in zip(LEADS[:6], LEADS[6:], strict=True)],
}
RHYTHM = {"3x4+1R": "II", "6x2": "none"}
SPEED, GAIN, DPI = 25.0, 10.0, 200.0
PAGE_W_MM, MARGIN_MM = 280.0, 15.0


def render_page(
    sig: dict[str, np.ndarray], fs: float, layout: str, out: Path
) -> dict[str, np.ndarray]:
    """Printout PNG of `sig` in `layout`; returns the printed truth (NaN
    outside each lead's slot; a lead printed twice keeps its longest slot)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = LAYOUTS[layout]
    row_mm = 180.0 / len(rows) if len(rows) > 4 else 40.0
    page_h = row_mm * len(rows) + 2 * MARGIN_MM
    fig = plt.figure(figsize=(PAGE_W_MM / 25.4, page_h / 25.4), dpi=DPI)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0, PAGE_W_MM)
    ax.set_ylim(0, page_h)
    for x in np.arange(0, PAGE_W_MM + 0.01, 1.0):
        ax.axvline(x, color="#f3b0b0", lw=0.25 if round(x) % 5 else 0.6, zorder=0)
    for y in np.arange(0, page_h + 0.01, 1.0):
        ax.axhline(y, color="#f3b0b0", lw=0.25 if round(y) % 5 else 0.6, zorder=0)
    x0 = MARGIN_MM + 10.0  # room for the calibration pulse
    printed: dict[str, np.ndarray] = {}
    n = round(10.0 * fs)
    for r, row in enumerate(rows):
        base = page_h - MARGIN_MM - (r + 0.6) * row_mm
        # 1 mV calibration pulse, 5 mm wide
        ax.plot(
            [x0 - 8, x0 - 7, x0 - 7, x0 - 2, x0 - 2, x0 - 1],
            [base, base, base + GAIN, base + GAIN, base, base],
            color="black",
            lw=0.8,
        )
        for name, t0, t1 in row:
            a, b = round(t0 * fs), round(t1 * fs)
            t = np.arange(a, b) / fs
            ax.plot(x0 + t * SPEED, base + sig[name][a:b] * GAIN, color="black", lw=0.8)
            ax.text(x0 + t0 * SPEED + 1, base + 6, name, fontsize=9)
            truth = np.full(n, np.nan)
            truth[a:b] = sig[name][a:b]
            if name not in printed or (b - a) > np.isfinite(printed[name]).sum():
                printed[name] = truth
    ax.axis("off")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return printed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--layouts", nargs="+", default=list(LAYOUTS))
    ap.add_argument(
        "--layout-configs",
        nargs="+",
        default=["lead_layouts_george-moody-2024.yml", "lead_layouts_all.yml"],
    )
    ap.add_argument("--ahus-root", type=Path, default=Path("external/ahus"))
    ap.add_argument("--ahus-python", type=Path, default=Path("external/.venv-ahus/bin/python"))
    ap.add_argument("--work", type=Path, default=Path("runs/f8/layout"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    root = args.ahus_root.resolve()
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    hashes = download_ptbxl(args.records, work / "ptbxl")
    cases_path = work / "cases.jsonl"
    done = set()
    if cases_path.exists():
        for line in cases_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["record"], r["layout"], r["layout_config"]))
    for rec in args.records:
        base = f"{rec:05d}_hr"
        wr = wfdb.rdrecord(str(work / "ptbxl" / base))
        fs = float(wr.fs)
        names = [n if n not in ("AVR", "AVL", "AVF") else "a" + n[1:] for n in wr.sig_name]
        sig = {nm: wr.p_signal[:, i].astype(np.float64) for i, nm in enumerate(names)}
        for layout in args.layouts:
            img = work / base / layout / "page.png"
            truth = render_page(sig, fs, layout, img)
            for lc in args.layout_configs:
                if (base, layout, lc) in done:
                    continue
                dig = AhusDigitizer(
                    ahus_root=root,
                    python_exe=args.ahus_python.absolute(),
                    base_config=root / "src/config/inference_wrapper_george-moody-2024.yml",
                    duration_s=ENGINE_DURATION_S,
                    device="cpu",
                    layout_config=lc,
                )
                case_dir = work / base / layout / lc.removesuffix(".yml")
                res = run_one(img, case_dir, "ahus", dig, truth, fs, RHYTHM[layout])
                layout_seen = None
                runs = sorted((case_dir / "runs").glob("run-*/run_report.json"))
                if runs:
                    layout_seen = json.loads(runs[0].read_text()).get("layout_detected")
                row = {
                    "record": base,
                    "layout": layout,
                    "layout_config": lc,
                    "layout_detected": layout_seen,
                    "wall_s": res["wall_s"],
                    "arms": res["arms"],
                }
                with open(cases_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
                ok = [
                    c["r_at_lag"] for c in res["arms"].get("engine", []) if c.get("status") == "ok"
                ]
                med = f"{np.median(ok):.3f}" if ok else "-"
                print(f"{base} {layout} {lc}: layout={layout_seen} r_med={med}", flush=True)

    rows = [json.loads(line) for line in cases_path.read_text().splitlines()]
    agg: dict = {}
    for layout in args.layouts:
        for lc in args.layout_configs:
            sel = [r for r in rows if r["layout"] == layout and r["layout_config"] == lc]
            key = f"{layout}|{lc}"
            agg[key] = {
                "layouts_detected": {
                    str(k): sum(r["layout_detected"] == k for r in sel)
                    for k in {r["layout_detected"] for r in sel}
                }
            }
            for arm in ("evidence", "engine"):
                cases = [c for r in sel for c in with_missing_leads(r["arms"].get(arm, []))]
                agg[key][arm] = aggregate(cases)
    res_json = {
        "generated": datetime.now(UTC).isoformat(),
        "scope": "PTB-XL signals rendered as clean printouts (synthetic images), Ahus; "
        "not photos; not clinical validation",
        "commit": git_rev(Path(__file__).resolve().parent.parent),
        "ptbxl_files": hashes,
        "aggregates": agg,
        "cases": rows,
    }
    args.out.write_text(json.dumps(res_json, indent=2), encoding="utf-8")
    print("| layout | layouts Ahus | detectado | eje | ok/filas | r@lag med [IQR] |")
    print("|---|---|---|---|---|---|")
    for key, a in agg.items():
        layout, lc = key.split("|")
        for arm in ("evidence", "engine"):
            x = a[arm]
            lo, hi = x["iqr_r_at_lag"] if x.get("iqr_r_at_lag") else (float("nan"),) * 2
            med = x.get("median_r_at_lag")
            print(
                f"| {layout} | {lc} | {a['layouts_detected']} | {arm} | {x['n_ok']}/{x['n_rows']} "
                f"| {med if med is None else round(med, 3)} [{lo:.2f}-{hi:.2f}] |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
