"""One command from a photo/PDF to an exported, quality-checked signal (O6 in
docs/mission.md).

`process_file` chains the existing steps on one input, in a fresh output
directory:

    ingest (+ page grid) -> digitize (engine) -> confirm_scale -> run_qc -> export

and writes `summary.json` plus `overview.png`, a 3x4 + II-rhythm redraw of the
digitized leads on millimetre paper to compare with the original by eye.

Time axis: `time_source="auto"` tries the image evidence (own grid x speed)
first and falls back to the engine's assumed canvas only when the evidence
axis is refused; the fallback and its reason are recorded in the summary and
the manifest of the confirmed run says which axis was used. Speed and gain are
never defaulted: they come from the caller (manual confirmation of what is
printed on the page).
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from ecg_photo.contracts import load_manifest, sha256_file
from ecg_photo.digitizers.base import Digitizer
from ecg_photo.export import ExportNotAllowed, export_csv, export_json, export_wfdb
from ecg_photo.ingest import estimate_page_grids, ingest, page_upsample
from ecg_photo.pipeline import RunPaths, confirm_scale, digitize_page
from ecg_photo.qc import QC_FILENAME, RHYTHM_S, SHORT_LEAD_S, qc_json, run_qc
from ecg_photo.render import PaperSpec, RenderNotAllowed, render_segment_pdf, render_segment_png

SUMMARY_SCHEMA = "ecg-photo-process/1"
TimeSource = Literal["auto", "evidence", "engine"]

# 3x4 + II rhythm (GE MAC2000 / ECG-image-kit): column of each short lead
COLUMNS = (("I", "II", "III"), ("aVR", "aVL", "aVF"), ("V1", "V2", "V3"), ("V4", "V5", "V6"))


@dataclass(frozen=True)
class ProcessOptions:
    speed_mm_s: float
    gain_mm_mV: float
    author: str
    reason: str
    page_id: str = "page-1"
    engine_duration_s: float = 10.0
    time_source: TimeSource = "auto"
    fs_hz: float | None = None
    printed_rr_ms: float | None = None
    printed_hr_bpm: float | None = None
    dpi: float = 300.0


def export_run(root: Path, out_dir: Path, paper: PaperSpec) -> dict:
    """CSV / PNG / PDF / WFDB per segment of a confirmed run; artifacts the
    contract refuses are listed in `skipped` with their reason code."""
    manifest = load_manifest(root / "manifest.json")
    out_dir.mkdir(parents=True, exist_ok=True)
    export_json(manifest, out_dir / "manifest.json")
    produced: list[str] = ["manifest.json"]
    skipped: list[dict[str, str]] = []
    for seg in manifest.segments:
        try:
            export_csv(root, seg, out_dir / f"{seg.segment_id}.csv")
            produced.append(f"{seg.segment_id}.csv")
        except ExportNotAllowed as e:
            skipped.append(
                {"segment_id": seg.segment_id, "artifact": "csv", "reason_code": str(e.reason)}
            )
        for artifact, fn in (("png", render_segment_png), ("pdf", render_segment_pdf)):
            try:
                fn(root, seg, out_dir / f"{seg.segment_id}.{artifact}", paper)
                produced.append(f"{seg.segment_id}.{artifact}")
            except RenderNotAllowed as e:
                skipped.append(
                    {
                        "segment_id": seg.segment_id,
                        "artifact": artifact,
                        "reason_code": str(e.reason),
                    }
                )
        try:
            export_wfdb(root, manifest, [seg.segment_id], out_dir, seg.segment_id)
            produced.append(f"{seg.segment_id}.hea/.dat (wfdb)")
        except ExportNotAllowed as e:
            skipped.append(
                {"segment_id": seg.segment_id, "artifact": "wfdb", "reason_code": str(e.reason)}
            )
    report = {"produced": produced, "skipped": skipped}
    (out_dir / "export_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def _lead_signals(run_dir: Path) -> dict[str, tuple[np.ndarray, float]]:
    """Observed span (leading/trailing gaps trimmed, inner gaps NaN) per lead."""
    manifest = load_manifest(run_dir / "manifest.json")
    out: dict[str, tuple[np.ndarray, float]] = {}
    for seg in manifest.segments:
        if seg.signal_path is None or seg.working_fs_hz is None:
            continue
        x = np.load(run_dir / seg.signal_path, allow_pickle=False).astype(np.float64)
        if seg.observed_mask_path and (run_dir / seg.observed_mask_path).exists():
            x = np.where(np.load(run_dir / seg.observed_mask_path, allow_pickle=False), x, np.nan)
        idx = np.nonzero(np.isfinite(x))[0]
        if len(idx):
            out[seg.lead_label] = (x[idx[0] : idx[-1] + 1], float(seg.working_fs_hz))
    return out


def render_overview(run_dir: Path, out_path: Path, *, title: str = "") -> None:
    """Redraw the digitized leads as a 3x4 + II-rhythm page at 25 mm/s and
    10 mm/mV on a millimetre grid (display only; the signals are in mV and s).
    Each short lead is drawn from the start of its 2.5 s slot; a lead that is
    missing leaves its slot empty."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    signals = _lead_signals(run_dir)
    row_mm, speed, gain = 30.0, 25.0, 10.0  # layout only
    width_mm, height_mm = RHYTHM_S * speed, 4 * row_mm
    fig = plt.figure(figsize=(width_mm / 25.4 * 1.1, height_mm / 25.4 * 1.1))
    ax = fig.add_axes((0.0, 0.0, 1.0, 0.95 if title else 1.0))
    for x in np.arange(0, width_mm + 0.01, 1.0):
        ax.axvline(x, color="#f4c7c7", lw=0.3 if x % 5 else 0.7, zorder=0)
    for y in np.arange(0, height_mm + 0.01, 1.0):
        ax.axhline(y, color="#f4c7c7", lw=0.3 if y % 5 else 0.7, zorder=0)

    def _draw(name: str, t0_s: float, max_s: float, base_mm: float) -> None:
        ax.text(t0_s * speed + 1, base_mm + 8, name, fontsize=7, zorder=3)
        if name not in signals:
            ax.text(t0_s * speed + 1, base_mm - 2, "(no digitalizada)", fontsize=6, color="#888")
            return
        sig, fs = signals[name]
        n = min(len(sig), round(max_s * fs))
        t = t0_s + np.arange(n) / fs
        ax.plot(t * speed, base_mm + sig[:n] * gain, color="black", lw=0.6, zorder=2)

    for col, names in enumerate(COLUMNS):
        for row, name in enumerate(names):
            _draw(name, col * SHORT_LEAD_S, SHORT_LEAD_S, height_mm - (row + 0.5) * row_mm)
    _draw("II", 0.0, RHYTHM_S, 0.5 * row_mm)
    ax.set_xlim(0, width_mm)
    ax.set_ylim(0, height_mm)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=8, y=0.995)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _confirm(run: RunPaths, opts: ProcessOptions) -> tuple[RunPaths, str, str | None]:
    """Confirmed run, time axis used and, if the evidence axis was refused,
    why. A refused attempt leaves no partial run directory behind."""
    order: list[Literal["evidence", "engine"]] = (
        ["evidence", "engine"] if opts.time_source == "auto" else [opts.time_source]
    )
    refused: str | None = None
    for source in order:
        before = {p.name for p in run.run_dir.parent.iterdir()}
        try:
            paths = confirm_scale(
                run.run_dir,
                gain_mm_mV=opts.gain_mm_mV,
                speed_mm_s=opts.speed_mm_s,
                fs_hz=opts.fs_hz,
                author=opts.author,
                reason=opts.reason,
                time_source=source,
            )
            return paths, source, refused
        except ValueError as e:
            for p in run.run_dir.parent.iterdir():
                if p.name not in before:
                    shutil.rmtree(p, ignore_errors=True)
            if source == order[-1]:
                raise
            refused = str(e)[:300]
    raise AssertionError("unreachable")


def process_file(file: Path, out: Path, digitizer: Digitizer, opts: ProcessOptions) -> dict:
    """Run the whole chain on one input into `out` (must not exist or be
    empty) and return the summary also written to `out/summary.json`.
    Rejections and engine errors propagate (IngestRejected, ValueError,
    RuntimeError); nothing is written as a result in that case."""
    file, out = Path(file), Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    study = out / "study"
    manifest = ingest(file, study)
    manifest, grids = estimate_page_grids(study, manifest)
    page = next((p for p in manifest.pages if p.page_id == opts.page_id), None)
    if page is None:
        raise ValueError(f"page {opts.page_id} not in {[p.page_id for p in manifest.pages]}")
    grid = grids[manifest.pages.index(page)]

    run = digitize_page(
        study,
        opts.page_id,
        digitizer,
        engine_duration_s=opts.engine_duration_s,
        runs_root=out / "runs",
    )
    run_report = json.loads((run.run_dir / "run_report.json").read_text(encoding="utf-8"))
    final, used, refused = _confirm(run, opts)

    qc = run_qc(final.run_dir, printed_rr_ms=opts.printed_rr_ms, printed_hr_bpm=opts.printed_hr_bpm)
    (final.run_dir / QC_FILENAME).write_text(qc_json(qc) + "\n", encoding="utf-8")
    paper = PaperSpec(speed_mm_s=opts.speed_mm_s, gain_mm_mV=opts.gain_mm_mV, dpi=opts.dpi)
    exported = export_run(final.run_dir, out / "export", paper)
    render_overview(
        final.run_dir,
        out / "overview.png",
        title=f"{file.name} - calidad: {qc.label.value} - eje temporal: {used}",
    )

    confirmed = load_manifest(final.run_dir / "manifest.json")
    summary = {
        "schema": SUMMARY_SCHEMA,
        "input": {"name": file.name, "sha256": sha256_file(file)},
        "page": {
            "page_id": page.page_id,
            "size_px": [page.width_px, page.height_px],
            "upsample": page_upsample(manifest, page.page_id),
            "grid_px_per_mm_x": grid.px_per_mm_x,
        },
        "engine": run_report.get("engine_id"),
        "layout_detected": run_report.get("layout_detected"),
        "leads_written": run_report.get("leads_written"),
        "skipped_leads": run_report.get("skipped_leads"),
        "time_source": used,
        "evidence_axis_refused": refused,
        "speed_mm_s": opts.speed_mm_s,
        "gain_mm_mV": opts.gain_mm_mV,
        "leads": [
            {
                "lead": s.lead_label,
                "duration_s": s.duration_s,
                "observed_duration_s": s.observed_duration_s,
                "fs_hz": s.working_fs_hz,
            }
            for s in confirmed.segments
        ],
        "qc": json.loads(qc_json(qc)),
        "run_dirs": {
            "engine": str(run.run_dir.relative_to(out)),
            "confirmed": str(final.run_dir.relative_to(out)),
        },
        "export": {"dir": "export", **exported},
        "overview": "overview.png",
        "note": "guidance only; not clinical validation",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
