"""Engine output -> contract segments, and manual scale confirmation.

Design principle (settled): the engines' canonical grids and units are
ENGINE ASSUMPTIONS, not evidence. `digitize_page` therefore writes segments
with speed_status=gain_status=unknown, units=None, signal_path=None, and the
engine values as a `temporal_trace_path` in units `arbitrary`. Only an
explicit manual confirmation (author+reason) turns them into mV/seconds.
"""

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ecg_photo.contracts import (
    CalibrationEvidence,
    LeadStatus,
    Manifest,
    ProcessingStep,
    RepresentationKind,
    ResolutionEvidence,
    ScaleStatus,
    Segment,
    TemporalTraceUnits,
    TimeRelation,
    dump_json,
    load_manifest,
    validate_revision_dir_report,
)
from ecg_photo.digitizers.base import Digitizer
from ecg_photo.signal import grid_from_observed_duration

ZERO_RUN_MIN = 50


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    manifest: Path


def new_run_id() -> str:
    return "run-" + uuid.uuid4().hex


def _mask_zero_runs(
    values: np.ndarray, support: np.ndarray, run_min: int
) -> tuple[np.ndarray, int]:
    """Mark runs of >= run_min consecutive exact zeros as not supported."""
    support = support.copy()
    masked = 0
    zero = values == 0.0
    n = len(values)
    i = 0
    while i < n:
        if zero[i]:
            j = i
            while j < n and zero[j]:
                j += 1
            if j - i >= run_min:
                support[i:j] = False
                masked += j - i
            i = j
        else:
            i += 1
    return support, masked


def _contiguous_resample(
    values: np.ndarray, support: np.ndarray, fs_in: float, fs_out: float, n_out: int
) -> tuple[np.ndarray, np.ndarray]:
    """Resample onto k/fs_out interpolating only inside contiguous supported runs."""
    out = np.full(n_out, np.nan)
    out_obs = np.zeros(n_out, dtype=np.bool_)
    t_in = np.arange(len(values)) / fs_in
    t_out = np.arange(n_out) / fs_out
    n = len(values)
    i = 0
    while i < n:
        if not support[i]:
            i += 1
            continue
        j = i
        while j < n and support[j]:
            j += 1
        # supported block [i, j): resample only within its span
        lo, hi = t_in[i], t_in[j - 1]
        sel = (t_out >= lo) & (t_out <= hi)
        out[sel] = np.interp(t_out[sel], t_in[i:j], values[i:j])
        out_obs[sel] = True
        i = j
    return out, out_obs


def digitize_page(
    revision_dir: Path,
    page_id: str,
    digitizer: Digitizer,
    *,
    engine_duration_s: float,
    zero_run_min: int = ZERO_RUN_MIN,
    runs_root: Path | None = None,
) -> RunPaths:
    revision_dir = Path(revision_dir)
    manifest = load_manifest(revision_dir / "manifest.json")
    page = next((p for p in manifest.pages if p.page_id == page_id), None)
    if page is None:
        raise ValueError(f"page {page_id} not in manifest")
    transform_id = next(
        (
            t.transform_id
            for t in manifest.transforms
            if t.target_frame_id == f"rectified-{page_id}"
        ),
        "",
    )

    run_id = new_run_id()
    run_dir = Path(runs_root) / run_id if runs_root else revision_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "pages").mkdir(exist_ok=True)
    for p in manifest.pages:
        src = revision_dir / p.raster_path
        if src.exists():
            (run_dir / "pages").joinpath(Path(p.raster_path).name).write_bytes(src.read_bytes())
        if p.grid_estimate_path and (revision_dir / p.grid_estimate_path).exists():
            dst = run_dir / p.grid_estimate_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes((revision_dir / p.grid_estimate_path).read_bytes())

    out = digitizer.run(revision_dir / page.raster_path, run_dir / "engine")
    engine_frame = f"engine-canonical:{out.engine_id}"

    segments: list[Segment] = []
    skipped_leads: list[dict[str, str]] = []
    (run_dir / "raw_paths").mkdir(exist_ok=True)
    (run_dir / "traces").mkdir(exist_ok=True)

    for lead, values in out.leads.items():
        values = np.asarray(values, dtype=np.float64)
        support = np.asarray(out.observed.get(lead, np.isfinite(values)), dtype=np.bool_).copy()
        steps: list[ProcessingStep] = [
            ProcessingStep(
                stage="engine_inference",
                implementation=out.engine_id,
                parameters={
                    "engine_commit": out.engine_commit,
                    "config_hash": out.config_hash,
                    "engine_duration_s_assumed": float(engine_duration_s),
                    "engine_fs_hz_assumed": float(out.fs_hz),
                    "wall_time_s": float(out.wall_time_s),
                    "layout_detected": out.layout_detected,
                },
                weights_sha256=",".join(out.weights_sha256) or None,
            )
        ]
        if out.extra.get("missing_encoded_as_zero"):
            support, masked = _mask_zero_runs(values, support, zero_run_min)
            if masked:
                steps.append(
                    ProcessingStep(
                        stage="zero_run_masking",
                        implementation="ecg_photo.pipeline",
                        parameters={
                            "zero_run_min": int(zero_run_min),
                            "masked_samples": int(masked),
                        },
                    )
                )
        if int(support.sum()) == 0:
            skipped_leads.append({"lead": lead, "reason": "NO_SUPPORT"})
            continue

        seg_id = f"seg-{lead}-{page_id}"
        raw = np.stack([np.arange(len(values), dtype=np.float64), values], axis=1)
        np.save(run_dir / "raw_paths" / f"{seg_id}.npy", raw)
        np.savez_compressed(run_dir / "raw_paths" / f"{seg_id}-support.npz", support=support)
        trace = np.where(support, values, np.nan)
        np.save(run_dir / "traces" / f"{seg_id}.npy", trace)

        if out.layout_detected:
            rep_ev = f"engine layout {out.layout_detected!r}; continuous strip slot"
        else:
            rep_ev = "engine did not report layout; output is a continuous canonical strip"
        w, h = int(page.width_px), int(page.height_px)
        segments.append(
            Segment(
                segment_id=seg_id,
                page_id=page_id,
                lead_label=lead,
                lead_status=LeadStatus.proposed,
                lead_evidence=(
                    f"column label emitted by engine {out.engine_id}; "
                    "engine gives no per-lead bounding box"
                ),
                representation_kind=RepresentationKind.rhythm_segment,
                representation_evidence=rep_ev,
                source_region=[(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)],
                transform_id=transform_id,
                raw_path=f"raw_paths/{seg_id}.npy",
                raw_coordinate_frame=engine_frame,
                raw_support_path=f"raw_paths/{seg_id}-support.npz",
                time_relation=TimeRelation.simultaneous,
                sync_group_id=f"{page_id}-{run_id}",
                speed_mm_s=None,
                speed_status=ScaleStatus.unknown,
                gain_mm_mV=None,
                gain_status=ScaleStatus.unknown,
                units=None,
                temporal_trace_path=f"traces/{seg_id}.npy",
                temporal_trace_units=TemporalTraceUnits.arbitrary,
                processing=steps,
            )
        )

    run_manifest = Manifest(
        schema_version="1.0.0",
        study_id=manifest.study_id,
        revision=manifest.revision,
        input_revision=manifest.revision,
        run_id=run_id,
        config_hash=out.config_hash,
        source=manifest.source,
        segments=segments,
        pages=manifest.pages,
        transforms=manifest.transforms,
    )
    dump_json(run_manifest, run_dir / "manifest.json")
    report = {
        "engine_id": out.engine_id,
        "wall_time_s": out.wall_time_s,
        "leads_written": [s.lead_label for s in segments],
        "skipped_leads": skipped_leads,
        "layout_detected": out.layout_detected,
    }
    (run_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    problems, _warnings = validate_revision_dir_report(run_dir, run_manifest)
    if problems:
        raise RuntimeError(f"run dir failed validation: {problems}")
    return RunPaths(run_dir=run_dir, manifest=run_dir / "manifest.json")


def _engine_assumption(seg: Segment, name: str) -> float:
    for st in seg.processing:
        if st.stage == "engine_inference":
            v = st.parameters.get(name)
            if isinstance(v, (int, float)):
                return float(v)
    raise ValueError(f"{seg.segment_id}: no engine_inference step with {name}")


def confirm_engine_scale(
    run_dir: Path,
    *,
    speed_mm_s: float,
    gain_mm_mV: float,
    author: str,
    reason: str,
    fs_hz: float | None = None,
) -> RunPaths:
    """New run dir whose segments become calibrated by manual confirmation.

    Manual confirmation asserts the engine's assumed scale is correct for this
    study; it does NOT derive scale from image evidence.
    """
    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir / "manifest.json")
    new_id = new_run_id()
    new_dir = run_dir.parent / new_id
    new_dir.mkdir(parents=True)
    if (run_dir / "pages").exists():
        (new_dir / "pages").mkdir()
        for f in (run_dir / "pages").iterdir():
            (new_dir / "pages" / f.name).write_bytes(f.read_bytes())

    segments: list[Segment] = []
    (new_dir / "signals").mkdir(exist_ok=True)
    (new_dir / "masks").mkdir(exist_ok=True)
    (new_dir / "raw_paths").mkdir(exist_ok=True)
    (new_dir / "traces").mkdir(exist_ok=True)

    for seg in manifest.segments:
        engine_fs = _engine_assumption(seg, "engine_fs_hz_assumed")
        engine_dur = _engine_assumption(seg, "engine_duration_s_assumed")
        target_fs = float(fs_hz) if fs_hz is not None else engine_fs
        if seg.temporal_trace_path is None:
            raise ValueError(f"{seg.segment_id}: no temporal_trace_path to confirm")
        trace = np.load(run_dir / seg.temporal_trace_path, allow_pickle=False)
        with np.load(run_dir / seg.raw_support_path, allow_pickle=False) as z:
            support = z["support"]

        grid = grid_from_observed_duration(engine_dur, target_fs)
        if abs(target_fs - engine_fs) > 1e-12:
            sig, observed = _contiguous_resample(
                np.where(support, trace, 0.0), support, engine_fs, target_fs, grid.n_samples
            )
        else:
            observed = support[: grid.n_samples]
            sig = np.where(observed, trace[: grid.n_samples], np.nan)

        sid = seg.segment_id
        np.save(new_dir / "signals" / f"{sid}.npy", sig)
        np.save(new_dir / "masks" / f"{sid}-observed.npy", observed)
        np.save(new_dir / "masks" / f"{sid}-valid.npy", observed.copy())
        np.save(new_dir / "masks" / f"{sid}-gapfill.npy", np.zeros(grid.n_samples, dtype=np.bool_))
        # carry raw paths forward so the new run is self-contained; the
        # engine-canonical trace is superseded by signal_path and dropped
        for rel in (seg.raw_path, seg.raw_support_path):
            dst = new_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes((run_dir / rel).read_bytes())

        evidence = [
            CalibrationEvidence(
                evidence_id=f"manual-{uuid.uuid4().hex[:12]}",
                kind="manual",
                quantity="speed_mm_s",  # type: ignore[arg-type]
                value=float(speed_mm_s),
                unit="mm/s",
                frame_id=seg.raw_coordinate_frame,
                method="manual confirmation of engine assumption",
                limitations="not derived from image evidence",
                author=author,
                reason=reason,
            ),
            CalibrationEvidence(
                evidence_id=f"manual-{uuid.uuid4().hex[:12]}",
                kind="manual",
                quantity="gain_mm_mV",  # type: ignore[arg-type]
                value=float(gain_mm_mV),
                unit="mm/mV",
                frame_id=seg.raw_coordinate_frame,
                method="manual confirmation of engine assumption",
                limitations="not derived from image evidence",
                author=author,
                reason=reason,
            ),
        ]
        steps = list(seg.processing) + [
            ProcessingStep(
                stage="manual_scale_confirmation",
                implementation="ecg_photo.pipeline",
                parameters={
                    "author": author,
                    "reason": reason,
                    "previous_speed_status": "unknown",
                    "previous_gain_status": "unknown",
                    "speed_mm_s": float(speed_mm_s),
                    "gain_mm_mV": float(gain_mm_mV),
                    "valid_mask": "copied from observed; no extra QC applied",
                },
            )
        ]
        segments.append(
            seg.model_copy(
                update={
                    "speed_mm_s": float(speed_mm_s),
                    "speed_status": ScaleStatus.manual,
                    "gain_mm_mV": float(gain_mm_mV),
                    "gain_status": ScaleStatus.manual,
                    "units": "mV",
                    "working_fs_hz": target_fs,
                    "n_samples": grid.n_samples,
                    "duration_s": grid.duration_s,
                    "observed_duration_s": grid.observed_duration_s,
                    "discarded_tail_s": grid.discarded_tail_s,
                    "signal_path": f"signals/{sid}.npy",
                    "temporal_trace_path": None,
                    "temporal_trace_units": None,
                    "observed_mask_path": f"masks/{sid}-observed.npy",
                    "valid_mask_path": f"masks/{sid}-valid.npy",
                    "gap_fill_mask_path": f"masks/{sid}-gapfill.npy",
                    "effective_dt_ms": None,
                    "effective_dv_mV": None,
                    "resolution_evidence": ResolutionEvidence(
                        source_frame_id=seg.raw_coordinate_frame,
                        method="none",
                        limitations=(
                            "engine canonical grid; native pixel resolution not traceable"
                        ),
                    ),
                    "calibration_evidence": list(seg.calibration_evidence) + evidence,
                    "processing": steps,
                }
            )
        )

    new_manifest = manifest.model_copy(update={"run_id": new_id, "segments": segments})
    dump_json(new_manifest, new_dir / "manifest.json")
    problems, _warnings = validate_revision_dir_report(new_dir, new_manifest)
    if problems:
        raise RuntimeError(f"confirmed run dir failed validation: {problems}")
    return RunPaths(run_dir=new_dir, manifest=new_dir / "manifest.json")
