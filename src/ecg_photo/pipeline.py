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
from typing import Literal

import numpy as np

from ecg_photo.calibration import resolve_scale
from ecg_photo.contracts import (
    CalibrationEvidence,
    LeadStatus,
    Manifest,
    ProcessingStep,
    ReasonCode,
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
from ecg_photo.transforms import apply_chain

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
        geo = out.geometry.get(lead) if out.geometry else None
        if geo is not None:
            raw = np.stack([geo.x_px, values], axis=1)
            raw_frame = geo.frame_id
            (run_dir / "raw_paths" / f"{seg_id}-geometry.json").write_text(
                json.dumps(geo.chain.model_dump(mode="json"), indent=2), encoding="utf-8"
            )
        else:
            raw = np.stack([np.arange(len(values), dtype=np.float64), values], axis=1)
            raw_frame = engine_frame
        np.save(run_dir / "raw_paths" / f"{seg_id}.npy", raw)
        np.savez_compressed(run_dir / "raw_paths" / f"{seg_id}-support.npz", support=support)
        trace = np.where(support, values, np.nan)
        np.save(run_dir / "traces" / f"{seg_id}.npy", trace)

        if out.layout_detected:
            rep_ev = f"engine layout {out.layout_detected!r}; continuous strip slot"
        else:
            rep_ev = "engine did not report layout; output is a continuous canonical strip"
        w, h = int(page.width_px), int(page.height_px)
        source_region = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
        if geo is not None and len(geo.chain.steps) >= 2:
            # ECG-Digitiser: the first affine step records the rotated-frame
            # mask rectangle; map its corners to file coords via the rest of
            # the chain (rotation about the centre).
            p0 = geo.chain.steps[0].parameters
            if "mask_x1" in p0:
                mx1, my1 = float(p0["mask_x1"]), float(p0["mask_y1"])  # type: ignore[arg-type]
                mw, mh = float(p0["mask_width"]), float(p0["mask_height"])  # type: ignore[arg-type]
                corners = np.array(
                    [[mx1, my1], [mx1 + mw, my1], [mx1 + mw, my1 + mh], [mx1, my1 + mh]]
                )
                tail_chain = geo.chain.model_copy(update={"steps": geo.chain.steps[1:]})
                source_region = [tuple(pt) for pt in apply_chain(tail_chain, corners).tolist()]
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
                source_region=source_region,
                transform_id=transform_id,
                raw_path=f"raw_paths/{seg_id}.npy",
                raw_coordinate_frame=raw_frame,
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


def _grid_evidence_for_page(run_dir: Path, page_id: str) -> list[CalibrationEvidence]:
    """CalibrationEvidence from the page's grid estimate (refined px_per_mm)."""
    grid_path = run_dir / "pages" / f"{page_id}.grid.json"
    if not grid_path.exists():
        return []
    g = json.loads(grid_path.read_text(encoding="utf-8"))
    out: list[CalibrationEvidence] = []
    for q, key in (("px_per_mm_x", "px_per_mm_x"), ("px_per_mm_y", "px_per_mm_y")):
        v = g.get(key)
        if v is None:
            continue
        out.append(
            CalibrationEvidence(
                evidence_id=f"grid-{page_id}-{key}",
                kind="grid_period",
                quantity=q,  # type: ignore[arg-type]
                value=float(v),
                unit="px/mm",
                frame_id="file",
                method=g.get("method", "grid estimate"),
                limitations=g.get("limitations", ""),
            )
        )
    return out


def _calibration_json_evidence(run_dir: Path) -> list[CalibrationEvidence]:
    """Evidence recorded by `calibrate` in the study's calibration.json."""
    for cand in (run_dir / "calibration.json", run_dir.parent.parent / "calibration.json"):
        if cand.exists():
            data = json.loads(cand.read_text(encoding="utf-8"))
            entries = data if isinstance(data, list) else data.get("evidence", [])
            return [CalibrationEvidence(**e) for e in entries]
    return []


def _contiguous_resample_t(
    t_in: np.ndarray, values: np.ndarray, support: np.ndarray, fs_out: float, n_out: int
) -> tuple[np.ndarray, np.ndarray]:
    """np.interp onto k/fs_out inside each contiguous supported block only."""
    out = np.full(n_out, np.nan)
    out_obs = np.zeros(n_out, dtype=np.bool_)
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
        lo, hi = t_in[i], t_in[j - 1]
        sel = (t_out >= lo) & (t_out <= hi)
        out[sel] = np.interp(t_out[sel], t_in[i:j], values[i:j])
        out_obs[sel] = True
        i = j
    return out, out_obs


def confirm_scale(
    run_dir: Path,
    *,
    gain_mm_mV: float,
    author: str,
    reason: str,
    speed_mm_s: float | None = None,
    fs_hz: float | None = None,
    time_source: Literal["evidence", "engine"] = "evidence",
) -> RunPaths:
    """New run dir whose segments become calibrated.

    time_source="evidence" derives the time axis from image evidence (page grid
    px/mm x resolved speed x engine trace pixel extent) and requires
    raw_coordinate_frame "file" plus a resolved px_per_mm_x and speed; it never
    falls back to the engine grid. "engine" keeps the old behaviour: the
    engine's assumed duration, and --speed is required (manual confirmation of
    the engine assumption). Gain is always manual (engine mV assumption).
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

    # resolve shared evidence once: page grid + study calibration.json + CLI
    page_ids = {s.page_id for s in manifest.segments}
    grid_evs = [e for pid in page_ids for e in _grid_evidence_for_page(run_dir, pid)]
    cal_evs = _calibration_json_evidence(run_dir)
    cli_speed_ev = (
        CalibrationEvidence(
            evidence_id=f"manual-{uuid.uuid4().hex[:12]}",
            kind="manual",
            quantity="speed_mm_s",  # type: ignore[arg-type]
            value=float(speed_mm_s),
            unit="mm/s",
            frame_id="file",
            method="manual confirmation of engine assumption",
            limitations="not derived from image evidence",
            author=author,
            reason=reason,
        )
        if speed_mm_s is not None
        else None
    )
    all_ev = grid_evs + cal_evs + ([cli_speed_ev] if cli_speed_ev else [])
    px_res = resolve_scale("px_per_mm_x", all_ev)
    speed_res = resolve_scale("speed_mm_s", all_ev)

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

        time_steps: list[ProcessingStep] = []
        seg_px: float | None = None
        seg_speed: float | None = None
        seg_speed_status = ScaleStatus.unknown
        effective_dt_ms: float | None = None
        if time_source == "evidence":
            if seg.raw_coordinate_frame != "file":
                raise ValueError(
                    f"{seg.segment_id}: {ReasonCode.TIME_SCALE_UNKNOWN} - raw frame is "
                    "engine-canonical, no page-pixel geometry"
                )
            if px_res.value is None:
                raise ValueError(
                    f"{seg.segment_id}: {ReasonCode.CALIBRATION_MISSING} - no px_per_mm_x "
                    "evidence (page grid estimate)"
                )
            if speed_res.value is None:
                raise ValueError(
                    f"{seg.segment_id}: {ReasonCode.TIME_SCALE_UNKNOWN} - no speed_mm_s "
                    "evidence or --speed"
                )
            raw = np.load(run_dir / seg.raw_path, allow_pickle=False)
            x_px = raw[:, 0]
            support = support & np.isfinite(x_px)
            idx = np.nonzero(support)[0]
            scale_fac = px_res.value * speed_res.value  # px per second
            t = (x_px - x_px[idx[0]]) / scale_fac
            dts = np.diff(t[support])
            dt_med = float(np.median(dts)) if len(dts) else 0.0
            observed_duration = float(t[idx[-1]]) + dt_med
            grid = grid_from_observed_duration(observed_duration, target_fs)
            sig, observed = _contiguous_resample_t(
                t, np.where(support, trace, 0.0), support, target_fs, grid.n_samples
            )
            seg_px = px_res.value
            seg_speed = speed_res.value
            seg_speed_status = speed_res.status
            effective_dt_ms = 1000.0 / scale_fac
            time_steps.append(
                ProcessingStep(
                    stage="time_axis_from_image_evidence",
                    implementation="ecg_photo.pipeline",
                    parameters={
                        "px_per_mm_x": float(px_res.value),
                        "speed_mm_s": float(speed_res.value),
                        "x_first": float(x_px[idx[0]]),
                        "x_last": float(x_px[idx[-1]]),
                        "extent_px": float(x_px[idx[-1]] - x_px[idx[0]]),
                        "observed_duration_s": observed_duration,
                    },
                )
            )
        else:
            if speed_mm_s is None:
                raise ValueError("--speed required when --time-source engine")
            grid = grid_from_observed_duration(engine_dur, target_fs)
            expected_n = grid_from_observed_duration(engine_dur, engine_fs).n_samples
            if trace.shape[0] != expected_n:
                raise ValueError(
                    f"{seg.segment_id}: engine trace has {trace.shape[0]} samples but engine grid "
                    f"({engine_dur} s @ {engine_fs} Hz) implies {expected_n}; "
                    "check engine_duration_s"
                )
            if abs(target_fs - engine_fs) > 1e-12:
                sig, observed = _contiguous_resample(
                    np.where(support, trace, 0.0), support, engine_fs, target_fs, grid.n_samples
                )
            else:
                observed = support[: grid.n_samples]
                sig = np.where(observed, trace[: grid.n_samples], np.nan)
            seg_speed = float(speed_mm_s)
            seg_speed_status = ScaleStatus.manual
            time_steps.append(
                ProcessingStep(
                    stage="manual_scale_confirmation",
                    implementation="ecg_photo.pipeline",
                    parameters={
                        "author": author,
                        "reason": reason,
                        "time_axis": "engine grid assumed (engine_duration_s)",
                    },
                )
            )

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

        evidence = []
        if cli_speed_ev is not None:
            evidence.append(cli_speed_ev)
        evidence.extend(e for e in grid_evs if e.quantity == "px_per_mm_x")
        evidence.append(
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
            )
        )
        steps = list(seg.processing) + time_steps
        if time_source == "evidence":
            steps.append(
                ProcessingStep(
                    stage="manual_scale_confirmation",
                    implementation="ecg_photo.pipeline",
                    parameters={
                        "author": author,
                        "reason": reason,
                        "previous_speed_status": "unknown",
                        "previous_gain_status": "unknown",
                        "gain_mm_mV": float(gain_mm_mV),
                        "valid_mask": "copied from observed; no extra QC applied",
                    },
                )
            )
        segments.append(
            seg.model_copy(
                update={
                    "speed_mm_s": seg_speed,
                    "speed_status": seg_speed_status,
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
                    "effective_dt_ms": effective_dt_ms,
                    "effective_dv_mV": None,
                    "px_per_mm_x": seg_px,
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


def apply_lead_corrections(manifest: Manifest, corrections: dict) -> Manifest:
    """Pure: apply rev<N>/corrections.json lead-label overrides to a manifest."""
    labels = (corrections or {}).get("lead_labels") or {}
    if not labels:
        return manifest
    segments = []
    for seg in manifest.segments:
        entry = labels.get(seg.segment_id)
        if entry is None:
            segments.append(seg)
            continue
        prev = entry.get("previous_label") or seg.lead_label
        segments.append(
            seg.model_copy(
                update={
                    "lead_label": entry["label"],
                    "lead_status": LeadStatus.confirmed,
                    "lead_evidence": (
                        f"manual correction by {entry.get('author', '?')} "
                        f"(rev {entry.get('revision', '?')}): {entry.get('reason', '')}; "
                        f"engine proposed {prev}"
                    ),
                }
            )
        )
    return manifest.model_copy(update={"segments": segments})
