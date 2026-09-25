from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from ecg_photo.contracts import (
    HashStatus,
    LeadStatus,
    Manifest,
    ProcessingStep,
    RepresentationKind,
    ResolutionEvidence,
    ScaleStatus,
    Segment,
    Source,
    SourceKind,
    TemporalTraceUnits,
    TimeRelation,
    dump_json,
    sha256_file,
    validate_revision_dir,
)
from ecg_photo.geometry import PixelScale, dt_ms_per_px, dv_mV_per_px
from ecg_photo.signal import (
    RawPath,
    grid_from_observed_duration,
    observed_duration_from_path,
    resample_raw_path,
)


@dataclass(frozen=True)
class SyntheticBeatSpec:
    p_onset_px: float
    qrs_onset_px: float
    qrs_offset_px: float
    t_offset_px: float
    r_peak_px: float


DEFAULT_BEAT = SyntheticBeatSpec(
    p_onset_px=100.0,
    qrs_onset_px=140.0,
    qrs_offset_px=165.0,
    t_offset_px=240.0,
    r_peak_px=155.0,
)
DEFAULT_RR_PX = 250.0
DEFAULT_SCALE = PixelScale(px_per_mm_x=10.0, px_per_mm_y=10.0)
DEFAULT_SPEED_MM_S = 25.0
DEFAULT_GAIN_MM_MV = 10.0
BASELINE_Y_PX = 200.0


def _gauss(x: npt.NDArray[np.float64], center: float, sigma: float) -> npt.NDArray[np.float64]:
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def synthetic_trajectory(
    n_beats: int,
    rr_px: float,
    beat: SyntheticBeatSpec,
    baseline_y_px: float,
    scale: PixelScale,
    gain_mm_mV: float,
    x_start_px: float,
    r_amp_mV: float = 1.0,
    p_amp_mV: float = 0.15,
    t_amp_mV: float = 0.3,
) -> RawPath:
    if n_beats < 1:
        raise ValueError("n_beats must be >= 1")
    extent_px = n_beats * rr_px
    x = x_start_px + np.arange(0.0, extent_px + 1.0, 1.0)
    px_per_mV = gain_mm_mV * scale.px_per_mm_y
    y = np.full(x.shape, baseline_y_px, dtype=np.float64)

    p_center = 0.5 * (beat.p_onset_px + beat.qrs_onset_px) - 2.0
    p_sigma = max((beat.qrs_onset_px - beat.p_onset_px) / 6.0, 1.0)
    t_center = 0.5 * (beat.qrs_offset_px + beat.t_offset_px)
    t_sigma = max((beat.t_offset_px - beat.qrs_offset_px) / 4.0, 1.0)
    r_sigma = 3.0

    for k in range(n_beats):
        off = k * rr_px
        y -= p_amp_mV * px_per_mV * _gauss(x, x_start_px + off + p_center, p_sigma)
        y -= r_amp_mV * px_per_mV * _gauss(x, x_start_px + off + beat.r_peak_px, r_sigma)
        y -= t_amp_mV * px_per_mV * _gauss(x, x_start_px + off + t_center, t_sigma)

    return RawPath(x_px=x, y_px=y, support=np.ones(x.shape, dtype=np.bool_))


def _write_support(path: Path, support: npt.NDArray[np.bool_]) -> None:
    np.savez_compressed(path, support=support)


def write_fixture_revision(
    root: Path,
    kind: Literal["calibrated", "gap", "gain_unknown", "time_unknown", "tail_2503"],
    *,
    fs_hz: float = 500.0,
) -> Manifest:
    root = Path(root)
    for sub in ("source", "raw_paths", "signals", "masks"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    scale = DEFAULT_SCALE
    speed = DEFAULT_SPEED_MM_S
    gain = DEFAULT_GAIN_MM_MV
    x0 = 100.0
    y0 = BASELINE_Y_PX
    beat = DEFAULT_BEAT

    if kind == "tail_2503":
        n_beats = 3
        extent_px = 2.503 * scale.px_per_mm_x * speed
        n_cols = int(np.ceil(extent_px)) + 1
        base = synthetic_trajectory(n_beats, DEFAULT_RR_PX, beat, y0, scale, gain, x0)
        x = x0 + np.linspace(0.0, extent_px, n_cols)
        y = np.interp(x, base.x_px, base.y_px)
        support = np.ones(n_cols, dtype=np.bool_)
        path = RawPath(x_px=x, y_px=y, support=support)
    else:
        path = synthetic_trajectory(3, DEFAULT_RR_PX, beat, y0, scale, gain, x0)
        if kind == "gap":
            support = path.support.copy()
            support[(path.x_px >= 420.0) & (path.x_px <= 470.0)] = False
            path = RawPath(x_px=path.x_px, y_px=path.y_px, support=support)

    seg_id = "seg-II-01"
    raw_rel = f"raw_paths/{seg_id}.npy"
    sup_rel = f"raw_paths/{seg_id}-support.npz"
    np.save(root / raw_rel, np.column_stack([path.x_px, path.y_px]))
    _write_support(root / sup_rel, path.support)

    src_rel = f"source/{seg_id}-source.npy"
    np.save(root / src_rel, np.column_stack([path.x_px, path.y_px]))
    sha = sha256_file(root / src_rel)

    time_known = kind != "time_unknown"
    gain_known = kind != "gain_unknown"

    signal_rel: str | None = None
    trace_rel: str | None = None
    trace_units: TemporalTraceUnits | None = None
    mask_rels: dict[str, str | None] = {
        "observed": None,
        "valid": None,
        "gap": None,
    }
    grid = None
    if time_known:
        observed_d = observed_duration_from_path(path, scale, speed)
        grid = grid_from_observed_duration(observed_d, fs_hz)
        trace = resample_raw_path(path, scale, speed, x0, y0, grid, gain if gain_known else None)
        if gain_known:
            signal_rel = f"signals/{seg_id}.npy"
            np.save(root / signal_rel, trace.values)
        else:
            trace_rel = f"signals/{seg_id}-temporal.npy"
            trace_units = TemporalTraceUnits.px
            np.save(root / trace_rel, trace.values)
        if grid.n_samples > 0:
            obs_rel = f"masks/{seg_id}-observed.npy"
            val_rel = f"masks/{seg_id}-valid.npy"
            gap_rel = f"masks/{seg_id}-gap-fill.npy"
            mask_rels["observed"] = obs_rel
            mask_rels["valid"] = val_rel
            mask_rels["gap"] = gap_rel
            np.save(root / obs_rel, trace.observed)
            np.save(root / val_rel, trace.valid)
            np.save(root / gap_rel, trace.gap_fill)

    seg = Segment(
        segment_id=seg_id,
        page_id="page-1",
        lead_label="II",
        lead_status=LeadStatus.confirmed,
        lead_evidence="synthetic_fixture",
        representation_kind=RepresentationKind.rhythm_segment,
        representation_evidence="synthetic_fixture",
        source_region=[
            (x0, 100.0),
            (float(path.x_px[-1]), 100.0),
            (float(path.x_px[-1]), 300.0),
            (x0, 300.0),
        ],
        transform_id="synthetic-identity",
        raw_path=raw_rel,
        raw_coordinate_frame="synthetic-frame-1",
        raw_support_path=sup_rel,
        time_relation=TimeRelation.unknown,
        sync_group_id=None,
        study_start_s=None,
        working_fs_hz=grid.working_fs_hz if grid else None,
        original_fs_hz=None,
        n_samples=grid.n_samples if grid else None,
        duration_s=grid.duration_s if grid else None,
        observed_duration_s=grid.observed_duration_s if grid else None,
        discarded_tail_s=grid.discarded_tail_s if grid else None,
        units="mV" if signal_rel else None,
        speed_mm_s=speed if time_known else None,
        speed_status=ScaleStatus.confirmed if time_known else ScaleStatus.unknown,
        gain_mm_mV=gain if gain_known else None,
        gain_status=ScaleStatus.confirmed if gain_known else ScaleStatus.unknown,
        effective_dt_ms=dt_ms_per_px(scale, speed) if time_known else None,
        effective_dv_mV=dv_mV_per_px(scale, gain) if gain_known else None,
        resolution_evidence=ResolutionEvidence(
            source_frame_id="synthetic-frame-1",
            method="closed_form_synthetic",
            limitations="geometric_pixel_equivalence_only",
        ),
        signal_path=signal_rel,
        temporal_trace_path=trace_rel,
        temporal_trace_units=trace_units,
        observed_mask_path=mask_rels["observed"],
        valid_mask_path=mask_rels["valid"],
        gap_fill_mask_path=mask_rels["gap"],
        processing=[
            ProcessingStep(
                stage="synthetic_fixture",
                implementation="closed_form_gaussian",
                parameters={"kind": kind, "fs_hz": fs_hz},
            )
        ],
    )

    manifest = Manifest(
        schema_version="1.0.0",
        study_id=f"fixture-{kind}",
        revision=1,
        input_revision=1,
        run_id=f"fixture-run-{kind}",
        config_hash=None,
        source=Source(
            sha256=sha,
            hash_status=HashStatus.computed,
            kind=SourceKind.image,
            page_count=1,
            original_filename="synthetic.png",
        ),
        segments=[seg],
    )

    dump_json(manifest, root / "manifest.json")
    problems = validate_revision_dir(root, manifest, strict=True)
    if problems:
        raise AssertionError(f"fixture {kind} failed validation: {problems}")
    return manifest
