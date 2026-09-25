from collections.abc import Sequence
from pathlib import Path

import numpy as np
import wfdb  # type: ignore[import-untyped]
from PIL import Image
from pypdf import PdfReader

from ecg_photo.contracts import (
    Manifest,
    ReasonCode,
    Segment,
    TemporalTraceUnits,
    TimeRelation,
    dump_json,
)
from ecg_photo.geometry import require_positive_finite
from ecg_photo.signal import GridSpec, sample_times_s


class ExportNotAllowed(ValueError):
    def __init__(self, reason: ReasonCode, detail: str = "") -> None:
        super().__init__(f"export not allowed: {reason} {detail}".strip())
        self.reason = reason


def export_json(manifest: Manifest, out_path: Path) -> None:
    dump_json(manifest, out_path)


def _signal_and_units(root: Path, seg: Segment) -> tuple[np.ndarray, np.ndarray, str]:
    if not seg.time_known():
        raise ExportNotAllowed(ReasonCode.TIME_SCALE_UNKNOWN)
    assert seg.n_samples is not None
    observed = (
        np.load(Path(root) / seg.observed_mask_path, allow_pickle=False)
        if seg.observed_mask_path
        else np.ones(seg.n_samples, dtype=np.bool_)
    )
    if seg.signal_path is not None:
        return np.load(Path(root) / seg.signal_path, allow_pickle=False), observed, "mV"
    if seg.temporal_trace_path is not None:
        units = (
            seg.temporal_trace_units.value
            if seg.temporal_trace_units is not None
            else TemporalTraceUnits.px.value
        )
        return (
            np.load(Path(root) / seg.temporal_trace_path, allow_pickle=False),
            observed,
            units,
        )
    raise ExportNotAllowed(ReasonCode.TIME_SCALE_UNKNOWN)


def _null(v: float | None) -> str:
    return "null" if v is None else repr(v)


def export_csv(root: Path, seg: Segment, out_path: Path) -> None:
    values, observed, units = _signal_and_units(root, seg)
    assert seg.working_fs_hz is not None
    assert seg.n_samples is not None
    assert seg.duration_s is not None
    assert seg.observed_duration_s is not None
    assert seg.discarded_tail_s is not None
    grid = GridSpec(
        working_fs_hz=seg.working_fs_hz,
        n_samples=seg.n_samples,
        duration_s=seg.duration_s,
        observed_duration_s=seg.observed_duration_s,
        discarded_tail_s=seg.discarded_tail_s,
    )
    t = sample_times_s(grid)
    valid = (
        np.load(Path(root) / seg.valid_mask_path, allow_pickle=False)
        if seg.valid_mask_path
        else observed.copy()
    )
    gap_fill = (
        np.load(Path(root) / seg.gap_fill_mask_path, allow_pickle=False)
        if seg.gap_fill_mask_path
        else np.zeros(seg.n_samples, dtype=np.bool_)
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# segment_id={seg.segment_id}\n")
        f.write(f"# working_fs_hz={_null(seg.working_fs_hz)}\n")
        f.write(f"# speed_mm_s={_null(seg.speed_mm_s)}\n")
        f.write(f"# gain_mm_mV={_null(seg.gain_mm_mV)}\n")
        f.write("sample_index,t_s,value,units,observed,valid,gap_fill\n")
        for k in range(seg.n_samples):
            v = "" if not observed[k] or not np.isfinite(values[k]) else repr(float(values[k]))
            f.write(
                f"{k},{t[k]:.9f},{v},{units},{bool(observed[k])},{bool(valid[k])},{bool(gap_fill[k])}\n"
            )


def export_wfdb(
    root: Path,
    manifest: Manifest,
    segment_ids: Sequence[str],
    out_dir: Path,
    record_name: str,
) -> Path:
    segs = [s for s in manifest.segments if s.segment_id in set(segment_ids)]
    if len(segs) != len(segment_ids):
        raise ExportNotAllowed(ReasonCode.LAYOUT_UNSUPPORTED, "unknown segment_id")
    for s in segs:
        if not s.time_known():
            raise ExportNotAllowed(ReasonCode.TIME_SCALE_UNKNOWN)
        if s.signal_path is None:
            raise ExportNotAllowed(ReasonCode.GAIN_UNKNOWN)
    fs_values = {s.working_fs_hz for s in segs}
    n_values = {s.n_samples for s in segs}
    if len(fs_values) != 1 or len(n_values) != 1:
        raise ExportNotAllowed(ReasonCode.RR_STRATEGY_UNVALIDATED, "grids differ")
    if len(segs) > 1:
        groups = {s.sync_group_id for s in segs}
        ok = (
            len(groups) == 1
            and None not in groups
            and all(s.time_relation == TimeRelation.simultaneous for s in segs)
        )
        if not ok:
            raise ExportNotAllowed(
                ReasonCode.RR_STRATEGY_UNVALIDATED,
                "segments not proven simultaneous",
            )
    fs = float(segs[0].working_fs_hz)  # type: ignore[arg-type]
    require_positive_finite("working_fs_hz", fs)
    cols = []
    comments = []
    for s in segs:
        assert s.signal_path is not None
        cols.append(np.load(Path(root) / s.signal_path, allow_pickle=False))
        if s.observed_mask_path:
            comments.append(f"{s.segment_id} observed_mask_path={s.observed_mask_path}")
    p_signal = np.column_stack(cols)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    k = len(segs)
    wfdb.wrsamp(
        record_name,
        fs=fs,
        units=["mV"] * k,
        sig_name=[s.lead_label for s in segs],
        p_signal=p_signal,
        fmt=["16"] * k,
        adc_gain=[1000.0] * k,
        baseline=[0] * k,
        write_dir=str(out_dir),
        comments=comments,
    )
    return out_dir / record_name


def readback_wfdb(
    path_no_ext: Path,
) -> tuple[np.ndarray, float, list[str], list[str]]:
    rec = wfdb.rdrecord(str(path_no_ext))
    return (
        np.asarray(rec.p_signal, dtype=np.float64),
        float(rec.fs),
        list(rec.sig_name),
        list(rec.units),
    )


def readback_png_scale(png_path: Path) -> tuple[int, int, float | None]:
    with Image.open(png_path) as im:
        dpi = im.info.get("dpi")
        dpi_x = float(dpi[0]) if dpi else None
        return im.width, im.height, dpi_x


def readback_pdf_page_size_mm(pdf_path: Path) -> tuple[float, float]:
    reader = PdfReader(str(pdf_path))
    box = reader.pages[0].mediabox
    w_pt = float(box.right) - float(box.left)
    h_pt = float(box.top) - float(box.bottom)
    return w_pt / 72.0 * 25.4, h_pt / 72.0 * 25.4


def readback_csv(csv_path: Path) -> dict[str, np.ndarray]:
    lines = Path(csv_path).read_text(encoding="utf-8").splitlines()
    rows = [ln for ln in lines if ln and not ln.startswith("#")]
    header = rows[0].split(",")
    cols: dict[str, list[str]] = {h: [] for h in header}
    for ln in rows[1:]:
        parts = ln.split(",")
        for h, p in zip(header, parts):
            cols[h].append(p)
    out: dict[str, np.ndarray] = {}
    for h, vals in cols.items():
        if h in ("observed", "valid", "gap_fill"):
            out[h] = np.array([v == "True" for v in vals], dtype=np.bool_)
        elif h == "units":
            out[h] = np.array(vals, dtype=object)
        elif h == "value":
            out[h] = np.array([float(v) if v else np.nan for v in vals], dtype=np.float64)
        else:
            out[h] = np.array([float(v) for v in vals], dtype=np.float64)
    return out
