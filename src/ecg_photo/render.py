from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ecg_photo.contracts import ReasonCode, Segment
from ecg_photo.geometry import require_positive_finite
from ecg_photo.signal import GridSpec, sample_times_s


class RenderNotAllowed(ValueError):
    def __init__(self, reason: ReasonCode) -> None:
        super().__init__(f"render not allowed: {reason}")
        self.reason = reason


@dataclass(frozen=True)
class PaperSpec:
    speed_mm_s: float
    gain_mm_mV: float
    dpi: float
    margin_mm: float = 10.0
    grid: bool = True


@dataclass(frozen=True)
class RenderInfo:
    width_px: int
    height_px: int
    dpi: float
    px_per_mm: float
    x0_px: float
    y0_px: float
    speed_mm_s: float
    gain_mm_mV: float
    page_width_mm: float
    page_height_mm: float


def _require_renderable(root: Path, seg: Segment, paper: PaperSpec) -> np.ndarray:
    if not seg.time_known():
        raise RenderNotAllowed(ReasonCode.TIME_SCALE_UNKNOWN)
    if seg.signal_path is None:
        raise RenderNotAllowed(ReasonCode.GAIN_UNKNOWN)
    require_positive_finite("speed_mm_s", paper.speed_mm_s)
    require_positive_finite("gain_mm_mV", paper.gain_mm_mV)
    require_positive_finite("dpi", paper.dpi)
    require_positive_finite("margin_mm", paper.margin_mm)
    values = np.load(Path(root) / seg.signal_path, allow_pickle=False)
    return values


def _build_figure(
    root: Path, seg: Segment, values: np.ndarray, paper: PaperSpec
) -> tuple[plt.Figure, RenderInfo]:
    assert seg.duration_s is not None
    assert seg.working_fs_hz is not None
    assert seg.n_samples is not None
    assert seg.observed_duration_s is not None
    assert seg.discarded_tail_s is not None

    page_width_mm = paper.margin_mm * 2.0 + seg.duration_s * paper.speed_mm_s
    page_height_mm = paper.margin_mm * 2.0 + 40.0
    px_per_mm = paper.dpi / 25.4
    width_px = round(page_width_mm * px_per_mm)
    height_px = round(page_height_mm * px_per_mm)

    fig = plt.figure(figsize=(page_width_mm / 25.4, page_height_mm / 25.4), dpi=paper.dpi)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0.0, page_width_mm)
    ax.set_ylim(0.0, page_height_mm)
    ax.axis("off")

    if paper.grid:
        for x in np.arange(0.0, page_width_mm + 1e-9, 1.0):
            major = abs(x % 5.0) < 1e-6 or abs((x % 5.0) - 5.0) < 1e-6
            ax.axvline(
                x, color="#f4b8b8" if major else "#fbe3e3", lw=0.8 if major else 0.4, zorder=1
            )
        for y in np.arange(0.0, page_height_mm + 1e-9, 1.0):
            major = abs(y % 5.0) < 1e-6 or abs((y % 5.0) - 5.0) < 1e-6
            ax.axhline(
                y, color="#f4b8b8" if major else "#fbe3e3", lw=0.8 if major else 0.4, zorder=1
            )

    y0_mm = paper.margin_mm + 20.0
    pulse_x0 = paper.margin_mm * 0.4
    pulse_w = 0.2 * paper.speed_mm_s
    pulse_h = 1.0 * paper.gain_mm_mV
    ax.plot(
        [pulse_x0, pulse_x0, pulse_x0 + pulse_w, pulse_x0 + pulse_w],
        [y0_mm, y0_mm + pulse_h, y0_mm + pulse_h, y0_mm],
        color="black",
        lw=1.2,
        zorder=3,
    )

    if seg.observed_mask_path:
        observed = np.load(Path(root) / seg.observed_mask_path, allow_pickle=False)
    else:
        observed = np.ones(seg.n_samples, dtype=np.bool_)
    grid = GridSpec(
        working_fs_hz=seg.working_fs_hz,
        n_samples=seg.n_samples,
        duration_s=seg.duration_s,
        observed_duration_s=seg.observed_duration_s,
        discarded_tail_s=seg.discarded_tail_s,
    )
    t = sample_times_s(grid)
    plot_values = np.where(observed, values, np.nan)
    ax.plot(
        paper.margin_mm + t * paper.speed_mm_s,
        y0_mm + plot_values * paper.gain_mm_mV,
        color="black",
        lw=0.8,
        zorder=2,
    )

    info = RenderInfo(
        width_px=width_px,
        height_px=height_px,
        dpi=paper.dpi,
        px_per_mm=px_per_mm,
        x0_px=paper.margin_mm * px_per_mm,
        y0_px=height_px - y0_mm * px_per_mm,
        speed_mm_s=paper.speed_mm_s,
        gain_mm_mV=paper.gain_mm_mV,
        page_width_mm=page_width_mm,
        page_height_mm=page_height_mm,
    )
    return fig, info


def render_segment_png(root: Path, seg: Segment, out_path: Path, paper: PaperSpec) -> RenderInfo:
    values = _require_renderable(root, seg, paper)
    fig, info = _build_figure(root, seg, values, paper)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path,
        dpi=paper.dpi,
        pil_kwargs={"dpi": (paper.dpi, paper.dpi)},
    )
    plt.close(fig)
    return info


def render_segment_pdf(root: Path, seg: Segment, out_path: Path, paper: PaperSpec) -> RenderInfo:
    values = _require_renderable(root, seg, paper)
    fig, info = _build_figure(root, seg, values, paper)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    return info


def t_to_x_px(info: RenderInfo, t_s: float) -> float:
    return info.x0_px + t_s * info.speed_mm_s * info.px_per_mm


def v_to_y_px(info: RenderInfo, v_mV: float) -> float:
    return info.y0_px - v_mV * info.gain_mm_mV * info.px_per_mm
