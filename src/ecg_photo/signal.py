from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ecg_photo.geometry import PixelScale, require_positive_finite

GRID_TOL = 1e-9


@dataclass(frozen=True)
class GridSpec:
    working_fs_hz: float
    n_samples: int
    duration_s: float
    observed_duration_s: float
    discarded_tail_s: float


def grid_from_observed_duration(
    observed_duration_s: float, fs_hz: float, tol: float = GRID_TOL
) -> GridSpec:
    fs = require_positive_finite("fs_hz", fs_hz)
    d = float(observed_duration_s)
    if not np.isfinite(d) or d < 0.0:
        raise ValueError(
            f"observed_duration_s must be finite and >= 0, got {observed_duration_s!r}"
        )
    raw = d * fs
    n = int(np.floor(raw))
    if raw - n > 1.0 - tol:
        n += 1
    duration_s = n / fs
    tail = max(d - duration_s, 0.0)
    assert 0.0 <= tail < 1.0 / fs + tol, f"tail {tail} out of bounds"
    return GridSpec(
        working_fs_hz=fs,
        n_samples=n,
        duration_s=duration_s,
        observed_duration_s=d,
        discarded_tail_s=tail,
    )


def sample_times_s(grid: GridSpec) -> npt.NDArray[np.float64]:
    return np.arange(grid.n_samples, dtype=np.float64) / grid.working_fs_hz


@dataclass
class RawPath:
    x_px: npt.NDArray[np.float64]
    y_px: npt.NDArray[np.float64]
    support: npt.NDArray[np.bool_]

    def __post_init__(self) -> None:
        x = np.asarray(self.x_px, dtype=np.float64)
        y = np.asarray(self.y_px, dtype=np.float64)
        s = np.asarray(self.support, dtype=np.bool_)
        if x.ndim != 1 or y.shape != x.shape or s.shape != x.shape:
            raise ValueError("x_px, y_px and support must be 1-D arrays of equal length")
        if x.size > 1 and not np.all(np.diff(x) > 0):
            raise ValueError("x_px must be strictly increasing")
        self.x_px = x
        self.y_px = y
        self.support = s


@dataclass
class GriddedTrace:
    values: npt.NDArray[np.float64]
    observed: npt.NDArray[np.bool_]
    valid: npt.NDArray[np.bool_]
    gap_fill: npt.NDArray[np.bool_]


def observed_duration_from_path(path: RawPath, scale: PixelScale, speed_mm_s: float) -> float:
    s = require_positive_finite("speed_mm_s", speed_mm_s)
    px = require_positive_finite("px_per_mm_x", scale.px_per_mm_x)
    return float(path.x_px[-1] - path.x_px[0]) / (px * s)


def resample_raw_path(
    path: RawPath,
    scale: PixelScale,
    speed_mm_s: float,
    x0_px: float,
    y0_px: float,
    grid: GridSpec,
    gain_mm_mV: float | None,
) -> GriddedTrace:
    s = require_positive_finite("speed_mm_s", speed_mm_s)
    px = require_positive_finite("px_per_mm_x", scale.px_per_mm_x)
    t = sample_times_s(grid)
    x_k = float(x0_px) + t * px * s
    n = grid.n_samples
    values = np.full(n, np.nan, dtype=np.float64)
    observed = np.zeros(n, dtype=np.bool_)

    x = path.x_px
    if n > 0 and x.size >= 2:
        idx = np.searchsorted(x, x_k, side="right") - 1
        in_range = (idx >= 0) & (idx < x.size - 1) & (x_k <= x[-1])
        for k in np.nonzero(in_range)[0]:
            i = int(idx[k])
            if not (path.support[i] and path.support[i + 1]):
                continue
            x_a, x_b = x[i], x[i + 1]
            frac = (x_k[k] - x_a) / (x_b - x_a)
            y_interp = path.y_px[i] + frac * (path.y_px[i + 1] - path.y_px[i])
            if gain_mm_mV is None:
                values[k] = float(y0_px) - y_interp
            else:
                g = require_positive_finite("gain_mm_mV", gain_mm_mV)
                py = require_positive_finite("px_per_mm_y", scale.px_per_mm_y)
                values[k] = (float(y0_px) - y_interp) / (py * g)
            observed[k] = True

    return GriddedTrace(
        values=values,
        observed=observed,
        valid=observed.copy(),
        gap_fill=np.zeros(n, dtype=np.bool_),
    )
