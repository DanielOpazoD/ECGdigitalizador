from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


class ScaleError(ValueError):
    pass


class FiducialOrderError(ValueError):
    pass


@dataclass(frozen=True)
class PixelScale:
    px_per_mm_x: float
    px_per_mm_y: float


@dataclass(frozen=True)
class Calibration:
    speed_mm_s: float | None
    gain_mm_mV: float | None


def require_positive_finite(name: str, value: float) -> float:
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise ScaleError(f"{name} must be positive and finite, got {value!r}")
    return v


def _check_scale(scale: PixelScale) -> PixelScale:
    require_positive_finite("px_per_mm_x", scale.px_per_mm_x)
    require_positive_finite("px_per_mm_y", scale.px_per_mm_y)
    return scale


def dt_ms_per_px(scale: PixelScale, speed_mm_s: float) -> float:
    s = require_positive_finite("speed_mm_s", speed_mm_s)
    px = require_positive_finite("px_per_mm_x", _check_scale(scale).px_per_mm_x)
    return 1000.0 / (px * s)


def dv_mV_per_px(scale: PixelScale, gain_mm_mV: float) -> float:
    g = require_positive_finite("gain_mm_mV", gain_mm_mV)
    py = require_positive_finite("px_per_mm_y", _check_scale(scale).px_per_mm_y)
    return 1.0 / (py * g)


def x_to_t_s(
    x_px: npt.NDArray[np.float64], x0_px: float, scale: PixelScale, speed_mm_s: float
) -> npt.NDArray[np.float64]:
    s = require_positive_finite("speed_mm_s", speed_mm_s)
    px = require_positive_finite("px_per_mm_x", _check_scale(scale).px_per_mm_x)
    return (np.asarray(x_px, dtype=np.float64) - float(x0_px)) / (px * s)


def y_to_mV(
    y_px: npt.NDArray[np.float64], y0_px: float, scale: PixelScale, gain_mm_mV: float
) -> npt.NDArray[np.float64]:
    g = require_positive_finite("gain_mm_mV", gain_mm_mV)
    py = require_positive_finite("px_per_mm_y", _check_scale(scale).px_per_mm_y)
    return (float(y0_px) - np.asarray(y_px, dtype=np.float64)) / (py * g)


def interval_ms(x_start_px: float, x_end_px: float, scale: PixelScale, speed_mm_s: float) -> float:
    if x_end_px < x_start_px:
        raise FiducialOrderError(f"x_end_px {x_end_px} precedes x_start_px {x_start_px}")
    dt = dt_ms_per_px(scale, speed_mm_s)
    return (float(x_end_px) - float(x_start_px)) * dt
