from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np

from ecg_photo.contracts import (
    Measurement,
    MeasurementStatus,
    QualityLabel,
    ReasonCode,
    Segment,
    SupportValue,
)
from ecg_photo.geometry import PixelScale, dt_ms_per_px


class MeasurementInputError(ValueError):
    pass


def _finite(name: str, value: float) -> float:
    v = float(value)
    if not np.isfinite(v):
        raise MeasurementInputError(f"{name} must be finite, got {value!r}")
    return v


def _positive(name: str, value: float) -> float:
    v = _finite(name, value)
    if v <= 0.0:
        raise MeasurementInputError(f"{name} must be > 0, got {value!r}")
    return v


def rr_s(t_qrs_k_s: float, t_qrs_k1_s: float) -> float:
    a = _finite("t_qrs_k_s", t_qrs_k_s)
    b = _finite("t_qrs_k1_s", t_qrs_k1_s)
    if b <= a:
        raise MeasurementInputError("t_qrs_k1_s must be after t_qrs_k_s")
    return b - a


def rr_ms_from_s(rr_seconds: float) -> float:
    return _positive("rr_s", rr_seconds) * 1000.0


def rr_s_from_ms(rr_ms: float) -> float:
    return _positive("rr_ms", rr_ms) / 1000.0


def hr_instant_bpm(rr_seconds: float) -> float:
    return 60.0 / _positive("rr_s", rr_seconds)


def hr_period_bpm(t_first_s: float, t_last_s: float, n_qrs: int) -> float:
    a = _finite("t_first_s", t_first_s)
    b = _finite("t_last_s", t_last_s)
    if n_qrs < 2:
        raise MeasurementInputError("n_qrs must be >= 2")
    if b <= a:
        raise MeasurementInputError("t_last_s must be after t_first_s")
    return 60.0 * (n_qrs - 1) / (b - a)


def mean_instant_hr_bpm(rr_list_s: Sequence[float]) -> float:
    if not rr_list_s:
        raise MeasurementInputError("rr_list_s must be non-empty")
    return float(np.mean([hr_instant_bpm(r) for r in rr_list_s]))


def pr_ms(p_onset_ms: float, qrs_onset_ms: float) -> float:
    p = _finite("p_onset_ms", p_onset_ms)
    q = _finite("qrs_onset_ms", qrs_onset_ms)
    if q <= p:
        raise MeasurementInputError("qrs_onset_ms must be after p_onset_ms")
    return q - p


def qrs_duration_ms(qrs_onset_ms: float, qrs_offset_ms: float) -> float:
    a = _finite("qrs_onset_ms", qrs_onset_ms)
    b = _finite("qrs_offset_ms", qrs_offset_ms)
    if b <= a:
        raise MeasurementInputError("qrs_offset_ms must be after qrs_onset_ms")
    return b - a


def qt_ms(qrs_onset_ms: float, t_offset_ms: float) -> float:
    a = _finite("qrs_onset_ms", qrs_onset_ms)
    b = _finite("t_offset_ms", t_offset_ms)
    if b <= a:
        raise MeasurementInputError("t_offset_ms must be after qrs_onset_ms")
    return b - a


def jt_ms(qt_ms_value: float, qrs_ms: float) -> float:
    qt = _positive("qt_ms", qt_ms_value)
    qrs = _positive("qrs_ms", qrs_ms)
    if qt < qrs:
        raise MeasurementInputError("qt_ms must be >= qrs_ms")
    return qt - qrs


def qtc_bazett_ms(qt_ms_value: float, rr_seconds: float) -> float:
    return _positive("qt_ms", qt_ms_value) / float(np.sqrt(_positive("rr_s", rr_seconds)))


def qtc_fridericia_ms(qt_ms_value: float, rr_seconds: float) -> float:
    return _positive("qt_ms", qt_ms_value) / float(_positive("rr_s", rr_seconds) ** (1.0 / 3.0))


def qtc_framingham_ms(qt_ms_value: float, rr_seconds: float) -> float:
    return _positive("qt_ms", qt_ms_value) + 154.0 * (1.0 - _positive("rr_s", rr_seconds))


def qtc_framingham_s(qt_s: float, rr_seconds: float) -> float:
    return _positive("qt_s", qt_s) + 0.154 * (1.0 - _positive("rr_s", rr_seconds))


def fiducials_px_to_ms(
    x_px_values: Mapping[str, float],
    x0_px: float,
    scale: PixelScale,
    speed_mm_s: float,
) -> dict[str, float]:
    dt = dt_ms_per_px(scale, speed_mm_s)
    out: dict[str, float] = {}
    for name, x in x_px_values.items():
        out[name] = (_finite(name, x) - float(x0_px)) * dt
    return out


def unavailable(
    name: str,
    unit: Literal["ms", "bpm", "mV", "s"],
    reason_codes: list[ReasonCode],
    measurement_id: str | None = None,
    method: str = "f1_core",
    **support: SupportValue,
) -> Measurement:
    if not reason_codes:
        raise MeasurementInputError("unavailable requires at least one reason code")
    return Measurement(
        measurement_id=measurement_id or f"unavailable-{name}",
        revision=1,
        name=name,
        value=None,
        unit=unit,
        status=MeasurementStatus.unavailable,
        reason_codes=reason_codes,
        method=method,
        support=dict(support),
        quality_label=QualityLabel.insufficient,
    )


def rr_allowed_between_segments(a: Segment, b: Segment) -> bool:
    return a.segment_id == b.segment_id
