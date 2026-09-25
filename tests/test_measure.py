import pytest

from ecg_photo.geometry import PixelScale
from ecg_photo.measure import (
    MeasurementInputError,
    fiducials_px_to_ms,
    hr_instant_bpm,
    hr_period_bpm,
    jt_ms,
    mean_instant_hr_bpm,
    pr_ms,
    qrs_duration_ms,
    qt_ms,
    qtc_bazett_ms,
    qtc_framingham_ms,
    qtc_framingham_s,
    qtc_fridericia_ms,
    rr_ms_from_s,
    rr_s,
    rr_s_from_ms,
)

SCALE = PixelScale(px_per_mm_x=10.0, px_per_mm_y=10.0)


def test_T05_intervals_from_px() -> None:
    f = fiducials_px_to_ms(
        {"p_onset": 100.0, "qrs_onset": 140.0, "qrs_offset": 165.0, "t_offset": 240.0},
        x0_px=0.0,
        scale=SCALE,
        speed_mm_s=25.0,
    )
    assert pr_ms(f["p_onset"], f["qrs_onset"]) == pytest.approx(160.0)
    assert qrs_duration_ms(f["qrs_onset"], f["qrs_offset"]) == pytest.approx(100.0)
    assert qt_ms(f["qrs_onset"], f["t_offset"]) == pytest.approx(400.0)


def test_T06_rr_and_hr() -> None:
    f = fiducials_px_to_ms({"r1": 100.0, "r2": 350.0}, 0.0, SCALE, 25.0)
    rr_sec = rr_s(f["r1"] / 1000.0, f["r2"] / 1000.0)
    assert rr_ms_from_s(rr_sec) == pytest.approx(1000.0)
    assert hr_instant_bpm(rr_sec) == pytest.approx(60.0)


def test_T07_qtc_bazett_fridericia() -> None:
    assert qtc_bazett_ms(400.0, 0.8) == pytest.approx(447.2136, abs=1e-3)
    assert qtc_fridericia_ms(400.0, 0.8) == pytest.approx(430.8869, abs=1e-3)


def test_T08_explicit_ms_s_conversions() -> None:
    assert rr_s_from_ms(1000.0) == pytest.approx(1.0)
    assert rr_ms_from_s(1.0) == pytest.approx(1000.0)
    with pytest.raises(MeasurementInputError):
        rr_s_from_ms(0.0)
    with pytest.raises(MeasurementInputError):
        rr_ms_from_s(-1.0)


def test_T09_period_vs_instantaneous_hr() -> None:
    assert hr_period_bpm(0.0, 2.0, 3) == pytest.approx(60.0)
    assert mean_instant_hr_bpm([0.5, 1.5]) == pytest.approx(80.0)


def test_T52_invalid_rr() -> None:
    for bad in (0.0, -0.5, float("nan"), float("inf")):
        with pytest.raises(MeasurementInputError):
            hr_instant_bpm(bad)
        with pytest.raises(MeasurementInputError):
            qtc_bazett_ms(400.0, bad)
    with pytest.raises(MeasurementInputError):
        rr_s(1.0, 1.0)
    with pytest.raises(MeasurementInputError):
        rr_s(float("nan"), 2.0)


def test_T53_invalid_fiducial_order() -> None:
    with pytest.raises(MeasurementInputError):
        qt_ms(500.0, 400.0)
    with pytest.raises(MeasurementInputError):
        qrs_duration_ms(165.0, 140.0)
    with pytest.raises(MeasurementInputError):
        pr_ms(200.0, 140.0)
    with pytest.raises(MeasurementInputError):
        jt_ms(300.0, 400.0)


def test_T55_framingham() -> None:
    assert qtc_framingham_ms(400.0, 0.8) == pytest.approx(430.8, abs=1e-6)
    assert qtc_framingham_ms(400.0, 1.0) == pytest.approx(400.0, abs=1e-6)
    assert qtc_framingham_s(0.4, 0.8) == pytest.approx(0.4308, abs=1e-6)
    assert qtc_framingham_s(0.4, 1.0) == pytest.approx(0.4, abs=1e-6)


def test_jt_and_hr_period_validations() -> None:
    assert jt_ms(400.0, 100.0) == pytest.approx(300.0)
    with pytest.raises(MeasurementInputError):
        hr_period_bpm(0.0, 1.0, 1)
    with pytest.raises(MeasurementInputError):
        hr_period_bpm(1.0, 0.0, 3)
    with pytest.raises(MeasurementInputError):
        mean_instant_hr_bpm([])
