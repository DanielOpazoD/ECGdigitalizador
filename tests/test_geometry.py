import numpy as np
import pytest

from ecg_photo.geometry import (
    FiducialOrderError,
    PixelScale,
    ScaleError,
    dt_ms_per_px,
    dv_mV_per_px,
    interval_ms,
    require_positive_finite,
    x_to_t_s,
    y_to_mV,
)

SCALE_10 = PixelScale(px_per_mm_x=10.0, px_per_mm_y=10.0)


def test_T01_dt_25mm_s_10px_mm() -> None:
    assert dt_ms_per_px(SCALE_10, 25.0) == pytest.approx(4.0)


def test_T02_dt_50mm_s_10px_mm() -> None:
    assert dt_ms_per_px(SCALE_10, 50.0) == pytest.approx(2.0)


def test_T03_rise_100px_is_1mV() -> None:
    assert dv_mV_per_px(SCALE_10, 10.0) == pytest.approx(0.01)
    v = y_to_mV(np.array([100.0]), 200.0, SCALE_10, 10.0)
    assert v[0] == pytest.approx(1.0)


def test_T04_image_y_down_positive_up() -> None:
    v = y_to_mV(np.array([100.0]), 200.0, SCALE_10, 10.0)
    assert v[0] > 0.0
    v_below = y_to_mV(np.array([300.0]), 200.0, SCALE_10, 10.0)
    assert v_below[0] < 0.0


def test_T10_x_translation_invariant() -> None:
    a = interval_ms(100.0, 350.0, SCALE_10, 25.0)
    b = interval_ms(1100.0, 1350.0, SCALE_10, 25.0)
    assert a == pytest.approx(b)


def test_T11_double_xy_and_scale_invariant() -> None:
    t1 = x_to_t_s(np.array([100.0, 350.0]), 100.0, SCALE_10, 25.0)
    s2 = PixelScale(px_per_mm_x=20.0, px_per_mm_y=20.0)
    t2 = x_to_t_s(np.array([200.0, 700.0]), 200.0, s2, 25.0)
    np.testing.assert_allclose(t1, t2)
    v1 = y_to_mV(np.array([100.0]), 200.0, SCALE_10, 10.0)
    v2 = y_to_mV(np.array([200.0]), 400.0, s2, 10.0)
    assert v2[0] == pytest.approx(v1[0])


def test_T12_rescale_only_x_updates_px_mm() -> None:
    s2 = PixelScale(px_per_mm_x=20.0, px_per_mm_y=10.0)
    t = x_to_t_s(np.array([700.0]), 200.0, s2, 25.0)
    assert t[0] == pytest.approx(1.0)
    v = y_to_mV(np.array([100.0]), 200.0, s2, 10.0)
    assert v[0] == pytest.approx(1.0)


def test_T13_speed_doubled_times_halved() -> None:
    x = np.array([100.0, 350.0])
    t25 = x_to_t_s(x, 100.0, SCALE_10, 25.0)
    t50 = x_to_t_s(x, 100.0, SCALE_10, 50.0)
    np.testing.assert_allclose(t50, t25 / 2.0)
    v25 = y_to_mV(np.array([100.0]), 200.0, SCALE_10, 10.0)
    v50 = y_to_mV(np.array([100.0]), 200.0, SCALE_10, 10.0)
    assert v25[0] == pytest.approx(v50[0])


def test_T14_gain_doubled_amplitudes_halved() -> None:
    v10 = y_to_mV(np.array([100.0]), 200.0, SCALE_10, 10.0)
    v20 = y_to_mV(np.array([100.0]), 200.0, SCALE_10, 20.0)
    assert v20[0] == pytest.approx(v10[0] / 2.0)


def test_T52_invalid_scales_raise() -> None:
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ScaleError):
            require_positive_finite("x", bad)
        with pytest.raises(ScaleError):
            dt_ms_per_px(SCALE_10, bad)
        with pytest.raises(ScaleError):
            dv_mV_per_px(SCALE_10, bad)
        with pytest.raises(ScaleError):
            dt_ms_per_px(PixelScale(bad, 10.0), 25.0)
        with pytest.raises(ScaleError):
            dv_mV_per_px(PixelScale(10.0, bad), 10.0)


def test_interval_order() -> None:
    with pytest.raises(FiducialOrderError):
        interval_ms(350.0, 100.0, SCALE_10, 25.0)
    assert interval_ms(350.0, 350.0, SCALE_10, 25.0) == 0.0
