import numpy as np
import pytest

from ecg_photo.geometry import PixelScale
from ecg_photo.signal import (
    RawPath,
    grid_from_observed_duration,
    observed_duration_from_path,
    resample_raw_path,
    sample_times_s,
)

SCALE = PixelScale(px_per_mm_x=10.0, px_per_mm_y=10.0)


def test_T15_grid_2_5s_500hz() -> None:
    g = grid_from_observed_duration(2.5, 500.0)
    assert g.n_samples == 1250
    assert g.duration_s == pytest.approx(2.5)
    t = sample_times_s(g)
    assert t[-1] == pytest.approx(2.498, abs=1e-9)


def test_T60_grid_2_503s() -> None:
    g = grid_from_observed_duration(2.503, 500.0)
    assert g.n_samples == 1251
    assert g.duration_s == pytest.approx(2.502, abs=1e-9)
    assert g.discarded_tail_s == pytest.approx(0.001, abs=1e-9)
    t = sample_times_s(g)
    assert t[-1] == pytest.approx(2.5, abs=1e-9)


def test_grid_rounding_and_empty() -> None:
    g = grid_from_observed_duration(0.001, 500.0)
    assert g.n_samples == 0
    g2 = grid_from_observed_duration(2.5 - 5e-13, 500.0)
    assert g2.n_samples == 1250


def _ramp_path() -> RawPath:
    x = np.arange(100.0, 851.0, 1.0)
    return RawPath(
        x_px=x,
        y_px=200.0 - (x - 100.0) * 0.01,
        support=np.ones(x.shape, dtype=np.bool_),
    )


def test_resample_linear_and_units() -> None:
    path = _ramp_path()
    d = observed_duration_from_path(path, SCALE, 25.0)
    assert d == pytest.approx(3.0)
    g = grid_from_observed_duration(d, 500.0)
    tr = resample_raw_path(path, SCALE, 25.0, 100.0, 200.0, g, 10.0)
    assert tr.values.shape == (g.n_samples,)
    assert tr.observed.all()
    assert tr.valid.all()
    assert not tr.gap_fill.any()
    t = sample_times_s(g)
    k = int(np.argmin(np.abs(t - 1.0)))
    assert tr.values[k] == pytest.approx(0.025, abs=1e-3)


def test_T16_resample_two_grids_agree() -> None:
    path = _ramp_path()
    d = observed_duration_from_path(path, SCALE, 25.0)
    x_fid = 350.0
    t_expected = (x_fid - 100.0) / (10.0 * 25.0)
    for fs in (500.0, 1000.0):
        g = grid_from_observed_duration(d, fs)
        tr = resample_raw_path(path, SCALE, 25.0, 100.0, 200.0, g, 10.0)
        t = sample_times_s(g)
        k = int(np.argmin(np.abs(t - t_expected)))
        assert abs(t[k] - t_expected) <= 1.0 / fs
        assert tr.observed[k]
    g500 = grid_from_observed_duration(d, 500.0)
    g1000 = grid_from_observed_duration(d, 1000.0)
    assert g1000.n_samples == 2 * g500.n_samples


def test_gap_no_interpolation() -> None:
    path = _ramp_path()
    support = path.support.copy()
    support[(path.x_px >= 420.0) & (path.x_px <= 470.0)] = False
    path = RawPath(x_px=path.x_px, y_px=path.y_px, support=support)
    g = grid_from_observed_duration(observed_duration_from_path(path, SCALE, 25.0), 500.0)
    tr = resample_raw_path(path, SCALE, 25.0, 100.0, 200.0, g, 10.0)
    t = sample_times_s(g)
    t_gap_start = (419.0 - 100.0) / 250.0
    t_gap_end = (470.5 - 100.0) / 250.0
    inside = (t >= t_gap_start) & (t <= t_gap_end)
    assert not tr.observed[inside].any()
    assert np.isnan(tr.values[inside]).all()
    assert np.isfinite(tr.values[~inside]).all()
    assert tr.observed[~inside].all()
    assert not tr.gap_fill.any()
    assert np.all(tr.valid <= tr.observed)


def test_px_units_when_gain_none() -> None:
    path = _ramp_path()
    g = grid_from_observed_duration(observed_duration_from_path(path, SCALE, 25.0), 500.0)
    tr = resample_raw_path(path, SCALE, 25.0, 100.0, 200.0, g, None)
    k = 500
    expected_px = 200.0 - (200.0 - (path.x_px[0] + k * 0.5 - 100.0) * 0.01)
    assert tr.values[k] == pytest.approx(expected_px, abs=1e-6)


def test_raw_path_validation() -> None:
    with pytest.raises(ValueError):
        RawPath(
            x_px=np.array([2.0, 1.0]),
            y_px=np.array([0.0, 0.0]),
            support=np.ones(2, dtype=np.bool_),
        )
    with pytest.raises(ValueError):
        RawPath(
            x_px=np.array([1.0, 2.0]),
            y_px=np.array([0.0]),
            support=np.ones(2, dtype=np.bool_),
        )
