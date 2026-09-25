import numpy as np
import pytest

from ecg_photo.contracts import TemporalTraceUnits, load_manifest, validate_revision_dir
from ecg_photo.fixtures import (
    DEFAULT_BEAT,
    DEFAULT_GAIN_MM_MV,
    DEFAULT_SCALE,
    DEFAULT_SPEED_MM_S,
    write_fixture_revision,
)
from ecg_photo.measure import rr_allowed_between_segments
from ecg_photo.signal import GridSpec, sample_times_s


@pytest.mark.parametrize("kind", ["calibrated", "gap", "gain_unknown", "time_unknown", "tail_2503"])
def test_fixture_kinds_validate(tmp_path, kind: str) -> None:
    root = tmp_path / kind
    write_fixture_revision(root, kind)
    m = load_manifest(root / "manifest.json")
    assert validate_revision_dir(root, m, strict=True) == []


def test_calibrated_fiducials_and_r_peak(tmp_path) -> None:
    root = tmp_path / "cal"
    m = write_fixture_revision(root, "calibrated")
    seg = m.segments[0]
    assert seg.n_samples == 1500
    assert seg.observed_duration_s == pytest.approx(3.0)
    assert seg.duration_s == pytest.approx(3.0)
    sig = np.load(root / seg.signal_path)
    obs = np.load(root / seg.observed_mask_path)
    assert obs.all()
    fs = seg.working_fs_hz
    grid = GridSpec(
        fs, seg.n_samples, seg.duration_s, seg.observed_duration_s, seg.discarded_tail_s
    )
    t = sample_times_s(grid)
    px_mm_s = DEFAULT_SCALE.px_per_mm_x * DEFAULT_SPEED_MM_S
    for fid_px in (DEFAULT_BEAT.p_onset_px, DEFAULT_BEAT.qrs_onset_px, DEFAULT_BEAT.t_offset_px):
        t_fid = fid_px / px_mm_s
        k = int(np.argmin(np.abs(t - t_fid)))
        assert abs(t[k] - t_fid) <= 1.0 / fs
    t_r = DEFAULT_BEAT.r_peak_px / px_mm_s
    k = int(np.argmin(np.abs(t - t_r)))
    assert sig[k] == pytest.approx(1.0, abs=0.02)
    assert np.nanmax(sig) == pytest.approx(1.0, abs=0.05)


def test_gap_fixture_masks(tmp_path) -> None:
    root = tmp_path / "gap"
    m = write_fixture_revision(root, "gap")
    seg = m.segments[0]
    obs = np.load(root / seg.observed_mask_path)
    sig = np.load(root / seg.signal_path)
    assert not obs.all()
    assert np.isnan(sig[~obs]).all()
    assert np.isfinite(sig[obs]).all()
    gap_fill = np.load(root / seg.gap_fill_mask_path)
    assert not gap_fill.any()
    valid = np.load(root / seg.valid_mask_path)
    assert np.all(valid <= obs)
    fs = seg.working_fs_hz
    grid = GridSpec(
        fs, seg.n_samples, seg.duration_s, seg.observed_duration_s, seg.discarded_tail_s
    )
    t = sample_times_s(grid)
    t_lo = (420.0 - 100.0) / 250.0
    t_hi = (470.0 - 100.0) / 250.0
    assert not obs[(t >= t_lo) & (t <= t_hi)].any()
    assert obs[t < t_lo - 0.01].all()


def test_T56_gain_unknown(tmp_path) -> None:
    root = tmp_path / "gu"
    m = write_fixture_revision(root, "gain_unknown")
    seg = m.segments[0]
    assert seg.signal_path is None
    assert seg.units is None
    assert seg.temporal_trace_path is not None
    assert seg.temporal_trace_units == TemporalTraceUnits.px
    assert seg.working_fs_hz is not None
    assert seg.gain_mm_mV is None
    trace = np.load(root / seg.temporal_trace_path)
    assert trace.shape == (seg.n_samples,)


def test_T57_time_unknown(tmp_path) -> None:
    root = tmp_path / "tu"
    m = write_fixture_revision(root, "time_unknown")
    seg = m.segments[0]
    for f in (
        seg.working_fs_hz,
        seg.n_samples,
        seg.duration_s,
        seg.observed_duration_s,
        seg.discarded_tail_s,
        seg.signal_path,
        seg.temporal_trace_path,
        seg.observed_mask_path,
        seg.units,
    ):
        assert f is None
    assert seg.gain_mm_mV == DEFAULT_GAIN_MM_MV
    assert seg.gain_status == "confirmed"
    raw = np.load(root / seg.raw_path)
    assert raw.ndim == 2 and raw.shape[1] == 2


def test_T60_tail_fixture(tmp_path) -> None:
    root = tmp_path / "tail"
    m = write_fixture_revision(root, "tail_2503")
    seg = m.segments[0]
    assert seg.n_samples == 1251
    assert seg.duration_s == pytest.approx(2.502, abs=1e-9)
    assert seg.observed_duration_s == pytest.approx(2.503, abs=1e-9)
    assert seg.discarded_tail_s == pytest.approx(0.001, abs=1e-9)


def test_T17_rr_between_segments(tmp_path) -> None:
    m = write_fixture_revision(tmp_path / "a", "calibrated")
    seg = m.segments[0]
    other = seg.model_copy(update={"segment_id": "seg-II-02"})
    assert not rr_allowed_between_segments(seg, other)
    assert rr_allowed_between_segments(seg, seg)
