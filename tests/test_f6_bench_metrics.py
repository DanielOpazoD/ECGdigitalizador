import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from metrics_common import segment_metrics, trim_to_observed


def test_segment_metrics_lag_and_amplitude() -> None:
    fs = 500.0
    t = np.arange(int(10 * fs)) / fs
    truth = np.sin(2 * np.pi * 1.2 * t) + 0.2 * np.sin(2 * np.pi * 5 * t)
    # est covers truth[1250:2500] (2.5 s slot starting at 2.5 s), no shift
    est = truth[1250:2500].copy()
    observed = np.ones(len(est), dtype=bool)

    m = segment_metrics(
        truth,
        est,
        observed,
        est_fs=fs,
        truth_fs=fs,
        observed_duration_s=2.5,
        expected_window_s=2.5,
    )
    assert m["status"] == "ok"
    assert m["best_lag_s"] == 2.5
    assert m["r_at_lag"] > 0.999
    assert abs(m["amp_ratio"] - 1.0) < 0.02
    assert m["rmse_mV"] < 1e-9
    assert m["offset_err_s"] == 0.0
    assert m["duration_err_pct"] == 0.0


def test_segment_metrics_mask_excludes_gap() -> None:
    fs = 500.0
    t = np.arange(int(10 * fs)) / fs
    truth = np.sin(2 * np.pi * 1.2 * t) + 0.2 * np.sin(2 * np.pi * 5 * t)
    est = truth[1250:2500].copy()
    observed = np.ones(len(est), dtype=bool)
    # corrupt a region but mark it unobserved: metrics must ignore it
    est[200:400] = 99.0
    observed[200:400] = False
    est[200:400] = np.nan

    m = segment_metrics(
        truth,
        est,
        observed,
        est_fs=fs,
        truth_fs=fs,
        observed_duration_s=2.1,
        expected_window_s=2.5,
    )
    assert m["status"] == "ok"
    assert m["coverage"] == 0.84
    assert m["r_at_lag"] > 0.999
    assert m["best_lag_s"] == 2.5


def test_segment_metrics_no_signal() -> None:
    fs = 500.0
    truth = np.zeros(int(10 * fs))
    est = np.full(100, np.nan)
    m = segment_metrics(
        truth,
        est,
        np.zeros(100, bool),
        est_fs=fs,
        truth_fs=fs,
        observed_duration_s=None,
        expected_window_s=2.5,
    )
    assert m["status"] == "no_signal"


def test_segment_metrics_snr_ignores_vertical_offset() -> None:
    fs = 500.0
    t = np.arange(int(10 * fs)) / fs
    truth = np.sin(2 * np.pi * 1.2 * t)
    est = truth[1250:2500] + 0.7  # baseline offset only -> removed
    rng = np.random.default_rng(0)
    noisy = est + rng.normal(0, 0.1, len(est))  # noise power 0.01 vs signal ~0.5
    obs = np.ones(len(est), dtype=bool)
    clean = segment_metrics(truth, est, obs, fs, fs, 2.5, 2.5)
    m = segment_metrics(truth, noisy, obs, fs, fs, 2.5, 2.5)
    assert clean["snr_db"] is None or clean["snr_db"] > 60
    assert 14.0 < m["snr_db"] < 20.0


def _slot_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Kaggle layout: truth defined only in its 2.5 s slot (starting at 2.5 s);
    estimate on the engine's 10 s page canvas, observed only in that slot."""
    fs = 1000.0
    t = np.arange(int(10 * fs)) / fs
    page = np.sin(2 * np.pi * 1.2 * t) + 0.2 * np.sin(2 * np.pi * 5 * t)
    truth_slot = page[2500:5000].copy()
    est = np.full(len(page), np.nan)
    est[2500:5000] = page[2500:5000]
    return truth_slot, est, np.isfinite(est), fs


def test_segment_metrics_no_overlap_is_a_case_not_a_crash() -> None:
    # real failure (F6-b, record 1512936796, ahus, engine axis): no lag overlapped,
    # truth was clamped to a constant and log10(0) raised "math domain error"
    truth_slot, est, obs, fs = _slot_case()
    m = segment_metrics(truth_slot, est, obs, fs, fs, 10.0, 2.5)
    assert m["status"] == "no_valid_lag"
    assert np.isnan(m["r_at_lag"])


def test_trim_to_observed_is_placement_agnostic() -> None:
    truth_slot, est, _obs, fs = _slot_case()
    # Ahus-like: lead at its page slot (2.5 s); ECG-Digitiser-like: from t=0
    at_zero = np.full(len(est), np.nan)
    at_zero[:2500] = est[2500:5000]
    for canvas, placed in ((est, 2.5), (at_zero, 0.0)):
        e, o, start = trim_to_observed(canvas, np.isfinite(canvas), fs)
        assert len(e) == 2500 and bool(o.all()) and start == placed
        m = segment_metrics(truth_slot, e, o, fs, fs, None, 2.5)
        assert m["status"] == "ok"
        assert m["r_at_lag"] > 0.999
        assert m["best_lag_s"] == 0.0
        assert m["duration_err_pct"] is None
    empty = np.full(100, np.nan)
    assert trim_to_observed(empty, np.zeros(100, bool), fs)[2] is None


def test_lag_slack_recovers_a_shifted_estimate_of_equal_length() -> None:
    # F6-b: estimate trimmed to its observed span and slot-cropped truth have the
    # same length, so without slack only lag 0 is searched; a 0.1 s shift in the
    # engine's placement then depresses r although the trace is right
    fs = 500.0
    t = np.arange(int(10 * fs)) / fs
    page = np.sin(2 * np.pi * 1.2 * t) + 0.2 * np.sin(2 * np.pi * 5 * t)
    truth_slot = page[1250:2500]
    est = page[1300:2550].copy()  # 0.1 s late, same length
    obs = np.ones(len(est), dtype=bool)
    no_slack = segment_metrics(truth_slot, est, obs, fs, fs, None, 2.5)
    slack = segment_metrics(truth_slot, est, obs, fs, fs, None, 2.5, lag_slack_s=0.2)
    assert no_slack["best_lag_s"] == 0.0
    assert slack["best_lag_s"] == 0.1
    assert slack["r_at_lag"] > 0.999
    assert slack["r_at_lag"] > no_slack["r_at_lag"]
    assert slack["rmse_mV"] < 1e-9  # scored only where the shifted estimate overlaps
    assert abs(slack["amp_ratio"] - 1.0) < 0.05


def test_lag_slack_zero_keeps_f6a_behaviour() -> None:
    fs = 500.0
    t = np.arange(int(10 * fs)) / fs
    truth = np.sin(2 * np.pi * 1.2 * t) + 0.2 * np.sin(2 * np.pi * 5 * t)
    est = truth[1250:2500] + np.random.default_rng(1).normal(0, 0.05, 1250)
    obs = np.ones(len(est), dtype=bool)
    a = segment_metrics(truth, est, obs, fs, fs, 2.5, 2.5)
    b = segment_metrics(truth, est, obs, fs, fs, 2.5, 2.5, lag_slack_s=0.0)
    assert a == b
