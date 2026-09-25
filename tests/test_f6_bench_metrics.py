import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from metrics_common import segment_metrics


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
