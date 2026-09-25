"""Shared metric helpers for F6 bench (importable from tests via sys.path)."""

import math

import numpy as np


def segment_metrics(
    truth: np.ndarray,
    est: np.ndarray,
    observed: np.ndarray,
    est_fs: float,
    truth_fs: float,
    observed_duration_s: float | None,
    expected_window_s: float,
) -> dict:
    """Metrics over observed samples only, lag-searched over the whole truth.

    truth: full-length reference signal (truth_fs).
    est: estimated signal (est_fs), NaN where unobserved (observed masks it too).
    """
    finite = np.isfinite(est) & observed
    n_obs = int(finite.sum())
    out: dict = {
        "coverage": float(np.mean(observed)) if len(observed) else 0.0,
        "observed_duration_s": observed_duration_s,
        "expected_window_s": expected_window_s,
        "duration_err_pct": (
            100.0 * (observed_duration_s / expected_window_s - 1.0)
            if observed_duration_s and expected_window_s > 0
            else None
        ),
        "r_at_lag": math.nan,
        "best_lag_s": None,
        "offset_err_s": None,
        "rmse_mV": math.nan,
        "amp_ratio": math.nan,
    }
    if n_obs < 10:
        out["status"] = "no_signal"
        return out

    est_500 = np.where(finite, est, np.nan)
    fs = float(est_fs)
    if abs(fs - truth_fs) > 1e-9:
        # resample observed samples onto the truth_fs grid (interpolate in
        # est-index space over observed positions only)
        idx = np.where(finite)[0]
        n_out = round(len(est) * truth_fs / fs)
        pos = np.arange(n_out) * (fs / truth_fs)  # output k -> input index
        e500 = np.full(n_out, np.nan)
        if len(idx) >= 2:
            e500 = np.interp(pos, idx.astype(float), est[idx], left=np.nan, right=np.nan)
        est_500 = e500
        finite = np.isfinite(est_500)
        fs = truth_fs

    t_est = np.arange(len(est_500)) / fs
    t_truth = np.arange(len(truth)) / truth_fs
    e_obs = est_500[finite]

    def pearson_at(lag_samples: int) -> float:
        shifted = t_est[finite] + lag_samples / fs
        in_range = (shifted >= 0.0) & (shifted <= t_truth[-1])
        if in_range.sum() < 10:
            return math.nan
        e = e_obs[in_range]
        tt = np.interp(shifted[in_range], t_truth, truth)
        if np.std(e) < 1e-12 or np.std(tt) < 1e-12:
            return math.nan
        return float(np.corrcoef(e, tt)[0, 1])

    max_lag = max(0, round((t_truth[-1] - t_est[-1]) * fs))
    best_lag, best_r = 0, -np.inf
    for lag in range(max_lag + 1):
        r = pearson_at(lag)
        if np.isfinite(r) and r > best_r:
            best_r, best_lag = r, lag

    best_lag_s = best_lag / fs
    truth_on_est = np.interp(t_est[finite] + best_lag_s, t_truth, truth)
    resid = truth_on_est - e_obs
    slots = np.array([0.0, 2.5, 5.0, 7.5])
    offset_err = (
        float(best_lag_s - slots[np.argmin(np.abs(slots - best_lag_s))])
        if expected_window_s < t_truth[-1]
        else None
    )
    p95e, p5e = np.percentile(e_obs, [95, 5])
    p95t, p5t = np.percentile(np.interp(t_est[finite] + best_lag_s, t_truth, truth), [95, 5])
    out.update(
        {
            "status": "ok",
            "r_at_lag": float(best_r) if np.isfinite(best_r) else math.nan,
            "best_lag_s": float(best_lag_s),
            "offset_err_s": offset_err,
            "rmse_mV": float(np.sqrt(np.mean(resid**2))),
            "amp_ratio": float((p95e - p5e) / (p95t - p5t)) if (p95t - p5t) > 0 else math.nan,
        }
    )
    return out
