"""Shared metric helpers for F6 bench (importable from tests via sys.path)."""

import math

import numpy as np


def crop_to_window(
    est: np.ndarray, observed: np.ndarray, est_fs: float, start_s: float, dur_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Cut a page-canvas estimate (t=0 at the page's first column) to one lead
    slot [start_s, start_s + dur_s). Used when the truth is only defined in
    that slot (NaN elsewhere) and the estimate lives on the engine's page time
    axis; otherwise the two time origins differ and nothing overlaps."""
    a = max(0, round(start_s * est_fs))
    b = min(len(est), round((start_s + dur_s) * est_fs))
    if b <= a:
        return est[:0], observed[:0]
    return est[a:b], observed[a:b]


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
        "snr_db": math.nan,
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

    if not np.isfinite(best_r):
        # no lag gives >= 10 overlapping samples with variance: interpolating
        # truth outside its span would clamp to a constant (zero power)
        out["status"] = "no_valid_lag"
        return out

    best_lag_s = best_lag / fs
    truth_on_est = np.interp(t_est[finite] + best_lag_s, t_truth, truth)
    resid = truth_on_est - e_obs
    slots = np.array([0.0, 2.5, 5.0, 7.5])
    offset_err = (
        float(best_lag_s - slots[np.argmin(np.abs(slots - best_lag_s))])
        if expected_window_s < t_truth[-1]
        else None
    )
    # competition-style SNR (not the official metric): best lag, both means removed
    t_c = truth_on_est - truth_on_est.mean()
    noise = t_c - (e_obs - e_obs.mean())
    p_noise = float(np.sum(noise**2))
    p_sig = float(np.sum(t_c**2))
    snr_db = 10.0 * math.log10(p_sig / p_noise) if p_noise > 0 and p_sig > 0 else None
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
            "snr_db": snr_db,
        }
    )
    return out
