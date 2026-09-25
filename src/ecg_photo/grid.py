"""Grid period estimation by periodicity (deterministic, numpy/scipy only).

Method: grayscale -> per-region mean column/row intensity profile -> detrend
-> autocorrelation via FFT -> first strong peak in [min,max] px = minor period
(assumed 1 mm); major = confirm a peak near ratio*minor. Returns None where no
peak passes the prominence threshold; nothing is invented.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d  # type: ignore[import-untyped]
from scipy.signal import find_peaks  # type: ignore[import-untyped]


@dataclass(frozen=True)
class GridEstimate:
    px_per_mm_x: float | None
    px_per_mm_y: float | None
    minor_period_px_x: float | None
    minor_period_px_y: float | None
    major_period_px_x: float | None
    major_period_px_y: float | None
    n_regions: int
    residual_rel_x: float | None
    residual_rel_y: float | None
    method: str
    limitations: str
    refine_method: str | None
    refine_n_lines_x: int | None
    refine_n_lines_y: int | None
    refine_rms_px_x: float | None
    refine_rms_px_y: float | None


METHOD = "autocorr-fft-grid-v1"
LIMITATIONS = (
    "assumes the minor grid step is 1 mm; if only a major period is found, "
    "px_per_mm = major/5 with that stated; does not detect warped grids"
)


def _autocorr_period(
    profile: np.ndarray,
    min_p: float,
    max_p: float,
    *,
    height: float = 0.3,
    prominence: float = 0.15,
) -> float | None:
    x = np.asarray(profile, dtype=np.float64)
    n = len(x)
    base = uniform_filter1d(x, size=max(3, int(2 * max_p)), mode="nearest")
    x = x - base
    x = x - np.mean(x)
    if n < 2 * int(min_p):
        return None
    f = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(f * np.conjugate(f))[:n]
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]
    lo, hi = int(np.ceil(min_p)), int(min(np.floor(max_p), n - 1))
    if hi <= lo:
        return None
    window = ac[lo : hi + 1]
    peaks, _props = find_peaks(window, height=height, prominence=prominence)
    if len(peaks) == 0:
        return None
    peak_idx = int(peaks[0]) + lo
    # parabolic refinement
    w0 = peak_idx - lo
    if 0 < w0 < len(window) - 1:
        y0, y1, y2 = window[w0 - 1], window[w0], window[w0 + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom
            if abs(delta) <= 1.0:
                return float(peak_idx + delta)
    return float(peak_idx)


def _region_estimate(
    profile: np.ndarray, min_p: float, max_p: float, ratio: int
) -> tuple[float | None, float | None]:
    first = _autocorr_period(profile, min_p, max_p)
    if first is None:
        return None, None
    # The first peak may already be the major period if the minor grid is weak;
    # prefer a confirmed subharmonic near first/ratio.
    sub = _autocorr_period(profile, min_p, first * 0.75, height=0.05, prominence=0.03)
    if sub is not None and abs(sub * ratio - first) <= first * 0.2:
        return sub, first
    target = ratio * first
    span = max(first * 0.5, 2.0)
    major = _autocorr_period(profile, target - span, target + span)
    return first, major


def _detrend(profile: np.ndarray, period_px: float) -> np.ndarray:
    x = np.asarray(profile, dtype=np.float64)
    base = uniform_filter1d(x, size=max(3, int(2 * period_px)), mode="nearest")
    return x - base


def _refine_period_by_lines(
    profile: np.ndarray, coarse_major_px: float
) -> tuple[float, float, int] | None:
    """Refine a coarse major-grid period by least squares on detected line
    positions. Returns (refined_period_px, rms_residual_px, n_lines) or None
    when fewer than 6 lines survive the spacing-consistency filter."""
    det = _detrend(profile, coarse_major_px)
    thr = float(np.percentile(det, 90))
    if thr <= 0:
        return None
    peaks, _ = find_peaks(det, distance=max(1, int(0.6 * coarse_major_px)), height=thr)
    if len(peaks) < 6:
        return None
    # assign each surviving peak an integer line index; a rejected gap of g
    # periods skips g indices so a missing line does not break the fit
    keep = [0]
    ks = [0]
    k = 0
    for i in range(1, len(peaks)):
        spacing = float(peaks[i] - peaks[keep[-1]])
        g = round(spacing / coarse_major_px)
        if g < 1 or abs(spacing / g - coarse_major_px) > 0.15 * coarse_major_px:
            continue
        keep.append(i)
        k += g
        ks.append(k)
    if len(keep) < 6:
        return None
    sel = peaks[keep].astype(np.float64)
    kk = np.asarray(ks, dtype=np.float64)
    a_mat = np.column_stack([np.ones_like(kk), kk])
    (a0, b), *_ = np.linalg.lstsq(a_mat, sel, rcond=None)
    resid = sel - (a0 + b * kk)
    rms = float(np.sqrt(np.mean(resid**2)))
    return float(b), rms, len(keep)


def estimate_grid(
    img: np.ndarray,
    *,
    regions: tuple[int, int] = (3, 3),
    minor_major_ratio: int = 5,
    min_period_px: float = 3.0,
    max_period_px: float = 60.0,
) -> GridEstimate:
    a = np.asarray(img)
    ink_bright = False  # True when lines are maxima in `a` (redness channel)
    if a.ndim == 3 and a.shape[2] >= 3:
        rgb = a[..., :3].astype(np.float64)
        # red-ish grid ink stands out in R - (G+B)/2; keeps traces dark
        redness = rgb[..., 0] - 0.5 * (rgb[..., 1] + rgb[..., 2])
        if float(redness.max()) > 8.0:
            a = np.clip(redness, 0.0, None)
            ink_bright = True
        else:
            a = rgb.mean(axis=2)
    a = np.asarray(a, dtype=np.float64)
    h, w = a.shape
    ry, rx = regions
    minors_x: list[float] = []
    minors_y: list[float] = []
    majors_x: list[float] = []
    majors_y: list[float] = []
    for iy in range(ry):
        for ix in range(rx):
            y0, y1 = iy * h // ry, (iy + 1) * h // ry
            x0, x1 = ix * w // rx, (ix + 1) * w // rx
            tile = a[y0:y1, x0:x1]
            col_profile = tile.mean(axis=0)
            row_profile = tile.mean(axis=1)
            mx, Mx = _region_estimate(col_profile, min_period_px, max_period_px, minor_major_ratio)
            my, My = _region_estimate(row_profile, min_period_px, max_period_px, minor_major_ratio)
            if mx is not None:
                minors_x.append(mx)
            if Mx is not None:
                majors_x.append(Mx)
            if my is not None:
                minors_y.append(my)
            if My is not None:
                majors_y.append(My)

    def agg(vals: list[float]) -> tuple[float | None, float | None]:
        if len(vals) < 3:
            # fewer than 3 supporting regions: not enough agreement to report
            return None, None
        med = float(np.median(vals))
        res = float(np.std(vals) / med) if med > 0 else None
        return med, res

    minor_x, res_x = agg(minors_x)
    minor_y, res_y = agg(minors_y)
    major_x, _ = agg(majors_x)
    major_y, _ = agg(majors_y)

    # refinement: least squares on major-line positions over the full
    # width/height profile, seeded by the aggregated coarse major period
    col_full = a.mean(axis=0) if ink_bright else -a.mean(axis=0)
    row_full = a.mean(axis=1) if ink_bright else -a.mean(axis=1)
    ref_x = _refine_period_by_lines(col_full, major_x) if major_x is not None else None
    ref_y = _refine_period_by_lines(row_full, major_y) if major_y is not None else None
    if ref_x is not None:
        major_x = ref_x[0]
        minor_x = major_x / minor_major_ratio
    if ref_y is not None:
        major_y = ref_y[0]
        minor_y = major_y / minor_major_ratio

    def px_per_mm(minor: float | None, major: float | None) -> float | None:
        if minor is not None:
            return minor / 1.0
        if major is not None:
            return major / minor_major_ratio
        return None

    method = METHOD
    limitations = LIMITATIONS
    if ref_x is not None or ref_y is not None:
        method = METHOD + " + major refined by line-position least squares"
    missing = [
        ax
        for ax, r, maj in (("x", ref_x, major_x), ("y", ref_y, major_y))
        if maj is not None and r is None
    ]
    if missing:
        limitations += f"; no line refinement on {','.join(missing)} (coarse kept)"

    return GridEstimate(
        px_per_mm_x=px_per_mm(minor_x, major_x),
        px_per_mm_y=px_per_mm(minor_y, major_y),
        minor_period_px_x=minor_x,
        minor_period_px_y=minor_y,
        major_period_px_x=major_x,
        major_period_px_y=major_y,
        n_regions=rx * ry,
        residual_rel_x=res_x,
        residual_rel_y=res_y,
        method=method,
        limitations=limitations,
        refine_method=(
            "line-position least squares" if (ref_x is not None or ref_y is not None) else None
        ),
        refine_n_lines_x=ref_x[2] if ref_x is not None else None,
        refine_n_lines_y=ref_y[2] if ref_y is not None else None,
        refine_rms_px_x=ref_x[1] if ref_x is not None else None,
        refine_rms_px_y=ref_y[1] if ref_y is not None else None,
    )
