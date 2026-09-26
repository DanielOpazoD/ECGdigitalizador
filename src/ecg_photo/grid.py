"""Grid period estimation by periodicity (deterministic, numpy/scipy only).

Method: grayscale -> per-region mean column/row intensity profile -> detrend
-> autocorrelation via FFT -> first strong peak in [min,max] px = candidate
minor period P. P is accepted as the 1 mm step only if the grid's own
structure confirms it: the thicker line every `ratio` steps makes the
autocorrelation at ratio*P stand out above 2P..(ratio-1)P. A single visible
period (e.g. minor lines blurred away in a photo) is ambiguous between 1 mm
and 5 mm and yields None with the candidate recorded, never a guess.
Returns None where no peak passes the prominence threshold; nothing is invented.
"""

from dataclasses import dataclass, replace

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
    ambiguous_period_px_x: float | None = None
    ambiguous_period_px_y: float | None = None
    # set only by resolve_ambiguous_period (page-size prior): mm per ambiguous period
    period_step_mm: float | None = None


METHOD = "autocorr-fft-grid-v1"
LIMITATIONS = (
    "assumes the minor grid step is 1 mm and every 5th line is a major line; a "
    "period without that minor/major structure is ambiguous (1 mm or 5 mm) and "
    "gives no estimate; does not detect warped grids"
)
# local-peak autocorrelation rise required at ratio*P over (ratio-1)*P to
# confirm P as the minor step. Observed rises: +0.19..+0.46 on real
# minor/major grids (renders 200/300 dpi, anti-aliased synthetic, imgkit-like
# page); -0.04 with minor lines blurred away; -0.28 on uniform lines; +0.11
# worst case when P was already the major period.
MAJOR_CONFIRM_MARGIN = 0.15


def _normalized_autocorr(profile: np.ndarray, max_p: float) -> np.ndarray | None:
    x = np.asarray(profile, dtype=np.float64)
    n = len(x)
    base = uniform_filter1d(x, size=max(3, int(2 * max_p)), mode="nearest")
    x = x - base
    x = x - np.mean(x)
    f = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(f * np.conjugate(f))[:n]
    if ac[0] <= 0:
        return None
    return ac / ac[0]


def _autocorr_period(
    profile: np.ndarray,
    min_p: float,
    max_p: float,
    *,
    height: float = 0.3,
    prominence: float = 0.15,
) -> float | None:
    n = len(profile)
    if n < 2 * int(min_p):
        return None
    ac = _normalized_autocorr(profile, max_p)
    if ac is None:
        return None
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


def _major_structure_confirmed(
    profile: np.ndarray, minor_px: float, ratio: int, max_p: float
) -> bool:
    """True if the autocorrelation peak near ratio*minor rises above the peak
    near (ratio-1)*minor by MAJOR_CONFIRM_MARGIN: a thicker line every `ratio`
    steps (a single period decays monotonically instead). Peaks are local maxima within +-max(1.5, 0.12*P)
    px so a small error in P does not miss thin lines at high multiples."""
    ac = _normalized_autocorr(profile, max(max_p, ratio * minor_px))
    if ac is None:
        return False
    radius = max(1.5, 0.12 * minor_px)

    def peak_near(k: int) -> float | None:
        lo = int(np.floor(k * minor_px - radius))
        hi = int(np.ceil(k * minor_px + radius))
        if lo < 0 or hi >= len(ac):
            return None
        return float(ac[lo : hi + 1].max())

    at_major, before = peak_near(ratio), peak_near(ratio - 1)
    if at_major is None or before is None:
        return False
    return at_major - before >= MAJOR_CONFIRM_MARGIN


def _region_estimate(
    profile: np.ndarray, min_p: float, max_p: float, ratio: int
) -> tuple[float | None, float | None, float | None]:
    """(minor, major, ambiguous_period) for one region profile."""
    first = _autocorr_period(profile, min_p, max_p)
    if first is None:
        return None, None, None
    # The first strong peak may be a multiple of a weaker fundamental (faint
    # minor lines): take the weaker peak as candidate when first is an integer
    # multiple 2..ratio of it. Either way the minor/major structure decides.
    candidate = first
    sub = _autocorr_period(profile, min_p, first * 0.75, height=0.05, prominence=0.03)
    if sub is not None:
        m = round(first / sub)
        if 2 <= m <= ratio and abs(first / sub - m) <= 0.1 * m:
            candidate = sub
    if not _major_structure_confirmed(profile, candidate, ratio, max_p):
        return None, None, candidate
    target = ratio * candidate
    span = max(candidate * 0.5, 2.0)
    major = _autocorr_period(profile, target - span, target + span)
    return candidate, major, None


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


def _grid_channel(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """2-D float image where grid lines stand out, and whether lines are maxima
    (True: redness channel) or minima (False: grayscale)."""
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[2] >= 3:
        rgb = a[..., :3].astype(np.float64)
        # red-ish grid ink stands out in R - (G+B)/2; keeps traces dark
        redness = rgb[..., 0] - 0.5 * (rgb[..., 1] + rgb[..., 2])
        if float(redness.max()) > 8.0:
            return np.clip(redness, 0.0, None), True
        return rgb.mean(axis=2), False
    return np.asarray(a, dtype=np.float64), False


def estimate_grid(
    img: np.ndarray,
    *,
    regions: tuple[int, int] = (3, 3),
    minor_major_ratio: int = 5,
    min_period_px: float = 3.0,
    max_period_px: float = 60.0,
) -> GridEstimate:
    a, ink_bright = _grid_channel(img)
    h, w = a.shape
    ry, rx = regions
    minors_x: list[float] = []
    minors_y: list[float] = []
    majors_x: list[float] = []
    majors_y: list[float] = []
    ambig_x: list[float] = []
    ambig_y: list[float] = []
    for iy in range(ry):
        for ix in range(rx):
            y0, y1 = iy * h // ry, (iy + 1) * h // ry
            x0, x1 = ix * w // rx, (ix + 1) * w // rx
            tile = a[y0:y1, x0:x1]
            col_profile = tile.mean(axis=0)
            row_profile = tile.mean(axis=1)
            mx, Mx, ax = _region_estimate(
                col_profile, min_period_px, max_period_px, minor_major_ratio
            )
            my, My, ay = _region_estimate(
                row_profile, min_period_px, max_period_px, minor_major_ratio
            )
            if ax is not None:
                ambig_x.append(ax)
            if ay is not None:
                ambig_y.append(ay)
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
    amb_x = float(np.median(ambig_x)) if minor_x is None and len(ambig_x) >= 3 else None
    amb_y = float(np.median(ambig_y)) if minor_y is None and len(ambig_y) >= 3 else None
    ambiguous = [f"{ax}={v:.2f}px" for ax, v in (("x", amb_x), ("y", amb_y)) if v is not None]
    if ambiguous:
        limitations += (
            f"; period {', '.join(ambiguous)} without minor/major structure: "
            "1 mm or 5 mm cannot be decided, no estimate on that axis"
        )

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
        ambiguous_period_px_x=amb_x,
        ambiguous_period_px_y=amb_y,
    )


# Page-size prior for an ambiguous period (1 mm or 5 mm). Assumption: the whole
# ECG page is in the frame, its printed width is PAGE_WIDTH_MM (a 10 s strip at
# 25 mm/s is 250 mm; letter/A4 landscape <= 300 mm) and it fills at least
# PAGE_MIN_FILL of the image width. Then image_width / P lies in
#   [250, 300 / PAGE_MIN_FILL]            if P is 1 mm
#   [250 / 5, 300 / (5 * PAGE_MIN_FILL)]  if P is 5 mm
# and anything else stays undecided. Observed (F6-b, 2026-09-25): 5 mm periods
# gave 56.7-82.9 (Kaggle scans/photos, MAC2000 phone photos); 1 mm periods
# 264-270 (clean full-page renders). A close-up of part of a page breaks the
# assumption; the measured strip duration must then disagree with the layout.
PAGE_WIDTH_MM = (250.0, 300.0)
PAGE_MIN_FILL = 0.35
PAGE_PRIOR = (
    "1 mm/5 mm resolved by page-size prior: whole page in frame, page width "
    f"{PAGE_WIDTH_MM[0]:.0f}-{PAGE_WIDTH_MM[1]:.0f} mm filling >= {PAGE_MIN_FILL:.0%} "
    "of the image width"
)


def period_step_mm_from_page(width_px: int, period_px: float) -> float | None:
    """1.0 or 5.0 (mm per period) if image_width/period fits only one of the
    two hypotheses under the page-size prior, else None."""
    if period_px <= 0:
        return None
    r = width_px / period_px
    lo, hi = PAGE_WIDTH_MM
    for step in (1.0, 5.0):
        if lo / step <= r <= hi / (step * PAGE_MIN_FILL):
            return step
    return None


def resolve_ambiguous_period(est: GridEstimate, img: np.ndarray) -> GridEstimate:
    """Resolve an ambiguous x period (see period_step_mm_from_page), refine it
    by line positions over the full width and apply the same step to y when
    the y period agrees within 15 %. Returns `est` unchanged when there is
    nothing to resolve or the prior does not decide. The raw ambiguous periods
    stay recorded; method/limitations name the prior."""
    if est.px_per_mm_x is not None or est.ambiguous_period_px_x is None:
        return est
    a, ink_bright = _grid_channel(img)
    _h, w = a.shape
    period_x = est.ambiguous_period_px_x
    step = period_step_mm_from_page(w, period_x)
    if step is None:
        return est
    col_full = a.mean(axis=0) if ink_bright else -a.mean(axis=0)
    ref_x = _refine_period_by_lines(col_full, period_x)
    if ref_x is not None:
        period_x = ref_x[0]
    px_y = est.px_per_mm_y
    period_y = est.ambiguous_period_px_y
    ref_y = None
    if px_y is None and period_y is not None and abs(period_y / period_x - 1.0) <= 0.15:
        row_full = a.mean(axis=1) if ink_bright else -a.mean(axis=1)
        ref_y = _refine_period_by_lines(row_full, period_y)
        px_y = (ref_y[0] if ref_y is not None else period_y) / step
    return replace(
        est,
        px_per_mm_x=period_x / step,
        px_per_mm_y=px_y,
        method=est.method + f" + {PAGE_PRIOR} (x period = {step:g} mm)",
        limitations=est.limitations
        + f"; {PAGE_PRIOR}: a close-up of part of the page breaks this assumption",
        refine_method="line-position least squares" if ref_x is not None else est.refine_method,
        refine_n_lines_x=ref_x[2] if ref_x is not None else est.refine_n_lines_x,
        refine_rms_px_x=ref_x[1] if ref_x is not None else est.refine_rms_px_x,
        refine_n_lines_y=ref_y[2] if ref_y is not None else est.refine_n_lines_y,
        refine_rms_px_y=ref_y[1] if ref_y is not None else est.refine_rms_px_y,
        period_step_mm=step,
    )
