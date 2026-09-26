import numpy as np
import pytest
from PIL import Image

from ecg_photo.fixtures import write_fixture_revision
from ecg_photo.grid import estimate_grid
from ecg_photo.render import PaperSpec, render_segment_png


def _paper(h: int, w: int, px: float, py: float, *, minor: bool = True) -> np.ndarray:
    """ECG paper: thin lines every px/py, thick darker line every 5th."""
    img = np.full((h, w), 240, dtype=np.uint8)
    for k, x in enumerate(np.arange(0, w, px)):
        if k % 5 == 0:
            img[:, max(0, round(x) - 1) : round(x) + 2] = 40
        elif minor:
            img[:, min(w - 1, round(x))] = 150
    for k, y in enumerate(np.arange(0, h, py)):
        if k % 5 == 0:
            img[max(0, round(y) - 1) : round(y) + 2, :] = 40
        elif minor:
            img[min(h - 1, round(y)), :] = 150
    return img


def test_synthetic_period() -> None:
    period = 9.3
    est = estimate_grid(_paper(800, 1200, period, period), min_period_px=3, max_period_px=60)
    assert est.minor_period_px_x == pytest.approx(period, abs=0.5)
    assert est.minor_period_px_y == pytest.approx(period, abs=0.5)
    assert est.px_per_mm_x == pytest.approx(period, abs=0.5)
    assert est.ambiguous_period_px_x is None


def test_single_period_is_ambiguous_not_guessed() -> None:
    # uniform lines: nothing says whether the step is 1 mm or 5 mm
    period = 9.3
    img = np.full((400, 600), 240, dtype=np.uint8)
    img[:, np.clip(np.round(np.arange(0, 600, period)).astype(int), 0, 599)] = 60
    img[np.clip(np.round(np.arange(0, 400, period)).astype(int), 0, 399), :] = 60
    est = estimate_grid(img, min_period_px=3, max_period_px=60)
    assert est.px_per_mm_x is None and est.px_per_mm_y is None
    assert est.ambiguous_period_px_x == pytest.approx(period, abs=0.5)
    assert "cannot be decided" in est.limitations


def test_minor_lines_blurred_away_gives_no_estimate() -> None:
    # F6-b smoke: blur erased the 1 mm lines; the 5 mm period used to be
    # reported as 1 mm (39.4 px/mm instead of 7.874, time axis 5x short)
    from scipy.ndimage import gaussian_filter

    img = _paper(1700, 2200, 7.874, 7.874).astype(np.float64)
    img = gaussian_filter(img, 2.5)
    est = estimate_grid(np.clip(img, 0, 255).astype(np.uint8))
    assert est.px_per_mm_x is None and est.px_per_mm_y is None
    assert est.ambiguous_period_px_x == pytest.approx(39.37, rel=0.02)


def test_noise_returns_none() -> None:
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, size=(300, 400), dtype=np.uint8)
    est = estimate_grid(img)
    assert est.px_per_mm_x is None
    assert est.px_per_mm_y is None
    assert est.refine_method is None


def test_synthetic_aa_grid_subpixel() -> None:
    # 200-dpi-like grid: 7.874 px minor, 39.37 px major, anti-aliased lines
    rng = np.random.default_rng(2)
    img = np.full((1700, 2200), 240.0)
    minor, major = 7.874, 39.37

    def line_field(pos: np.ndarray, period: float, width: float, depth: float) -> np.ndarray:
        k = np.round(pos / period) * period
        return depth * np.clip(1.0 - np.abs(pos - k) / width, 0.0, 1.0)

    xs = np.arange(2200, dtype=np.float64)
    ys = np.arange(1700, dtype=np.float64)
    img -= line_field(xs, minor, 0.6, 80.0)[None, :]
    img -= line_field(xs, major, 1.0, 100.0)[None, :]
    img -= line_field(ys, minor, 0.6, 80.0)[:, None]
    img -= line_field(ys, major, 1.0, 100.0)[:, None]
    img += rng.normal(0, 2.0, img.shape)
    est = estimate_grid(np.clip(img, 0, 255).astype(np.uint8), min_period_px=3, max_period_px=60)
    assert est.px_per_mm_x == pytest.approx(minor, rel=0.002)
    assert est.px_per_mm_y == pytest.approx(minor, rel=0.002)


def test_x_only_stretched() -> None:
    px, py = 12.0, 6.0
    est = estimate_grid(_paper(900, 1200, px, py), min_period_px=3, max_period_px=60)
    assert est.px_per_mm_x == pytest.approx(px, abs=1.0)
    assert est.px_per_mm_y == pytest.approx(py, abs=1.0)
    assert est.px_per_mm_x != est.px_per_mm_y


@pytest.mark.parametrize("dpi,expected", [(300.0, 300.0 / 25.4), (200.0, 200.0 / 25.4)])
def test_render_png_grid(tmp_path, dpi: float, expected: float) -> None:
    root = tmp_path / "rev"
    m = write_fixture_revision(root, "calibrated")
    seg = m.segments[0]
    out = tmp_path / f"seg-{int(dpi)}.png"
    render_segment_png(root, seg, out, PaperSpec(speed_mm_s=25.0, gain_mm_mV=10.0, dpi=dpi))
    img = np.asarray(Image.open(out).convert("RGB"))
    est = estimate_grid(img, min_period_px=3, max_period_px=max(60, expected * 2))
    assert est.px_per_mm_x == pytest.approx(expected, rel=0.003)
    assert est.px_per_mm_y == pytest.approx(expected, rel=0.003)


def test_page_prior_resolves_blurred_full_page() -> None:
    # same blurred page as above: the grid alone cannot decide, the page-size
    # prior (2200 px / 39.4 px = 56 -> 5 mm) recovers 7.874 px/mm on both axes
    from scipy.ndimage import gaussian_filter

    from ecg_photo.grid import resolve_ambiguous_period

    img = gaussian_filter(_paper(1700, 2200, 7.874, 7.874).astype(np.float64), 2.5)
    img8 = np.clip(img, 0, 255).astype(np.uint8)
    est = resolve_ambiguous_period(estimate_grid(img8), img8)
    assert est.period_step_mm == 5.0
    assert est.px_per_mm_x == pytest.approx(7.874, rel=0.01)
    assert est.px_per_mm_y == pytest.approx(7.874, rel=0.01)
    assert est.ambiguous_period_px_x == pytest.approx(39.37, rel=0.02)  # still recorded
    assert "page-size prior" in est.method and "close-up" in est.limitations


def test_page_prior_resolves_1mm_period_on_full_page() -> None:
    # uniform 1 mm lines across a full 280 mm page at 7.5 px/mm: 2100/7.5 = 280
    from ecg_photo.grid import resolve_ambiguous_period

    p = 7.5
    img = np.full((1500, 2100), 240, dtype=np.uint8)
    img[:, np.clip(np.round(np.arange(0, 2100, p)).astype(int), 0, 2099)] = 60
    img[np.clip(np.round(np.arange(0, 1500, p)).astype(int), 0, 1499), :] = 60
    raw = estimate_grid(img)
    assert raw.px_per_mm_x is None  # grid alone: ambiguous
    est = resolve_ambiguous_period(raw, img)
    assert est.period_step_mm == 1.0
    assert est.px_per_mm_x == pytest.approx(p, rel=0.01)


def test_page_prior_leaves_gray_zone_undecided() -> None:
    from ecg_photo.grid import period_step_mm_from_page, resolve_ambiguous_period

    # 600 px / 9.3 px = 64.5 fits 5 mm; 1900 px / 9.3 px = 204 fits neither
    assert period_step_mm_from_page(600, 9.3) == 5.0
    assert period_step_mm_from_page(1900, 9.3) is None
    assert period_step_mm_from_page(2000, 7.5) == 1.0
    assert period_step_mm_from_page(2000, 0.0) is None
    img = np.full((400, 1900), 240, dtype=np.uint8)
    img[:, np.clip(np.round(np.arange(0, 1900, 9.3)).astype(int), 0, 1899)] = 60
    img[np.clip(np.round(np.arange(0, 400, 9.3)).astype(int), 0, 399), :] = 60
    raw = estimate_grid(img)
    assert resolve_ambiguous_period(raw, img) == raw


def test_page_prior_does_not_override_a_confirmed_grid() -> None:
    from ecg_photo.grid import resolve_ambiguous_period

    img = _paper(800, 1200, 9.3, 9.3)
    raw = estimate_grid(img)
    assert raw.px_per_mm_x is not None
    assert resolve_ambiguous_period(raw, img) is raw


def test_page_prior_refused_on_perspective() -> None:
    # 5 mm lines only, x period growing 12 % from top to bottom (phone photo
    # taken at an angle): one global scale would be wrong along some rows
    from ecg_photo.grid import resolve_ambiguous_period

    h, w = 1700, 2200
    img = np.full((h, w), 240, dtype=np.uint8)
    for i in range(5):
        p = 39.37 * (0.94 + 0.03 * i)
        xs = np.clip(np.round(np.arange(0, w, p)).astype(int), 0, w - 1)
        img[i * h // 5 : (i + 1) * h // 5, xs] = 60
    img[np.clip(np.round(np.arange(0, h, 39.37)).astype(int), 0, h - 1), :] = 60
    raw = estimate_grid(img)
    assert raw.px_per_mm_x is None and raw.ambiguous_period_px_x is not None
    est = resolve_ambiguous_period(raw, img)
    assert est.px_per_mm_x is None and est.period_step_mm is None
    assert est.band_spread_rel_x is not None and est.band_spread_rel_x > 0.03
    assert "not applied" in est.limitations and "perspective" in est.limitations
