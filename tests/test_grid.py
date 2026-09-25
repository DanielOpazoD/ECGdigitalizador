import numpy as np
import pytest
from PIL import Image

from ecg_photo.fixtures import write_fixture_revision
from ecg_photo.grid import estimate_grid
from ecg_photo.render import PaperSpec, render_segment_png


def test_synthetic_period() -> None:
    period = 9.3
    img = np.full((400, 600), 240, dtype=np.uint8)
    xs = np.clip(np.round(np.arange(0, 600, period)).astype(int), 0, 599)
    ys = np.clip(np.round(np.arange(0, 400, period)).astype(int), 0, 399)
    img[:, xs] = 60
    img[ys, :] = 60
    est = estimate_grid(img, min_period_px=3, max_period_px=60)
    assert est.minor_period_px_x == pytest.approx(period, abs=0.5)
    assert est.minor_period_px_y == pytest.approx(period, abs=0.5)
    assert est.px_per_mm_x == pytest.approx(period, abs=0.5)


def test_noise_returns_none() -> None:
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, size=(300, 400), dtype=np.uint8)
    est = estimate_grid(img)
    assert est.px_per_mm_x is None
    assert est.px_per_mm_y is None


def test_x_only_stretched() -> None:
    px, py = 12.0, 6.0
    img = np.full((300, 400), 240, dtype=np.uint8)
    img[:, np.round(np.arange(0, 400, px)).astype(int)] = 60
    img[np.round(np.arange(0, 300, py)).astype(int), :] = 60
    est = estimate_grid(img, min_period_px=3, max_period_px=60)
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
    assert est.px_per_mm_x == pytest.approx(expected, rel=0.02)
    assert est.px_per_mm_y == pytest.approx(expected, rel=0.02)
