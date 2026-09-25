import numpy as np
import pytest
from PIL import Image

from ecg_photo.contracts import ReasonCode, load_manifest
from ecg_photo.export import readback_pdf_page_size_mm, readback_png_scale
from ecg_photo.fixtures import write_fixture_revision
from ecg_photo.render import (
    PaperSpec,
    RenderNotAllowed,
    render_segment_pdf,
    render_segment_png,
    t_to_x_px,
    v_to_y_px,
)


def _calibrated(tmp_path):
    root = tmp_path / "rev"
    m = write_fixture_revision(root, "calibrated")
    return root, m.segments[0]


def test_T35_png_scale_and_r_peak(tmp_path) -> None:
    root, seg = _calibrated(tmp_path)
    paper = PaperSpec(speed_mm_s=25.0, gain_mm_mV=10.0, dpi=254.0)
    png = tmp_path / "out.png"
    info = render_segment_png(root, seg, png, paper)
    w, h, dpi = readback_png_scale(png)
    assert dpi is not None and abs(dpi - 254.0) <= 0.5
    assert w == round(info.page_width_mm * 10.0)
    assert w == info.width_px
    assert h == info.height_px
    assert abs(t_to_x_px(info, 1.0) - info.x0_px - 250.0) <= 0.01
    assert abs(info.y0_px - v_to_y_px(info, 1.0) - 100.0) <= 0.01

    with Image.open(png) as im:
        arr = np.asarray(im.convert("RGB")).astype(int)
    black = np.all(arr < 80, axis=-1)
    # R peak of first beat: offset r_peak_px=155 from beat start at raw x=100
    # -> raw x=255 -> t = (255-100)/250 s = 0.62 s
    t_r = 0.62
    x_col = t_to_x_px(info, t_r)
    cols = range(round(x_col) - 2, round(x_col) + 3)
    rows = []
    for c in cols:
        ys = np.nonzero(black[:, c])[0]
        rows.extend(ys.tolist())
    assert rows, "no dark trace pixels near expected R column"
    r_row = float(np.min(rows))
    assert abs(r_row - v_to_y_px(info, 1.0)) <= 3.0


def test_T36_pdf_page_size(tmp_path) -> None:
    root, seg = _calibrated(tmp_path)
    paper = PaperSpec(speed_mm_s=25.0, gain_mm_mV=10.0, dpi=300.0)
    pdf = tmp_path / "out.pdf"
    info = render_segment_pdf(root, seg, pdf, paper)
    w_mm, h_mm = readback_pdf_page_size_mm(pdf)
    assert abs(w_mm - info.page_width_mm) <= 0.05
    assert abs(h_mm - info.page_height_mm) <= 0.05


def test_T36_gap_not_bridged_in_png(tmp_path) -> None:
    root = tmp_path / "rev"
    m = write_fixture_revision(root, "gap")
    seg = m.segments[0]
    paper = PaperSpec(speed_mm_s=25.0, gain_mm_mV=10.0, dpi=254.0)
    png = tmp_path / "gap.png"
    info = render_segment_png(root, seg, png, paper)
    with Image.open(png) as im:
        arr = np.asarray(im.convert("RGB")).astype(int)
    black = np.all(arr < 80, axis=-1)
    x1 = t_to_x_px(info, (420.0 - 100.0) / 250.0)
    x2 = t_to_x_px(info, (470.0 - 100.0) / 250.0)
    band = black[:, round(x1) + 3 : round(x2) - 3]
    assert band.sum() == 0


def test_render_not_allowed(tmp_path) -> None:
    for kind, reason in (
        ("gain_unknown", ReasonCode.GAIN_UNKNOWN),
        ("time_unknown", ReasonCode.TIME_SCALE_UNKNOWN),
    ):
        root = tmp_path / kind
        m = write_fixture_revision(root, kind)
        seg = m.segments[0]
        paper = PaperSpec(speed_mm_s=25.0, gain_mm_mV=10.0, dpi=300.0)
        with pytest.raises(RenderNotAllowed) as ei:
            render_segment_png(root, seg, tmp_path / f"{kind}.png", paper)
        assert ei.value.reason == reason


def test_render_loads_manifest(tmp_path) -> None:
    root, _ = _calibrated(tmp_path)
    m = load_manifest(root / "manifest.json")
    assert m.segments[0].signal_path == "signals/seg-II-01.npy"
