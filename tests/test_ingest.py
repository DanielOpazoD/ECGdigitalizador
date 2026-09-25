import io

import numpy as np
import pytest
from PIL import Image

from ecg_photo.contracts import ReasonCode, SourceKind, load_manifest
from ecg_photo.ingest import (
    IngestRejected,
    SupportedInputs,
    admit,
    ingest,
    safe_study_id,
    sniff_kind,
)


def _png_bytes(size: tuple[int, int] = (40, 30)) -> bytes:
    img = Image.fromarray(np.full((size[1], size[0], 3), 200, dtype=np.uint8))
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


def test_sniff_kind(tmp_path) -> None:
    p = tmp_path / "a.png"
    p.write_bytes(_png_bytes())
    assert sniff_kind(p) == SourceKind.image
    bad = tmp_path / "b.png"
    bad.write_bytes(b"hello world, this is text")
    with pytest.raises(IngestRejected, match="UNSUPPORTED_TYPE"):
        sniff_kind(bad)


def test_t43_pixel_limit_before_decode(tmp_path) -> None:
    p = tmp_path / "big.png"
    p.write_bytes(_png_bytes((20, 20)))
    limits = SupportedInputs(max_file_mib=25, max_pdf_pages=10, max_pixels_per_page=100)
    with pytest.raises(IngestRejected, match="PIXEL_LIMIT"):
        admit(p, limits)


def test_t44_safe_study_id_and_filename_escape(tmp_path) -> None:
    evil = tmp_path / "x.png"
    evil.write_bytes(_png_bytes())
    limits = SupportedInputs(max_file_mib=25, max_pdf_pages=10, max_pixels_per_page=10**7)
    src = admit(evil, limits)
    sid = safe_study_id("<script>alert(1)</script>.png")
    assert sid.startswith("study-") and "<script>" not in sid
    assert src.original_filename == "x.png"
    evil2 = tmp_path / "..\\x.png"
    evil2.write_bytes(_png_bytes())
    src2 = admit(evil2, limits)
    assert "\\x00" not in src2.original_filename
    assert len(src2.original_filename) <= 255


def test_ingest_png(tmp_path) -> None:
    p = tmp_path / "in.png"
    p.write_bytes(_png_bytes((64, 48)))
    m = ingest(p, tmp_path / "rev")
    assert len(m.pages) == 1
    assert m.pages[0].extraction == "image_file"
    assert (tmp_path / "rev/pages/page-1.png").exists()
    assert (tmp_path / "rev/manifest.json").exists()
    assert load_manifest(tmp_path / "rev/manifest.json").study_id == m.study_id


def test_ingest_exif6_jpeg(tmp_path) -> None:
    arr = np.arange(3 * 2 * 3, dtype=np.uint8).reshape(2, 3, 3)
    img = Image.fromarray(arr)
    exif = img.getexif()
    exif[0x0112] = 6
    p = tmp_path / "rot.jpg"
    img.save(p, exif=exif)
    m = ingest(p, tmp_path / "rev")
    pg = m.pages[0]
    assert pg.exif_orientation == 6
    assert (pg.width_px, pg.height_px) == (2, 3)  # swapped by rotation
    chain = m.transforms[0]
    assert any(s.kind == "rotate90" and s.parameters["k"] == 3 for s in chain.steps)


def test_t42_two_page_pdf(tmp_path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = tmp_path / "two.pdf"
    with PdfPages(pdf_path) as pdf:
        for _ in range(2):
            fig = plt.figure(figsize=(4, 3))
            fig.text(0.5, 0.5, "x")
            pdf.savefig(fig)
            plt.close(fig)
    m = ingest(pdf_path, tmp_path / "rev", render_dpi=150.0)
    assert len(m.pages) == 2
    assert all(pg.extraction in ("pdf_rendered", "pdf_embedded_raster") for pg in m.pages)
    # mediabox 4x3 in -> 288x216 pt -> at 150 dpi: 600x450 px
    assert abs(m.pages[0].width_px - 600) <= 2
    assert abs(m.pages[0].height_px - 450) <= 2
    assert m.source.page_count == 2


def test_admit_pdf_page_limit(tmp_path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = tmp_path / "two.pdf"
    with PdfPages(pdf_path) as pdf:
        for _ in range(2):
            fig = plt.figure()
            pdf.savefig(fig)
            plt.close(fig)
    limits = SupportedInputs(max_file_mib=25, max_pdf_pages=1, max_pixels_per_page=10**7)
    with pytest.raises(IngestRejected, match="PAGE_LIMIT"):
        admit(pdf_path, limits)


def test_admit_file_size_limit(tmp_path) -> None:
    p = tmp_path / "big.png"
    p.write_bytes(_png_bytes() * 50)
    limits = SupportedInputs(
        max_file_mib=p.stat().st_size / (1024 * 1024) / 2,
        max_pdf_pages=10,
        max_pixels_per_page=10**7,
    )
    with pytest.raises(IngestRejected, match="FILE_SIZE_LIMIT"):
        admit(p, limits)


def test_reason_codes_exist() -> None:
    assert ReasonCode.PIXEL_LIMIT
    assert ReasonCode.UNSUPPORTED_TYPE
    assert ReasonCode.PAGE_LIMIT
    assert ReasonCode.FILE_SIZE_LIMIT
