"""Admission and ingest of real inputs (T42-T44).

Admission is decided by magic bytes and declared dimensions only — a payload
is never fully decoded before the pixel/page/size limits are checked.
"""

import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium  # type: ignore[import-untyped]
import yaml
from PIL import Image
from pypdf import PdfReader

from ecg_photo.contracts import (
    HashStatus,
    Manifest,
    Page,
    ReasonCode,
    Source,
    SourceKind,
    TransformChain,
    TransformStep,
    dump_json,
    sha256_file,
)
from ecg_photo.transforms import exif_to_steps

EXIF_ORIENTATION_TAG = 0x0112
MAGIC = {
    SourceKind.image: (b"\xff\xd8\xff", b"\x89PNG"),
    SourceKind.pdf: (b"%PDF-",),
}


class IngestRejected(ValueError):
    def __init__(self, reason_code: ReasonCode) -> None:
        super().__init__(f"ingest rejected: {reason_code}")
        self.reason_code = reason_code


@dataclass(frozen=True)
class SupportedInputs:
    max_file_mib: float
    max_pdf_pages: int
    max_pixels_per_page: int


def load_supported_inputs(path: Path | None = None) -> SupportedInputs:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "supported_inputs.yml"
        if not path.exists():
            path = Path("configs/supported_inputs.yml")
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return SupportedInputs(
        max_file_mib=float(cfg["max_file_mib"]),
        max_pdf_pages=int(cfg["max_pdf_pages"]),
        max_pixels_per_page=int(cfg["max_pixels_per_page"]),
    )


def safe_study_id(original_filename: str) -> str:
    """uuid4-based study id; never derived from the filename."""
    return f"study-{uuid.uuid4().hex}"


def escape_filename(name: str) -> str:
    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", name)
    return cleaned[:255]


def sniff_kind(path: Path) -> SourceKind:
    with Path(path).open("rb") as f:
        head = f.read(8)
    if head[:3] == b"\xff\xd8\xff" or head[:4] == b"\x89PNG":
        return SourceKind.image
    if head[:5] == b"%PDF-":
        return SourceKind.pdf
    raise IngestRejected(ReasonCode.UNSUPPORTED_TYPE)


def admit(path: Path, limits: SupportedInputs) -> Source:
    path = Path(path)
    size_mib = path.stat().st_size / (1024 * 1024)
    if size_mib > limits.max_file_mib:
        raise IngestRejected(ReasonCode.FILE_SIZE_LIMIT)
    kind = sniff_kind(path)
    if kind == SourceKind.image:
        try:
            with Image.open(path) as img:
                w, h = img.size
        except Image.DecompressionBombError:
            raise IngestRejected(ReasonCode.PIXEL_LIMIT) from None
        except OSError:
            raise IngestRejected(ReasonCode.UNSUPPORTED_TYPE) from None
        if w * h > limits.max_pixels_per_page:
            raise IngestRejected(ReasonCode.PIXEL_LIMIT)
        return Source(
            sha256=sha256_file(path),
            hash_status=HashStatus.computed,
            kind=kind,
            page_count=1,
            original_filename=escape_filename(path.name),
        )
    if kind == SourceKind.pdf:
        try:
            n_pages = len(PdfReader(str(path)).pages)
        except Exception as e:
            raise IngestRejected(ReasonCode.UNSUPPORTED_TYPE) from e
        if n_pages > limits.max_pdf_pages:
            raise IngestRejected(ReasonCode.PAGE_LIMIT)
        return Source(
            sha256=sha256_file(path),
            hash_status=HashStatus.computed,
            kind=kind,
            page_count=n_pages,
            original_filename=escape_filename(path.name),
        )
    raise IngestRejected(ReasonCode.UNSUPPORTED_TYPE)


def _exif_chain(orientation: int, w: int, h: int, page_id: str) -> TransformChain:
    steps: list[TransformStep] = exif_to_steps(orientation, w, h)
    return TransformChain(
        transform_id=f"{page_id}-exif",
        source_frame_id="file",
        target_frame_id=f"rectified-{page_id}",
        steps=steps,
    )


def _embedded_raster_dpi(px_w: int, px_h: int, box_w_pt: float, box_h_pt: float) -> float | None:
    if box_w_pt <= 0 or box_h_pt <= 0:
        return None
    dpi_x = px_w * 72.0 / box_w_pt
    dpi_y = px_h * 72.0 / box_h_pt
    if abs(dpi_x - dpi_y) / max(dpi_x, dpi_y) > 0.01:
        return None
    return (dpi_x + dpi_y) / 2.0


def _ingest_pdf(
    path: Path, pages_dir: Path, render_dpi: float
) -> list[tuple[Page, TransformChain]]:
    reader = PdfReader(str(path))
    pdfium_doc = pdfium.PdfDocument(str(path))
    out: list[tuple[Page, TransformChain]] = []
    for i, page in enumerate(reader.pages):
        page_id = f"page-{i + 1}"
        mediabox = page.mediabox
        box_w_pt = float(mediabox.width)
        box_h_pt = float(mediabox.height)
        embedded = page.images
        png_path = pages_dir / f"{page_id}.png"
        if len(embedded) == 1:
            img = embedded[0]
            pil_img = img.image
            if pil_img is None:
                continue
            w, h = pil_img.size
            covers = (
                w * h
                >= 0.9
                * (box_w_pt * box_h_pt)
                / (72 * 72)
                * (_embedded_raster_dpi(w, h, box_w_pt, box_h_pt) or 72) ** 2
            )
            dpi_decl = _embedded_raster_dpi(w, h, box_w_pt, box_h_pt)
            if covers:
                pil_img.save(png_path)
                pg = Page(
                    page_id=page_id,
                    index=i,
                    width_px=w,
                    height_px=h,
                    dpi_declared=dpi_decl,
                    raster_path=f"pages/{page_id}.png",
                    exif_orientation=None,
                    extraction="pdf_embedded_raster",
                    render_dpi=None,
                )
                out.append((pg, _exif_chain(1, w, h, page_id)))
                continue
        pdfium_page = pdfium_doc[i]
        bitmap = pdfium_page.render(scale=render_dpi / 72.0)
        rendered = bitmap.to_pil()
        w, h = rendered.size
        rendered.save(png_path)
        pg = Page(
            page_id=page_id,
            index=i,
            width_px=w,
            height_px=h,
            dpi_declared=render_dpi,
            raster_path=f"pages/{page_id}.png",
            exif_orientation=None,
            extraction="pdf_rendered",
            render_dpi=render_dpi,
        )
        out.append((pg, _exif_chain(1, w, h, page_id)))
    return out


def ingest(path: Path, out_dir: Path, *, render_dpi: float = 300.0) -> Manifest:
    path = Path(path)
    out_dir = Path(out_dir)
    limits = load_supported_inputs()
    source = admit(path, limits)

    (out_dir / "source").mkdir(parents=True, exist_ok=True)
    (out_dir / "pages").mkdir(parents=True, exist_ok=True)
    assert source.sha256 is not None
    dest = out_dir / "source" / f"{source.sha256}{path.suffix.lower()}"
    shutil.copy2(path, dest)

    pages: list[Page] = []
    transforms: list[TransformChain] = []
    if source.kind == SourceKind.image:
        with Image.open(path) as img:
            try:
                w, h = img.size
            except Image.DecompressionBombError:
                raise IngestRejected(ReasonCode.PIXEL_LIMIT) from None
            orientation = int(img.getexif().get(EXIF_ORIENTATION_TAG, 1) or 1)
            oriented = ImageOps_safe_transpose(img)
            ow, oh = oriented.size
            png_path = out_dir / "pages" / "page-1.png"
            oriented.save(png_path)
        pages.append(
            Page(
                page_id="page-1",
                index=0,
                width_px=ow,
                height_px=oh,
                dpi_declared=float(img.info.get("dpi", (0, 0))[0]) or None
                if img.info.get("dpi")
                else None,
                raster_path="pages/page-1.png",
                exif_orientation=orientation,
                extraction="image_file",
                render_dpi=None,
            )
        )
        transforms.append(_exif_chain(orientation, w, h, "page-1"))
    elif source.kind == SourceKind.pdf:
        for pg, chain in _ingest_pdf(path, out_dir / "pages", render_dpi):
            pages.append(pg)
            transforms.append(chain)
    else:
        raise IngestRejected(ReasonCode.UNSUPPORTED_TYPE)

    manifest = Manifest(
        schema_version="1.0.0",
        study_id=safe_study_id(path.name),
        revision=1,
        input_revision=1,
        run_id=f"ingest-{uuid.uuid4().hex[:12]}",
        config_hash=None,
        source=source,
        segments=[],
        pages=pages,
        transforms=transforms,
    )
    dump_json(manifest, out_dir / "manifest.json")
    return manifest


def ImageOps_safe_transpose(img: Image.Image) -> Image.Image:
    from PIL import ImageOps

    orientation = int(img.getexif().get(EXIF_ORIENTATION_TAG, 1) or 1)
    if orientation == 1:
        return img.copy()
    return ImageOps.exif_transpose(img)
