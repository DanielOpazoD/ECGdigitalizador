"""Admission and ingest of real inputs (T42-T44).

Admission is decided by magic bytes and declared dimensions only — a payload
is never fully decoded before the pixel/page/size limits are checked.
"""

import json
import re
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
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
from ecg_photo.grid import GridEstimate, estimate_grid, resolve_ambiguous_period
from ecg_photo.transforms import exif_to_steps

EXIF_ORIENTATION_TAG = 0x0112
# Low-resolution photos (e.g. re-compressed by messaging apps) are upsampled
# before the engine. F7 (Kaggle phone photos, Ahus, 10 records): median r@lag
# 0.831 at 4032 px, 0.834 at 1600 px, 0.740 at 1000 px and 0.845 at 1000 px
# upsampled x2 (QC insufficient 9/10 -> 1/10). Integer factor so pixel centres
# map exactly; recorded as an affine step of the file -> page transform.
MIN_PAGE_WIDTH_PX = 1600
TARGET_PAGE_WIDTH_PX = 2000
MAX_UPSAMPLE = 4
MIN_UPSAMPLE_SOURCE_PX = 400  # narrower is not a readable ECG page: left as is
UPSAMPLE_VERSION = "f7-lanczos-upsample-v1"
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
    """Metadata-safe filename: control chars out, HTML escaped (T44)."""
    import html

    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", name)
    return html.escape(cleaned)[:255]


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


def upsample_factor(width_px: int) -> int:
    """Integer factor that brings a page narrower than MIN_PAGE_WIDTH_PX to at
    least TARGET_PAGE_WIDTH_PX (capped at MAX_UPSAMPLE); 1 otherwise."""
    if width_px < MIN_UPSAMPLE_SOURCE_PX or width_px >= MIN_PAGE_WIDTH_PX:
        return 1
    return int(min(MAX_UPSAMPLE, -(-TARGET_PAGE_WIDTH_PX // width_px)))


def page_upsample(manifest: Manifest, page_id: str) -> int:
    """Upsampling factor applied to a page raster at ingest (1 if none)."""
    for t in manifest.transforms:
        if t.target_frame_id == f"rectified-{page_id}":
            for st in t.steps:
                if st.procedure_version == UPSAMPLE_VERSION:
                    return int(st.parameters["upsample"])  # type: ignore[arg-type]
    return 1


def _upsample_step(k: int, w: int, h: int) -> TransformStep:
    # pixel-centre convention of an integer Lanczos resize: x' = k (x + 0.5) - 0.5
    c = 0.5 * k - 0.5
    return TransformStep(
        kind="affine",
        parameters={"matrix": [float(k), 0.0, c, 0.0, float(k), c], "upsample": k},
        input_size=(w, h),
        output_size=(w * k, h * k),
        procedure_version=UPSAMPLE_VERSION,
    )


def _exif_chain(
    orientation: int, w: int, h: int, page_id: str, upsample: int = 1
) -> TransformChain:
    steps: list[TransformStep] = exif_to_steps(orientation, w, h)
    if upsample > 1:
        ow, oh = (h, w) if orientation in (5, 6, 7, 8) else (w, h)
        steps = [*steps, _upsample_step(upsample, ow, oh)]
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
            k = upsample_factor(oriented.size[0])
            if k > 1:
                oriented = oriented.resize(
                    (oriented.size[0] * k, oriented.size[1] * k), Image.Resampling.LANCZOS
                )
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
        transforms.append(_exif_chain(orientation, w, h, "page-1", upsample=k))
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


def estimate_page_grids(out_dir: Path, manifest: Manifest) -> tuple[Manifest, list[GridEstimate]]:
    """Estimate the grid of every page raster, write `pages/<page>.grid.json`,
    record `grid_estimate_path` and rewrite `manifest.json`. A null estimate is
    still written (with its limitations) so the absence of evidence is explicit."""
    grids: list[GridEstimate] = []
    pages: list[Page] = []
    for pg in manifest.pages:
        img = np.asarray(Image.open(out_dir / pg.raster_path))
        grid = resolve_ambiguous_period(estimate_grid(img), img)
        rel = f"pages/{pg.page_id}.grid.json"
        (out_dir / rel).write_text(
            json.dumps(asdict(grid), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        pages.append(pg.model_copy(update={"grid_estimate_path": rel}))
        grids.append(grid)
    manifest = manifest.model_copy(update={"pages": pages})
    dump_json(manifest, out_dir / "manifest.json")
    return manifest, grids
