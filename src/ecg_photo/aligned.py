"""Grid scale in the engine's perspective-corrected (aligned) frame.

Ahus rectifies the page before reading the traces: it maps a quadrilateral of
the (resampled) photo onto an axis-aligned canvas with a homography, and its
canonical samples are linear in the columns of that canvas. The geometry
chain written by the adapter records this ([affine canonical -> aligned col,
homography aligned -> resampled, affine resampled -> page]). Warping the page
raster into the aligned frame removes the perspective, so one grid scale is
valid there even for phone photos, and the time axis of a lead is its extent
in aligned columns over that scale.

F7 (Kaggle, Ahus): measuring the grid in the aligned frame gave a scale on
10/10 scans and 9-10/10 photos of every type (page frame: 0-3/10 photos) with
median strip-duration errors 0.0-0.8 %; outliers were engine failures (canvas
placed on part of the page) or a photo of two overlapping sheets, which the
quality report flags as RHYTHM_DURATION_MISMATCH.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ecg_photo.contracts import TransformChain
from ecg_photo.grid import GridEstimate, estimate_grid, resolve_ambiguous_period

AHUS_GEOMETRY_VERSION = "f4c-ahus-geometry"
ALIGNED_FRAME = "engine-aligned:ahus"


@dataclass(frozen=True)
class AlignedGeometry:
    col_per_sample: float  # aligned columns per canonical sample
    col_first: float  # aligned column of canonical sample 0
    h_aligned_to_resampled: np.ndarray  # 3x3
    resample_scale: float  # resampled px per page px
    aligned_size: tuple[int, int]


def aligned_geometry(chain: TransformChain) -> AlignedGeometry | None:
    """The aligned-frame parameters of an Ahus geometry chain, or None for any
    other chain (other engines, older runs)."""
    steps = chain.steps
    if (
        len(steps) != 3
        or any(s.procedure_version != AHUS_GEOMETRY_VERSION for s in steps)
        or [s.kind for s in steps] != ["affine", "homography", "affine"]
    ):
        return None
    m0 = [float(v) for v in steps[0].parameters["matrix"]]  # type: ignore[union-attr]
    h = np.asarray(steps[1].parameters["matrix"], dtype=np.float64).reshape(3, 3)
    m2 = [float(v) for v in steps[2].parameters["matrix"]]  # type: ignore[union-attr]
    if not m2[0] > 0:
        return None
    return AlignedGeometry(
        col_per_sample=m0[0],
        col_first=m0[2],
        h_aligned_to_resampled=h,
        resample_scale=1.0 / m2[0],
        aligned_size=(int(steps[0].output_size[0]), int(steps[0].output_size[1])),
    )


def warp_to_aligned(page: np.ndarray, geo: AlignedGeometry) -> np.ndarray:
    """Page raster (page px) resampled into the aligned frame."""
    s = geo.resample_scale
    # output (aligned) -> input (page): page = S^-1 . H_inv . aligned
    back = np.diag([1.0 / s, 1.0 / s, 1.0]) @ geo.h_aligned_to_resampled
    back = back / back[2, 2]
    img = Image.fromarray(np.asarray(page, dtype=np.uint8))
    out = img.transform(
        geo.aligned_size,
        Image.Transform.PERSPECTIVE,
        tuple(float(v) for v in back.reshape(-1)[:8]),
        Image.Resampling.BICUBIC,
    )
    return np.asarray(out)


def aligned_grid(page_png: Path, geo: AlignedGeometry) -> GridEstimate:
    """Grid estimate (incl. the page-size prior and its uniformity check) of
    the page warped into the aligned frame; px/mm are aligned columns."""
    page = np.asarray(Image.open(page_png).convert("RGB"))
    img = warp_to_aligned(page, geo)
    return resolve_ambiguous_period(estimate_grid(img), img)


def grid_summary(est: GridEstimate) -> dict:
    d = asdict(est)
    return {
        k: d[k]
        for k in (
            "px_per_mm_x",
            "period_step_mm",
            "band_spread_rel_x",
            "ambiguous_period_px_x",
            "method",
            "limitations",
        )
    }
