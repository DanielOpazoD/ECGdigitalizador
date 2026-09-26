import numpy as np
import pytest
from PIL import Image

from ecg_photo.aligned import aligned_geometry, aligned_grid, warp_to_aligned
from ecg_photo.contracts import TransformChain, TransformStep
from ecg_photo.grid import estimate_grid, resolve_ambiguous_period

V = "f4c-ahus-geometry"
AW, AH = 3000, 1960
PX_MM = 10.2  # aligned px per mm; 5 mm lines every 51 px


def _chain(h_inv: np.ndarray, scale: float) -> TransformChain:
    def step(kind, params, i, o) -> TransformStep:
        return TransformStep(
            kind=kind, parameters=params, input_size=i, output_size=o, procedure_version=V
        )

    return TransformChain(
        transform_id="engine-geometry-ahus",
        source_frame_id="engine-canonical:ahus",
        target_frame_id="file",
        steps=[
            step("affine", {"matrix": [0.3, 0.0, 12.0, 0.0, 1.0, 0.0]}, (10000, 1), (AW, AH)),
            step("homography", {"matrix": h_inv.reshape(-1).tolist()}, (AW, AH), (3200, 2400)),
            step(
                "affine", {"matrix": [1 / scale, 0, 0, 0, 1 / scale, 0]}, (3200, 2400), (3200, 2400)
            ),
        ],
    )


def _aligned_paper() -> np.ndarray:
    """Only 5 mm lines (minor lines lost, as in phone photos): ambiguous grid."""
    img = np.full((AH, AW, 3), 245, dtype=np.uint8)
    p = 5 * PX_MM
    for x in np.round(np.arange(0, AW, p)).astype(int):
        img[:, x : x + 2] = (230, 80, 80)
    for y in np.round(np.arange(0, AH, p)).astype(int):
        img[y : y + 2, :] = (230, 80, 80)
    return img


def test_aligned_frame_undoes_perspective(tmp_path) -> None:
    # page photographed at an angle: keystone homography aligned -> page
    h_inv = np.array([[0.9, 0.06, 120.0], [0.0, 0.85, 150.0], [0.0, 4e-5, 1.0]])
    geo = aligned_geometry(_chain(h_inv, scale=1.0))
    assert geo is not None and geo.col_per_sample == 0.3 and geo.col_first == 12.0
    # render the page: page pixel -> aligned pixel is inv(h_inv)
    fwd = np.linalg.inv(h_inv)
    fwd /= fwd[2, 2]
    page = Image.fromarray(_aligned_paper()).transform(
        (3200, 2400), Image.Transform.PERSPECTIVE, tuple(fwd.reshape(-1)[:8]), Image.BICUBIC
    )
    page_arr = np.asarray(page)
    # in the page frame the scale varies with the perspective: no global scale
    raw = resolve_ambiguous_period(estimate_grid(page_arr), page_arr)
    assert raw.px_per_mm_x is None
    png = tmp_path / "page.png"
    page.save(png)
    est = aligned_grid(png, geo)
    assert est.period_step_mm == 5.0
    assert est.px_per_mm_x == pytest.approx(PX_MM, rel=0.01)
    back = warp_to_aligned(page_arr, geo)
    assert back.shape[:2] == (AH, AW)


def test_other_chains_have_no_aligned_frame() -> None:
    other = TransformChain(
        transform_id="engine-geometry-ecg-digitiser-II",
        source_frame_id="engine-canonical:ecg_digitiser",
        target_frame_id="file",
        steps=[
            TransformStep(
                kind="affine",
                parameters={"matrix": [1.0, 0, 0, 0, 1, 0]},
                input_size=(10, 1),
                output_size=(10, 1),
                procedure_version="f4c-digitiser-geometry",
            )
        ],
    )
    assert aligned_geometry(other) is None
