import numpy as np
import pytest
from PIL import Image, ImageOps

from ecg_photo.contracts import TransformChain, TransformStep
from ecg_photo.geometry import PixelScale, interval_ms, x_to_t_s
from ecg_photo.measure import hr_instant_bpm
from ecg_photo.transforms import (
    apply_chain,
    exif_to_steps,
    homography_folds,
    invert_chain,
)


def chain_with(steps: list[TransformStep]) -> TransformChain:
    return TransformChain(
        transform_id="t1", source_frame_id="file", target_frame_id="rect", steps=steps
    )


def step(
    kind: str, params: dict, insize: tuple[int, int], outsize: tuple[int, int]
) -> TransformStep:
    return TransformStep(
        kind=kind,  # type: ignore[arg-type]
        parameters=params,
        input_size=insize,
        output_size=outsize,
        procedure_version="test",
    )


SCALE = PixelScale(px_per_mm_x=10.0, px_per_mm_y=10.0)


def test_t10_translate_x_invariant() -> None:
    pts = np.array([[100.0, 50.0], [350.0, 50.0], [600.0, 50.0]])
    c = chain_with(
        [step("affine", {"matrix": [1.0, 0.0, 1000.0, 0.0, 1.0, 0.0]}, (2000, 100), (3000, 100))]
    )
    moved = apply_chain(c, pts)
    iv0 = interval_ms(pts[0, 0], pts[1, 0], SCALE, 25.0)
    iv1 = interval_ms(moved[0, 0], moved[1, 0], SCALE, 25.0)
    assert iv1 == pytest.approx(iv0)
    t = x_to_t_s(moved[:, 0], moved[0, 0], SCALE, 25.0)
    assert hr_instant_bpm(t[2] - t[1]) == pytest.approx(hr_instant_bpm(1.0))


def test_t11_scale_both_invariant() -> None:
    pts = np.array([[100.0, 50.0], [350.0, 60.0]])
    c = chain_with(
        [step("affine", {"matrix": [2.0, 0.0, 0.0, 0.0, 2.0, 0.0]}, (1000, 100), (2000, 200))]
    )
    moved = apply_chain(c, pts)
    scale2 = PixelScale(px_per_mm_x=20.0, px_per_mm_y=20.0)
    assert interval_ms(moved[0, 0], moved[1, 0], scale2, 25.0) == pytest.approx(
        interval_ms(pts[0, 0], pts[1, 0], SCALE, 25.0)
    )


def test_t12_scale_x_only_preserves_time() -> None:
    pts = np.array([[100.0, 50.0], [350.0, 60.0]])
    c = chain_with(
        [step("affine", {"matrix": [2.0, 0.0, 0.0, 0.0, 1.0, 0.0]}, (1000, 100), (2000, 100))]
    )
    moved = apply_chain(c, pts)
    scale_x2 = PixelScale(px_per_mm_x=20.0, px_per_mm_y=10.0)
    assert interval_ms(moved[0, 0], moved[1, 0], scale_x2, 25.0) == pytest.approx(
        interval_ms(pts[0, 0], pts[1, 0], SCALE, 25.0)
    )
    # y unchanged -> voltage preserved with same px/mm_y
    assert moved[0, 1] == pts[0, 1]


def test_t39_crop_rotate90_homography_roundtrip() -> None:
    H = [1.05, 0.01, 3.0, -0.02, 0.98, 5.0, 1e-5, -2e-5, 1.0]
    steps = [
        step("crop", {"x0": 10.0, "y0": 20.0, "w": 300.0, "h": 200.0}, (400, 300), (300, 200)),
        step("rotate90", {"k": 1}, (300, 200), (200, 300)),
        step("homography", {"matrix": H}, (200, 300), (200, 300)),
    ]
    c = chain_with(steps)
    rng = np.random.default_rng(0)
    pts = rng.uniform([15.0, 25.0], [390.0, 290.0], size=(20, 2))
    fwd = apply_chain(c, pts)
    back = invert_chain(c, fwd)
    assert np.max(np.abs(back - pts)) < 1e-6


def test_homography_folds_detection() -> None:
    identity = np.eye(3)
    assert homography_folds(identity, (100, 100)) is False
    mild = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1e-5, 0.0, 1.0]])
    assert homography_folds(mild, (100, 100)) is False
    strong = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.02, 0.0, 1.0]])
    assert homography_folds(strong, (100, 100)) is True


@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_corners_match_pil(orientation: int) -> None:
    w, h = 3, 2
    arr = np.arange(w * h, dtype=np.uint8).reshape(h, w) * 20 + 10
    img = Image.fromarray(arr, mode="L")
    img.getexif()[0x0112] = orientation
    transposed = np.asarray(ImageOps.exif_transpose(img))
    th, tw = transposed.shape

    steps = exif_to_steps(orientation, w, h)
    c = chain_with(steps)
    corners = np.array([[0.0, 0.0], [w - 1.0, 0.0], [0.0, h - 1.0], [w - 1.0, h - 1.0]])
    mapped = apply_chain(c, corners)
    assert (tw, th) == ((w, h) if orientation in (1, 2, 3, 4) else (h, w))
    # every mapped corner lands on an output corner
    mapped_set = {tuple(np.round(p).astype(int)) for p in mapped}
    expected = {(0, 0), (tw - 1, 0), (0, th - 1), (tw - 1, th - 1)}
    assert mapped_set == expected
    # pixel identity check: value at source corner equals value at mapped corner
    for src, dst in zip(corners, mapped, strict=True):
        sx, sy = int(src[0]), int(src[1])
        dx, dy = round(dst[0]), round(dst[1])
        assert transposed[dy, dx] == arr[sy, sx]
