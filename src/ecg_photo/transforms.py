"""Ordered transform chains: EXIF, crop, rotate90, affine, homography.

Each step maps points from its input frame to its output frame. A chain maps
``source_frame_id`` -> ``target_frame_id``; every step's math must be
reversible so points can be projected back to the original file frame.
"""

import numpy as np

from ecg_photo.contracts import TransformChain, TransformStep

PROCEDURE_VERSION = "f3a-transforms-1"


def _num(v: float | str | list[float], name: str) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    raise ValueError(f"transform parameter {name} must be numeric, got {v!r}")


def _pts(pts: np.ndarray) -> np.ndarray:
    a = np.asarray(pts, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("pts must be (N,2)")
    return a


def _rot90_forward(pts: np.ndarray, k: int, w: float, h: float) -> np.ndarray:
    """Rotate point coordinates CCW by k quarter-turns (PIL ROTATE_90 sense)."""
    x, y = pts[:, 0].copy(), pts[:, 1].copy()
    cw, ch = w, h
    for _ in range(k % 4):
        x, y = y, cw - 1.0 - x
        cw, ch = ch, cw
    return np.stack([x, y], axis=1)


def _rot90_inverse(pts: np.ndarray, k: int, out_w: float, out_h: float) -> np.ndarray:
    return _rot90_forward(pts, (-k) % 4, out_w, out_h)


def _affine_matrix(parameters: dict) -> np.ndarray:
    m = np.asarray(parameters["matrix"], dtype=np.float64)
    if m.shape == (6,):
        return m.reshape(2, 3)
    raise ValueError("affine 'matrix' must be 6 values (2x3 row-major)")


def _homography_matrix(parameters: dict) -> np.ndarray:
    m = np.asarray(parameters["matrix"], dtype=np.float64)
    if m.shape == (9,):
        return m.reshape(3, 3)
    raise ValueError("homography 'matrix' must be 9 values (row-major)")


def _apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    ph = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    q = ph @ H.T
    w = q[:, 2:3]
    if np.any(w == 0):
        raise ValueError("homography maps a point to infinity")
    return q[:, :2] / w


def _exif_sub_steps(orientation: int, w: int, h: int) -> list[TransformStep]:
    return exif_to_steps(orientation, w, h)


def _apply_step(step: TransformStep, pts: np.ndarray) -> np.ndarray:
    w, h = step.input_size
    if step.kind == "exif_orientation":
        out = pts
        for sub in _exif_sub_steps(int(_num(step.parameters["orientation"], "orientation")), w, h):
            out = _apply_step(sub, out)
        return out
    if step.kind == "crop":
        x0 = _num(step.parameters["x0"], "x0")
        y0 = _num(step.parameters["y0"], "y0")
        return pts - np.array([x0, y0])
    if step.kind == "rotate90":
        return _rot90_forward(pts, int(_num(step.parameters["k"], "k")), w, h)
    if step.kind == "affine":
        A = _affine_matrix(step.parameters)
        return pts @ A[:, :2].T + A[:, 2]
    if step.kind == "homography":
        return _apply_homography(_homography_matrix(step.parameters), pts)
    raise ValueError(f"unknown step kind {step.kind}")


def _invert_step(step: TransformStep, pts: np.ndarray) -> np.ndarray:
    w, h = step.input_size
    ow, oh = step.output_size
    if step.kind == "exif_orientation":
        out = pts
        for sub in reversed(
            _exif_sub_steps(int(_num(step.parameters["orientation"], "orientation")), w, h)
        ):
            out = _invert_step(sub, out)
        return out
    if step.kind == "crop":
        x0 = _num(step.parameters["x0"], "x0")
        y0 = _num(step.parameters["y0"], "y0")
        return pts + np.array([x0, y0])
    if step.kind == "rotate90":
        return _rot90_inverse(pts, int(_num(step.parameters["k"], "k")), ow, oh)
    if step.kind == "affine":
        A = _affine_matrix(step.parameters)
        A3 = np.vstack([A, [0.0, 0.0, 1.0]])
        inv = np.linalg.inv(A3)[:2]
        return pts @ inv[:, :2].T + inv[:, 2]
    if step.kind == "homography":
        return _apply_homography(np.linalg.inv(_homography_matrix(step.parameters)), pts)
    raise ValueError(f"unknown step kind {step.kind}")


def apply_chain(chain: TransformChain, pts: np.ndarray) -> np.ndarray:
    out = _pts(pts)
    for step in chain.steps:
        out = _apply_step(step, out)
    return out


def invert_chain(chain: TransformChain, pts: np.ndarray) -> np.ndarray:
    out = _pts(pts)
    for step in reversed(chain.steps):
        out = _invert_step(step, out)
    return out


def _step(
    kind: str,
    parameters: dict,
    input_size: tuple[int, int],
    output_size: tuple[int, int],
) -> TransformStep:
    return TransformStep(
        kind=kind,  # type: ignore[arg-type]
        parameters=parameters,
        input_size=input_size,
        output_size=output_size,
        procedure_version=PROCEDURE_VERSION,
    )


def exif_to_steps(orientation: int, w: int, h: int) -> list[TransformStep]:
    """EXIF orientation 1..8 as rotate90/affine steps matching PIL exif_transpose."""
    if not 1 <= orientation <= 8:
        raise ValueError(f"EXIF orientation must be 1..8, got {orientation}")
    wm1, hm1 = w - 1.0, h - 1.0
    if orientation == 1:
        return []
    if orientation == 2:
        return [_step("affine", {"matrix": [-1.0, 0.0, wm1, 0.0, 1.0, 0.0]}, (w, h), (w, h))]
    if orientation == 3:
        return [_step("rotate90", {"k": 2}, (w, h), (w, h))]
    if orientation == 4:
        return [_step("affine", {"matrix": [1.0, 0.0, 0.0, 0.0, -1.0, hm1]}, (w, h), (w, h))]
    if orientation == 5:
        return [_step("affine", {"matrix": [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]}, (w, h), (h, w))]
    if orientation == 6:
        return [_step("rotate90", {"k": 3}, (w, h), (h, w))]
    if orientation == 7:
        return [
            _step(
                "affine",
                {"matrix": [0.0, -1.0, hm1, -1.0, 0.0, wm1]},
                (w, h),
                (h, w),
            )
        ]
    return [_step("rotate90", {"k": 1}, (w, h), (h, w))]


def homography_folds(H: np.ndarray, size: tuple[int, int]) -> bool:
    """True if the homography folds the image (Jacobian determinant sign change).

    A fold means the mapping is not locally invertible everywhere in the frame,
    so the recorded chain could not be trusted for reprojection.
    """
    H = np.asarray(H, dtype=np.float64).reshape(3, 3)
    w, h = size
    xs = np.linspace(0.0, w - 1.0, 25)
    ys = np.linspace(0.0, h - 1.0, 25)
    gx, gy = np.meshgrid(xs, ys)
    eps = max(w, h) * 1e-4

    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    mapped = _apply_homography(H, pts)
    dx = _apply_homography(H, pts + np.array([eps, 0.0])) - mapped
    dy = _apply_homography(H, pts + np.array([0.0, eps])) - mapped
    det = dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0]
    det = det[np.isfinite(det)]
    if len(det) == 0:
        return True
    return bool(np.any(det > 0) and np.any(det < 0))
