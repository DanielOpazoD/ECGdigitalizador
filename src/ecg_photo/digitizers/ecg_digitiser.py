import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import wfdb  # type: ignore[import-untyped]

from ecg_photo.contracts import TransformChain, TransformStep
from ecg_photo.digitizers.base import (
    EngineNotReady,
    EngineOutput,
    EngineSpec,
    LeadGeometry,
    WeightSpec,
    load_engine_specs,
    verify_weights,
)

PATCH_MARKERS = ("float(rot_angle)", "_geometry.json")
PATCH_FILE = "src/run/digitize.py"


class EcgDigitiserDigitizer:
    def __init__(
        self,
        root: Path,
        python_exe: Path,
        model_dir: Path,
        *,
        engine_spec: EngineSpec | None = None,
        expected_patches_applied: bool = True,
    ) -> None:
        # absolute paths: the engine subprocess runs with cwd=root (see Ahus)
        self.root = Path(root).resolve()
        self.python_exe = Path(python_exe).absolute()
        self.model_dir = Path(model_dir)
        self.expected_patches_applied = expected_patches_applied
        if engine_spec is None:
            engine_spec = load_engine_specs()["ecg_digitiser"]
        self.spec = engine_spec

    def _check_patch(self) -> None:
        if not self.expected_patches_applied:
            return
        digitize_py = self.root / PATCH_FILE
        text = digitize_py.read_text(encoding="utf-8") if digitize_py.exists() else ""
        if not all(m in text for m in PATCH_MARKERS):
            raise EngineNotReady("ecg-digitiser patches 0001/0002 not applied")

    def _weights_under_model_dir(self) -> list[WeightSpec]:
        model_rel = str(self.model_dir).rstrip("/")
        return [
            w
            for w in self.spec.weights
            if w.path == model_rel or w.path.startswith(model_rel + "/")
        ]

    def run(self, image_path: Path, work_dir: Path) -> EngineOutput:
        # Patch check first, then weights: the patch check needs no weight files,
        # so a fake root without weights can still exercise it cheaply.
        self._check_patch()
        sub_spec = EngineSpec(
            engine_id=self.spec.engine_id,
            repo=self.spec.repo,
            commit=self.spec.commit,
            license=self.spec.license,
            weights=tuple(self._weights_under_model_dir()),
            patches=self.spec.patches,
        )
        problems = verify_weights(self.root, sub_spec)
        if problems:
            raise EngineNotReady("ecg_digitiser weights invalid: " + "; ".join(problems))

        image_path = Path(image_path).resolve()
        work_dir = Path(work_dir).resolve()
        in_dir = work_dir / "in"
        out_dir = work_dir / "out"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, in_dir / image_path.name)

        cmd = [
            str(self.python_exe),
            "-m",
            "src.run.digitize",
            "-d",
            str(in_dir),
            "-m",
            str(self.model_dir),
            "-o",
            str(out_dir),
            "-v",
        ]
        env = {
            **os.environ,
            "PATH": str(self.python_exe.parent) + os.pathsep + os.environ.get("PATH", ""),
            "PYTHONPATH": str(self.root),
        }
        t0 = time.monotonic()
        proc = subprocess.run(
            cmd,
            check=False,
            cwd=self.root,
            env=env,
            timeout=3600,
            capture_output=True,
            text=True,
        )
        wall = time.monotonic() - t0
        if proc.returncode != 0:
            raise EngineNotReady(
                f"ecg_digitiser subprocess failed (rc={proc.returncode}): {proc.stderr[-2000:]}"
            )

        model_dir_abs = (
            self.model_dir if self.model_dir.is_absolute() else self.root / self.model_dir
        )
        h = hashlib.sha256()
        for name in ("dataset.json", "plans.json"):
            matches = sorted(model_dir_abs.rglob(name))
            if not matches:
                raise EngineNotReady(f"{name} not found under {model_dir_abs}")
            h.update(matches[0].read_bytes())
        h.update(str(self.model_dir).encode("utf-8"))
        config_hash = h.hexdigest()

        record_name = image_path.stem
        leads, observed, fs_hz, extra = parse_wfdb_output(out_dir, record_name)
        n_samples = len(next(iter(leads.values())))
        geometry = parse_digitiser_geometry(out_dir, record_name, n_samples, set(leads), fs_hz)
        return EngineOutput(
            engine_id=self.spec.engine_id,
            engine_commit=self.spec.commit,
            weights_sha256=tuple(w.sha256 for w in sub_spec.weights),
            config_hash=config_hash,
            fs_hz=fs_hz,
            leads=leads,
            observed=observed,
            layout_detected=None,
            extra=extra,
            wall_time_s=wall,
            geometry=geometry,
        )


def parse_wfdb_output(
    out_dir: Path, record_name: str
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    float,
    dict[str, str | float | int | bool | None],
]:
    record = wfdb.rdrecord(str(Path(out_dir) / record_name))
    leads = {
        name: np.asarray(record.p_signal[:, i], dtype=np.float64)
        for i, name in enumerate(record.sig_name)
    }
    # The engine applies np.nan_to_num before wrsamp, so every sample is finite;
    # missing regions are encoded as exact 0.0 and reported per lead.
    observed = {name: np.isfinite(arr) for name, arr in leads.items()}
    extra: dict[str, str | float | int | bool | None] = {"missing_encoded_as_zero": True}
    for name, arr in leads.items():
        extra[f"exact_zero_fraction_{name}"] = float(np.mean(arr == 0))
    return leads, observed, float(record.fs), extra


def parse_digitiser_geometry(
    out_dir: Path, record_name: str, n_samples: int, lead_names: set[str], fs_hz: float
) -> dict[str, LeadGeometry] | None:
    """Build sample-index -> page-pixel-x geometry from <record>_geometry.json.

    The engine resamples each lead mask's columns onto its canonical fs_hz grid
    using its assumed sec_per_pixel (2.5 s per 2.5-s slot width), so canonical
    sample i -> rotated column x1 + (i/fs_hz)/sec_per_pixel, then rotated ->
    original by the inverse of torchvision's rotate(angle) about the image
    centre (forward map verified empirically: p_rot = c + [[cos, sin],
    [-sin, cos]] . (p - c) in pixel coords for a counter-clockwise image
    rotation of `angle` degrees). Samples whose mapped column falls beyond the
    mask width are padding and carry NaN x — they carry no geometry.
    """
    out_dir = Path(out_dir)
    geo_path = out_dir / f"{record_name}_geometry.json"
    if not geo_path.exists():
        return None
    g = json.loads(geo_path.read_text(encoding="utf-8"))
    rot_deg = float(g["rot_angle"])
    spp = float(g["sec_per_pixel_engine_assumed"])
    rw, rh = (int(v) for v in g["rotated_shape"])
    ow, oh = (int(v) for v in g["original_shape"])
    cx, cy = rw / 2.0, rh / 2.0
    th = np.deg2rad(rot_deg)
    c, s = float(np.cos(th)), float(np.sin(th))

    out: dict[str, LeadGeometry] = {}
    for lead, box in g["leads"].items():
        if lead not in lead_names:
            continue
        x1, y1 = float(box["x1"]), float(box["y1"])
        w, h = float(box["width"]), float(box["height"])
        y_mid = y1 + h / 2.0
        px_per_sample = 1.0 / (fs_hz * spp)
        x_rot = x1 + np.arange(n_samples, dtype=np.float64) * px_per_sample
        x_file = c * (x_rot - cx) - s * (y_mid - cy) + cx
        x_px = np.where(x_rot < x1 + w, x_file, np.nan)

        chain = TransformChain(
            transform_id=f"engine-geometry-ecg-digitiser-{lead}",
            source_frame_id="engine-canonical:ecg_digitiser",
            target_frame_id="file",
            steps=[
                TransformStep(
                    kind="affine",
                    parameters={
                        "matrix": [px_per_sample, 0.0, x1, 0.0, 0.0, y_mid],
                        "mask_x1": x1,
                        "mask_y1": y1,
                        "mask_width": w,
                        "mask_height": h,
                        "sec_per_pixel_engine_assumed": spp,
                        "engine_fs_hz": fs_hz,
                    },
                    input_size=(n_samples, 1),
                    output_size=(rw, rh),
                    procedure_version="f4c-digitiser-geometry",
                ),
                TransformStep(
                    kind="affine",
                    parameters={
                        "matrix": [
                            c,
                            -s,
                            cx - c * cx + s * cy,
                            s,
                            c,
                            cy - s * cx - c * cy,
                        ],
                        "rot_angle_deg": rot_deg,
                    },
                    input_size=(rw, rh),
                    output_size=(ow, oh),
                    procedure_version="f4c-digitiser-geometry",
                ),
            ],
        )
        out[lead] = LeadGeometry(
            frame_id="file",
            x_px=x_px,
            y_ref_px=None,
            chain=chain,
            method=(
                "sample i -> rotated col x1 + (i/fs)/sec_per_pixel (engine "
                "resamples mask columns onto its canonical grid) -> "
                "original px via inverse rotation about image centre"
            ),
            limitations=(
                "x evaluated at the mask's vertical centre row; the true signal "
                "row varies per column, so x carries a small residual error "
                "proportional to rot_angle; samples mapping beyond the mask "
                "width are padding with no geometry"
            ),
        )
    return out or None
