import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

from ecg_photo.contracts import TransformChain, TransformStep
from ecg_photo.digitizers.base import (
    EngineNotReady,
    EngineOutput,
    EngineSpec,
    LeadGeometry,
    load_engine_specs,
    verify_weights,
)
from ecg_photo.transforms import homography_folds, homography_from_points

LEAD_NAMES_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
PATCH_MARKER = "save_geometry_json"
PATCH_FILE = "src/digitize.py"


class AhusDigitizer:
    def __init__(
        self,
        ahus_root: Path,
        python_exe: Path,
        base_config: Path,
        duration_s: float,
        device: str = "cpu",
        layout_config: str = "lead_layouts_george-moody-2024.yml",
        target_num_samples: int | None = None,
        engine_spec: EngineSpec | None = None,
        expected_patches_applied: bool = True,
    ) -> None:
        # absolute paths: the engine subprocess runs with cwd=ahus_root, where
        # relative paths given by the caller would not resolve. The venv
        # interpreter is made absolute, not resolved (it is a symlink).
        self.ahus_root = Path(ahus_root).resolve()
        self.python_exe = Path(python_exe).absolute()
        self.base_config = Path(base_config).resolve()
        self.duration_s = float(duration_s)
        self.device = device
        self.layout_config = layout_config
        self.target_num_samples = target_num_samples
        self.expected_patches_applied = expected_patches_applied
        if engine_spec is None:
            engine_spec = load_engine_specs()["ahus"]
        self.spec = engine_spec

    def _check_patch(self) -> None:
        if not self.expected_patches_applied:
            return
        digitize_py = self.ahus_root / PATCH_FILE
        if not digitize_py.exists() or PATCH_MARKER not in digitize_py.read_text(encoding="utf-8"):
            raise EngineNotReady("ahus geometry patch 0001 not applied")

    def _layout_path(self) -> Path:
        """Layout set file: a name in Ahus' src/config/, or a path to our own
        (e.g. configs/ahus_lead_layouts.yml)."""
        own = Path(self.layout_config)
        if own.suffix in (".yml", ".yaml") and own.exists() and own.parent != Path("."):
            return own.resolve()
        return self.ahus_root / "src" / "config" / self.layout_config

    def _effective_config(self, work_dir: Path) -> Path:
        cfg = yaml.safe_load(self.base_config.read_text(encoding="utf-8"))
        kw = cfg["MODEL"]["KWARGS"]
        kw["device"] = self.device
        kw["config"]["LAYOUT_IDENTIFIER"]["KWARGS"]["device"] = self.device
        kw["config"]["LAYOUT_IDENTIFIER"]["config_path"] = str(self._layout_path())
        if self.target_num_samples is not None:
            kw["config"]["LAYOUT_IDENTIFIER"]["KWARGS"]["target_num_samples"] = (
                self.target_num_samples
            )
        in_dir = work_dir / "in"
        out_dir = work_dir / "out"
        in_dir.mkdir(parents=True, exist_ok=True)
        cfg["DATA"]["images_path"] = str(in_dir)
        cfg["DATA"]["output_path"] = str(out_dir)
        cfg_path = work_dir / "effective_config.yml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return cfg_path

    def run(self, image_path: Path, work_dir: Path) -> EngineOutput:
        self._check_patch()
        problems = verify_weights(self.ahus_root, self.spec)
        if problems:
            raise EngineNotReady("ahus weights invalid: " + "; ".join(problems))
        work_dir = Path(work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = self._effective_config(work_dir)
        # the layout set is part of the configuration: hash its content too
        config_hash = hashlib.sha256(
            cfg_path.read_bytes()
            + (self._layout_path().read_bytes() if self._layout_path().exists() else b"")
        ).hexdigest()

        in_dir = work_dir / "in"
        shutil.copy2(image_path, in_dir / Path(image_path).name)

        env = {**os.environ, "PYTHONPATH": str(self.ahus_root)}
        t0 = time.monotonic()
        proc = subprocess.run(
            [
                str(self.python_exe),
                "src/digitize.py",
                "--config",
                str(cfg_path),
            ],
            cwd=self.ahus_root,
            env=env,
            check=False,
            timeout=3600,
            capture_output=True,
            text=True,
        )
        wall = time.monotonic() - t0
        if proc.returncode != 0:
            # the cause is at the end of stderr (progress output comes first)
            raise EngineNotReady(
                f"ahus subprocess failed (rc={proc.returncode}): {proc.stderr[-2000:]}"
            )

        out_dir = work_dir / "out"
        leads, observed, fs_hz, layout = parse_ahus_output(out_dir, self.duration_s)
        geometry = parse_ahus_geometry(out_dir, n_samples=len(next(iter(leads.values()))))
        return EngineOutput(
            engine_id=self.spec.engine_id,
            engine_commit=self.spec.commit,
            weights_sha256=tuple(w.sha256 for w in self.spec.weights),
            config_hash=config_hash,
            fs_hz=fs_hz,
            leads=leads,
            observed=observed,
            layout_detected=layout,
            extra={
                "device": self.device,
                "layout_config": self.layout_config,
                "duration_s": self.duration_s,
            },
            wall_time_s=wall,
            geometry=geometry,
        )


def parse_ahus_output(
    out_dir: Path, duration_s: float
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float, str | None]:
    out_dir = Path(out_dir)
    csv_files = sorted(out_dir.glob("*_timeseries_canonical.csv"))
    if not csv_files:
        raise FileNotFoundError(f"no *_timeseries_canonical.csv in {out_dir}")
    path = csv_files[0]
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(v) if v else np.nan for v in r] for r in reader if r]
    data = np.array(rows, dtype=np.float64) / 1000.0
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    leads = {name: data[:, i] for i, name in enumerate(header)}
    observed = {name: np.isfinite(arr) for name, arr in leads.items()}
    fs_hz = float(data.shape[0]) / float(duration_s)

    layout: str | None = None
    meta = out_dir / "digitization_metadata.csv"
    if meta.exists():
        with open(meta, newline="", encoding="utf-8") as f:
            mrows = list(csv.DictReader(f))
        if mrows:
            layout = mrows[-1].get("lead_layout") or None
    return leads, observed, fs_hz, layout


def parse_ahus_geometry(out_dir: Path, n_samples: int) -> dict[str, LeadGeometry] | None:
    """Build canonical-index -> page-pixel-x geometry from <base>_geometry.json.

    Mapping: canonical k -> aligned col (affine over the valid span) ->
    resampled px (inverse of the destination->source perspective) -> original
    px (1/resample_scale). Returns None when the file is absent; raises
    EngineNotReady when apply_dewarping is set (non-linear x warp unsupported)
    or when the inverse homography folds.
    """
    out_dir = Path(out_dir)
    geo_files = sorted(out_dir.glob("*_geometry.json"))
    if not geo_files:
        return None
    g = json.loads(geo_files[0].read_text(encoding="utf-8"))
    if g.get("apply_dewarping"):
        raise EngineNotReady("dewarping geometry not supported")

    src = np.asarray(g["source_points"], dtype=np.float64)
    dst = np.asarray(g["destination_points"], dtype=np.float64)
    # perspective() maps source -> destination; we need aligned -> resampled
    H_inv = homography_from_points(dst, src)
    aligned_w, aligned_h = (int(v) for v in g["aligned_size"])
    if homography_folds(H_inv, (aligned_w, aligned_h)):
        return None

    span = g["valid_span"]
    scale = float(g["resample_scale"])
    first, last = float(span[0]), float(span[1])
    n = int(g.get("target_num_samples") or n_samples)
    k = np.arange(n, dtype=np.float64)
    col = first + k * (last - first) / (n - 1)

    # resampled px via the inverse homography (evaluate at the canvas mid-row;
    # the homography row-mixing means a y is needed — use the aligned centre row)
    y_mid = aligned_h / 2.0
    pts = np.stack([col, np.full_like(col, y_mid)], axis=1)
    ph = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    q = ph @ H_inv.T
    x_resampled = q[:, 0] / q[:, 2]
    x_file = x_resampled / scale

    chain = TransformChain(
        transform_id="engine-geometry-ahus",
        source_frame_id="engine-canonical:ahus",
        target_frame_id="file",
        steps=[
            TransformStep(
                kind="affine",
                parameters={
                    "matrix": [
                        (last - first) / (n - 1),
                        0.0,
                        first,
                        0.0,
                        1.0,
                        0.0,
                    ]
                },
                input_size=(n, 1),
                output_size=(aligned_w, aligned_h),
                procedure_version="f4c-ahus-geometry",
            ),
            TransformStep(
                kind="homography",
                parameters={"matrix": H_inv.reshape(-1).tolist(), "y_eval": y_mid},
                input_size=(aligned_w, aligned_h),
                output_size=(int(g["resampled_size"][0]), int(g["resampled_size"][1])),
                procedure_version="f4c-ahus-geometry",
            ),
            TransformStep(
                kind="affine",
                parameters={
                    "matrix": [
                        1.0 / scale,
                        0.0,
                        0.0,
                        0.0,
                        1.0 / scale,
                        0.0,
                    ]
                },
                input_size=(int(g["resampled_size"][0]), int(g["resampled_size"][1])),
                output_size=(int(g["original_size"][0]), int(g["original_size"][1])),
                procedure_version="f4c-ahus-geometry",
            ),
        ],
    )
    geo = LeadGeometry(
        frame_id="file",
        x_px=x_file,
        y_ref_px=None,
        chain=chain,
        method=(
            "canonical k -> aligned col (affine over engine valid_span) -> "
            "resampled px (inverse source<->destination homography) -> original px"
        ),
        limitations=(
            "x evaluated at the aligned image centre row; perspective row-mixing "
            "means x has a small y dependence not resolved per sample; "
            "valid_span and canvas sizes come from the engine patch output"
        ),
    )
    return {name: geo for name in LEAD_NAMES_12}
