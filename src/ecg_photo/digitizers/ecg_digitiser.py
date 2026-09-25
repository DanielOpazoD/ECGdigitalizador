import hashlib
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import wfdb  # type: ignore[import-untyped]

from ecg_photo.digitizers.base import (
    EngineNotReady,
    EngineOutput,
    EngineSpec,
    WeightSpec,
    load_engine_specs,
    verify_weights,
)

PATCH_MARKER = "float(rot_angle)"
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
        self.root = Path(root)
        self.python_exe = Path(python_exe)
        self.model_dir = Path(model_dir)
        self.expected_patches_applied = expected_patches_applied
        if engine_spec is None:
            engine_spec = load_engine_specs()["ecg_digitiser"]
        self.spec = engine_spec

    def _check_patch(self) -> None:
        if not self.expected_patches_applied:
            return
        digitize_py = self.root / PATCH_FILE
        if not digitize_py.exists() or PATCH_MARKER not in digitize_py.read_text(encoding="utf-8"):
            raise EngineNotReady("patch 0001 not applied")

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

        image_path = Path(image_path)
        work_dir = Path(work_dir)
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
