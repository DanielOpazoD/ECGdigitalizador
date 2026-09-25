import csv
import hashlib
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

from ecg_photo.digitizers.base import (
    EngineNotReady,
    EngineOutput,
    EngineSpec,
    load_engine_specs,
    verify_weights,
)

LEAD_NAMES_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


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
    ) -> None:
        self.ahus_root = Path(ahus_root)
        self.python_exe = Path(python_exe)
        self.base_config = Path(base_config)
        self.duration_s = float(duration_s)
        self.device = device
        self.layout_config = layout_config
        self.target_num_samples = target_num_samples
        if engine_spec is None:
            engine_spec = load_engine_specs()["ahus"]
        self.spec = engine_spec

    def _effective_config(self, work_dir: Path) -> Path:
        cfg = yaml.safe_load(self.base_config.read_text(encoding="utf-8"))
        kw = cfg["MODEL"]["KWARGS"]
        kw["device"] = self.device
        kw["config"]["LAYOUT_IDENTIFIER"]["KWARGS"]["device"] = self.device
        kw["config"]["LAYOUT_IDENTIFIER"]["config_path"] = f"src/config/{self.layout_config}"
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
        problems = verify_weights(self.ahus_root, self.spec)
        if problems:
            raise EngineNotReady("ahus weights invalid: " + "; ".join(problems))
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = self._effective_config(work_dir)
        config_hash = hashlib.sha256(cfg_path.read_bytes()).hexdigest()

        in_dir = work_dir / "in"
        shutil.copy2(image_path, in_dir / Path(image_path).name)

        env = {**os.environ, "PYTHONPATH": str(self.ahus_root)}
        t0 = time.monotonic()
        subprocess.run(
            [
                str(self.python_exe),
                "src/digitize.py",
                "--config",
                str(cfg_path),
            ],
            cwd=self.ahus_root,
            env=env,
            check=True,
            timeout=3600,
            capture_output=True,
            text=True,
        )
        wall = time.monotonic() - t0

        out_dir = work_dir / "out"
        leads, observed, fs_hz, layout = parse_ahus_output(out_dir, self.duration_s)
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
