import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from ecg_photo.contracts import sha256_file


class EngineNotReady(RuntimeError):
    pass


@dataclass(frozen=True)
class WeightSpec:
    path: str
    sha256: str


@dataclass(frozen=True)
class EngineSpec:
    engine_id: str
    repo: str
    commit: str
    license: str
    weights: tuple[WeightSpec, ...]


@dataclass
class EngineOutput:
    engine_id: str
    engine_commit: str
    weights_sha256: tuple[str, ...]
    config_hash: str
    fs_hz: float
    leads: dict[str, np.ndarray] = field(default_factory=dict)
    observed: dict[str, np.ndarray] = field(default_factory=dict)
    layout_detected: str | None = None
    extra: dict[str, str | float | int | bool | None] = field(default_factory=dict)
    wall_time_s: float = 0.0


class Digitizer(Protocol):
    spec: EngineSpec

    def run(self, image_path: Path, work_dir: Path) -> EngineOutput: ...


def load_engine_specs(path: Path = Path("configs/checkpoints.json")) -> dict[str, EngineSpec]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    specs: dict[str, EngineSpec] = {}
    for e in data["engines"]:
        specs[e["engine_id"]] = EngineSpec(
            engine_id=e["engine_id"],
            repo=e["repo"],
            commit=e["commit"],
            license=e["license"],
            weights=tuple(WeightSpec(path=w["path"], sha256=w["sha256"]) for w in e["weights"]),
        )
    return specs


def verify_weights(root: Path, spec: EngineSpec) -> list[str]:
    problems: list[str] = []
    for w in spec.weights:
        p = Path(root) / w.path
        if not p.exists():
            problems.append(f"{spec.engine_id}: missing weight {w.path}")
            continue
        actual = sha256_file(p)
        if actual != w.sha256:
            problems.append(
                f"{spec.engine_id}: sha256 mismatch for {w.path}: expected {w.sha256}, got {actual}"
            )
    return problems
