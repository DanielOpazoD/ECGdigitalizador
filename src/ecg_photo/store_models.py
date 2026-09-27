"""Data models, errors and small utilities of the study store (split from
store.py; `ecg_photo.store` re-exports every public name)."""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ecg_photo.paths import config_file


class StoreError(RuntimeError):
    code = "STORE_ERROR"


class RevisionConflict(StoreError):
    code = "REVISION_CONFLICT"


class QueueFull(StoreError):
    code = "QUEUE_FULL"


class EngineNotConfigured(StoreError):
    code = "ENGINE_NOT_CONFIGURED"


class StudyNotFound(StoreError):
    code = "STUDY_NOT_FOUND"


class RunNotFound(StoreError):
    code = "RUN_NOT_FOUND"


class ArtifactNotFound(StoreError):
    code = "ARTIFACT_NOT_FOUND"


class RunRevisionMismatch(StoreError):
    code = "RUN_REVISION_MISMATCH"


class StoreLocked(StoreError):
    code = "STORE_LOCKED"


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: str | None = None
    engine_duration_s: float | None = None
    engine_spec: dict[str, str] | None = None  # commit + weights sha256 (frozen)
    time_source: Literal["evidence", "engine"] = "evidence"
    gain_mm_mV: float | None = None
    speed_mm_s: float | None = None
    fs_hz: float | None = None
    layout_profile: str | None = None
    page_id: str = "page-1"


_ALLOWED_LEAD_LABELS = {
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "unknown",
}


class StudyPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: str | None = None
    engine_duration_s: float | None = None
    time_source: Literal["evidence", "engine"] | None = None
    speed_mm_s: float | None = None
    gain_mm_mV: float | None = None
    fs_hz: float | None = None
    layout_profile: str | None = None
    page_id: str | None = None
    lead_labels: dict[str, str] | None = None

    @field_validator("lead_labels")
    @classmethod
    def _check_lead_labels(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        if v is None:
            return v
        for seg_id, label in v.items():
            if label not in _ALLOWED_LEAD_LABELS:
                raise ValueError(
                    f"invalid lead label {label!r} for segment {seg_id!r}; "
                    f"allowed: {sorted(_ALLOWED_LEAD_LABELS)}"
                )
        return v


class StudyState(BaseModel):
    study_id: str
    active_revision: int
    config_hash: str
    selected_run_id: str | None = None
    published_run_id: str | None = None
    deleted: bool = False
    revisions: list[int] = Field(default_factory=lambda: [1])
    original_filename: str | None = None


class Job(BaseModel):
    run_id: str
    input_revision: int
    config_hash: str
    status: Literal[
        "queued", "running", "completed", "completed_unpublished", "failed", "cancelled"
    ] = "queued"
    stage: str = "queued"
    error: str | None = None
    created_at: str = ""
    finished_at: str | None = None
    cancel_requested: bool = False


class Artifact(BaseModel):
    artifact_id: str
    study_id: str
    run_id: str
    revision: int
    filename_safe: str
    mime: str
    path_rel: str
    sha256: str


@dataclass(frozen=True)
class ExecutionLimits:
    max_active_jobs: int = 1
    max_pending_jobs: int = 5
    public_network_binding: bool = False


def load_execution_limits(path: Path | None = None) -> ExecutionLimits:
    path = path or config_file("default.yml")
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["execution"]
    return ExecutionLimits(
        max_active_jobs=int(cfg["max_active_jobs"]),
        max_pending_jobs=int(cfg["max_pending_jobs"]),
        public_network_binding=bool(cfg["public_network_binding"]),
    )


MIME_BY_EXT = {
    ".png": "image/png",
    ".pdf": "application/pdf",
    ".csv": "text/csv",
    ".json": "application/json",
    ".hea": "text/plain",
    ".dat": "application/octet-stream",
}


def utc_now_iso() -> str:
    import datetime

    return datetime.datetime.now(datetime.UTC).isoformat()


def write_json_atomic(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if isinstance(payload, BaseModel):
        tmp.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    else:
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class RecoveryReport:
    """What `Store.recover` found after an unclean stop (T47). Entries are
    "study_id/run_id" (or study_id) strings, safe to print as JSON."""

    requeue: list[tuple[str, str]] = field(default_factory=list)
    interrupted: list[str] = field(default_factory=list)
    superseded: list[str] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    published: list[str] = field(default_factory=list)
    unpublished: list[str] = field(default_factory=list)
    finished_deletes: list[str] = field(default_factory=list)
    orphans_removed: list[str] = field(default_factory=list)
    orphans_kept: list[str] = field(default_factory=list)
    tmp_removed: int = 0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["requeue"] = [f"{s}/{r}" for s, r in self.requeue]
        return d


def _config_hash(config: RunConfig) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(config.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    ).hexdigest()
