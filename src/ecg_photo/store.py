"""Study store / coordinator — no FastAPI here, testable on its own.

Layout STORE/<study_id>/:
    state.json           atomic (tmp + os.replace)
    rev<N>/              revision dir (rev1 = ingest output); config.json +
                         calibration.json live inside the revision
    runs/<run_id>/job.json + result/
    artifacts.json       registered export artifacts
"""

import json
import os
import shutil
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ecg_photo.contracts import (
    CalibrationEvidence,
    dump_json,
    load_manifest,
    sha256_file,
    validate_revision_dir,
)
from ecg_photo.digitizers.base import EngineSpec, load_engine_specs
from ecg_photo.ingest import SupportedInputs, admit, escape_filename, ingest, safe_study_id


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
    path = path or Path("configs/default.yml")
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


def _now() -> str:
    import datetime

    return datetime.datetime.now(datetime.UTC).isoformat()


def _atomic_json(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if isinstance(payload, BaseModel):
        tmp.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    else:
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _config_hash(config: RunConfig) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(config.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    ).hexdigest()


class Store:
    def __init__(self, root: Path, limits: SupportedInputs, execution: ExecutionLimits) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.limits = limits
        self.execution = execution
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # -- internals -----------------------------------------------------
    def _study_dir(self, study_id: str) -> Path:
        return self.root / study_id

    def _lock(self, study_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(study_id, threading.Lock())

    def _state_path(self, study_id: str) -> Path:
        return self._study_dir(study_id) / "state.json"

    def _read_state(self, study_id: str) -> StudyState | None:
        p = self._state_path(study_id)
        if not p.exists():
            return None
        return StudyState(**json.loads(p.read_text(encoding="utf-8")))

    def _write_state(self, state: StudyState) -> None:
        _atomic_json(self._state_path(state.study_id), state)

    def _rev_dir(self, study_id: str, revision: int) -> Path:
        return self._study_dir(study_id) / f"rev{revision}"

    def _config_path(self, study_id: str, revision: int) -> Path:
        return self._rev_dir(study_id, revision) / "config.json"

    def _read_config(self, study_id: str, revision: int) -> RunConfig:
        return RunConfig(**json.loads(self._config_path(study_id, revision).read_text()))

    def _job_path(self, study_id: str, run_id: str) -> Path:
        return self._study_dir(study_id) / "runs" / run_id / "job.json"

    def _read_job(self, study_id: str, run_id: str) -> Job | None:
        p = self._job_path(study_id, run_id)
        if not p.exists():
            return None
        return Job(**json.loads(p.read_text(encoding="utf-8")))

    def write_job(self, study_id: str, job: Job) -> None:
        _atomic_json(self._job_path(study_id, run_id=job.run_id), job)

    def run_dir(self, study_id: str, run_id: str) -> Path:
        return self._study_dir(study_id) / "runs" / run_id

    def job_for_run(self, run_id: str) -> tuple[str, Job] | None:
        for sdir in self.root.iterdir():
            if not sdir.is_dir():
                continue
            job = self._read_job(sdir.name, run_id)
            if job is not None:
                return sdir.name, job
        return None

    # -- studies -------------------------------------------------------
    def create_study(
        self,
        upload_path: Path,
        original_filename: str,
        *,
        default_engine: str | None = None,
        options: StudyPatch | None = None,
    ) -> StudyState:
        admit(upload_path, self.limits)  # raises IngestRejected on limit breach
        study_id = safe_study_id(original_filename)
        while (self.root / study_id).exists():
            study_id = safe_study_id(original_filename)
        sdir = self._study_dir(study_id)
        rev_dir = sdir / "rev1"
        rev_dir.mkdir(parents=True)
        ingest(upload_path, rev_dir)

        config = self._default_config(default_engine)
        if options is not None:
            config = config.model_copy(
                update={k: v for k, v in options.model_dump(exclude_unset=True).items()}
            )
            specs = load_engine_specs()
            if config.engine is not None:
                config.engine_spec = (
                    self._engine_spec_dict(specs[config.engine]) if config.engine in specs else None
                )
            entries = []
            for quantity, new_val, unit in (
                ("speed_mm_s", options.speed_mm_s, "mm/s"),
                ("gain_mm_mV", options.gain_mm_mV, "mm/mV"),
            ):
                if new_val is None:
                    continue
                entries.append(
                    CalibrationEvidence(
                        evidence_id=f"manual-{uuid.uuid4().hex[:12]}",
                        kind="manual",
                        quantity=quantity,  # type: ignore[arg-type]
                        value=new_val,
                        unit=unit,
                        frame_id="study",
                        method="initial options via API",
                        limitations="manual operator value; not verified against raster",
                        author="uploader",
                        reason="initial options",
                        previous_value=None,
                    ).model_dump(mode="json")
                )
            if entries:
                (rev_dir / "calibration.json").write_text(
                    json.dumps(entries, indent=2), encoding="utf-8"
                )
        cfg_hash = _config_hash(config)
        (rev_dir / "config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")
        state = StudyState(
            study_id=study_id,
            active_revision=1,
            config_hash=cfg_hash,
            original_filename=escape_filename(original_filename),
        )
        self._write_state(state)
        return state

    def _default_config(self, engine: str | None) -> RunConfig:
        specs = load_engine_specs()
        engine_spec: dict[str, str] | None = None
        if engine is not None:
            engine_spec = self._engine_spec_dict(specs[engine])
        return RunConfig(engine=engine, engine_spec=engine_spec)

    @staticmethod
    def _engine_spec_dict(spec: EngineSpec) -> dict[str, str]:
        d = {"commit": spec.commit}
        for i, w in enumerate(spec.weights):
            d[f"weight_{i}_sha256"] = w.sha256
        return d

    def get(self, study_id: str) -> StudyState | None:
        state = self._read_state(study_id)
        if state is None or state.deleted:
            return None
        return state

    def config_for(self, study_id: str, revision: int) -> RunConfig:
        return self._read_config(study_id, revision)

    def published_result_stale(self, study_id: str) -> bool:
        state = self._read_state(study_id)
        if state is None or state.published_run_id is None:
            return False
        job = self._read_job(study_id, state.published_run_id)
        return job is None or job.input_revision != state.active_revision

    def patch(
        self,
        study_id: str,
        expected_revision: int,
        changes: StudyPatch,
        author: str,
        reason: str,
    ) -> StudyState:
        with self._lock(study_id):
            state = self.get(study_id)
            if state is None:
                raise StudyNotFound(study_id)
            if state.active_revision != expected_revision:
                raise RevisionConflict(
                    f"expected_revision {expected_revision} != active {state.active_revision}"
                )
            old_rev = state.active_revision
            new_rev = old_rev + 1
            src, dst = self._rev_dir(study_id, old_rev), self._rev_dir(study_id, new_rev)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("config.json"))

            manifest = load_manifest(dst / "manifest.json")
            manifest = manifest.model_copy(
                update={
                    "revision": new_rev,
                    "input_revision": old_rev,
                    "selected_run_id": None,
                }
            )
            dump_json(manifest, dst / "manifest.json")

            # calibration.json: append manual evidence for explicit speed/gain
            cal_path = dst / "calibration.json"
            entries: list[dict] = json.loads(cal_path.read_text()) if cal_path.exists() else []
            prev = self._read_config(study_id, old_rev)
            for quantity, new_val, unit in (
                ("speed_mm_s", changes.speed_mm_s, "mm/s"),
                ("gain_mm_mV", changes.gain_mm_mV, "mm/mV"),
            ):
                if new_val is None:
                    continue
                entries.append(
                    CalibrationEvidence(
                        evidence_id=f"manual-{uuid.uuid4().hex[:12]}",
                        kind="manual",
                        quantity=quantity,  # type: ignore[arg-type]
                        value=new_val,
                        unit=unit,
                        frame_id="study",
                        method="patch via API",
                        limitations="manual operator value; not verified against raster",
                        author=author,
                        reason=reason,
                        previous_value=getattr(prev, quantity),
                    ).model_dump(mode="json")
                )
            if entries:
                cal_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

            cfg = prev.model_copy(
                update={k: v for k, v in changes.model_dump(exclude_unset=True).items()}
            )
            specs = load_engine_specs()
            if changes.engine is not None:
                cfg.engine_spec = (
                    self._engine_spec_dict(specs[changes.engine])
                    if changes.engine in specs
                    else None
                )
            elif cfg.engine is not None:
                cfg.engine_spec = (
                    self._engine_spec_dict(specs[cfg.engine]) if cfg.engine in specs else None
                )
            cfg_hash = _config_hash(cfg)
            (dst / "config.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")

            state = state.model_copy(
                update={
                    "active_revision": new_rev,
                    "config_hash": cfg_hash,
                    "selected_run_id": None,
                    "revisions": state.revisions + [new_rev],
                }
            )
            self._write_state(state)
            return state

    # -- runs ----------------------------------------------------------
    def create_run(self, study_id: str, expected_revision: int, config_hash: str) -> Job:
        with self._lock(study_id):
            state = self.get(study_id)
            if state is None:
                raise StudyNotFound(study_id)
            if state.active_revision != expected_revision:
                raise RevisionConflict(
                    f"expected_revision {expected_revision} != active {state.active_revision}"
                )
            if state.config_hash != config_hash:
                raise RevisionConflict("config_hash mismatch")
            cfg = self._read_config(study_id, expected_revision)
            if cfg.engine is None:
                raise EngineNotConfigured("engine is None; PATCH one first")
            run_id = "run-" + uuid.uuid4().hex[:16]
            job = Job(
                run_id=run_id,
                input_revision=expected_revision,
                config_hash=config_hash,
                created_at=_now(),
            )
            self.run_dir(study_id, run_id).mkdir(parents=True)
            self.write_job(study_id, job)
            state = state.model_copy(update={"selected_run_id": run_id})
            self._write_state(state)
            return job

    def get_job(self, study_id: str, run_id: str) -> Job:
        job = self._read_job(study_id, run_id)
        if job is None:
            raise RunNotFound(run_id)
        return job

    def request_cancel(self, study_id: str, run_id: str) -> Job:
        with self._lock(study_id):
            job = self.get_job(study_id, run_id)
            if job.status in ("completed", "failed", "cancelled"):
                return job
            if job.status == "queued":
                job.status = "cancelled"
                job.stage = "cancelled"
                job.finished_at = _now()
            job.cancel_requested = True
            self.write_job(study_id, job)
            return job

    def publish(self, study_id: str, run_id: str) -> bool:
        with self._lock(study_id):
            state = self._read_state(study_id)
            job = self._read_job(study_id, run_id)
            result_dir = self.run_dir(study_id, run_id) / "result"
            ok = (
                state is not None
                and not state.deleted
                and job is not None
                and job.status == "completed"
                and not job.cancel_requested
                and state.active_revision == job.input_revision
                and state.config_hash == job.config_hash
                and state.selected_run_id == run_id
                and result_dir.exists()
            )
            if ok:
                try:
                    manifest = load_manifest(result_dir / "manifest.json")
                    ok = not validate_revision_dir(result_dir, manifest)
                except Exception:  # noqa: BLE001 - invalid result must not publish
                    ok = False
            if ok and state is not None:
                state = state.model_copy(update={"published_run_id": run_id})
                self._write_state(state)
                return True
            if job is not None and job.status == "completed":
                job.status = "completed_unpublished"
                job.stage = "unpublished"
                job.finished_at = _now()
                self.write_job(study_id, job)
            return False

    # -- artifacts -----------------------------------------------------
    def _artifacts_path(self, study_id: str) -> Path:
        return self._study_dir(study_id) / "artifacts.json"

    def _read_artifacts(self, study_id: str) -> dict[str, Artifact]:
        p = self._artifacts_path(study_id)
        if not p.exists():
            return {}
        return {k: Artifact(**v) for k, v in json.loads(p.read_text()).items()}

    def register_exports(
        self, study_id: str, revision: int, run_id: str, formats: list[str]
    ) -> list[Artifact]:
        from ecg_photo.export import export_revision_outputs

        with self._lock(study_id):
            state = self.get(study_id)
            if state is None:
                raise StudyNotFound(study_id)
            job = self.get_job(study_id, run_id)
            if job.input_revision != revision or job.status != "completed":
                raise RunRevisionMismatch(
                    f"run {run_id} does not belong to revision {revision} or is not completed"
                )
            result_dir = self.run_dir(study_id, run_id) / "result"
            manifest = load_manifest(result_dir / "manifest.json")
            produced = export_revision_outputs(
                result_dir,
                manifest,
                formats,
                self.run_dir(study_id, run_id) / "exports",
                study_id=study_id,
                run_id=run_id,
            )
            arts = self._read_artifacts(study_id)
            out: list[Artifact] = []
            for path in produced:
                ext = path.suffix
                art = Artifact(
                    artifact_id=f"art-{uuid.uuid4().hex[:12]}",
                    study_id=study_id,
                    run_id=run_id,
                    revision=revision,
                    filename_safe=f"{study_id}-{run_id[:8]}-{path.name}",
                    mime=MIME_BY_EXT.get(ext, "application/octet-stream"),
                    path_rel=str(path.relative_to(self._study_dir(study_id))),
                    sha256=sha256_file(path),
                )
                arts[art.artifact_id] = art
                out.append(art)
            _atomic_json(
                self._artifacts_path(study_id),
                {k: v.model_dump(mode="json") for k, v in arts.items()},
            )
            return out

    def artifact_path(self, study_id: str, artifact_id: str) -> tuple[Path, Artifact] | None:
        arts = self._read_artifacts(study_id)
        art = arts.get(artifact_id)
        if art is None:
            return None
        p = self._study_dir(study_id) / art.path_rel
        if not p.exists():
            return None
        return p, art

    # -- delete --------------------------------------------------------
    def delete(self, study_id: str) -> None:
        with self._lock(study_id):
            state = self._read_state(study_id)
            if state is None or state.deleted:
                return
            state = state.model_copy(update={"deleted": True})
            self._write_state(state)
            for name in os.listdir(self._study_dir(study_id)):
                p = self._study_dir(study_id) / name
                if p.name == "state.json":
                    continue
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)

    def late_worker_result(self, study_id: str, run_id: str, result_src: Path) -> bool:
        """Move a finished worker result under runs/<run_id>/result unless the
        study was deleted in the meantime; on delete, wipe it instead."""
        with self._lock(study_id):
            state = self._read_state(study_id)
            if state is None or state.deleted:
                shutil.rmtree(result_src, ignore_errors=True)
                return False
            dst = self.run_dir(study_id, run_id) / "result"
            shutil.rmtree(dst, ignore_errors=True)
            shutil.move(str(result_src), str(dst))
            return True
