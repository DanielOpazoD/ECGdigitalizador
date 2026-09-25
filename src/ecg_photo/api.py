"""FastAPI app — routes per spec (API, CLI y operacion privada)."""

import json
import re
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from ecg_photo.contracts import load_manifest
from ecg_photo.ingest import IngestRejected
from ecg_photo.store import (
    Artifact,
    EngineNotConfigured,
    QueueFull,
    RevisionConflict,
    RunNotFound,
    Store,
    StoreError,
    StudyNotFound,
    StudyPatch,
)
from ecg_photo.worker import Worker

_ID_RE = re.compile(r"^[a-z0-9-]{1,64}$")

_REASON_TO_STATUS = {
    "UNSUPPORTED_TYPE": 415,
    "PIXEL_LIMIT": 413,
    "PAGE_LIMIT": 413,
    "FILE_SIZE_LIMIT": 413,
    "DECODE_ERROR": 422,
}


def _err(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def _check_id(value: str) -> str:
    if not _ID_RE.match(value):
        raise _err(404, "NOT_FOUND", "invalid identifier")
    return value


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int
    config_hash: str


class PatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int
    changes: StudyPatch
    author: str
    reason: str


class ExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revision: int
    run_id: str
    formats: list[str] = Field(default_factory=list)


def create_app(store: Store, worker: Worker) -> FastAPI:
    app = FastAPI(title="ecg-photo")

    @app.exception_handler(IngestRejected)
    async def _ingest_rejected(_req, exc: IngestRejected):
        code = str(exc.reason_code)
        status = _REASON_TO_STATUS.get(code, 422)
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=status, content={"detail": {"code": code, "message": str(exc)}}
        )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/studies", status_code=201)
    def create_study(
        file: UploadFile = File(...),  # noqa: B008 - FastAPI idiom
        options: str = Form(default=""),
    ) -> dict:
        limit_bytes = int(store.limits.max_file_mib * 1024 * 1024)
        tmp = Path(tempfile.mkstemp(prefix="upload-", dir=store.root)[1])
        try:
            written = 0
            with open(tmp, "wb") as fh:
                while True:
                    chunk = file.file.read(1024 * 512)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > limit_bytes:
                        raise _err(413, "FILE_SIZE_LIMIT", "file exceeds max_file_mib")
                    fh.write(chunk)
            if options.strip():
                StudyPatch(**json.loads(options))  # validates keys; values applied via PATCH
            filename = file.filename or "upload"
            state = store.create_study(tmp, filename)
        finally:
            tmp.unlink(missing_ok=True)
        return {
            "study_id": state.study_id,
            "revision": state.active_revision,
            "config_hash": state.config_hash,
        }

    @app.get("/studies/{study_id}")
    def get_study(study_id: str) -> dict:
        study_id = _check_id(study_id)
        state = store.get(study_id)
        if state is None:
            raise _err(404, "STUDY_NOT_FOUND", "unknown or deleted study")
        rev_dir = store.root / study_id / f"rev{state.active_revision}"
        pages: list[dict] = []
        manifest = None
        mp = rev_dir / "manifest.json"
        if mp.exists():
            manifest = load_manifest(mp)
            pages = [p.model_dump(mode="json") for p in manifest.pages]
        segments: list[dict] = []
        run_id = state.published_run_id
        if run_id:
            rp = store.run_dir(study_id, run_id) / "result" / "manifest.json"
            if rp.exists():
                rm = load_manifest(rp)
                segments = [{**s.model_dump(mode="json"), "run_id": run_id} for s in rm.segments]
        return {
            **state.model_dump(mode="json"),
            "pages": pages,
            "segments": segments,
            "published_result_stale": store.published_result_stale(study_id),
        }

    @app.post("/studies/{study_id}/runs", status_code=202)
    def create_run(study_id: str, body: RunRequest) -> dict:
        study_id = _check_id(study_id)
        try:
            job = store.create_run(study_id, body.expected_revision, body.config_hash)
        except StudyNotFound:
            raise _err(404, "STUDY_NOT_FOUND", "unknown study") from None
        except RevisionConflict as e:
            raise _err(409, "REVISION_CONFLICT", str(e)) from None
        except EngineNotConfigured as e:
            raise _err(422, "ENGINE_NOT_CONFIGURED", str(e)) from None
        try:
            worker.submit(study_id, job.run_id)
        except QueueFull as e:
            raise _err(503, "QUEUE_FULL", str(e)) from None
        return {"run_id": job.run_id}

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        run_id = _check_id(run_id)
        found = store.job_for_run(run_id)
        if found is None:
            raise _err(404, "RUN_NOT_FOUND", "unknown run")
        _sid, job = found
        return job.model_dump(mode="json")

    @app.post("/runs/{run_id}/cancel", status_code=202)
    def cancel_run(run_id: str) -> dict:
        run_id = _check_id(run_id)
        found = store.job_for_run(run_id)
        if found is None:
            raise _err(404, "RUN_NOT_FOUND", "unknown run")
        study_id, _job = found
        try:
            job = store.request_cancel(study_id, run_id)
        except RunNotFound:
            raise _err(404, "RUN_NOT_FOUND", "unknown run") from None
        return {"run_id": job.run_id, "status": job.status}

    @app.patch("/studies/{study_id}")
    def patch_study(study_id: str, body: PatchRequest) -> dict:
        study_id = _check_id(study_id)
        try:
            state = store.patch(
                study_id, body.expected_revision, body.changes, body.author, body.reason
            )
        except StudyNotFound:
            raise _err(404, "STUDY_NOT_FOUND", "unknown study") from None
        except RevisionConflict as e:
            raise _err(409, "REVISION_CONFLICT", str(e)) from None
        return {
            "study_id": state.study_id,
            "revision": state.active_revision,
            "config_hash": state.config_hash,
        }

    @app.post("/studies/{study_id}/exports", status_code=201)
    def create_exports(study_id: str, body: ExportRequest) -> dict:
        study_id = _check_id(study_id)
        from ecg_photo.store import RunRevisionMismatch

        try:
            arts = store.register_exports(study_id, body.revision, body.run_id, body.formats)
        except StudyNotFound:
            raise _err(404, "STUDY_NOT_FOUND", "unknown study") from None
        except RunNotFound:
            raise _err(404, "RUN_NOT_FOUND", "unknown run") from None
        except RunRevisionMismatch as e:
            raise _err(409, "RUN_REVISION_MISMATCH", str(e)) from None
        return {
            "artifacts": [a.model_dump(mode="json") for a in arts],
        }

    @app.get("/studies/{study_id}/artifacts/{artifact_id}")
    def get_artifact(study_id: str, artifact_id: str):
        study_id = _check_id(study_id)
        artifact_id = _check_id(artifact_id)
        found = store.artifact_path(study_id, artifact_id)
        if found is None:
            raise _err(404, "ARTIFACT_NOT_FOUND", "unknown artifact")
        path, art = found
        return FileResponse(path, filename=art.filename_safe, media_type=art.mime)

    @app.delete("/studies/{study_id}", status_code=204)
    def delete_study(study_id: str) -> None:
        study_id = _check_id(study_id)
        store.delete(study_id)

    return app


__all__ = [
    "Artifact",
    "StoreError",
    "create_app",
]
