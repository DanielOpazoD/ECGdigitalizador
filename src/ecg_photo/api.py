"""FastAPI app — routes per spec (API, CLI y operacion privada)."""

import json
import re
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from ecg_photo.contracts import load_manifest
from ecg_photo.ingest import IngestRejected
from ecg_photo.process import OVERVIEW_FILENAME
from ecg_photo.qc import read_qc_report
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
_UI_INDEX = Path(__file__).parent / "ui" / "index.html"

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

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/ui", status_code=307)

    @app.get("/ui", include_in_schema=False)
    def ui() -> FileResponse:
        return FileResponse(
            _UI_INDEX, media_type="text/html", headers={"Cache-Control": "no-store"}
        )

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
            patch = StudyPatch(**json.loads(options)) if options.strip() else None
            filename = file.filename or "upload"
            state = store.create_study(tmp, filename, options=patch)
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
            "config": store.config_for(study_id, state.active_revision).model_dump(mode="json"),
            "corrections": store.corrections_for(study_id, state.active_revision),
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

    # -- read endpoints for the UI ------------------------------------
    @app.get("/studies/{study_id}/pages/{page_id}/raster")
    def page_raster(study_id: str, page_id: str):
        study_id = _check_id(study_id)
        page_id = _check_id(page_id)
        state = store.get(study_id)
        if state is None:
            raise _err(404, "STUDY_NOT_FOUND", "unknown study")
        rev_dir = store.rev_dir(study_id, state.active_revision)
        manifest = load_manifest(rev_dir / "manifest.json")
        page = next((p for p in manifest.pages if p.page_id == page_id), None)
        if page is None:
            raise _err(404, "PAGE_NOT_FOUND", "unknown page")
        path = rev_dir / page.raster_path
        if not path.exists():
            raise _err(404, "PAGE_NOT_FOUND", "raster missing")
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/studies/{study_id}/pages/{page_id}/grid")
    def page_grid(study_id: str, page_id: str) -> dict:
        study_id = _check_id(study_id)
        page_id = _check_id(page_id)
        state = store.get(study_id)
        if state is None:
            raise _err(404, "STUDY_NOT_FOUND", "unknown study")
        rev_dir = store.rev_dir(study_id, state.active_revision)
        manifest = load_manifest(rev_dir / "manifest.json")
        page = next((p for p in manifest.pages if p.page_id == page_id), None)
        grid_rel = getattr(page, "grid_estimate_path", None) if page else None
        path = rev_dir / grid_rel if grid_rel else rev_dir / "pages" / f"{page_id}.grid.json"
        if not path.exists():
            raise _err(404, "GRID_NOT_FOUND", "no grid estimate for this page")
        return json.loads(path.read_text())

    def _run_result_dir(study_id: str, run_id: str) -> Path:
        job = store._read_job(study_id, run_id)
        if job is None:
            raise _err(404, "RUN_NOT_FOUND", "run not in this study")
        result = store.run_dir(study_id, run_id) / "result"
        if not result.exists():
            raise _err(404, "RESULT_NOT_FOUND", "run has no published result")
        return result

    @app.get("/studies/{study_id}/runs/{run_id}/segments")
    def run_segments(study_id: str, run_id: str) -> dict:
        study_id = _check_id(study_id)
        run_id = _check_id(run_id)
        if store.get(study_id) is None:
            raise _err(404, "STUDY_NOT_FOUND", "unknown study")
        result = _run_result_dir(study_id, run_id)
        manifest = load_manifest(result / "manifest.json")
        out = []
        for s in manifest.segments:
            has_geo = (result / "raw_paths" / f"{s.segment_id}-geometry.json").exists()
            out.append(
                {
                    "segment_id": s.segment_id,
                    "lead_label": s.lead_label,
                    "lead_status": str(s.lead_status),
                    "representation_kind": str(s.representation_kind),
                    "speed_status": str(s.speed_status),
                    "gain_status": str(s.gain_status),
                    "units": s.units,
                    "n_samples": s.n_samples,
                    "working_fs_hz": s.working_fs_hz,
                    "observed_duration_s": s.observed_duration_s,
                    "raw_coordinate_frame": s.raw_coordinate_frame,
                    "has_geometry": has_geo,
                    "source_region": [list(t) for t in s.source_region],
                }
            )
        return {"segments": out}

    @app.get("/studies/{study_id}/runs/{run_id}/qc")
    def run_qc_report(study_id: str, run_id: str) -> dict:
        """Truth-free quality report written by the worker (qc.json)."""
        study_id = _check_id(study_id)
        run_id = _check_id(run_id)
        if store.get(study_id) is None:
            raise _err(404, "STUDY_NOT_FOUND", "unknown study")
        report = read_qc_report(_run_result_dir(study_id, run_id))
        if report is None:
            raise _err(404, "QC_NOT_FOUND", "run has no quality report (scale not confirmed)")
        return report

    @app.get("/studies/{study_id}/runs/{run_id}/overview")
    def run_overview(study_id: str, run_id: str):
        """3x4 + II redraw of the published leads on mm paper (display only)."""
        study_id = _check_id(study_id)
        run_id = _check_id(run_id)
        if store.get(study_id) is None:
            raise _err(404, "STUDY_NOT_FOUND", "unknown study")
        path = _run_result_dir(study_id, run_id) / OVERVIEW_FILENAME
        if not path.exists():
            raise _err(404, "OVERVIEW_NOT_FOUND", "run has no overview (scale not confirmed)")
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/studies/{study_id}/runs/{run_id}/segments/{segment_id}/trace")
    def run_segment_trace(study_id: str, run_id: str, segment_id: str) -> dict:
        study_id = _check_id(study_id)
        run_id = _check_id(run_id)
        if store.get(study_id) is None:
            raise _err(404, "STUDY_NOT_FOUND", "unknown study")
        result = _run_result_dir(study_id, run_id)
        manifest = load_manifest(result / "manifest.json")
        seg = next((s for s in manifest.segments if s.segment_id == segment_id), None)
        if seg is None:
            raise _err(404, "SEGMENT_NOT_FOUND", "unknown segment")

        import math

        import numpy as np

        values = None
        observed = None
        x_px = None
        t_s = None
        if seg.raw_path and (result / seg.raw_path).exists():
            raw = np.load(result / seg.raw_path, allow_pickle=False)
            if seg.raw_coordinate_frame == "file" and raw.ndim == 2 and raw.shape[1] == 2:
                x_px = [None if math.isnan(v) else v for v in raw[:, 0].tolist()]
            vals = raw[:, -1] if raw.ndim == 2 else raw
            values = [None if math.isnan(v) else v for v in vals.tolist()]
        if seg.signal_path and (result / seg.signal_path).exists():
            sig = np.load(result / seg.signal_path, allow_pickle=False)
            values = [None if math.isnan(v) else v for v in sig.tolist()]
            fs = float(seg.working_fs_hz or 0.0)
            if fs > 0:
                t_s = (np.arange(len(sig)) / fs).tolist()
        if seg.observed_mask_path and (result / seg.observed_mask_path).exists():
            observed = np.load(result / seg.observed_mask_path, allow_pickle=False).tolist()
        elif values is not None:
            observed = [v is not None for v in values]

        n = len(values) if values is not None else 0
        resp: dict = {
            "segment_id": seg.segment_id,
            "frame": seg.raw_coordinate_frame,
            "x_px": x_px,
            "values": values,
            "units": "mV" if seg.signal_path else "arbitrary",
            "t_s": t_s,
            "observed": observed,
            "effective_dt_ms": seg.effective_dt_ms,
            "px_per_mm_x": seg.px_per_mm_x,
            "speed_mm_s": seg.speed_mm_s,
            "speed_status": str(seg.speed_status),
            "gain_status": str(seg.gain_status),
        }
        if n > 20000:
            k = math.ceil(n / 20000)
            for key in ("values", "observed", "x_px", "t_s"):
                if resp[key] is not None:
                    resp[key] = resp[key][::k]
            resp["decimation"] = k
        return resp

    @app.get("/studies/{study_id}/runs")
    def list_runs(study_id: str) -> dict:
        study_id = _check_id(study_id)
        if store.get(study_id) is None:
            raise _err(404, "STUDY_NOT_FOUND", "unknown study")
        return {"runs": [j.model_dump(mode="json") for j in store.list_jobs(study_id)]}

    return app


__all__ = [
    "Artifact",
    "StoreError",
    "create_app",
]
