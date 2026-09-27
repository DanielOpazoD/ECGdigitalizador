"""Read-only routes over a run's published result: segments, trace, quality
report, intervals, overview image, PDF report and the study's run list."""

import math
from pathlib import Path

import numpy as np
from fastapi import APIRouter
from fastapi.responses import FileResponse

from ecg_photo.api_common import api_error as _err
from ecg_photo.api_common import check_id as _check_id
from ecg_photo.contracts import load_manifest
from ecg_photo.intervals import read_intervals_report
from ecg_photo.process import OVERVIEW_FILENAME
from ecg_photo.qc import read_qc_report
from ecg_photo.report import REPORT_FILENAME
from ecg_photo.store import RunNotFound, Store


def results_router(store: Store) -> APIRouter:
    router = APIRouter()

    def _run_result_dir(study_id: str, run_id: str) -> Path:
        """Published result directory of a run, or the matching 404."""
        _check_id(study_id)
        _check_id(run_id)
        if store.get(study_id) is None:
            raise _err(404, "STUDY_NOT_FOUND", "unknown study")
        try:
            store.get_job(study_id, run_id)
        except RunNotFound:
            raise _err(404, "RUN_NOT_FOUND", "run not in this study") from None
        result = store.run_dir(study_id, run_id) / "result"
        if not result.exists():
            raise _err(404, "RESULT_NOT_FOUND", "run has no published result")
        return result

    def _result_file(study_id: str, run_id: str, name: str, code: str) -> Path:
        path = _run_result_dir(study_id, run_id) / name
        if not path.exists():
            raise _err(404, code, f"run has no {name} (scale not confirmed)")
        return path

    @router.get("/studies/{study_id}/runs/{run_id}/segments")
    def run_segments(study_id: str, run_id: str) -> dict:
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

    @router.get("/studies/{study_id}/runs/{run_id}/qc")
    def run_qc_report(study_id: str, run_id: str) -> dict:
        """Truth-free quality report written by the worker (qc.json)."""
        report = read_qc_report(_run_result_dir(study_id, run_id))
        if report is None:
            raise _err(404, "QC_NOT_FOUND", "run has no quality report (scale not confirmed)")
        return report

    @router.get("/studies/{study_id}/runs/{run_id}/intervals")
    def run_intervals(study_id: str, run_id: str) -> dict:
        """PR / QRS / QT / QTc written by the worker (intervals.json)."""
        report = read_intervals_report(_run_result_dir(study_id, run_id))
        if report is None:
            raise _err(404, "INTERVALS_NOT_FOUND", "run has no intervals (scale not confirmed)")
        return report

    @router.get("/studies/{study_id}/runs/{run_id}/overview")
    def run_overview(study_id: str, run_id: str):
        """3x4 + II redraw of the published leads on mm paper (display only)."""
        path = _result_file(study_id, run_id, OVERVIEW_FILENAME, "OVERVIEW_NOT_FOUND")
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})

    @router.get("/studies/{study_id}/runs/{run_id}/report")
    def run_report_pdf(study_id: str, run_id: str):
        """One-page PDF: redrawn ECG, measurements, quality, scales, limits."""
        path = _result_file(study_id, run_id, REPORT_FILENAME, "REPORT_NOT_FOUND")
        return FileResponse(
            path,
            media_type="application/pdf",
            filename=f"ecg-{study_id}-{run_id}.pdf",
            headers={"Cache-Control": "no-store"},
        )

    @router.get("/studies/{study_id}/runs/{run_id}/segments/{segment_id}/trace")
    def run_segment_trace(study_id: str, run_id: str, segment_id: str) -> dict:
        result = _run_result_dir(study_id, run_id)
        manifest = load_manifest(result / "manifest.json")
        seg = next((s for s in manifest.segments if s.segment_id == segment_id), None)
        if seg is None:
            raise _err(404, "SEGMENT_NOT_FOUND", "unknown segment")

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

    @router.get("/studies/{study_id}/runs")
    def list_runs(study_id: str) -> dict:
        study_id = _check_id(study_id)
        if store.get(study_id) is None:
            raise _err(404, "STUDY_NOT_FOUND", "unknown study")
        return {"runs": [j.model_dump(mode="json") for j in store.list_jobs(study_id)]}

    return router
