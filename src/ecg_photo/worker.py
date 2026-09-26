"""Single inference worker: bounded queue, one active job, thread-based."""

import os
import queue
import shutil
import threading
from collections.abc import Callable
from pathlib import Path

import yaml

from ecg_photo.digitizers.base import Digitizer
from ecg_photo.pipeline import apply_lead_corrections, confirm_scale, digitize_page
from ecg_photo.process import write_overview
from ecg_photo.qc import write_qc_report
from ecg_photo.store import QueueFull, RunConfig, Store, _now

EngineFactory = Callable[[RunConfig], Digitizer]


def default_engines(config_path: Path | None = None) -> dict[str, EngineFactory]:
    """Engine registry: 'ahus'/'ecg-digitiser' from configs/engines.local.yml
    (external paths, not versioned); 'fake' only with ECG_PHOTO_ENABLE_FAKE_ENGINE=1."""
    engines: dict[str, EngineFactory] = {}
    path = Path(config_path) if config_path else Path("configs/engines.local.yml")
    if path.exists():
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if cfg.get("ahus"):

            def _ahus(rc: RunConfig, c=cfg["ahus"]) -> Digitizer:
                from ecg_photo.digitizers.ahus import AhusDigitizer

                return AhusDigitizer(
                    ahus_root=Path(c["root"]),
                    python_exe=Path(c["python"]),
                    base_config=Path(c["config"]),
                    duration_s=float(rc.engine_duration_s or 10.0),
                )

            engines["ahus"] = _ahus
        if cfg.get("ecg_digitiser"):

            def _dig(rc: RunConfig, c=cfg["ecg_digitiser"]) -> Digitizer:
                from ecg_photo.digitizers.ecg_digitiser import EcgDigitiserDigitizer

                return EcgDigitiserDigitizer(
                    root=Path(c["root"]),
                    python_exe=Path(c["python"]),
                    model_dir=Path(c.get("model_dir", "models/M3")),
                )

            engines["ecg-digitiser"] = _dig
    if os.environ.get("ECG_PHOTO_ENABLE_FAKE_ENGINE") == "1":
        import numpy as np

        from ecg_photo.digitizers.base import EngineOutput, EngineSpec

        class _FakeDigitizer:
            spec = EngineSpec(engine_id="fake", repo="r", commit="c" * 8, license="l", weights=())

            def run(self, image_path: Path, work_dir: Path) -> EngineOutput:
                n = 1500
                ii = np.sin(np.linspace(0, 6 * np.pi, n))
                v1 = ii.copy()
                v1[500:800] = np.nan
                return EngineOutput(
                    engine_id="fake",
                    engine_commit="c" * 40,
                    weights_sha256=(),
                    config_hash="h" * 64,
                    fs_hz=500.0,
                    leads={"II": ii, "V1": v1, "V4": np.full(n, np.nan)},
                    observed={
                        "II": np.isfinite(ii),
                        "V1": np.isfinite(v1),
                        "V4": np.zeros(n, bool),
                    },
                    layout_detected=None,
                )

        engines["fake"] = lambda rc: _FakeDigitizer()
    return engines


class Worker(threading.Thread):
    def __init__(self, store: Store, engines: dict[str, EngineFactory]) -> None:
        super().__init__(daemon=True, name="ecg-photo-worker")
        self.store = store
        self.engines = engines
        self.queue: queue.Queue[tuple[str, str] | None] = queue.Queue(
            maxsize=store.execution.max_pending_jobs
        )

    def submit(self, study_id: str, run_id: str) -> None:
        try:
            self.queue.put_nowait((study_id, run_id))
        except queue.Full as e:
            raise QueueFull("inference queue is full") from e

    def run(self) -> None:
        while True:
            item = self.queue.get()
            if item is None:
                return
            self.process(*item)

    def process(self, study_id: str, run_id: str) -> None:
        """Run one job to a terminal status; exceptions become `failed` (job
        isolation). Used by the thread loop and synchronously by `batch`."""
        try:
            self._process(study_id, run_id)
        except Exception as e:  # noqa: BLE001 - job isolation, no traceback out
            try:
                job = self.store.get_job(study_id, run_id)
                job.status = "failed"
                job.stage = "failed"
                job.error = f"{type(e).__name__}: {e}"[:500]
                job.finished_at = _now()
                self.store.write_job(study_id, job)
            except Exception:  # noqa: BLE001,S110 - job may already be gone
                pass

    def _process(self, study_id: str, run_id: str) -> None:
        store = self.store
        job = store.get_job(study_id, run_id)
        if job.cancel_requested:
            job.status = "cancelled"
            job.stage = "cancelled"
            job.finished_at = _now()
            store.write_job(study_id, job)
            return
        cfg = store.config_for(study_id, job.input_revision)
        if cfg.engine is None or cfg.engine not in self.engines:
            job.status = "failed"
            job.stage = "failed"
            job.error = f"engine {cfg.engine!r} not configured on this worker"
            job.finished_at = _now()
            store.write_job(study_id, job)
            return

        job.status = "running"
        job.stage = "engine_inference"
        store.write_job(study_id, job)

        rev_dir = store.root / study_id / f"rev{job.input_revision}"
        work_root = store.run_dir(study_id, run_id) / "work"
        digitizer = self.engines[cfg.engine](cfg)
        paths = digitize_page(
            rev_dir,
            cfg.page_id,
            digitizer,
            engine_duration_s=float(cfg.engine_duration_s or 10.0),
            runs_root=work_root,
        )
        # apply rev<N>/corrections.json (lead-label overrides) before confirm_scale
        corr = store.corrections_for(study_id, job.input_revision)
        if corr.get("lead_labels"):
            from ecg_photo.contracts import dump_json as _dump
            from ecg_photo.contracts import load_manifest as _load

            m = _load(paths.manifest)
            _dump(apply_lead_corrections(m, corr), paths.manifest)

        final = paths
        if cfg.gain_mm_mV is not None:
            job.stage = "confirm_scale"
            store.write_job(study_id, job)
            final = confirm_scale(
                paths.run_dir,
                gain_mm_mV=float(cfg.gain_mm_mV),
                speed_mm_s=cfg.speed_mm_s,
                fs_hz=cfg.fs_hz,
                author="worker",
                reason=f"config {cfg_hash_of(job)}",
                time_source=cfg.time_source,
            )

        if final is not paths:
            # signals exist only after confirm_scale: attach the truth-free
            # quality report; a QC failure never fails the job
            write_qc_report(final.run_dir)
            write_overview(final.run_dir)

        job = store.get_job(study_id, run_id)  # re-read: cancel may have landed
        if job.cancel_requested:
            job.status = "cancelled"
            job.stage = "cancelled"
            job.finished_at = _now()
            store.write_job(study_id, job)
            shutil.rmtree(final.run_dir, ignore_errors=True)
            return

        job.status = "completed"
        job.stage = "awaiting_publish"
        job.finished_at = _now()
        store.write_job(study_id, job)

        if not store.late_worker_result(study_id, run_id, final.run_dir):
            shutil.rmtree(work_root, ignore_errors=True)
            return
        shutil.rmtree(work_root, ignore_errors=True)
        store.publish(study_id, run_id)

    def shutdown(self) -> None:
        self.queue.put(None)


def resume_after_restart(store: Store, worker: Worker) -> dict:
    """T47: reconcile the store (`Store.recover`) and re-enqueue the runs that
    were still waiting and current. Call with the process lock held, before
    serving requests. Runs that no longer fit the queue fail explicitly."""
    rep = store.recover()
    out = rep.as_dict()
    out["requeued"] = []
    out["queue_overflow"] = []
    for sid, rid in rep.requeue:
        try:
            worker.submit(sid, rid)
            out["requeued"].append(f"{sid}/{rid}")
        except QueueFull:
            store.fail_job(sid, rid, "QUEUE_FULL: queue full while resuming after restart")
            out["queue_overflow"].append(f"{sid}/{rid}")
    del out["requeue"]
    return out


def cfg_hash_of(job) -> str:
    return job.config_hash[:8]
