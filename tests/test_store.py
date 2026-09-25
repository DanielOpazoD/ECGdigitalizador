import threading
import time
from pathlib import Path

import numpy as np
import pytest
from test_pipeline import FakeDigitizer, _fake_output, _study

from ecg_photo.ingest import load_supported_inputs
from ecg_photo.store import (
    EngineNotConfigured,
    ExecutionLimits,
    QueueFull,
    RevisionConflict,
    Store,
    StudyPatch,
)
from ecg_photo.worker import Worker


def _store(tmp_path: Path, max_pending: int = 5) -> Store:
    return Store(
        tmp_path / "store",
        load_supported_inputs(),
        ExecutionLimits(max_active_jobs=1, max_pending_jobs=max_pending),
    )


def _make_study(tmp_path: Path, store: Store):
    png = _study(tmp_path / "src") / "pages" / "page-1.png"
    return store.create_study(png, "ecg.png")


def test_create_study_and_conflict(tmp_path) -> None:
    store = _store(tmp_path)
    state = _make_study(tmp_path, store)
    assert state.active_revision == 1
    assert (store.root / state.study_id / "rev1" / "manifest.json").exists()

    with pytest.raises(RevisionConflict):
        store.patch(state.study_id, 2, StudyPatch(gain_mm_mV=10), author="t", reason="r")
    with pytest.raises(RevisionConflict):
        store.create_run(state.study_id, 99, state.config_hash)
    with pytest.raises(EngineNotConfigured):
        store.create_run(state.study_id, 1, state.config_hash)

    st2 = store.patch(
        state.study_id,
        1,
        StudyPatch(engine="fake", engine_duration_s=3.0, gain_mm_mV=10, speed_mm_s=25),
        author="t",
        reason="r",
    )
    assert st2.active_revision == 2 and st2.config_hash != state.config_hash
    assert (store.root / state.study_id / "rev2" / "config.json").exists()
    cal = (store.root / state.study_id / "rev2" / "calibration.json").read_text()
    assert "speed_mm_s" in cal and "t" in cal


def test_run_complete_and_publish(tmp_path) -> None:
    store = _store(tmp_path)
    state = _make_study(tmp_path, store)
    st2 = store.patch(
        state.study_id,
        1,
        StudyPatch(
            engine="fake",
            engine_duration_s=3.0,
            gain_mm_mV=10,
            speed_mm_s=25,
            time_source="engine",
        ),
        author="t",
        reason="r",
    )
    worker = Worker(
        store, {"fake": lambda rc: FakeDigitizer(_fake_output(np.sin(np.linspace(0, 9, 1500))))}
    )
    worker.start()
    try:
        job = store.create_run(st2.study_id, 2, st2.config_hash)
        worker.submit(st2.study_id, job.run_id)
        deadline = time.time() + 30
        while time.time() < deadline:
            job = store.get_job(st2.study_id, job.run_id)
            if job.status in ("completed", "completed_unpublished", "failed", "cancelled"):
                break
            time.sleep(0.05)
        assert job.status == "completed", job.error
        final = store.get(st2.study_id)
        assert final is not None and final.published_run_id == job.run_id
    finally:
        worker.shutdown()
        worker.join(5)


class _SlowDigitizer:
    spec = FakeDigitizer(_fake_output(np.zeros(4))).spec

    def __init__(self, release: threading.Event) -> None:
        self.release = release

    def run(self, image_path, work_dir):
        self.release.wait(30)
        return _fake_output(np.sin(np.linspace(0, 9, 1500)))


def test_cancel_queued_and_delete_during_run(tmp_path) -> None:
    store = _store(tmp_path, max_pending=1)
    state = _make_study(tmp_path, store)
    st2 = store.patch(
        state.study_id,
        1,
        StudyPatch(engine="fake", engine_duration_s=3.0, time_source="engine"),
        author="t",
        reason="r",
    )
    release = threading.Event()
    worker = Worker(store, {"fake": lambda rc: _SlowDigitizer(release)})
    worker.start()
    try:
        j1 = store.create_run(st2.study_id, 2, st2.config_hash)
        worker.submit(st2.study_id, j1.run_id)
        # wait until running
        deadline = time.time() + 10
        while time.time() < deadline:
            if store.get_job(st2.study_id, j1.run_id).status == "running":
                break
            time.sleep(0.02)
        j2 = store.create_run(st2.study_id, 2, st2.config_hash)
        worker.submit(st2.study_id, j2.run_id)
        with pytest.raises(QueueFull):
            j3 = store.create_run(st2.study_id, 2, st2.config_hash)
            worker.submit(st2.study_id, j3.run_id)
        store.request_cancel(st2.study_id, j2.run_id)
        assert store.get_job(st2.study_id, j2.run_id).status == "cancelled"
        release.set()
        deadline = time.time() + 30
        while time.time() < deadline:
            if store.get_job(st2.study_id, j1.run_id).status in (
                "completed",
                "completed_unpublished",
            ):
                break
            time.sleep(0.05)
        # j2 was selected over j1 then cancelled while queued -> nothing publishes
        assert store.get_job(st2.study_id, j1.run_id).status == "completed_unpublished"
        assert store.get(st2.study_id).published_run_id is None
        assert (store.run_dir(st2.study_id, j1.run_id) / "result").exists()
    finally:
        release.set()
        worker.shutdown()
        worker.join(5)


def test_delete_marks_state(tmp_path) -> None:
    store = _store(tmp_path)
    state = _make_study(tmp_path, store)
    store.delete(state.study_id)
    assert store.get(state.study_id) is None
    raw = store._read_state(state.study_id)
    assert raw is not None and raw.deleted
