"""T47: restart recovery and the one-process-per-store lock."""

import time
from pathlib import Path

import numpy as np
import pytest
from test_pipeline import FakeDigitizer, _fake_output, _study

from ecg_photo.ingest import load_supported_inputs
from ecg_photo.store import ExecutionLimits, Store, StoreLocked, StudyPatch
from ecg_photo.worker import Worker, resume_after_restart


def _store(tmp_path: Path, max_pending: int = 5) -> Store:
    return Store(
        tmp_path / "store",
        load_supported_inputs(),
        ExecutionLimits(max_active_jobs=1, max_pending_jobs=max_pending),
    )


def _engines():
    return {"fake": lambda rc: FakeDigitizer(_fake_output(np.sin(np.linspace(0, 9, 1500))))}


def _ready_study(tmp_path: Path, store: Store, tag: str = "a"):
    png = _study(tmp_path / f"src-{tag}") / "pages" / "page-1.png"
    st = store.create_study(png, "ecg.png")
    return store.patch(
        st.study_id,
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


def _set(store: Store, sid: str, run_id: str, **fields) -> None:
    job = store.get_job(sid, run_id)
    for k, v in fields.items():
        setattr(job, k, v)
    store.write_job(sid, job)


def test_process_lock_is_exclusive(tmp_path) -> None:
    a = _store(tmp_path)
    b = _store(tmp_path)
    a.acquire_process_lock()
    try:
        with pytest.raises(StoreLocked):
            b.acquire_process_lock()
    finally:
        a.release_process_lock()
    b.acquire_process_lock()
    b.release_process_lock()


def test_running_job_is_interrupted_not_retried(tmp_path) -> None:
    store = _store(tmp_path)
    st = _ready_study(tmp_path, store)
    job = store.create_run(st.study_id, 2, st.config_hash)
    _set(store, st.study_id, job.run_id, status="running", stage="engine_inference")
    work = store.run_dir(st.study_id, job.run_id) / "work"
    work.mkdir()
    (work / "partial.npy").write_bytes(b"x")

    rep = store.recover()
    j = store.get_job(st.study_id, job.run_id)
    assert j.status == "failed" and j.stage == "interrupted"
    assert "engine_inference" in (j.error or "")
    assert not work.exists()
    assert rep.requeue == [] and rep.interrupted == [f"{st.study_id}/{job.run_id}"]
    assert store.get(st.study_id).published_run_id is None


def test_queued_current_run_is_resumed_and_published(tmp_path) -> None:
    store = _store(tmp_path)
    st = _ready_study(tmp_path, store)
    job = store.create_run(st.study_id, 2, st.config_hash)  # queued, never submitted

    worker = Worker(store, _engines())
    worker.start()
    try:
        out = resume_after_restart(store, worker)
        assert out["requeued"] == [f"{st.study_id}/{job.run_id}"]
        deadline = time.time() + 30
        while time.time() < deadline:
            if store.get(st.study_id).published_run_id == job.run_id:
                break
            time.sleep(0.05)
        assert store.get(st.study_id).published_run_id == job.run_id
    finally:
        worker.shutdown()
        worker.join(5)


def test_queued_superseded_and_cancel_requested(tmp_path) -> None:
    store = _store(tmp_path)
    st = _ready_study(tmp_path, store)
    old = store.create_run(st.study_id, 2, st.config_hash)
    new = store.create_run(st.study_id, 2, st.config_hash)  # now the selected run
    st_b = _ready_study(tmp_path, store, "b")
    wanted_cancel = store.create_run(st_b.study_id, 2, st_b.config_hash)
    _set(store, st_b.study_id, wanted_cancel.run_id, cancel_requested=True)

    rep = store.recover()
    assert rep.requeue == [(st.study_id, new.run_id)]
    j_old = store.get_job(st.study_id, old.run_id)
    assert j_old.status == "cancelled" and j_old.stage == "superseded"
    assert store.get_job(st_b.study_id, wanted_cancel.run_id).status == "cancelled"
    assert rep.cancelled == [f"{st_b.study_id}/{wanted_cancel.run_id}"]


def test_queued_run_of_older_revision_is_not_resumed(tmp_path) -> None:
    store = _store(tmp_path)
    st = _ready_study(tmp_path, store)
    job = store.create_run(st.study_id, 2, st.config_hash)
    store.patch(st.study_id, 2, StudyPatch(gain_mm_mV=20), author="t", reason="r")
    rep = store.recover()
    assert rep.requeue == []
    assert store.get_job(st.study_id, job.run_id).stage == "superseded"


def test_completed_before_publish_goes_through_publish(tmp_path) -> None:
    store = _store(tmp_path)
    st = _ready_study(tmp_path, store)
    job = store.create_run(st.study_id, 2, st.config_hash)
    Worker(store, _engines()).process(st.study_id, job.run_id)
    assert store.get(st.study_id).published_run_id == job.run_id

    # simulate a stop between 'completed' and publication
    state = store.get(st.study_id)
    store._write_state(state.model_copy(update={"published_run_id": None}))
    _set(store, st.study_id, job.run_id, status="completed", stage="awaiting_publish")
    rep = store.recover()
    assert rep.published == [f"{st.study_id}/{job.run_id}"]
    assert store.get(st.study_id).published_run_id == job.run_id

    # same, but the result never reached runs/<id>/result -> failed, not published
    job2 = store.create_run(st.study_id, 2, st.config_hash)
    _set(store, st.study_id, job2.run_id, status="completed", stage="awaiting_publish")
    rep = store.recover()
    assert store.get_job(st.study_id, job2.run_id).status == "failed"
    assert rep.interrupted == [f"{st.study_id}/{job2.run_id}"]
    assert store.get(st.study_id).published_run_id == job.run_id


def test_invalid_result_is_not_published_on_recovery(tmp_path) -> None:
    store = _store(tmp_path)
    st = _ready_study(tmp_path, store)
    job = store.create_run(st.study_id, 2, st.config_hash)
    Worker(store, _engines()).process(st.study_id, job.run_id)
    state = store.get(st.study_id)
    store._write_state(state.model_copy(update={"published_run_id": None}))
    _set(store, st.study_id, job.run_id, status="completed", stage="awaiting_publish")
    (store.run_dir(st.study_id, job.run_id) / "result" / "manifest.json").write_text("{}")
    rep = store.recover()
    assert rep.unpublished == [f"{st.study_id}/{job.run_id}"]
    assert store.get_job(st.study_id, job.run_id).status == "completed_unpublished"
    assert store.get(st.study_id).published_run_id is None


def test_cleanup_deletes_orphans_and_tmp(tmp_path) -> None:
    store = _store(tmp_path)
    st = _ready_study(tmp_path, store)
    # delete interrupted after marking state: leftovers remain
    state = store.get(st.study_id)
    store._write_state(state.model_copy(update={"deleted": True}))
    # upload interrupted before state.json: only rev1 -> removed
    orphan = store.root / "study-orphan"
    (orphan / "rev1").mkdir(parents=True)
    # unknown content without state -> kept, reported
    odd = store.root / "study-odd"
    (odd / "notes").mkdir(parents=True)
    (store.root / "upload-abc123").write_bytes(b"partial upload")
    (store.root / st.study_id / "state.json.tmp").write_text("{")

    rep = store.recover()
    assert rep.finished_deletes == [st.study_id]
    assert sorted(p.name for p in (store.root / st.study_id).iterdir()) == ["state.json"]
    assert store.get(st.study_id) is None
    assert not orphan.exists() and rep.orphans_removed == ["study-orphan"]
    assert odd.exists() and rep.orphans_kept == ["study-odd"]
    assert not (store.root / "upload-abc123").exists()
    assert rep.tmp_removed == 2


def test_resume_overflow_fails_explicitly(tmp_path) -> None:
    store = _store(tmp_path, max_pending=1)
    st_a = _ready_study(tmp_path, store, "a")
    st_b = _ready_study(tmp_path, store, "b")
    ja = store.create_run(st_a.study_id, 2, st_a.config_hash)
    jb = store.create_run(st_b.study_id, 2, st_b.config_hash)
    worker = Worker(store, _engines())  # not started: the queue only fills
    out = resume_after_restart(store, worker)
    assert out["requeued"] == [f"{st_a.study_id}/{ja.run_id}"]
    assert out["queue_overflow"] == [f"{st_b.study_id}/{jb.run_id}"]
    jb2 = store.get_job(st_b.study_id, jb.run_id)
    assert jb2.status == "failed" and "QUEUE_FULL" in (jb2.error or "")
