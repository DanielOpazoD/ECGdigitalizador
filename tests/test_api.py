import json
import threading
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from test_pipeline import FakeDigitizer, _fake_output, _study

from ecg_photo.api import create_app
from ecg_photo.ingest import load_supported_inputs
from ecg_photo.store import ExecutionLimits, Store
from ecg_photo.worker import Worker


def _client(tmp_path: Path, max_pending: int = 5, engines=None):
    store = Store(
        tmp_path / "store",
        load_supported_inputs(),
        ExecutionLimits(max_active_jobs=1, max_pending_jobs=max_pending),
    )
    if engines is None:
        engines = {"fake": lambda rc: FakeDigitizer(_fake_output(np.sin(np.linspace(0, 9, 1500))))}
    worker = Worker(store, engines)
    worker.start()
    return TestClient(create_app(store, worker)), store, worker


def _upload(client: TestClient, tmp_path: Path, filename: str = "ecg.png") -> dict:
    png = _study(tmp_path / "src") / "pages" / "page-1.png"
    with open(png, "rb") as fh:
        r = client.post("/studies", files={"file": (filename, fh, "image/png")})
    assert r.status_code == 201, r.text
    return r.json()


def _wait_job(client: TestClient, run_id: str, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = client.get(f"/runs/{run_id}").json()
        if j["status"] in ("completed", "completed_unpublished", "failed", "cancelled"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def _patch_engine(client: TestClient, sid: str, rev: int) -> dict:
    r = client.patch(
        f"/studies/{sid}",
        json={
            "expected_revision": rev,
            "changes": {
                "engine": "fake",
                "engine_duration_s": 3.0,
                "gain_mm_mV": 10,
                "speed_mm_s": 25,
                "time_source": "engine",
            },
            "author": "t",
            "reason": "r",
        },
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_upload_get_and_engine_none(tmp_path) -> None:
    client, _store, worker = _client(tmp_path)
    try:
        st = _upload(client, tmp_path)
        g = client.get(f"/studies/{st['study_id']}")
        assert g.status_code == 200
        body = g.json()
        assert body["active_revision"] == 1 and body["pages"]
        r = client.post(
            f"/studies/{st['study_id']}/runs",
            json={"expected_revision": 1, "config_hash": st["config_hash"]},
        )
        assert r.status_code == 422
    finally:
        worker.shutdown()


def test_patch_run_publish_export(tmp_path) -> None:
    client, _store, worker = _client(tmp_path)
    try:
        st = _upload(client, tmp_path)
        sid = st["study_id"]
        bad = client.patch(
            f"/studies/{sid}",
            json={
                "expected_revision": 1,
                "changes": {"bogus_key": 1},
                "author": "t",
                "reason": "r",
            },
        )
        assert bad.status_code == 422
        conflict = client.patch(
            f"/studies/{sid}",
            json={
                "expected_revision": 99,
                "changes": {},
                "author": "t",
                "reason": "r",
            },
        )
        assert conflict.status_code == 409

        st2 = _patch_engine(client, sid, 1)
        assert st2["revision"] == 2 and st2["config_hash"] != st["config_hash"]

        r = client.post(
            f"/studies/{sid}/runs",
            json={"expected_revision": 2, "config_hash": st2["config_hash"]},
        )
        assert r.status_code == 202, r.text
        run_id = r.json()["run_id"]
        job = _wait_job(client, run_id)
        assert job["status"] == "completed", job

        g = client.get(f"/studies/{sid}").json()
        assert g["published_run_id"] == run_id
        assert g["segments"] and g["segments"][0]["run_id"] == run_id
        assert g["published_result_stale"] is False

        ex = client.post(
            f"/studies/{sid}/exports",
            json={"revision": 2, "run_id": run_id, "formats": ["csv", "json", "png"]},
        )
        assert ex.status_code == 201, ex.text
        arts = ex.json()["artifacts"]
        assert arts
        a0 = arts[0]
        dl = client.get(f"/studies/{sid}/artifacts/{a0['artifact_id']}")
        assert dl.status_code == 200 and dl.content
        assert client.get(f"/studies/{sid}/artifacts/art-nope").status_code == 404

        # exports rejected for a run of a different revision / not completed
        bad2 = client.post(
            f"/studies/{sid}/exports",
            json={"revision": 1, "run_id": run_id, "formats": ["csv"]},
        )
        assert bad2.status_code == 409
    finally:
        worker.shutdown()


def test_t40_patch_during_run_keeps_old_result(tmp_path) -> None:
    release = threading.Event()
    client, store, worker = _client(
        tmp_path,
        engines={"fake": lambda rc: _Slow(release)},
    )
    try:
        st = _upload(client, tmp_path)
        sid = st["study_id"]
        st2 = _patch_engine(client, sid, 1)
        r = client.post(
            f"/studies/{sid}/runs",
            json={"expected_revision": 2, "config_hash": st2["config_hash"]},
        )
        run_id = r.json()["run_id"]
        deadline = time.time() + 10
        while time.time() < deadline:
            if client.get(f"/runs/{run_id}").json()["status"] == "running":
                break
            time.sleep(0.02)
        _patch_engine(client, sid, 2)
        release.set()
        job = _wait_job(client, run_id)
        assert job["status"] == "completed_unpublished"
        g = client.get(f"/studies/{sid}").json()
        assert g["published_run_id"] != run_id
        assert (store.run_dir(sid, run_id) / "result").exists()
    finally:
        release.set()
        worker.shutdown()


def test_retry_only_last_selected_publishes(tmp_path) -> None:
    client, _store, worker = _client(tmp_path)
    try:
        st = _upload(client, tmp_path)
        sid = st["study_id"]
        st2 = _patch_engine(client, sid, 1)
        runs = []
        for _ in range(2):
            r = client.post(
                f"/studies/{sid}/runs",
                json={"expected_revision": 2, "config_hash": st2["config_hash"]},
            )
            runs.append(r.json()["run_id"])
        for run_id in runs:
            _wait_job(client, run_id)
        jobs = {r: client.get(f"/runs/{r}").json()["status"] for r in runs}
        assert jobs[runs[1]] == "completed"
        assert jobs[runs[0]] == "completed_unpublished"
        assert client.get(f"/studies/{sid}").json()["published_run_id"] == runs[1]
    finally:
        worker.shutdown()


def test_t50_delete_during_run(tmp_path) -> None:
    release = threading.Event()
    client, store, worker = _client(tmp_path, engines={"fake": lambda rc: _Slow(release)})
    try:
        st = _upload(client, tmp_path)
        sid = st["study_id"]
        st2 = _patch_engine(client, sid, 1)
        r = client.post(
            f"/studies/{sid}/runs",
            json={"expected_revision": 2, "config_hash": st2["config_hash"]},
        )
        run_id = r.json()["run_id"]
        deadline = time.time() + 10
        while time.time() < deadline:
            if client.get(f"/runs/{run_id}").json()["status"] == "running":
                break
            time.sleep(0.02)
        d = client.delete(f"/studies/{sid}")
        assert d.status_code == 204
        release.set()
        deadline = time.time() + 30
        time.sleep(0.3)
        sdir = store.root / sid
        assert client.get(f"/studies/{sid}").status_code == 404
        for p in sdir.rglob("result"):
            raise AssertionError("late result recreated after delete")
    finally:
        release.set()
        worker.shutdown()


def test_t43_t44_limits_and_filename(tmp_path) -> None:
    from test_ingest import _png_bytes

    from ecg_photo.ingest import SupportedInputs

    # separate store with a tiny pixel cap to trigger PIXEL_LIMIT
    store2 = Store(
        tmp_path / "store2",
        SupportedInputs(max_file_mib=25, max_pdf_pages=10, max_pixels_per_page=100),
        ExecutionLimits(max_active_jobs=1, max_pending_jobs=5),
    )
    worker2 = Worker(store2, {})
    worker2.start()
    client2 = TestClient(create_app(store2, worker2))
    r = client2.post("/studies", files={"file": ("b.png", _png_bytes((20, 20)), "image/png")})
    assert r.status_code == 413 and r.json()["detail"]["code"] == "PIXEL_LIMIT"
    worker2.shutdown()

    client, _store, worker = _client(tmp_path)
    try:
        with open(tmp_path / "fake.zip", "wb") as fh:
            fh.write(b"PK\x03\x04" + b"\x00" * 64)
        with open(tmp_path / "fake.zip", "rb") as fh:
            r = client.post("/studies", files={"file": ("a.zip", fh, "application/zip")})
        assert r.status_code == 415

        st = _upload(client, tmp_path, filename="../../etc/passwd<script>")
        g = client.get(f"/studies/{st['study_id']}").json()
        assert g["original_filename"] is not None
        assert "<script>" not in g["original_filename"]
        assert "/" not in st["study_id"]

        assert client.get("/health").json() == {"status": "ok"}
        assert "store" not in client.get("/health").text.lower()
    finally:
        worker.shutdown()


class _Slow:
    spec = FakeDigitizer(_fake_output(np.zeros(4))).spec

    def __init__(self, release: threading.Event) -> None:
        self.release = release

    def run(self, image_path, work_dir):
        self.release.wait(30)
        return _fake_output(np.sin(np.linspace(0, 9, 1500)))


def test_upload_options_applied_to_rev1(tmp_path) -> None:
    client, _store, worker = _client(tmp_path)
    try:
        png = _study(tmp_path / "src") / "pages" / "page-1.png"
        with open(png, "rb") as fh:
            r = client.post(
                "/studies",
                files={"file": ("ecg.png", fh, "image/png")},
                data={
                    "options": json.dumps({"engine": "fake", "gain_mm_mV": 10, "speed_mm_s": 25})
                },
            )
        assert r.status_code == 201, r.text
        st = r.json()
        g = client.get(f"/studies/{st['study_id']}").json()
        assert g["active_revision"] == 1
        assert g["config_hash"] != "" and g["config_hash"] == st["config_hash"]
        cfg = json.loads((_store.root / st["study_id"] / "rev1" / "config.json").read_text())
        assert cfg["engine"] == "fake" and cfg["gain_mm_mV"] == 10
        cal = json.loads((_store.root / st["study_id"] / "rev1" / "calibration.json").read_text())
        assert any(e["quantity"] == "gain_mm_mV" and e["author"] == "uploader" for e in cal)
        # a study without options gets a different hash
        st2 = _upload(client, tmp_path / "b")
        assert st2["config_hash"] != st["config_hash"]
    finally:
        worker.shutdown()
