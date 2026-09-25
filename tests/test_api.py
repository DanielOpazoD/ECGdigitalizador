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


def _run_and_publish(client: TestClient, tmp_path: Path) -> tuple[str, str]:
    st = _upload(client, tmp_path)
    sid = st["study_id"]
    st2 = _patch_engine(client, sid, 1)
    r = client.post(
        f"/studies/{sid}/runs",
        json={"expected_revision": 2, "config_hash": st2["config_hash"]},
    )
    run_id = r.json()["run_id"]
    job = _wait_job(client, run_id)
    assert job["status"] == "completed", job
    return sid, run_id


def test_read_endpoints(tmp_path) -> None:
    client, _store, worker = _client(tmp_path)
    try:
        sid, run_id = _run_and_publish(client, tmp_path)

        r = client.get(f"/studies/{sid}/pages/page-1/raster")
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        assert r.headers.get("cache-control") == "no-store"
        assert client.get(f"/studies/{sid}/pages/page-9/raster").status_code == 404

        g = client.get(f"/studies/{sid}/pages/page-1/grid")
        assert g.status_code == 404 or g.status_code == 200  # grid optional at ingest

        segs = client.get(f"/studies/{sid}/runs/{run_id}/segments")
        assert segs.status_code == 200
        seg_list = segs.json()["segments"]
        assert seg_list and "lead_label" in seg_list[0] and "has_geometry" in seg_list[0]
        assert client.get(f"/studies/{sid}/runs/run-nope/segments").status_code == 404

        seg_id = seg_list[0]["segment_id"]
        tr = client.get(f"/studies/{sid}/runs/{run_id}/segments/{seg_id}/trace")
        assert tr.status_code == 200, tr.text
        body = tr.json()
        assert body["units"] == "mV" and body["t_s"] is not None
        assert body["observed"] and body["x_px"] is None  # engine frame, no geometry
        assert (
            client.get(f"/studies/{sid}/runs/{run_id}/segments/seg-nope/trace").status_code == 404
        )

        runs = client.get(f"/studies/{sid}/runs")
        assert runs.status_code == 200
        assert [j["run_id"] for j in runs.json()["runs"]] == [run_id]
        assert "created_at" in runs.json()["runs"][0]
    finally:
        worker.shutdown()


def test_trace_decimation_and_null_gaps(tmp_path) -> None:
    from ecg_photo.digitizers.base import EngineOutput

    n = 45000
    sig = np.sin(np.linspace(0, 20, n))
    sig[1000:5000] = np.nan
    v1 = np.sin(np.linspace(0, 12, n))
    v1[20000:25000] = np.nan

    def _big_output(rc) -> EngineOutput:
        return EngineOutput(
            engine_id="fake",
            engine_commit="c" * 40,
            weights_sha256=(),
            config_hash="h" * 64,
            fs_hz=500.0,
            leads={"II": sig, "V1": v1},
            observed={"II": np.isfinite(sig), "V1": np.isfinite(v1)},
            layout_detected=None,
        )

    client, _store, worker = _client(
        tmp_path,
        engines={"fake": lambda rc: FakeDigitizer(_big_output(rc))},
    )
    try:
        st = _upload(client, tmp_path)
        sid = st["study_id"]
        st2 = client.patch(
            f"/studies/{sid}",
            json={
                "expected_revision": 1,
                "changes": {
                    "engine": "fake",
                    "engine_duration_s": 90,
                    "gain_mm_mV": 10,
                    "speed_mm_s": 25,
                    "time_source": "engine",
                },
                "author": "t",
                "reason": "r",
            },
        ).json()
        r = client.post(
            f"/studies/{sid}/runs",
            json={"expected_revision": 2, "config_hash": st2["config_hash"]},
        )
        run_id = r.json()["run_id"]
        assert _wait_job(client, run_id)["status"] == "completed"
        segs = client.get(f"/studies/{sid}/runs/{run_id}/segments").json()["segments"]
        seg_id = segs[0]["segment_id"]
        tr = client.get(f"/studies/{sid}/runs/{run_id}/segments/{seg_id}/trace").json()
        assert tr.get("decimation") == 3  # ceil(45000/20000)
        assert len(tr["values"]) == len(sig[::3])
        # NaN gaps -> nulls in values, False in observed; never 0
        assert any(v is None for v in tr["values"])
        assert any(o is False for o in tr["observed"])
        # V1 has NaN gaps -> nulls in values, False in observed
        v1 = next(s for s in segs if s["lead_label"] == "V1")
        tv1 = client.get(f"/studies/{sid}/runs/{run_id}/segments/{v1['segment_id']}/trace").json()
        assert any(v is None for v in tv1["values"])
        assert any(o is False for o in tv1["observed"])
    finally:
        worker.shutdown()


def test_lead_label_corrections(tmp_path) -> None:
    client, _store, worker = _client(tmp_path)
    try:
        sid, run_id = _run_and_publish(client, tmp_path)
        segs = client.get(f"/studies/{sid}/runs/{run_id}/segments").json()["segments"]
        v1 = next(s for s in segs if s["lead_label"] == "V1")
        # invalid label -> 422
        bad = client.patch(
            f"/studies/{sid}",
            json={
                "expected_revision": 2,
                "changes": {"lead_labels": {v1["segment_id"]: "BOGUS"}},
                "author": "t",
                "reason": "r",
            },
        )
        assert bad.status_code == 422
        # valid correction -> rev3 + corrections.json
        st3 = client.patch(
            f"/studies/{sid}",
            json={
                "expected_revision": 2,
                "changes": {"lead_labels": {v1["segment_id"]: "aVR"}},
                "author": "t",
                "reason": "mislabelled",
            },
        )
        assert st3.status_code == 200 and st3.json()["revision"] == 3
        g = client.get(f"/studies/{sid}").json()
        corr = g["corrections"]["lead_labels"][v1["segment_id"]]
        assert corr["label"] == "aVR" and corr["previous_label"] == "V1"
        # next run applies it
        r = client.post(
            f"/studies/{sid}/runs",
            json={"expected_revision": 3, "config_hash": st3.json()["config_hash"]},
        )
        job = _wait_job(client, r.json()["run_id"])
        assert job["status"] == "completed", job
        segs2 = client.get(f"/studies/{sid}/runs/{r.json()['run_id']}/segments").json()["segments"]
        fixed = next(s for s in segs2 if s["segment_id"] == v1["segment_id"])
        assert fixed["lead_label"] == "aVR" and fixed["lead_status"] == "confirmed"
    finally:
        worker.shutdown()


def test_ui_grid_and_study_config_keys(tmp_path) -> None:
    client, _store, worker = _client(tmp_path)
    try:
        st = _upload(client, tmp_path)
        sid = st["study_id"]
        # ingest now estimates page grids -> file exists in store
        assert (_store.root / sid / "rev1" / "pages" / "page-1.grid.json").exists()
        g = client.get(f"/studies/{sid}/pages/page-1/grid")
        assert g.status_code == 200 and "px_per_mm_x" in g.json()

        root = client.get("/", follow_redirects=False)
        assert root.status_code == 307 and root.headers["location"] == "/ui"
        ui = client.get("/ui")
        assert ui.status_code == 200
        assert "text/html" in ui.headers["content-type"]
        assert "<title" in ui.text

        body = client.get(f"/studies/{sid}").json()
        assert "config" in body and "corrections" in body
        assert body["config"]["engine"] is None
    finally:
        worker.shutdown()


def test_artifact_filename_safe_is_basename(tmp_path) -> None:
    client, _store, worker = _client(tmp_path)
    try:
        sid, run_id = _run_and_publish(client, tmp_path)
        ex = client.post(
            f"/studies/{sid}/exports",
            json={"revision": 2, "run_id": run_id, "formats": ["csv"]},
        )
        assert ex.status_code == 201
        for art in ex.json()["artifacts"]:
            assert art["filename_safe"] == art["path_rel"].split("/")[-1]
            assert art["filename_safe"].count(sid) == 1
    finally:
        worker.shutdown()
