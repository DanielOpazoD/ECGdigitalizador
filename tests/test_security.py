import json
from pathlib import Path

from fastapi.testclient import TestClient
from test_api import _client, _patch_engine, _upload

from ecg_photo.api import create_app
from ecg_photo.cli import main


def test_access_token(tmp_path: Path) -> None:
    client, store, worker = _client(tmp_path)
    try:
        tok = TestClient(create_app(store, worker, token="s3cret-token"))
        # pages that carry no study data stay reachable
        assert tok.get("/health").status_code == 200
        assert tok.get("/ui").status_code == 200
        # everything else needs the token
        r = tok.get("/studies/study-x")
        assert r.status_code == 401 and r.json()["detail"]["code"] == "UNAUTHORIZED"
        assert (
            tok.get("/studies/study-x", headers={"Authorization": "Bearer nope"}).status_code == 401
        )
        ok = {"Authorization": "Bearer s3cret-token"}
        assert tok.get("/studies/study-x", headers=ok).status_code == 404  # authorised, unknown
        # images and downloads: ?token=
        assert tok.get("/studies/study-x?token=s3cret-token").status_code == 404
        # and without a token configured nothing changes
        assert client.get("/studies/study-x").status_code == 404
    finally:
        worker.shutdown()


def test_delete_leaves_no_trace_and_is_audited(tmp_path: Path) -> None:
    client, store, worker = _client(tmp_path)
    try:
        st = _upload(client, tmp_path, filename="juan_perez_ecg.png")
        sid = st["study_id"]
        _patch_engine(client, sid, 1)
        assert client.delete(f"/studies/{sid}").status_code == 204
        study_dir = store.root / sid
        assert sorted(p.name for p in study_dir.iterdir()) == ["state.json"]
        assert client.get(f"/studies/{sid}").status_code == 404
        # no file anywhere in the store keeps the (patient-identifying) name
        for p in store.root.rglob("*"):
            if p.is_file():
                assert b"juan_perez" not in p.read_bytes(), p
        events = [
            json.loads(line) for line in (store.root / "audit.jsonl").read_text().splitlines()
        ]
        assert [e["event"] for e in events] == [
            "study_created",
            "revision_created",
            "study_deleted",
        ]
        assert events[1]["author"] == "t" and all(e["study_id"] == sid for e in events)
    finally:
        worker.shutdown()


def test_serving_beyond_this_computer_needs_a_token(tmp_path: Path, capsys) -> None:
    base = ["serve", "--store", str(tmp_path / "s"), "--host", "0.0.0.0"]
    assert main(base) == 2
    assert "non-loopback" in capsys.readouterr().out
    assert main([*base, "--allow-non-loopback"]) == 2
    assert "needs an access token" in capsys.readouterr().out
