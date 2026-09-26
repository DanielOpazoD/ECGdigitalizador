"""T48/T49: batch processing with per-file isolation and resume."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from test_pipeline import FakeDigitizer, _fake_output, _study

from ecg_photo.batch import BatchError, list_inputs, run_batch
from ecg_photo.cli import main
from ecg_photo.ingest import load_supported_inputs
from ecg_photo.store import ExecutionLimits, Store, StudyPatch

OPTIONS = StudyPatch(
    engine="fake",
    engine_duration_s=3.0,
    gain_mm_mV=10,
    speed_mm_s=25,
    time_source="engine",
)


def _store(tmp_path: Path) -> Store:
    return Store(
        tmp_path / "store",
        load_supported_inputs(),
        ExecutionLimits(max_active_jobs=1, max_pending_jobs=5),
    )


def _inputs(tmp_path: Path) -> Path:
    src = _study(tmp_path / "src") / "pages" / "page-1.png"
    d = tmp_path / "in"
    d.mkdir()
    shutil.copy(src, d / "a.png")
    shutil.copy(src, d / "b.png")
    (d / "c.txt").write_text("not an ecg")
    (d / ".hidden.png").write_bytes(src.read_bytes())
    return d


class _FlakyDigitizer(FakeDigitizer):
    """Fails on the calls listed in `fail_on` (1-based, counted across files)."""

    calls = 0

    def __init__(self, fail_on: set[int]) -> None:
        super().__init__(_fake_output(np.sin(np.linspace(0, 9, 1500))))
        self.fail_on = fail_on

    def run(self, image_path, work_dir):
        type(self).calls += 1
        if type(self).calls in self.fail_on:
            raise RuntimeError("engine crashed")
        return super().run(image_path, work_dir)


def _engines(fail_on: set[int] = frozenset()):  # type: ignore[assignment]
    _FlakyDigitizer.calls = 0
    return {"fake": lambda rc: _FlakyDigitizer(set(fail_on))}


def test_batch_isolates_failures_and_writes_report(tmp_path) -> None:
    store = _store(tmp_path)
    inputs = list_inputs(_inputs(tmp_path))
    assert [p.name for p in inputs] == ["a.png", "b.png", "c.txt"]
    report_path = tmp_path / "report.json"
    rep = run_batch(store, _engines({1}), inputs, OPTIONS, report_path, author="op", reason="lote")
    by = {e["input"]: e for e in rep["entries"]}
    assert by["a.png"]["status"] == "failed" and "engine crashed" in by["a.png"]["error"]
    assert by["b.png"]["status"] == "published"
    # quality label of the published result travels in the batch report
    assert by["b.png"]["qc_label"] == "insufficient"  # fake engine: one lead
    assert "MISSING_LEAD" in by["b.png"]["qc_flags"]
    assert "qc_label" not in by["a.png"]
    assert by["c.txt"]["status"] == "rejected"
    assert by["c.txt"]["reason_code"] == "UNSUPPORTED_TYPE"
    assert rep["summary"] == {"failed": 1, "published": 1, "rejected": 1}
    on_disk = json.loads(report_path.read_text())
    assert on_disk["summary"] == rep["summary"]

    # published study is a normal store study, with manual evidence from the batch
    sid = by["b.png"]["study_id"]
    assert store.get(sid).published_run_id == by["b.png"]["run_id"]
    cal = json.loads((store.rev_dir(sid, 1) / "calibration.json").read_text())
    assert {c["author"] for c in cal} == {"op"}
    assert {c["method"] for c in cal} == {"initial options via batch"}


def test_batch_resume_retries_only_unfinished(tmp_path) -> None:
    store = _store(tmp_path)
    inputs = list_inputs(_inputs(tmp_path))
    report_path = tmp_path / "report.json"
    first = run_batch(store, _engines({1}), inputs, OPTIONS, report_path, author="o", reason="r")
    a_first = next(e for e in first["entries"] if e["input"] == "a.png")
    b_first = next(e for e in first["entries"] if e["input"] == "b.png")

    with pytest.raises(BatchError, match="--resume"):
        run_batch(store, _engines(), inputs, OPTIONS, report_path, author="o", reason="r")
    other = OPTIONS.model_copy(update={"gain_mm_mV": 20})
    with pytest.raises(BatchError, match="options differ"):
        run_batch(
            store, _engines(), inputs, other, report_path, author="o", reason="r", resume=True
        )

    second = run_batch(
        store, _engines(), inputs, OPTIONS, report_path, author="o", reason="r", resume=True
    )
    by = {e["input"]: e for e in second["entries"]}
    assert _FlakyDigitizer.calls == 1  # only a.png ran again
    assert by["a.png"]["status"] == "published" and by["a.png"]["attempts"] == 2
    assert by["a.png"]["study_id"] == a_first["study_id"]  # same study, new run
    assert by["a.png"]["run_id"] != a_first["run_id"]
    assert by["b.png"] == b_first
    assert second["summary"] == {"published": 2, "rejected": 1}
    studies = [p for p in store.root.iterdir() if p.is_dir()]
    assert len(studies) == 2


def test_batch_resume_after_crash_reuses_queued_run(tmp_path) -> None:
    store = _store(tmp_path)
    inputs = list_inputs(_inputs(tmp_path))[:1]
    report_path = tmp_path / "report.json"
    # simulate: study + run created, process killed before the engine ran
    st = store.create_study(inputs[0], inputs[0].name, options=OPTIONS)
    job = store.create_run(st.study_id, 1, st.config_hash)
    from ecg_photo.batch import SCHEMA, options_hash
    from ecg_photo.contracts import sha256_file

    report_path.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "options_hash": options_hash(OPTIONS),
                "entries": [
                    {
                        "input": "a.png",
                        "sha256": sha256_file(inputs[0]),
                        "study_id": st.study_id,
                        "run_id": job.run_id,
                        "status": "processing",
                        "attempts": 1,
                    }
                ],
            }
        )
    )
    assert store.recover().requeue == [(st.study_id, job.run_id)]
    rep = run_batch(
        store, _engines(), inputs, OPTIONS, report_path, author="o", reason="r", resume=True
    )
    e = rep["entries"][0]
    assert e["status"] == "published" and e["run_id"] == job.run_id
    assert len(store.list_jobs(st.study_id)) == 1


def test_batch_rejects_bad_options(tmp_path) -> None:
    store = _store(tmp_path)
    rp = tmp_path / "r.json"
    with pytest.raises(BatchError, match="engine"):
        run_batch(store, _engines(), [], StudyPatch(), rp, author="o", reason="r")
    with pytest.raises(BatchError, match="not configured"):
        run_batch(store, _engines(), [], StudyPatch(engine="ahus"), rp, author="o", reason="r")
    with pytest.raises(BatchError, match="lead_labels"):
        run_batch(
            store,
            _engines(),
            [],
            OPTIONS.model_copy(update={"lead_labels": {"seg-1": "II"}}),
            rp,
            author="o",
            reason="r",
        )


def test_cli_batch_and_store_lock(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("ECG_PHOTO_ENABLE_FAKE_ENGINE", "1")
    in_dir = _inputs(tmp_path)
    (in_dir / "c.txt").unlink()
    store_dir = tmp_path / "store"
    args = [
        "batch",
        str(in_dir),
        "--store",
        str(store_dir),
        "--report",
        str(tmp_path / "rep.json"),
        "--engine",
        "fake",
        "--duration",
        "3",
        "--time-source",
        "engine",
        "--gain",
        "10",
        "--speed",
        "25",
    ]
    assert main(args) == 2  # manual speed/gain without author/reason
    assert "--author" in capsys.readouterr().out
    args += ["--author", "op", "--reason", "lote"]

    holder = _store(tmp_path)
    holder.acquire_process_lock()
    try:
        assert main(args) == 2
        assert "STORE_LOCKED" in capsys.readouterr().out
    finally:
        holder.release_process_lock()

    assert main(args) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["summary"] == {"published": 2}
