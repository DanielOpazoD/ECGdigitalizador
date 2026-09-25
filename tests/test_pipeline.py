import hashlib
import json
from pathlib import Path

import numpy as np
import wfdb

from ecg_photo.cli import main
from ecg_photo.contracts import (
    ScaleStatus,
    load_manifest,
    validate_revision_dir,
)
from ecg_photo.digitizers.base import EngineOutput, EngineSpec
from ecg_photo.fixtures import write_fixture_revision
from ecg_photo.ingest import ingest
from ecg_photo.pipeline import confirm_engine_scale, digitize_page


class FakeDigitizer:
    def __init__(self, out: EngineOutput) -> None:
        self.spec = EngineSpec(engine_id="fake", repo="r", commit="c" * 8, license="l", weights=())
        self._out = out

    def run(self, image_path: Path, work_dir: Path) -> EngineOutput:
        return self._out


def _fake_output(truth: np.ndarray, *, zero_encoded: bool = False) -> EngineOutput:
    n = 1500  # 3 s @ 500 Hz
    ii = truth[:n]
    v1 = truth[:n].copy()
    v1[500:800] = np.nan if not zero_encoded else 0.0
    if zero_encoded:
        v1 = np.nan_to_num(v1)
    v4 = np.full(n, np.nan) if not zero_encoded else np.zeros(n)
    extra: dict = {}
    if zero_encoded:
        extra = {"missing_encoded_as_zero": True}
    return EngineOutput(
        engine_id="fake",
        engine_commit="c" * 40,
        weights_sha256=(),
        config_hash="h" * 64,
        fs_hz=500.0,
        leads={"II": ii, "V1": v1, "V4": v4},
        observed={
            "II": np.isfinite(ii),
            "V1": np.isfinite(v1) if not zero_encoded else np.ones(n, bool),
            "V4": np.zeros(n, bool),
        },
        layout_detected=None,
        extra=extra,
    )


def _study(tmp_path: Path) -> Path:
    rev = tmp_path / "rev"
    m = write_fixture_revision(rev, "calibrated")
    from ecg_photo.render import PaperSpec, render_segment_png

    png = tmp_path / "page.png"
    render_segment_png(rev, m.segments[0], png, PaperSpec(25.0, 10.0, 150.0))
    study = tmp_path / "study"
    ingest(png, study)
    return study


def test_digitize_page_unknown_then_confirm(tmp_path) -> None:
    rev = tmp_path / "rev"
    m = write_fixture_revision(rev, "calibrated")
    truth = np.load(rev / m.segments[0].signal_path)
    study = _study(tmp_path)

    paths = digitize_page(
        study, "page-1", FakeDigitizer(_fake_output(truth)), engine_duration_s=3.0
    )
    rm = load_manifest(paths.manifest)
    assert validate_revision_dir(paths.run_dir, rm) == []
    assert {s.lead_label for s in rm.segments} == {"II", "V1"}
    for s in rm.segments:
        assert s.speed_status == ScaleStatus.unknown
        assert s.gain_status == ScaleStatus.unknown
        assert s.units is None and s.signal_path is None
        assert s.temporal_trace_units == "arbitrary"
    report = json.loads((paths.run_dir / "run_report.json").read_text())
    assert report["skipped_leads"] == [{"lead": "V4", "reason": "NO_SUPPORT"}]

    # export of unconfirmed run skips everything
    exp = tmp_path / "exp0"
    assert main(["export", str(paths.run_dir), "--out", str(exp)]) == 0
    rep = json.loads((exp / "export_report.json").read_text())
    assert rep["produced"] == ["manifest.json"]
    assert {s["reason_code"] for s in rep["skipped"]} == {"TIME_SCALE_UNKNOWN"}

    before_hash = hashlib.sha256(paths.manifest.read_bytes()).hexdigest()
    p2 = confirm_engine_scale(
        paths.run_dir, speed_mm_s=25.0, gain_mm_mV=10.0, author="t", reason="r"
    )
    assert hashlib.sha256(paths.manifest.read_bytes()).hexdigest() == before_hash

    m2 = load_manifest(p2.manifest)
    assert validate_revision_dir(p2.run_dir, m2) == []
    ii = next(s for s in m2.segments if s.lead_label == "II")
    sig = np.load(p2.run_dir / ii.signal_path)
    obs = np.load(p2.run_dir / ii.observed_mask_path)
    np.testing.assert_allclose(sig[obs], truth[:1500][obs], atol=1e-9)
    v1 = next(s for s in m2.segments if s.lead_label == "V1")
    sig1 = np.load(p2.run_dir / v1.signal_path)
    obs1 = np.load(p2.run_dir / v1.observed_mask_path)
    assert np.isnan(sig1[500:800]).all()
    assert not obs1[500:800].any()
    assert len(sig1) == v1.n_samples
    assert ii.speed_status == ScaleStatus.manual

    exp2 = tmp_path / "exp2"
    assert main(["export", str(p2.run_dir), "--out", str(exp2)]) == 0
    rep2 = json.loads((exp2 / "export_report.json").read_text())
    assert "seg-II-page-1.png" in rep2["produced"]
    rec = wfdb.rdrecord(str(exp2 / "seg-V1-page-1"))
    assert rec.fs == 500.0 and rec.sig_name == ["V1"]
    assert np.isnan(rec.p_signal[500:800, 0]).all()
    rec_ii = wfdb.rdrecord(str(exp2 / "seg-II-page-1"))
    np.testing.assert_allclose(rec_ii.p_signal[obs, 0], truth[:1500][obs], atol=1e-3)


def test_zero_run_masking(tmp_path) -> None:
    rev = tmp_path / "rev"
    m = write_fixture_revision(rev, "calibrated")
    truth = np.load(rev / m.segments[0].signal_path)
    study = _study(tmp_path)
    paths = digitize_page(
        study,
        "page-1",
        FakeDigitizer(_fake_output(truth, zero_encoded=True)),
        engine_duration_s=3.0,
    )
    rm = load_manifest(paths.manifest)
    v1 = next(s for s in rm.segments if s.lead_label == "V1")
    with np.load(paths.run_dir / v1.raw_support_path) as z:
        sup = z["support"]
    assert not sup[500:800].any()  # 300-sample zero run masked
    # the fixture baseline is exactly 0.0 for the first 90 samples -> also masked
    assert sup[90:500].all()
    step = next(st for st in v1.processing if st.stage == "zero_run_masking")
    assert step.parameters["masked_samples"] == 390


def test_cli_confirm_scale(tmp_path, capsys) -> None:
    rev = tmp_path / "rev"
    m = write_fixture_revision(rev, "calibrated")
    truth = np.load(rev / m.segments[0].signal_path)
    study = _study(tmp_path)
    paths = digitize_page(
        study, "page-1", FakeDigitizer(_fake_output(truth)), engine_duration_s=3.0
    )
    assert (
        main(
            [
                "confirm-scale",
                str(paths.run_dir),
                "--speed",
                "25",
                "--gain",
                "10",
                "--author",
                "t",
                "--reason",
                "r",
            ]
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert Path(out["run_dir"]).exists()
    assert Path(out["run_dir"]) != paths.run_dir


def test_cli_digitize_missing_engine_args(tmp_path, capsys) -> None:
    rc = main(
        ["digitize", str(tmp_path), "--page", "page-1", "--engine", "ahus", "--duration", "10"]
    )
    assert rc == 2
    assert "missing args" in capsys.readouterr().out


def test_confirm_resample_different_fs(tmp_path) -> None:
    rev = tmp_path / "rev"
    m = write_fixture_revision(rev, "calibrated")
    truth = np.load(rev / m.segments[0].signal_path)
    study = _study(tmp_path)
    paths = digitize_page(
        study, "page-1", FakeDigitizer(_fake_output(truth)), engine_duration_s=3.0
    )
    p2 = confirm_engine_scale(
        paths.run_dir, speed_mm_s=25.0, gain_mm_mV=10.0, author="t", reason="r", fs_hz=250.0
    )
    m2 = load_manifest(p2.manifest)
    ii = next(s for s in m2.segments if s.lead_label == "II")
    assert ii.working_fs_hz == 250.0
    assert ii.n_samples == 750
    sig = np.load(p2.run_dir / ii.signal_path)
    obs = np.load(p2.run_dir / ii.observed_mask_path)
    interp_truth = np.interp(np.arange(750) / 250.0, np.arange(1500) / 500.0, truth[:1500])
    np.testing.assert_allclose(sig[obs], interp_truth[obs], atol=1e-9)
    v1 = next(s for s in m2.segments if s.lead_label == "V1")
    obs1 = np.load(p2.run_dir / v1.observed_mask_path)
    assert not obs1[250:400].any()  # gap preserved at half fs
