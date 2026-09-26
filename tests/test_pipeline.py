import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import wfdb

from ecg_photo.cli import main
from ecg_photo.contracts import (
    ScaleStatus,
    TransformChain,
    TransformStep,
    load_manifest,
    validate_revision_dir,
)
from ecg_photo.digitizers.base import EngineOutput, EngineSpec, LeadGeometry
from ecg_photo.fixtures import write_fixture_revision
from ecg_photo.ingest import ingest
from ecg_photo.pipeline import confirm_scale, digitize_page
from ecg_photo.transforms import homography_from_points


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
    p2 = confirm_scale(
        paths.run_dir,
        speed_mm_s=25.0,
        gain_mm_mV=10.0,
        author="t",
        reason="r",
        time_source="engine",
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
                "--time-source",
                "engine",
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
    p2 = confirm_scale(
        paths.run_dir,
        speed_mm_s=25.0,
        gain_mm_mV=10.0,
        author="t",
        reason="r",
        fs_hz=250.0,
        time_source="engine",
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


def _fake_geometry(n: int, *, kind: str = "affine") -> dict[str, LeadGeometry]:
    chain = TransformChain(
        transform_id="fake-geo",
        source_frame_id="engine-canonical:fake",
        target_frame_id="file",
        steps=[
            TransformStep(
                kind="affine",
                parameters={"matrix": [0.5, 0.0, 50.0, 0.0, 0.0, 0.0]},
                input_size=(n, 1),
                output_size=(2000, 2000),
                procedure_version="test",
            )
        ],
    )
    return {
        lead: LeadGeometry(
            frame_id="file",
            x_px=50.0 + 0.5 * np.arange(n, dtype=np.float64),
            y_ref_px=None,
            chain=chain,
            method="test affine x=50+0.5k",
            limitations="synthetic",
        )
        for lead in ("II", "V1", "V4")
    }


def _write_grid(study: Path, px_per_mm_x: float) -> None:
    grid = {
        "px_per_mm_x": px_per_mm_x,
        "px_per_mm_y": px_per_mm_x,
        "method": "test",
        "limitations": "synthetic",
    }
    (study / "pages" / "page-1.grid.json").write_text(json.dumps(grid), encoding="utf-8")
    man = load_manifest(study / "manifest.json")
    for i, pg in enumerate(man.pages):
        man.pages[i] = pg.model_copy(update={"grid_estimate_path": f"pages/{pg.page_id}.grid.json"})
    from ecg_photo.contracts import dump_json

    dump_json(man, study / "manifest.json")


def test_confirm_scale_evidence(tmp_path) -> None:
    rev = tmp_path / "rev"
    m = write_fixture_revision(rev, "calibrated")
    truth = np.load(rev / m.segments[0].signal_path)
    study = _study(tmp_path)
    _write_grid(study, 8.0)

    out = _fake_output(truth)
    out.geometry = _fake_geometry(1500)
    paths = digitize_page(study, "page-1", FakeDigitizer(out), engine_duration_s=3.0)
    rm = load_manifest(paths.manifest)
    assert all(s.raw_coordinate_frame == "file" for s in rm.segments)
    assert (paths.run_dir / "raw_paths" / "seg-II-page-1-geometry.json").exists()

    # extent = 0.5*1499 px = 749.5; 8*25 = 200 px/s -> t_last 3.7475 s,
    # +median dt (0.0025) -> observed 3.75 s; n_samples = 1875 at 500 Hz
    p2 = confirm_scale(paths.run_dir, gain_mm_mV=10.0, author="t", reason="r", speed_mm_s=25.0)
    m2 = load_manifest(p2.manifest)
    assert validate_revision_dir(p2.run_dir, m2) == []
    ii = next(s for s in m2.segments if s.lead_label == "II")
    assert ii.observed_duration_s == pytest.approx(3.75)
    assert ii.n_samples == 1875
    assert ii.speed_status == ScaleStatus.manual
    assert ii.effective_dt_ms == pytest.approx(5.0)
    assert ii.px_per_mm_x == 8.0
    assert any(st.stage == "time_axis_from_image_evidence" for st in ii.processing)
    sig = np.load(p2.run_dir / ii.signal_path)
    obs = np.load(p2.run_dir / ii.observed_mask_path)
    t_true = np.arange(1500) * 0.0025
    t_out = np.arange(ii.n_samples) / 500.0
    np.testing.assert_allclose(sig[obs], np.interp(t_out[obs], t_true, truth[:1500]), atol=1e-9)
    v1 = next(s for s in m2.segments if s.lead_label == "V1")
    sig1 = np.load(p2.run_dir / v1.signal_path)
    obs1 = np.load(p2.run_dir / v1.observed_mask_path)
    # gap t in [500*0.0025, 799*0.0025] = [1.25, 1.9975] -> unobserved cols
    lo, hi = int(1.25 * 500), int(1.9975 * 500)
    assert np.isnan(sig1[lo : hi + 1]).all() or not obs1[lo : hi + 1].any()


def test_confirm_scale_evidence_requires_geometry_and_speed(tmp_path) -> None:
    rev = tmp_path / "rev"
    m = write_fixture_revision(rev, "calibrated")
    truth = np.load(rev / m.segments[0].signal_path)
    study = _study(tmp_path)
    paths = digitize_page(
        study, "page-1", FakeDigitizer(_fake_output(truth)), engine_duration_s=3.0
    )
    # no geometry (engine-canonical raw frame) -> TIME_SCALE_UNKNOWN
    with pytest.raises(ValueError, match="TIME_SCALE_UNKNOWN"):
        confirm_scale(paths.run_dir, gain_mm_mV=10.0, author="t", reason="r", speed_mm_s=25.0)

    # geometry + grid but no speed evidence -> TIME_SCALE_UNKNOWN
    _write_grid(study, 8.0)
    out = _fake_output(truth)
    out.geometry = _fake_geometry(1500)
    paths2 = digitize_page(study, "page-1", FakeDigitizer(out), engine_duration_s=3.0)
    with pytest.raises(ValueError, match="TIME_SCALE_UNKNOWN"):
        confirm_scale(paths2.run_dir, gain_mm_mV=10.0, author="t", reason="r")


def test_homography_from_points_round_trip() -> None:
    src = np.array([[10.0, 20.0], [400.0, 15.0], [405.0, 300.0], [8.0, 295.0]])
    dst = np.array([[100.0, 80.0], [1400.0, 90.0], [1390.0, 900.0], [110.0, 890.0]])
    H = homography_from_points(src, dst)
    pts = np.concatenate([src, np.ones((4, 1))], axis=1)
    back = (pts @ H.T)[:, :2] / (pts @ H.T)[:, 2:3]
    np.testing.assert_allclose(back, dst, atol=1e-6)
    # inverse round-trip
    H_inv = homography_from_points(dst, src)
    pts2 = np.concatenate([dst, np.ones((4, 1))], axis=1)
    np.testing.assert_allclose((pts2 @ H_inv.T)[:, :2] / (pts2 @ H_inv.T)[:, 2:3], src, atol=1e-6)


def test_digitiser_rotation_chain_round_trip() -> None:
    # 10-deg rotation: rotated->file affine step; invert must return the point
    rot_deg = 10.0
    rw, rh = 2000, 1000
    cx, cy = rw / 2.0, rh / 2.0
    th = np.deg2rad(rot_deg)
    c, s = np.cos(th), np.sin(th)
    step_rot = TransformStep(
        kind="affine",
        parameters={
            "matrix": [c, -s, cx - c * cx + s * cy, s, c, cy - s * cx - c * cy],
            "rot_angle_deg": rot_deg,
        },
        input_size=(rw, rh),
        output_size=(rw, rh),
        procedure_version="test",
    )
    chain = TransformChain(
        transform_id="t", source_frame_id="rot", target_frame_id="file", steps=[step_rot]
    )
    from ecg_photo.transforms import apply_chain, invert_chain

    pt = np.array([[1500.0, 400.0]])
    file_pt = apply_chain(chain, pt)
    np.testing.assert_allclose(invert_chain(chain, file_pt), pt, atol=1e-6)
    # sanity: forward torchvision map moves (cx+30, cy) to (cx+30cos, cy-30sin)
    rot_pt = np.array([[cx + 30 * c, cy - 30 * s]])
    np.testing.assert_allclose(apply_chain(chain, rot_pt), [[cx + 30, cy]], atol=1e-6)


def test_engine_duration_mismatch_raises(tmp_path) -> None:
    rev = tmp_path / "rev"
    m = write_fixture_revision(rev, "calibrated")
    truth = np.load(rev / m.segments[0].signal_path)
    study = _study(tmp_path)

    paths = digitize_page(
        study, "page-1", FakeDigitizer(_fake_output(truth)), engine_duration_s=10.0
    )
    with pytest.raises(ValueError, match="check engine_duration_s"):
        confirm_scale(
            paths.run_dir,
            speed_mm_s=25.0,
            gain_mm_mV=10.0,
            author="t",
            reason="r",
            time_source="engine",
        )


def test_confirm_scale_evidence_keeps_run_when_a_lead_has_no_mapped_support(tmp_path) -> None:
    # seen in the F6-b smoke run with ECG-Digitiser: a lead has observed samples
    # but its geometry maps none of them to a page column (x_px NaN)
    rev = tmp_path / "rev"
    m = write_fixture_revision(rev, "calibrated")
    truth = np.load(rev / m.segments[0].signal_path)
    study = _study(tmp_path)
    _write_grid(study, 8.0)
    out = _fake_output(truth)
    out.geometry = _fake_geometry(1500)
    g = out.geometry["V1"]
    out.geometry["V1"] = LeadGeometry(
        frame_id=g.frame_id,
        x_px=np.full(1500, np.nan),
        y_ref_px=None,
        chain=g.chain,
        method=g.method,
        limitations=g.limitations,
    )
    paths = digitize_page(study, "page-1", FakeDigitizer(out), engine_duration_s=3.0)

    p2 = confirm_scale(paths.run_dir, gain_mm_mV=10.0, author="t", reason="r", speed_mm_s=25.0)
    m2 = load_manifest(p2.manifest)
    assert validate_revision_dir(p2.run_dir, m2) == []
    v1 = next(s for s in m2.segments if s.lead_label == "V1")
    assert v1.signal_path is None and v1.speed_status == ScaleStatus.unknown
    assert v1.processing[-1].parameters["skipped"].startswith("no observed samples")
    ii = next(s for s in m2.segments if s.lead_label == "II")
    assert ii.signal_path is not None and ii.speed_status == ScaleStatus.manual
