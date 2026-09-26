import json
from pathlib import Path

import numpy as np
import pytest
from test_pipeline import FakeDigitizer

from ecg_photo.cli import main
from ecg_photo.contracts import TransformChain, TransformStep
from ecg_photo.digitizers.base import EngineOutput, LeadGeometry
from ecg_photo.fixtures import write_fixture_revision
from ecg_photo.process import COLUMNS, ProcessOptions, overview_columns, process_file
from ecg_photo.render import PaperSpec, render_segment_png

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
FS = 500.0
N = 5000  # engine canvas: 10 s @ 500 Hz


def _page(tmp_path: Path) -> Path:
    rev = tmp_path / "rev"
    m = write_fixture_revision(rev, "calibrated")
    png = tmp_path / "page.png"
    render_segment_png(rev, m.segments[0], png, PaperSpec(25.0, 10.0, 150.0))
    return png


def _engine_output(*, geometry: bool, layout: str = "3x4+1R") -> EngineOutput:
    """12 leads on the engine's 10 s canvas; short leads observed only in
    their 2.5 s slot, II over the whole strip (beats every 0.8 s); with
    layout 6x2, limb leads 0-5 s and V1-V6 5-10 s."""
    t = np.arange(N) / FS
    beat = np.exp(-(((t % 0.8) - 0.4) ** 2) / (2 * 0.012**2))
    leads, observed, geo = {}, {}, {}
    for k, name in enumerate(LEADS):
        obs = np.ones(N, bool)
        if layout == "standard_6x2":
            obs = t < 5.0 if k < 6 else t >= 5.0
        elif name != "II":
            col = next(
                c
                for c, grp in enumerate(("I III", "aVR aVL aVF", "V1 V2 V3", "V4 V5 V6"))
                if name in grp.split()
            )
            obs = (t >= 2.5 * col) & (t < 2.5 * (col + 1))
        leads[name] = np.where(obs, (0.5 + 0.1 * k) * beat, np.nan)
        observed[name] = obs
        if geometry:
            chain = TransformChain(
                transform_id="fake-geo",
                source_frame_id="engine-canonical:fake",
                target_frame_id="file",
                steps=[
                    TransformStep(
                        kind="affine",
                        parameters={"matrix": [0.4, 0.0, 20.0, 0.0, 0.0, 0.0]},
                        input_size=(N, 1),
                        output_size=(2100, 1500),
                        procedure_version="test",
                    )
                ],
            )
            geo[name] = LeadGeometry(
                frame_id="file",
                x_px=20.0 + 0.4 * np.arange(N, dtype=np.float64),
                y_ref_px=None,
                chain=chain,
                method="test affine",
                limitations="synthetic",
            )
    return EngineOutput(
        engine_id="fake",
        engine_commit="c" * 40,
        weights_sha256=(),
        config_hash="h" * 64,
        fs_hz=FS,
        leads=leads,
        observed=observed,
        layout_detected=layout,
        geometry=geo or None,
    )


OPTS = ProcessOptions(speed_mm_s=25.0, gain_mm_mV=10.0, author="t", reason="r", printed_rr_ms=800)


def test_process_falls_back_to_engine_axis_and_says_why(tmp_path: Path) -> None:
    out = tmp_path / "out"
    s = process_file(_page(tmp_path), out, FakeDigitizer(_engine_output(geometry=False)), OPTS)
    assert s["time_source"] == "engine"
    assert "TIME_SCALE_UNKNOWN" in s["evidence_axis_refused"]
    # the refused evidence attempt left no partial run behind: engine + confirmed
    assert len(list((out / "runs").iterdir())) == 2
    assert s["qc"]["label"] is not None and s["qc"]["rr_error_pct"] is not None
    assert abs(s["qc"]["rr_error_pct"]) < 2
    assert (out / "overview.png").stat().st_size > 0
    assert (out / "summary.json").exists() and (out / "export" / "export_report.json").exists()
    assert "seg-II-page-1.csv" in s["export"]["produced"]
    assert json.loads((out / "summary.json").read_text())["input"]["name"] == "page.png"


def test_process_uses_evidence_axis_when_available(tmp_path: Path) -> None:
    out = tmp_path / "out"
    s = process_file(_page(tmp_path), out, FakeDigitizer(_engine_output(geometry=True)), OPTS)
    assert s["time_source"] == "evidence" and s["evidence_axis_refused"] is None
    px_per_mm = s["page"]["grid_px_per_mm_x"]
    assert px_per_mm is not None
    ii = next(x for x in s["leads"] if x["lead"] == "II")
    # 0.4 px per engine sample over 5000 samples at the page's own grid x 25 mm/s
    assert ii["observed_duration_s"] == pytest.approx(0.4 * N / (px_per_mm * 25.0), rel=1e-3)


def test_process_evidence_only_refuses_without_leftovers(tmp_path: Path) -> None:
    out = tmp_path / "out"
    opts = ProcessOptions(
        speed_mm_s=25.0, gain_mm_mV=10.0, author="t", reason="r", time_source="evidence"
    )
    with pytest.raises(ValueError, match="TIME_SCALE_UNKNOWN"):
        process_file(_page(tmp_path), out, FakeDigitizer(_engine_output(geometry=False)), opts)
    assert len(list((out / "runs").iterdir())) == 1  # only the engine run
    assert not (out / "summary.json").exists()


def test_process_refuses_non_empty_out(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "x").write_text("x")
    with pytest.raises(ValueError, match="not empty"):
        process_file(_page(tmp_path), out, FakeDigitizer(_engine_output(geometry=False)), OPTS)


def test_cli_process(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("ECG_PHOTO_ENABLE_FAKE_ENGINE", "1")
    base = ["process", str(_page(tmp_path)), "--speed", "25", "--gain", "10"]
    base += ["--author", "t", "--reason", "r", "--engines-config", str(tmp_path / "none.yml")]
    assert main([*base, "--out", str(tmp_path / "a"), "--engine", "nope"]) == 2
    assert "not configured" in capsys.readouterr().out
    assert main([*base, "--out", str(tmp_path / "b"), "--engine", "fake", "--duration", "3"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["time_source"] == "engine" and res["qc_label"] is not None
    assert Path(res["overview"]).exists()


def test_process_6x2_layout(tmp_path: Path) -> None:
    out = tmp_path / "out"
    dig = FakeDigitizer(_engine_output(geometry=False, layout="standard_6x2"))
    s = process_file(_page(tmp_path), out, dig, OPTS)
    assert s["layout_detected"] == "standard_6x2" and s["qc"]["layout"] == "standard_6x2"
    assert {q["expected_s"] for q in s["qc"]["leads"]} == {5.0}
    assert s["qc"]["rr_lead"] == "II" and abs(s["qc"]["rr_error_pct"]) < 2
    assert (out / "overview.png").stat().st_size > 0


def test_overview_columns() -> None:
    assert overview_columns(2.5) == COLUMNS
    assert [len(c) for c in overview_columns(5.0)] == [6, 6]
    assert [len(c) for c in overview_columns(10.0)] == [12]
