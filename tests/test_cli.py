import json
from pathlib import Path

import pytest

from ecg_photo.cli import main


def test_cli_run_demo(tmp_path, capsys) -> None:
    out = tmp_path / "demo"
    rc = main(["run-demo", "--out", str(out)])
    assert rc == 0
    report = json.loads((out / "export" / "export_report.json").read_text())
    assert report["skipped"] == []
    produced = " ".join(report["produced"])
    for ext in ("csv", "png", "pdf"):
        assert f"seg-II-01.{ext}" in produced
    assert "wfdb" in produced


def test_cli_export_gain_unknown_report(tmp_path) -> None:
    rev = tmp_path / "rev"
    assert main(["fixture", "--kind", "gain_unknown", "--out", str(rev)]) == 0
    out = tmp_path / "exp"
    assert main(["export", str(rev), "--out", str(out)]) == 0
    report = json.loads((out / "export_report.json").read_text())
    assert "seg-II-01.csv" in report["produced"]
    skipped = {(s["artifact"], s["reason_code"]) for s in report["skipped"]}
    assert ("png", "GAIN_UNKNOWN") in skipped
    assert ("pdf", "GAIN_UNKNOWN") in skipped
    assert ("wfdb", "GAIN_UNKNOWN") in skipped


def test_cli_validate(tmp_path) -> None:
    rev = tmp_path / "rev"
    assert main(["fixture", "--kind", "calibrated", "--out", str(rev)]) == 0
    assert main(["validate", str(rev)]) == 0
    (rev / "manifest.json").unlink()
    # manifest gone -> load fails; recreate via fixture then delete a file instead
    assert main(["fixture", "--kind", "calibrated", "--out", str(rev)]) == 0
    Path(rev / "masks/seg-II-01-observed.npy").unlink()
    assert main(["validate", str(rev)]) == 1


def test_cli_ingest_and_calibrate(tmp_path, capsys) -> None:
    rev = tmp_path / "rev"
    assert main(["fixture", "--kind", "calibrated", "--out", str(rev)]) == 0
    exp = tmp_path / "exp"
    assert main(["export", str(rev), "--out", str(exp), "--dpi", "300"]) == 0
    png = exp / "seg-II-01.png"
    assert png.exists()

    study = tmp_path / "study"
    capsys.readouterr()
    assert main(["ingest", str(png), "--out", str(study)]) == 0
    summary = json.loads(capsys.readouterr().out)
    pg = summary["pages"][0]
    assert pg["extraction"] == "image_file"
    assert pg["px_per_mm_x"] == pytest.approx(300.0 / 25.4, rel=0.05)
    assert (study / "manifest.json").exists()
    assert (study / "pages/page-1.grid.json").exists()
    m = json.loads((study / "manifest.json").read_text())
    assert m["pages"][0]["grid_estimate_path"] == "pages/page-1.grid.json"

    capsys.readouterr()
    assert (
        main(
            [
                "calibrate",
                str(study),
                "--page",
                "page-1",
                "--speed",
                "25",
                "--gain",
                "10",
                "--author",
                "op",
                "--reason",
                "etiqueta impresa",
            ]
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert out["revision"] == 2
    m2 = json.loads((study / "manifest.json").read_text())
    assert m2["revision"] == 2
    assert (study / "manifest.rev1.json").exists()
    cal = json.loads((study / "calibration.json").read_text())
    assert {e["quantity"] for e in cal} == {"speed_mm_s", "gain_mm_mV"}
    assert all(e["kind"] == "manual" and e["author"] == "op" for e in cal)


def test_cli_time_unknown_export(tmp_path) -> None:
    rev = tmp_path / "rev"
    assert main(["fixture", "--kind", "time_unknown", "--out", str(rev)]) == 0
    out = tmp_path / "exp"
    assert main(["export", str(rev), "--out", str(out)]) == 0
    report = json.loads((out / "export_report.json").read_text())
    codes = {(s["artifact"], s["reason_code"]) for s in report["skipped"]}
    assert ("csv", "TIME_SCALE_UNKNOWN") in codes
    assert ("png", "TIME_SCALE_UNKNOWN") in codes
