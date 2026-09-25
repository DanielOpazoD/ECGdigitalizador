import json
from pathlib import Path

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


def test_cli_time_unknown_export(tmp_path) -> None:
    rev = tmp_path / "rev"
    assert main(["fixture", "--kind", "time_unknown", "--out", str(rev)]) == 0
    out = tmp_path / "exp"
    assert main(["export", str(rev), "--out", str(out)]) == 0
    report = json.loads((out / "export_report.json").read_text())
    codes = {(s["artifact"], s["reason_code"]) for s in report["skipped"]}
    assert ("csv", "TIME_SCALE_UNKNOWN") in codes
    assert ("png", "TIME_SCALE_UNKNOWN") in codes
