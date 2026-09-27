import json
import shutil
from pathlib import Path

import pytest
from test_pipeline import FakeDigitizer
from test_process import _engine_output, _page

from ecg_photo.cli import main
from ecg_photo.concordance import (
    read_printed_csv,
    run_concordance,
    summarize,
    write_outputs,
    write_template,
)
from ecg_photo.process import ProcessOptions

OPTS = ProcessOptions(speed_mm_s=25.0, gain_mm_mV=10.0, author="t", reason="r")


def _folder(tmp_path: Path) -> Path:
    folder = tmp_path / "fotos"
    folder.mkdir()
    page = _page(tmp_path)
    shutil.copy(page, folder / "ecg1.png")
    shutil.copy(page, folder / "ecg2.png")
    (folder / "roto.jpg").write_bytes(b"not an image")
    (folder / "notas.txt").write_text("ignored")
    return folder


def test_printed_csv_spanish_spreadsheet(tmp_path: Path) -> None:
    # Excel in Spanish: ';' separator and decimal comma; blank = not printed
    p = tmp_path / "v.csv"
    p.write_text("file;hr_bpm;pr_ms;qrs_ms;qt_ms;qtc_ms;axis_deg\necg1.png;75;160;92,5;;;270\n")
    v = read_printed_csv(p)["ecg1.png"]
    assert v.hr_bpm == 75 and v.qrs_ms == 92.5 and v.qt_ms is None and v.axis_deg == -90
    p.write_text("file,hr_bpm,pr_ms,qrs_ms,qt_ms,qtc_ms,axis_deg\necg1.png,900,,,,,\n")
    with pytest.raises(ValueError, match="row 2"):
        read_printed_csv(p)


def test_concordance_end_to_end_and_resume(tmp_path: Path) -> None:
    folder = _folder(tmp_path)
    template = tmp_path / "valores.csv"
    write_template(sorted(folder.glob("*.*")), template)
    assert template.read_text().splitlines()[0].startswith("file;hr_bpm")
    template.write_text(
        "file;hr_bpm;pr_ms;qrs_ms;qt_ms;qtc_ms;axis_deg\n"
        "ecg1.png;75;;;;;\n"
        "ecg2.png;90;;;;;\n"
        "roto.jpg;60;;;;;\n"
    )
    printed = read_printed_csv(template)
    out = tmp_path / "out"
    dig = FakeDigitizer(_engine_output(geometry=False))
    rows = run_concordance(folder, printed, out, dig, OPTS)
    by_file = {r.file: r for r in rows}
    assert by_file["ecg1.png"].status == "processed" and by_file["roto.jpg"].status == "rejected"
    assert "notas.txt" not in by_file
    s = summarize(rows)
    fc = s["measures"]["FC"]["all"]
    # the fake engine beats every 0.8 s (75/min): ecg1 agrees, ecg2 (90) does not
    assert fc["n"] == 2 and fc["within_tolerance"] == 0.5
    assert s["processed"] == 2 and "roto.jpg" in s["not_processed"]
    files = write_outputs(rows, s, out)
    assert Path(files["plot"]).stat().st_size > 0
    assert "Concordancia con lo impreso (2 de 3" in Path(files["summary"]).read_text()
    assert "ecg1.png;processed" in Path(files["table"]).read_text()

    # resumable: processed photos are read back, not digitized again
    class Boom:
        spec = dig.spec

        def run(self, *_a, **_k):
            raise AssertionError("must not run again")

    again = run_concordance(folder, printed, out, Boom(), OPTS)
    assert [r.status for r in again] == [r.status for r in rows]


def test_cli_concordance(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("ECG_PHOTO_ENABLE_FAKE_ENGINE", "1")
    folder = _folder(tmp_path)
    csv_path = tmp_path / "v.csv"
    base = ["concordance", str(folder), "--printed", str(csv_path)]
    assert main([*base, "--template"]) == 0
    assert "ecg2.png" in csv_path.read_text()
    capsys.readouterr()
    # scale is never assumed
    assert main([*base, "--engine", "fake"]) == 2
    assert "never assumed" in capsys.readouterr().out
    rc = main(
        [*base, "--engine", "fake", "--speed", "25", "--gain", "10", "--out", str(tmp_path / "o")]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["photos"] == 3


def test_ok_subset_uses_the_status_before_the_printed_comparison() -> None:
    # a QRS that is ok on its own but disagrees with the printed value was
    # demoted to doubtful by the comparison; the concordance must still count
    # it as ok, otherwise 'ok values within tolerance' is circular
    from ecg_photo.concordance import _row_from_summary
    from ecg_photo.printed import PrintedValues

    summary = {
        "qc": {"label": "good"},
        "intervals": {
            "qrs_ms": 140.0,
            "status": {"qrs_ms": "doubtful"},
            "printed": {"qrs_ms": {"printed": 90.0, "agrees": False, "status_before": "ok"}},
        },
    }
    row = _row_from_summary("a.png", summary, PrintedValues(qrs_ms=90))
    assert row.measure_status["QRS"] == "ok"
    s = summarize([row])
    assert s["measures"]["QRS"]["ok"]["n"] == 1
    assert s["measures"]["QRS"]["ok"]["within_tolerance"] == 0.0
