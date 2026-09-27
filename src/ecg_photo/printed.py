"""Values printed by the electrocardiograph on the sheet header (heart rate
or RR, PR, QRS, QT, QTc, QRS axis), recorded for a confirmed run.

`apply_printed` stores them in `printed.json` (with author and time) and
rewrites the advisory files derived from the run with them: the quality
report (RR against the printed RR/HR), the intervals report (each interval
against its printed value, F11/F12 tolerances) and the PDF report. The
signals and the manifest are not touched; the same inputs always give the
same files.
"""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ecg_photo.intervals import write_intervals_report
from ecg_photo.qc import QC_FILENAME, qc_json, run_qc
from ecg_photo.report import write_report
from ecg_photo.store_models import utc_now_iso, write_json_atomic

PRINTED_FILENAME = "printed.json"


class PrintedValues(BaseModel):
    """Header values as printed; any subset (None = not printed / not read)."""

    model_config = ConfigDict(extra="forbid")

    hr_bpm: float | None = None
    rr_ms: float | None = None
    pr_ms: float | None = None
    qrs_ms: float | None = None
    qt_ms: float | None = None
    qtc_ms: float | None = None
    axis_deg: float | None = None

    @field_validator("hr_bpm")
    @classmethod
    def _hr(cls, v: float | None) -> float | None:
        if v is not None and not 15.0 <= v <= 350.0:
            raise ValueError("hr_bpm must be within 15-350")
        return v

    @field_validator("rr_ms", "pr_ms", "qrs_ms", "qt_ms", "qtc_ms")
    @classmethod
    def _ms(cls, v: float | None) -> float | None:
        if v is not None and not 10.0 <= v <= 4000.0:
            raise ValueError("intervals must be within 10-4000 ms")
        return v

    @field_validator("axis_deg")
    @classmethod
    def _axis(cls, v: float | None) -> float | None:
        # electrocardiographs print -180..180 or 0..360 (e.g. 270 = -90)
        if v is None:
            return v
        if not -180.0 <= v <= 360.0:
            raise ValueError("axis_deg must be within -180..360")
        return (v + 180.0) % 360.0 - 180.0

    @model_validator(mode="after")
    def _one_rate(self) -> "PrintedValues":
        if self.hr_bpm is not None and self.rr_ms is not None:
            raise ValueError("give either hr_bpm or rr_ms, not both")
        return self

    def interval_keys(self) -> dict[str, float | None]:
        """Keys of intervals.compare_printed."""
        return {
            "pr_ms": self.pr_ms,
            "qrs_ms": self.qrs_ms,
            "qt_ms": self.qt_ms,
            "qtc_bazett_ms": self.qtc_ms,
            "qrs_axis_deg": self.axis_deg,
        }


def apply_printed(
    run_dir: Path, values: PrintedValues, *, author: str, source_name: str | None = None
) -> dict:
    """Record the printed values of a confirmed run and rewrite qc.json,
    intervals.json and report.pdf with them. Raises FileNotFoundError when the
    run has no calibrated result (no qc/intervals to compare with)."""
    run_dir = Path(run_dir)
    if not (run_dir / "manifest.json").exists():
        raise FileNotFoundError("run has no manifest")
    qc = run_qc(run_dir, printed_rr_ms=values.rr_ms, printed_hr_bpm=values.hr_bpm)
    record = {
        "values": values.model_dump(),
        "author": author,
        "recorded_at": utc_now_iso(),
    }
    write_json_atomic(run_dir / PRINTED_FILENAME, record)
    (run_dir / QC_FILENAME).write_text(qc_json(qc) + "\n", encoding="utf-8")
    intervals = write_intervals_report(run_dir, printed=values.interval_keys())
    report = write_report(run_dir, source_name=source_name)
    return {
        "printed": record,
        "qc": json.loads(qc_json(qc)),
        "intervals": intervals,
        "report": report,
    }


def read_printed(run_dir: Path) -> dict | None:
    p = Path(run_dir) / PRINTED_FILENAME
    if not p.exists():
        return None
    result: dict = json.loads(p.read_text(encoding="utf-8"))
    return result
