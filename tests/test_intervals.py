import json
from pathlib import Path

import numpy as np
import pytest

from ecg_photo.intervals import (
    INTERVALS_FILENAME,
    compare_printed,
    demote_on_qc,
    measure_signals,
    read_intervals_report,
    write_intervals_report,
)
from ecg_photo.qc import STANDARD_LEADS

FS = 500.0
RR = 0.8
# synthetic beat timing (s, relative to QRS onset): P 0.16 s before, QRS
# 0.09 s wide, T ends 0.38 s after QRS onset -> PR 160, QRS 90, QT 380 ms
P_ON, QRS_W, T_END = -0.16, 0.09, 0.38


def _beat_train(n_s: float, amp: float = 1.0, p_wave: bool = True) -> np.ndarray:
    t = np.arange(round(n_s * FS)) / FS
    x = np.zeros_like(t)
    for q0 in np.arange(0.3, n_s - 0.5, RR):
        # QRS: triangle Q-R-S over [q0, q0 + QRS_W]
        u = (t - q0) / QRS_W
        x += amp * np.where(
            (u >= 0) & (u <= 1), np.interp(u, [0, 0.2, 0.5, 0.8, 1], [0, -0.1, 1, -0.2, 0]), 0
        )
        # T: half-sine from q0 + 0.2 to q0 + T_END
        ut = (t - (q0 + 0.2)) / (T_END - 0.2)
        x += 0.3 * amp * np.where((ut >= 0) & (ut <= 1), np.sin(np.pi * ut), 0)
        if p_wave:
            up = (t - (q0 + P_ON)) / 0.1
            x += 0.15 * np.where((up >= 0) & (up <= 1), np.sin(np.pi * up), 0)
    return x


def _page(p_wave: bool = True) -> dict[str, tuple[np.ndarray, float]]:
    rng = np.random.default_rng(0)
    return {
        n: (_beat_train(10.0, 0.6 + 0.1 * k, p_wave) + 0.005 * rng.standard_normal(5000), FS)
        for k, n in enumerate(STANDARD_LEADS)
    }


def test_intervals_of_known_beats() -> None:
    iv = measure_signals(_page(), rr_ms=RR * 1000)
    assert iv.pr_ms == pytest.approx(160, abs=15)
    assert iv.qrs_ms == pytest.approx(90, abs=12)
    assert iv.qt_ms == pytest.approx(380, abs=25)
    assert iv.status == {"pr_ms": "ok", "qrs_ms": "ok", "qt_ms": "ok"}
    assert iv.qtc_bazett_ms == pytest.approx(iv.qt_ms / RR**0.5)
    assert iv.hr_bpm == pytest.approx(75)


def test_no_p_wave_means_no_pr() -> None:
    # atrial fibrillation: no P wave -> PR unavailable, never a default
    iv = measure_signals(_page(p_wave=False))
    assert iv.pr_ms is None and iv.status["pr_ms"] == "unavailable"
    assert iv.qrs_ms is not None and iv.qtc_bazett_ms is None
    assert iv.reasons["qtc"] == "RR not measured"


def test_few_leads_are_doubtful() -> None:
    page = _page()
    iv = measure_signals({k: page[k] for k in ("I", "II")})
    assert iv.status["qrs_ms"] == "doubtful" and "2 lead" in iv.reasons["qrs_ms"]


def test_short_printed_leads(tmp_path: Path) -> None:
    # 3x4 printout: 2.5 s per lead (II 10 s) still gives the intervals
    page = {n: (x[:1250] if n != "II" else x, fs) for n, (x, fs) in _page().items()}
    iv = measure_signals(page)
    assert iv.qrs_ms == pytest.approx(90, abs=12) and iv.n_leads["qrs_ms"] >= 10


def test_insufficient_quality_demotes_ok() -> None:
    iv = demote_on_qc(measure_signals(_page(), rr_ms=RR * 1000), "insufficient")
    assert set(iv.status.values()) == {"doubtful"}
    assert iv.reasons["qt_ms"] == "digitization quality insufficient"
    assert iv.qt_ms is not None  # the value stays visible, only its status changes
    good = demote_on_qc(measure_signals(_page()), "good")
    assert set(good.status.values()) == {"ok"}


def test_compare_with_printed_values() -> None:
    iv = measure_signals(_page(), rr_ms=RR * 1000)
    assert iv.qrs_ms is not None and iv.qt_ms is not None
    printed = {"qrs_ms": iv.qrs_ms, "qt_ms": iv.qt_ms + 200.0, "pr_ms": None}
    iv = compare_printed(iv, printed)
    assert iv.printed["qrs_ms"]["agrees"] is True and iv.status["qrs_ms"] == "ok"
    # far from the printed QT: the doubt is on the measurement, the value stays
    assert iv.printed["qt_ms"]["agrees"] is False and iv.printed["qt_ms"]["diff_ms"] < -150
    assert iv.status["qt_ms"] == "doubtful" and "printed qt_ms" in iv.reasons["qt_ms"]
    assert "pr_ms" not in iv.printed  # not printed -> not compared
    with pytest.raises(ValueError):
        compare_printed(measure_signals(_page()), {"qt_ms": -1.0})


def test_report_never_raises(tmp_path: Path) -> None:
    d = write_intervals_report(tmp_path)  # no manifest
    assert "error" in d and json.loads((tmp_path / INTERVALS_FILENAME).read_text())["error"]
    assert read_intervals_report(tmp_path) == d
    assert read_intervals_report(tmp_path / "none") is None
