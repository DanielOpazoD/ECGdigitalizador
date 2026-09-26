from pathlib import Path

import numpy as np
import pytest

from ecg_photo.contracts import QualityLabel, dump_json
from ecg_photo.fixtures import write_fixture_revision
from ecg_photo.qc import identity_residual, run_qc
from ecg_photo.signal import grid_from_observed_duration

FS = 500.0
# GE MAC2000 / Kaggle 3x4 + II rhythm: slot start (s) of each short lead
SLOTS = {
    "I": 0.0,
    "III": 0.0,
    "aVR": 2.5,
    "aVL": 2.5,
    "aVF": 2.5,
    "V1": 5.0,
    "V2": 5.0,
    "V3": 5.0,
    "V4": 7.5,
    "V5": 7.5,
    "V6": 7.5,
}


def _limb_page(t: np.ndarray) -> dict[str, np.ndarray]:
    """A physically consistent 12-lead page (Einthoven/Goldberger hold)."""
    # one steep positive complex per 833 ms cycle (sin**15 alone also has an
    # equally steep negative lobe: two QRS-like complexes per cycle)
    beat = np.maximum(np.sin(2 * np.pi * 1.2 * t), 0.0) ** 15
    lead_i = 0.6 * beat + 0.1 * np.sin(2 * np.pi * 0.3 * t)
    lead_ii = 1.0 * beat + 0.2 * np.sin(2 * np.pi * 1.2 * t + 1.0)
    v = {f"V{k}": (0.3 + 0.2 * k) * np.roll(beat, 5 * k) for k in range(1, 7)}
    return {
        "I": lead_i,
        "II": lead_ii,
        "III": lead_ii - lead_i,
        "aVR": -(lead_i + lead_ii) / 2,
        "aVL": lead_i - lead_ii / 2,
        "aVF": lead_ii - lead_i / 2,
        **v,
    }


def _write_run(root: Path, page: dict[str, np.ndarray], drop: tuple[str, ...] = ()) -> Path:
    """Confirmed-run layout: every lead on a 10 s canvas, NaN outside its slot."""
    m = write_fixture_revision(root, "calibrated", fs_hz=FS)
    base = m.segments[0]
    grid = grid_from_observed_duration(10.0, FS)
    n = grid.n_samples
    segs = []
    for name, sig in page.items():
        if name in drop:
            continue
        canvas = np.full(n, np.nan)
        if name == "II":
            canvas[:] = sig[:n]
        else:
            a = int(SLOTS[name] * FS)
            canvas[a : a + int(2.5 * FS)] = sig[a : a + int(2.5 * FS)]
        obs = np.isfinite(canvas)
        sid = f"seg-{name}"
        np.save(root / "signals" / f"{sid}.npy", np.nan_to_num(canvas))
        np.save(root / "masks" / f"{sid}-observed.npy", obs)
        segs.append(
            base.model_copy(
                update={
                    "segment_id": sid,
                    "lead_label": name,
                    "signal_path": f"signals/{sid}.npy",
                    "observed_mask_path": f"masks/{sid}-observed.npy",
                    "n_samples": n,
                    "duration_s": grid.duration_s,
                    "observed_duration_s": grid.observed_duration_s,
                    "discarded_tail_s": grid.discarded_tail_s,
                }
            )
        )
    dump_json(m.model_copy(update={"segments": segs}), root / "manifest.json")
    return root


def _page() -> dict[str, np.ndarray]:
    return _limb_page(np.arange(int(10 * FS)) / FS)


def test_consistent_page_is_good(tmp_path: Path) -> None:
    rep = run_qc(_write_run(tmp_path, _page()))
    assert rep.label == QualityLabel.good
    assert rep.flags == []
    assert rep.einthoven_residual is not None and rep.einthoven_residual < 0.01
    assert rep.goldberger_residual is not None and rep.goldberger_residual < 0.01
    ii = next(q for q in rep.leads if q.lead == "II")
    assert ii.expected_s == 10.0 and ii.coverage == 1.0


def test_flat_and_missing_leads_are_insufficient(tmp_path: Path) -> None:
    # real case (MAC2000 photo at 48/min): aVL came back empty and V6 cut short
    page = _page()
    page["aVL"] = np.zeros_like(page["aVL"])
    page["V6"] = np.where(np.arange(len(page["V6"])) < int(9.2 * FS), page["V6"], np.nan)
    rep = run_qc(_write_run(tmp_path, page, drop=("V2",)))
    by = {q.lead: q for q in rep.leads}
    assert "FLAT_TRACE" in by["aVL"].flags
    assert "LOW_COVERAGE" in by["V6"].flags
    assert by["V2"].status == "missing"
    assert rep.label == QualityLabel.insufficient


def test_inverted_limb_lead_is_inconsistent(tmp_path: Path) -> None:
    # a trace digitized upside down breaks I + III = II
    page = _page()
    page["III"] = -page["III"]
    rep = run_qc(_write_run(tmp_path, page))
    assert rep.einthoven_residual is not None and rep.einthoven_residual > 1.0
    assert "LIMB_LEADS_INCONSISTENT" in rep.flags
    assert rep.label == QualityLabel.insufficient


def test_identity_residual_tolerates_small_row_offset() -> None:
    t = np.arange(int(2.5 * FS)) / FS
    a = np.sin(2 * np.pi * 1.1 * t) ** 9
    b = 0.5 * np.cos(2 * np.pi * 0.7 * t)
    target = np.roll(a + b, 8)  # 16 ms misalignment between printed rows
    r = identity_residual([a, b], target, FS)
    assert r is not None and r < 0.05
    assert identity_residual([a[:5], b[:5]], target[:5], FS) is None


def test_rr_compared_with_printed_values(tmp_path: Path) -> None:
    # synthetic beats every 1/1.2 s = 833.3 ms (72 bpm) on the II rhythm strip
    run = _write_run(tmp_path, _page())
    rep = run_qc(run, printed_hr_bpm=72)
    assert rep.n_beats is not None and rep.n_beats >= 10
    assert rep.rr_measured_ms is not None and abs(rep.rr_measured_ms - 833.3) < 5
    assert rep.rr_error_pct is not None and abs(rep.rr_error_pct) < 1
    assert "RR_MISMATCH_PRINTED" not in rep.flags and rep.label == QualityLabel.good
    # the time axis of a digitization 10 % off the printout is not usable
    bad = run_qc(run, printed_rr_ms=750)
    assert bad.rr_error_pct is not None and bad.rr_error_pct > 5
    assert "RR_MISMATCH_PRINTED" in bad.flags and bad.label == QualityLabel.insufficient
    assert run_qc(run).rr_error_pct is None  # nothing printed, nothing compared


def test_rr_not_measurable_without_rhythm_strip(tmp_path: Path) -> None:
    rep = run_qc(_write_run(tmp_path, _page(), drop=("II",)), printed_rr_ms=800)
    assert "RR_NOT_MEASURABLE" in rep.flags
    with pytest.raises(ValueError):
        run_qc(tmp_path, printed_hr_bpm=0)


def _strip(
    qrs_times: np.ndarray,
    *,
    t_amp: float = 0.35,
    p_times: np.ndarray | None = None,
    drift_mV: float = 0.0,
    qrs_amp: np.ndarray | None = None,
    t_delay: float = 0.25,
) -> np.ndarray:
    """10 s rhythm strip at FS: narrow QRS, broad T 0.25 s later, optional P
    waves and linear baseline drift (curved paper in a photo)."""
    t = np.arange(int(10 * FS)) / FS
    x = drift_mV * (0.5 - t / 10.0)
    amps = np.ones(len(qrs_times)) if qrs_amp is None else qrs_amp
    for tq, a in zip(qrs_times, amps, strict=True):
        x += a * (
            np.exp(-((t - tq) ** 2) / (2 * 0.012**2))
            - 0.25 * np.exp(-((t - tq - 0.03) ** 2) / (2 * 0.01**2))
        )
        x += t_amp * np.exp(-((t - tq - t_delay) ** 2) / (2 * 0.06**2))
    for tp in p_times if p_times is not None else qrs_times - 0.16:
        x += 0.15 * np.exp(-((t - tp) ** 2) / (2 * 0.025**2))
    return x


@pytest.mark.parametrize(
    "case,qrs,kw,printed",
    [
        # SVT at 207 bpm: the former 0.3 s refractory counted every other beat
        ("svt_rr290", np.arange(0.2, 9.9, 0.29), {}, 290.0),
        # sinus 77 bpm on curved paper: 0.9 mV drift and tall T waves
        ("drift_tall_t", np.arange(0.3, 9.9, 0.776), {"drift_mV": 0.9, "t_amp": 0.6}, 776.0),
        # 2:1 AV block: P every 0.696 s, QRS every 1.392 s, tall late T (QT 500 ms)
        (
            "avb_2to1",
            np.arange(0.5, 9.9, 1.392),
            {"p_times": np.arange(0.34, 9.9, 0.696), "t_amp": 0.55, "t_delay": 0.35},
            1392.0,
        ),
    ],
)
def test_rr_detector_real_photo_failures(tmp_path: Path, case, qrs, kw, printed) -> None:
    page = _page()
    page["II"] = _strip(qrs, **kw)
    rep = run_qc(_write_run(tmp_path / case, page), printed_rr_ms=printed)
    assert rep.n_beats == len(qrs), case
    assert rep.rr_error_pct is not None and abs(rep.rr_error_pct) < 2.0, case
    assert "RR_MISMATCH_PRINTED" not in rep.flags


def test_rr_irregular_rhythm_uses_mean_like_the_device(tmp_path: Path) -> None:
    # AF-like: irregular RR plus one large aberrant beat; the device prints the
    # mean RR, which the median misses by more than the 5 % tolerance here
    rr = np.array(
        [
            0.38,
            0.62,
            0.41,
            0.70,
            0.36,
            0.66,
            0.40,
            0.58,
            0.39,
            0.72,
            0.37,
            0.60,
            0.42,
            0.68,
            0.40,
            0.55,
        ]
    )
    qrs = 0.3 + np.concatenate([[0.0], np.cumsum(rr)])
    amp = np.ones(len(qrs))
    amp[4] = 3.0  # aberrant beat must not hide the normal ones
    page = _page()
    page["II"] = _strip(qrs, qrs_amp=amp)
    printed = float(np.mean(rr) * 1000)
    rep = run_qc(_write_run(tmp_path, page), printed_rr_ms=printed)
    assert rep.n_beats == len(qrs)
    assert rep.rr_error_pct is not None and abs(rep.rr_error_pct) < 1.0
    assert rep.rr_median_ms is not None and abs(rep.rr_median_ms / printed - 1) > 0.05


def test_rr_mean_ignores_missed_beat(tmp_path: Path) -> None:
    # a beat lost in the digitized trace (gap of ~2 RR) must not inflate the
    # mean RR: with the plain mean F7 Kaggle had 7/104 strips > 5 % off the truth
    qrs = np.delete(np.arange(0.3, 9.9, 0.8), 6)
    page = _page()
    page["II"] = _strip(qrs)
    rep = run_qc(_write_run(tmp_path, page), printed_rr_ms=800.0)
    assert rep.n_beats == len(qrs)
    assert rep.rr_error_pct is not None and abs(rep.rr_error_pct) < 1.0
    assert "RR_MISMATCH_PRINTED" not in rep.flags
