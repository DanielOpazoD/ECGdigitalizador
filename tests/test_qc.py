from pathlib import Path

import numpy as np

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
    beat = np.sin(2 * np.pi * 1.2 * t) ** 15
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
