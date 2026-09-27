import numpy as np

from ecg_photo.intervals import measure_signals
from ecg_photo.report import rhythm_row
from ecg_photo.rhythm import IRREGULARITY_THRESHOLD, rhythm_alert, rhythm_features

FS = 500.0


def _strip(beat_times: np.ndarray, n_s: float = 10.0) -> np.ndarray:
    t = np.arange(round(n_s * FS)) / FS
    x = np.zeros_like(t)
    for b in beat_times:
        x += np.exp(-((t - b) ** 2) / (2 * 0.012**2))
    return x + 0.005 * np.random.default_rng(0).standard_normal(len(t))


def test_regular_strip() -> None:
    f = rhythm_features(_strip(np.arange(0.3, 9.8, 0.8)), FS, p_leads=10)
    assert f is not None and f["irregularity"] < IRREGULARITY_THRESHOLD / 2
    assert rhythm_alert(f)["status"] == "regular"


def test_isolated_premature_beat_stays_regular() -> None:
    # one early beat (and its compensatory pause) must not raise the alert
    beats = list(np.arange(0.3, 9.8, 0.8))
    beats[6] -= 0.3
    f = rhythm_features(_strip(np.array(beats)), FS, p_leads=10)
    assert rhythm_alert(f)["status"] == "regular"


def test_irregular_without_p_is_flagged_as_consistent_with_af() -> None:
    rr = np.random.default_rng(3).uniform(0.45, 1.1, 14)
    beats = 0.3 + np.cumsum(np.concatenate([[0.0], rr]))
    beats = beats[beats < 9.8]
    a = rhythm_alert(rhythm_features(_strip(beats), FS, p_leads=2))
    assert a["status"] == "irregular" and a["no_clear_p_wave"] is True
    assert "not a diagnosis" in a["message"]
    assert rhythm_row(a)[2].startswith("dudoso: RR irregular sin onda P clara")
    b = rhythm_alert(rhythm_features(_strip(beats), FS, p_leads=10))
    assert b["status"] == "irregular" and b["no_clear_p_wave"] is False


def test_too_short_is_unavailable_and_measure_reports_it() -> None:
    assert rhythm_alert(rhythm_features(_strip(np.array([0.3, 1.1, 1.9])), FS)) == {
        "status": "unavailable",
        "message": "fewer than 5 beats in the strip",
    }
    iv = measure_signals({"II": (_strip(np.arange(0.3, 9.8, 0.8)), FS)})
    assert iv.rhythm["lead"] == "II" and iv.rhythm["status"] == "regular"
    assert rhythm_row({})[1] == "—"
