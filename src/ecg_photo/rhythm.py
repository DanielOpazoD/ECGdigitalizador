"""Irregular-rhythm alert from the 10 s rhythm strip (F14).

irregularity = median |successive RR difference| / median RR. Medians on
purpose: an isolated premature beat makes 2 large differences out of ~10 and
barely moves it; in atrial fibrillation most differences are large. The
alert is raised above IRREGULARITY_THRESHOLD (95th percentile of regular
sinus records, PTB-XL even ids); "no clear P wave" is added when fewer than
P_LEADS_MIN printed leads show one. It is a prompt to look at the tracing,
never a diagnosis.
"""

import numpy as np

from ecg_photo.qc import detect_qrs, observed_span

MIN_BEATS = 5
# 95th percentile of regular sinus records (PTB-XL, even ecg_ids): about 5 %
# of sinus strips are flagged; F14 reports the odd ids
IRREGULARITY_THRESHOLD = 0.067
# among irregular strips, AF shows a P wave in fewer printed leads (median 5)
# than other irregular rhythms (median 8); < 6 separates them best on the
# even ids (a modest discriminator: "consistent with", never a diagnosis)
P_LEADS_MIN = 6


def rhythm_features(x: np.ndarray, fs: float, p_leads: int = 0) -> dict | None:
    """Irregularity of a rhythm strip; None with fewer than MIN_BEATS beats."""
    span, _ = observed_span(x, fs)
    beats = detect_qrs(span, fs) if len(span) else np.array([], int)
    if len(beats) < MIN_BEATS:
        return None
    rr = np.diff(beats) / fs * 1000.0
    d = np.abs(np.diff(rr))
    return {
        "n_beats": len(beats),
        "rr_median_ms": float(np.median(rr)),
        "irregularity": float(np.median(d) / np.median(rr)),
        "rr_cv": float(np.std(rr) / np.mean(rr)),
        "p_leads": int(p_leads),
    }


def rhythm_alert(features: dict | None) -> dict:
    """{'status': 'regular' | 'irregular' | 'unavailable', 'message': ...}."""
    if features is None:
        return {"status": "unavailable", "message": f"fewer than {MIN_BEATS} beats in the strip"}
    irr = features["irregularity"]
    if irr <= IRREGULARITY_THRESHOLD:
        return {"status": "regular", "irregularity": round(irr, 3)}
    no_p = features.get("p_leads", 0) < P_LEADS_MIN
    return {
        "status": "irregular",
        "irregularity": round(irr, 3),
        "no_clear_p_wave": no_p,
        "message": "irregular RR"
        + (" without a clear P wave: consistent with atrial fibrillation" if no_p else "")
        + "; look at the tracing (not a diagnosis)",
    }
