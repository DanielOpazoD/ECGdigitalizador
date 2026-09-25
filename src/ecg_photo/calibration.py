"""Evidence-based calibration resolution.

Manual evidence (with author+reason) wins -> status manual/known; if it
contradicts another evidence by more than tolerance it still wins but records
CALIBRATION_CONFLICT. Without manual: all non-manual evidence must agree within
tolerance -> known (median); disagreement -> review_required + conflict; none
-> unknown + CALIBRATION_MISSING. Grid periods can only supply px_per_mm_*;
speed/gain require declared_text, calibration_pulse or manual.
"""

from dataclasses import dataclass

import numpy as np

from ecg_photo.contracts import CalibrationEvidence, ReasonCode, ScaleStatus

GRID_QUANTITIES = {"px_per_mm_x", "px_per_mm_y"}
NONGRID_ALLOWED = {"declared_text", "calibration_pulse", "manual"}


@dataclass(frozen=True)
class Resolved:
    value: float | None
    status: ScaleStatus
    reason_codes: list[ReasonCode]
    used_evidence_ids: list[str]


def _eligible(ev: CalibrationEvidence, quantity: str) -> bool:
    if ev.quantity != quantity:
        return False
    if quantity in GRID_QUANTITIES:
        return ev.kind in ("grid_period", "manual")
    return ev.kind in NONGRID_ALLOWED


def resolve_scale(
    quantity: str,
    evidence: list[CalibrationEvidence],
    *,
    tolerance_rel: float = 0.02,
) -> Resolved:
    evs = [ev for ev in evidence if _eligible(ev, quantity) and ev.value is not None]
    if not evs:
        return Resolved(
            value=None,
            status=ScaleStatus.unknown,
            reason_codes=[ReasonCode.CALIBRATION_MISSING],
            used_evidence_ids=[],
        )

    manuals = [ev for ev in evs if ev.kind == "manual" and ev.author and ev.reason]
    if manuals:
        chosen = manuals[-1]
        conflict = any(
            ev is not chosen
            and ev.value is not None
            and chosen.value is not None
            and abs(ev.value - chosen.value) > tolerance_rel * abs(chosen.value)
            for ev in evs
        )
        return Resolved(
            value=chosen.value,
            status=ScaleStatus.manual,
            reason_codes=[ReasonCode.CALIBRATION_CONFLICT] if conflict else [],
            used_evidence_ids=[chosen.evidence_id],
        )

    values = [float(ev.value) for ev in evs if ev.value is not None]
    med = float(np.median(values))
    agree = all(abs(v - med) <= tolerance_rel * abs(med) for v in values)
    if agree:
        return Resolved(
            value=med,
            status=ScaleStatus.confirmed,
            reason_codes=[],
            used_evidence_ids=[ev.evidence_id for ev in evs],
        )
    return Resolved(
        value=None,
        status=ScaleStatus.review_required,
        reason_codes=[ReasonCode.CALIBRATION_CONFLICT],
        used_evidence_ids=[ev.evidence_id for ev in evs],
    )
