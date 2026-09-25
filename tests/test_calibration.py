import pytest

from ecg_photo.calibration import resolve_scale
from ecg_photo.contracts import CalibrationEvidence, ReasonCode, ScaleStatus
from ecg_photo.fixtures import write_fixture_revision


def ev(
    eid: str,
    kind: str,
    quantity: str,
    value: float,
    author: str | None = None,
    reason: str | None = None,
) -> CalibrationEvidence:
    return CalibrationEvidence(
        evidence_id=eid,
        kind=kind,  # type: ignore[arg-type]
        quantity=quantity,  # type: ignore[arg-type]
        value=value,
        unit="mm/s" if quantity == "speed_mm_s" else "px/mm",
        frame_id="rect",
        method="test",
        limitations="none",
        author=author,
        reason=reason,
    )


def test_no_evidence_unknown() -> None:
    r = resolve_scale("speed_mm_s", [])
    assert r.status == ScaleStatus.unknown
    assert r.value is None
    assert r.reason_codes == [ReasonCode.CALIBRATION_MISSING]


def test_agreeing_evidence_known() -> None:
    evs = [
        ev("a", "declared_text", "speed_mm_s", 25.0),
        ev("b", "calibration_pulse", "speed_mm_s", 25.1),
    ]
    r = resolve_scale("speed_mm_s", evs)
    assert r.status == ScaleStatus.confirmed
    assert r.value == pytest.approx(25.05)
    assert not r.reason_codes
    assert set(r.used_evidence_ids) == {"a", "b"}


def test_conflicting_evidence_review() -> None:
    evs = [
        ev("a", "declared_text", "speed_mm_s", 25.0),
        ev("b", "calibration_pulse", "speed_mm_s", 50.0),
    ]
    r = resolve_scale("speed_mm_s", evs)
    assert r.status == ScaleStatus.review_required
    assert r.value is None
    assert r.reason_codes == [ReasonCode.CALIBRATION_CONFLICT]


def test_manual_wins_and_conflict_flagged() -> None:
    evs = [
        ev("a", "declared_text", "speed_mm_s", 25.0),
        ev("m", "manual", "speed_mm_s", 50.0, author="dr", reason="label says 50"),
    ]
    r = resolve_scale("speed_mm_s", evs)
    assert r.status == ScaleStatus.manual
    assert r.value == 50.0
    assert r.reason_codes == [ReasonCode.CALIBRATION_CONFLICT]


def test_manual_agreeing_no_conflict() -> None:
    evs = [
        ev("a", "declared_text", "speed_mm_s", 25.0),
        ev("m", "manual", "speed_mm_s", 25.1, author="dr", reason="check"),
    ]
    r = resolve_scale("speed_mm_s", evs)
    assert r.status == ScaleStatus.manual
    assert r.reason_codes == []


def test_manual_without_author_not_privileged() -> None:
    evs = [
        ev("a", "declared_text", "speed_mm_s", 25.0),
        ev("m", "manual", "speed_mm_s", 50.0),  # missing author/reason
    ]
    r = resolve_scale("speed_mm_s", evs)
    assert r.status == ScaleStatus.review_required


def test_grid_kind_cannot_supply_speed() -> None:
    evs = [ev("g", "grid_period", "speed_mm_s", 25.0)]
    r = resolve_scale("speed_mm_s", evs)
    assert r.status == ScaleStatus.unknown
    assert r.reason_codes == [ReasonCode.CALIBRATION_MISSING]


def test_grid_kind_supplies_px_per_mm() -> None:
    evs = [ev("g", "grid_period", "px_per_mm_x", 7.87)]
    r = resolve_scale("px_per_mm_x", evs)
    assert r.status == ScaleStatus.confirmed
    assert r.value == pytest.approx(7.87)


def test_segment_roundtrip(tmp_path) -> None:
    m = write_fixture_revision(tmp_path / "rev", "calibrated")
    seg = m.segments[0]
    evs = [
        ev("t", "declared_text", "speed_mm_s", 25.0),
        ev("m", "manual", "speed_mm_s", 50.0, author="dr", reason="fixed"),
    ]
    seg2 = seg.model_copy(update={"calibration_evidence": evs})
    r = resolve_scale("speed_mm_s", seg2.calibration_evidence)
    assert r.status == ScaleStatus.manual
    assert r.value == 50.0
    assert r.reason_codes == [ReasonCode.CALIBRATION_CONFLICT]
