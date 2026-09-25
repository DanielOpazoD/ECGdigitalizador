import json

import numpy as np
import pytest
from pydantic import ValidationError

from ecg_photo.contracts import (
    Manifest,
    Measurement,
    MeasurementStatus,
    QualityLabel,
    ReasonCode,
    Segment,
    dump_json,
    load_manifest,
    to_jsonable,
    validate_revision_dir,
)
from ecg_photo.fixtures import write_fixture_revision

SPEC_EXAMPLE = json.loads("""{
  "schema_version": "1.0.0",
  "study_id": "example-study",
  "revision": 1,
  "input_revision": 1,
  "run_id": "example-run",
  "config_hash": null,
  "source": {
    "sha256": null,
    "hash_status": "example_only",
    "kind": "image",
    "page_count": 1
  },
  "segments": [
    {
      "segment_id": "seg-I-01",
      "page_id": "page-1",
      "lead_label": "I",
      "lead_status": "confirmed",
      "lead_evidence": "manual_example",
      "representation_kind": "rhythm_segment",
      "representation_evidence": "manual_example",
      "source_region": [[100, 200], [725, 200], [725, 500], [100, 500]],
      "transform_id": "transform-example",
      "raw_path": "raw_paths/seg-I-01.npy",
      "raw_coordinate_frame": "oriented-page-1",
      "raw_support_path": "raw_paths/seg-I-01-support.npz",
      "time_relation": "unknown",
      "sync_group_id": null,
      "study_start_s": null,
      "working_fs_hz": 500,
      "original_fs_hz": null,
      "n_samples": 1250,
      "duration_s": 2.5,
      "observed_duration_s": 2.5,
      "discarded_tail_s": 0,
      "units": "mV",
      "speed_mm_s": 25,
      "speed_status": "confirmed",
      "gain_mm_mV": 10,
      "gain_status": "confirmed",
      "effective_dt_ms": 4,
      "effective_dv_mV": 0.01,
      "resolution_evidence": {
        "source_frame_id": "native-page-1",
        "method": "illustrative_identity_mapping",
        "limitations": "geometric_pixel_equivalence_only"
      },
      "signal_path": "signals/seg-I-01.npy",
      "temporal_trace_path": null,
      "temporal_trace_units": null,
      "observed_mask_path": "masks/seg-I-01-observed.npy",
      "valid_mask_path": "masks/seg-I-01-valid.npy",
      "gap_fill_mask_path": "masks/seg-I-01-gap-fill.npy",
      "processing": [{"stage": "illustrative_fixture", "implementation": "example_only"}]
    }
  ]
}""")


def test_T38_json_nan_and_inf(tmp_path) -> None:
    data = {"a": [1.0, np.nan], "b": np.float64(2.5), "c": np.array([1, 2])}
    out = tmp_path / "d.json"
    dump_json(data, out)
    text = out.read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == {"a": [1.0, None], "b": 2.5, "c": [1, 2]}
    with pytest.raises(ValueError):
        to_jsonable(float("inf"))
    with pytest.raises(ValueError):
        dump_json({"x": float("inf")}, tmp_path / "bad.json")


def test_spec_example_parses_but_strict_rejects(tmp_path) -> None:
    m = Manifest.model_validate(SPEC_EXAMPLE)
    problems = validate_revision_dir(tmp_path, m, strict=True)
    assert any("hash_status" in p or "sha256" in p for p in problems)
    assert problems  # files also missing


def test_relpath_rejected(tmp_path) -> None:
    m = write_fixture_revision(tmp_path, "calibrated")
    seg = m.segments[0].model_dump()
    for bad in ("../x.npy", "/abs/x.npy", "a\\b.npy"):
        seg["raw_path"] = bad
        with pytest.raises(ValidationError):
            Segment.model_validate(seg)
    seg["raw_path"] = ""
    with pytest.raises(ValidationError):
        Segment.model_validate(seg)


def test_extra_fields_rejected(tmp_path) -> None:
    m = write_fixture_revision(tmp_path, "calibrated")
    seg = m.segments[0].model_dump()
    seg["bogus"] = 1
    with pytest.raises(ValidationError):
        Segment.model_validate(seg)
    man = m.model_dump()
    man["bogus"] = 1
    with pytest.raises(ValidationError):
        Manifest.model_validate(man)


def test_time_unknown_with_n_samples_fails(tmp_path) -> None:
    m = write_fixture_revision(tmp_path, "time_unknown")
    seg = m.segments[0].model_dump()
    seg["n_samples"] = 10
    with pytest.raises(ValidationError):
        Segment.model_validate(seg)


def test_signal_path_with_gain_unknown_fails(tmp_path) -> None:
    m = write_fixture_revision(tmp_path, "gain_unknown")
    seg = m.segments[0].model_dump()
    seg["signal_path"] = "signals/x.npy"
    seg["units"] = "mV"
    with pytest.raises(ValidationError):
        Segment.model_validate(seg)


def test_units_mV_without_signal_path_fails(tmp_path) -> None:
    m = write_fixture_revision(tmp_path, "gain_unknown")
    seg = m.segments[0].model_dump()
    seg["units"] = "mV"
    with pytest.raises(ValidationError):
        Segment.model_validate(seg)


def test_duration_fs_mismatch_fails(tmp_path) -> None:
    m = write_fixture_revision(tmp_path, "calibrated")
    seg = m.segments[0].model_dump()
    seg["duration_s"] = 1.234
    with pytest.raises(ValidationError):
        Segment.model_validate(seg)
    seg["duration_s"] = m.segments[0].duration_s
    seg["discarded_tail_s"] = 0.01
    with pytest.raises(ValidationError):
        Segment.model_validate(seg)


def test_sync_group_requires_simultaneous(tmp_path) -> None:
    m = write_fixture_revision(tmp_path, "calibrated")
    man = m.model_dump()
    s2 = dict(man["segments"][0])
    s2["segment_id"] = "seg-II-02"
    man["segments"][0]["sync_group_id"] = "g1"
    s2["sync_group_id"] = "g1"
    man["segments"].append(s2)
    with pytest.raises(ValidationError):
        Manifest.model_validate(man)
    for s in man["segments"]:
        s["time_relation"] = "simultaneous"
    Manifest.model_validate(man)


def test_measurement_status_value_rules() -> None:
    base: dict = {
        "measurement_id": "m1",
        "revision": 1,
        "name": "QT",
        "unit": "ms",
        "method": "m",
        "support": {},
        "quality_label": QualityLabel.insufficient,
    }
    with pytest.raises(ValidationError):
        Measurement(value=None, status=MeasurementStatus.available, reason_codes=[], **base)
    with pytest.raises(ValidationError):
        Measurement(
            value=1.0,
            status=MeasurementStatus.unavailable,
            reason_codes=[ReasonCode.T_END_UNCERTAIN],
            **base,
        )
    with pytest.raises(ValidationError):
        Measurement(value=None, status=MeasurementStatus.unavailable, reason_codes=[], **base)
    Measurement(
        value=None,
        status=MeasurementStatus.unavailable,
        reason_codes=[ReasonCode.T_END_UNCERTAIN],
        **base,
    )
    Measurement(value=400.0, status=MeasurementStatus.available, reason_codes=[], **base)


def test_load_manifest_roundtrip(tmp_path) -> None:
    write_fixture_revision(tmp_path, "calibrated")
    m = load_manifest(tmp_path / "manifest.json")
    assert m.segments[0].segment_id == "seg-II-01"
    assert validate_revision_dir(tmp_path, m, strict=True) == []
