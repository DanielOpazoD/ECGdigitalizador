import numpy as np
import pytest

from ecg_photo.contracts import TimeRelation, load_manifest
from ecg_photo.export import (
    ExportNotAllowed,
    export_csv,
    export_wfdb,
    readback_csv,
    readback_wfdb,
)
from ecg_photo.fixtures import write_fixture_revision


def test_T37_wfdb_roundtrip(tmp_path) -> None:
    root = tmp_path / "rev"
    m = write_fixture_revision(root, "calibrated")
    seg = m.segments[0]
    out = tmp_path / "wfdb"
    rec_base = export_wfdb(root, m, [seg.segment_id], out, "r1")
    sig, fs, names, units = readback_wfdb(rec_base)
    assert fs == 500.0
    assert names == ["II"]
    assert units == ["mV"]
    saved = np.load(root / seg.signal_path)
    assert sig.shape == (1500, 1)
    np.testing.assert_allclose(sig[:, 0], saved, atol=1e-3)


def test_T37_wfdb_gap_nans_preserved(tmp_path) -> None:
    root = tmp_path / "rev"
    m = write_fixture_revision(root, "gap")
    seg = m.segments[0]
    rec_base = export_wfdb(root, m, [seg.segment_id], tmp_path / "wfdb", "g1")
    sig, _, _, _ = readback_wfdb(rec_base)
    obs = np.load(root / seg.observed_mask_path)
    assert np.isnan(sig[~obs, 0]).all()
    assert np.isfinite(sig[obs, 0]).all()


def test_T37_wfdb_rejected(tmp_path) -> None:
    for kind in ("gain_unknown", "time_unknown"):
        root = tmp_path / kind
        m = write_fixture_revision(root, kind)
        with pytest.raises(ExportNotAllowed):
            export_wfdb(root, m, [m.segments[0].segment_id], tmp_path / "w", "x")


def test_T37_wfdb_two_segments_not_simultaneous(tmp_path) -> None:
    root = tmp_path / "rev"
    m = write_fixture_revision(root, "calibrated")
    seg2 = m.segments[0].model_copy(update={"segment_id": "seg-III-02", "lead_label": "III"})
    m2 = m.model_copy(update={"segments": [m.segments[0], seg2]})
    with pytest.raises(ExportNotAllowed):
        export_wfdb(root, m2, ["seg-II-01", "seg-III-02"], tmp_path / "w", "x")
    seg2s = seg2.model_copy(
        update={"time_relation": TimeRelation.simultaneous, "sync_group_id": "g1"}
    )
    seg1s = m.segments[0].model_copy(
        update={"time_relation": TimeRelation.simultaneous, "sync_group_id": "g1"}
    )
    m3 = m.model_copy(update={"segments": [seg1s, seg2s]})
    rec_base = export_wfdb(root, m3, ["seg-II-01", "seg-III-02"], tmp_path / "w2", "x2")
    sig, _, names, _ = readback_wfdb(rec_base)
    assert sig.shape == (1500, 2)
    assert names == ["II", "III"]


def test_csv_roundtrip_and_T38(tmp_path) -> None:
    root = tmp_path / "rev"
    m = write_fixture_revision(root, "gap")
    seg = m.segments[0]
    csv_path = tmp_path / "s.csv"
    export_csv(root, seg, csv_path)
    text = csv_path.read_text()
    assert "nan" not in text.lower()
    assert text.startswith("# segment_id=seg-II-01\n")
    assert "sample_index,t_s,value,units,observed,valid,gap_fill\n" in text
    cols = readback_csv(csv_path)
    saved = np.load(root / seg.signal_path)
    obs = np.load(root / seg.observed_mask_path)
    np.testing.assert_allclose(cols["value"][obs], saved[obs], atol=1e-9)
    assert np.isnan(cols["value"][~obs]).all()
    np.testing.assert_array_equal(cols["observed"], obs)
    np.testing.assert_array_equal(cols["gap_fill"], np.zeros(seg.n_samples, dtype=bool))
    assert (cols["units"] == "mV").all()
    assert cols["t_s"][1] == pytest.approx(1.0 / seg.working_fs_hz)


def test_csv_px_units_for_gain_unknown(tmp_path) -> None:
    root = tmp_path / "rev"
    m = write_fixture_revision(root, "gain_unknown")
    seg = m.segments[0]
    csv_path = tmp_path / "s.csv"
    export_csv(root, seg, csv_path)
    text = csv_path.read_text()
    assert "# gain_mm_mV=null" in text
    cols = readback_csv(csv_path)
    assert (cols["units"] == "px").all()


def test_csv_time_unknown_rejected(tmp_path) -> None:
    root = tmp_path / "rev"
    m = write_fixture_revision(root, "time_unknown")
    with pytest.raises(ExportNotAllowed):
        export_csv(root, m.segments[0], tmp_path / "s.csv")


def test_export_loads_manifest(tmp_path) -> None:
    root = tmp_path / "rev"
    write_fixture_revision(root, "calibrated")
    m = load_manifest(root / "manifest.json")
    assert m.segments[0].n_samples == 1500
