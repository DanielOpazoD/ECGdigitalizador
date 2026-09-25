import numpy as np
import pytest

from ecg_photo.digitizers import (
    EngineSpec,
    WeightSpec,
    load_engine_specs,
    verify_weights,
)
from ecg_photo.digitizers.ahus import parse_ahus_output


def test_load_engine_specs() -> None:
    specs = load_engine_specs()
    ahus = specs["ahus"]
    assert ahus.commit == "97a15087d4abcda843da8c58ee74b1d8f47e6f9a"
    assert len(ahus.weights) == 2
    assert specs["ecg_image_kit"].weights == ()


def test_verify_weights(tmp_path) -> None:
    spec = EngineSpec(
        engine_id="x",
        repo="r",
        commit="c",
        license="l",
        weights=(
            WeightSpec(path="w1.pt", sha256="0" * 64),
            WeightSpec(path="w2.pt", sha256="1" * 64),
        ),
    )
    problems = verify_weights(tmp_path, spec)
    assert len(problems) == 2
    assert "missing" in problems[0]
    (tmp_path / "w1.pt").write_bytes(b"abc")
    problems = verify_weights(tmp_path, spec)
    assert len(problems) == 2
    assert "mismatch" in problems[0]


def test_parse_ahus_output(tmp_path) -> None:
    header = "I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6"
    lines = [header]
    lines.append(",".join(["1000.0"] + ["0.0"] * 10 + ["500.0"]))
    lines.append("," + ",".join(["0.0"] * 10 + ["500.0"]))
    lines.append(",".join(["-1000.0"] * 11 + ["0.0"]))
    (tmp_path / "img_timeseries_canonical.csv").write_text("\n".join(lines) + "\n")
    (tmp_path / "digitization_metadata.csv").write_text(
        "file_path,matching_cost,is_flipped,lead_layout\nimg,0.09,False,3x4+1R\n"
    )
    leads, observed, fs_hz, layout = parse_ahus_output(tmp_path, duration_s=3.0)
    assert fs_hz == pytest.approx(1.0)
    assert layout == "3x4+1R"
    assert leads["I"][0] == pytest.approx(1.0)
    assert leads["I"][2] == pytest.approx(-1.0)
    assert np.isnan(leads["I"][1])
    assert observed["I"].tolist() == [True, False, True]
    assert leads["V6"][0] == pytest.approx(0.5)
    with pytest.raises(FileNotFoundError):
        parse_ahus_output(tmp_path / "empty", 10.0)
