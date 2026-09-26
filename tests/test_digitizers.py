from pathlib import Path

import numpy as np
import pytest
import wfdb

from ecg_photo.digitizers import (
    EngineNotReady,
    EngineSpec,
    WeightSpec,
    load_engine_specs,
    verify_weights,
)
from ecg_photo.digitizers.ahus import parse_ahus_output
from ecg_photo.digitizers.ecg_digitiser import EcgDigitiserDigitizer, parse_wfdb_output


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


def test_parse_wfdb_output(tmp_path) -> None:
    sig = np.array(
        [
            [0.0, 0.1, 0.0],
            [0.2, 0.2, 0.0],
            [0.4, 0.0, 0.0],
            [0.2, 0.3, 0.0],
            [0.0, 0.4, 0.0],
            [0.1, 0.0, 0.0],
            [0.3, 0.2, 0.0],
            [0.0, 0.1, 0.0],
            [0.5, 0.3, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    wfdb.wrsamp(
        "rec",
        fs=500.0,
        units=["mV"] * 3,
        sig_name=["I", "II", "V1"],
        p_signal=sig,
        fmt=["16"] * 3,
        write_dir=str(tmp_path),
    )
    leads, observed, fs_hz, extra = parse_wfdb_output(tmp_path, "rec")
    assert fs_hz == pytest.approx(500.0)
    assert set(leads) == {"I", "II", "V1"}
    assert all(o.all() for o in observed.values())
    assert extra["missing_encoded_as_zero"] is True
    assert extra["exact_zero_fraction_V1"] == pytest.approx(1.0)
    assert extra["exact_zero_fraction_I"] == pytest.approx(0.4, abs=0.15)
    assert leads["I"].dtype == np.float64


def test_ecg_digitiser_patch_check_before_weights(tmp_path) -> None:
    # Patch text is checked before weights, so a fake root with no weight
    # files and a digitize.py lacking float(rot_angle) raises EngineNotReady.
    root = tmp_path / "fake_root"
    (root / "src/run").mkdir(parents=True)
    (root / "src/run/digitize.py").write_text("image_rotated = rotate(image, rot_angle)\n")
    d = EcgDigitiserDigitizer(
        root=root,
        python_exe=tmp_path / "python",
        model_dir="models/M3",
    )
    with pytest.raises(EngineNotReady, match="patches 0001/0002 not applied"):
        d.run(tmp_path / "img.png", tmp_path / "work")


def test_ahus_relative_work_dir_and_stderr_tail(tmp_path, monkeypatch) -> None:
    # F7: a relative work dir produced a relative --config path; the engine
    # subprocess runs with cwd=ahus_root, where it did not exist, and the
    # failure surfaced as a bare CalledProcessError without the engine's stderr
    import subprocess

    import yaml

    import ecg_photo.digitizers.ahus as ahus_mod
    from ecg_photo.digitizers.ahus import AhusDigitizer

    root = tmp_path / "ahus"
    root.mkdir()
    base = tmp_path / "base.yml"
    base.write_text(
        yaml.safe_dump(
            {
                "MODEL": {"KWARGS": {"config": {"LAYOUT_IDENTIFIER": {"KWARGS": {}}}}},
                "DATA": {},
            }
        )
    )
    img = tmp_path / "page.png"
    img.write_bytes(b"png")
    seen: dict = {}

    def fake_run(cmd, cwd, **kw):
        cfg = Path(cmd[-1])
        seen["cfg_absolute"] = cfg.is_absolute()
        seen["cfg_exists_from_cwd"] = (Path(cwd) / cfg).exists()
        seen["images_path"] = yaml.safe_load(cfg.read_text())["DATA"]["images_path"]
        return subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="12%|###| progress\n" * 50 + "ValueError: boom"
        )

    monkeypatch.setattr(ahus_mod, "verify_weights", lambda *_a, **_k: [])
    monkeypatch.setattr(ahus_mod.subprocess, "run", fake_run)
    spec = load_engine_specs()["ahus"]  # read from the repo before leaving it
    monkeypatch.chdir(tmp_path)
    d = AhusDigitizer(
        ahus_root=Path("ahus"),
        python_exe=Path("venv/bin/python"),
        base_config=Path("base.yml"),
        duration_s=10.0,
        engine_spec=spec,
        expected_patches_applied=False,
    )
    with pytest.raises(EngineNotReady, match="ValueError: boom"):
        d.run(img, Path("work/rel"))
    assert seen["cfg_absolute"] and seen["cfg_exists_from_cwd"]
    assert Path(seen["images_path"]).is_absolute()
    assert d.python_exe.is_absolute() and d.ahus_root.is_absolute()
