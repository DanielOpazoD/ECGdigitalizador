import json
import sys
from pathlib import Path

import yaml

from ecg_photo.cli import main
from ecg_photo.doctor import report, run_checks


def test_no_engine_registry_is_not_usable(tmp_path: Path) -> None:
    checks, usable = run_checks(tmp_path / "engines.local.yml")
    assert not usable
    by_name = {c.name: c for c in checks}
    assert not by_name["engine registry"].ok
    assert "install.sh" in by_name["engine registry"].detail
    assert by_name["config checkpoints.json"].ok
    assert "NO está listo" in report(checks, usable)


def test_broken_engine_is_reported(tmp_path: Path) -> None:
    # a checkout without the patch, with a wrong weight and an interpreter
    # without torch: every problem is named, and no engine is ready
    root = tmp_path / "ahus"
    (root / "src").mkdir(parents=True)
    (root / "src" / "digitize.py").write_text("print('unpatched')\n")
    (root / "weights").mkdir()
    (root / "weights" / "unet_weights_07072025.pt").write_bytes(b"not the weights")
    reg = tmp_path / "engines.local.yml"
    reg.write_text(yaml.safe_dump({"ahus": {"root": str(root), "python": sys.executable}}))
    checks, usable = run_checks(reg)
    by_name = {c.name: c for c in checks}
    assert not by_name["ahus: patches (patches/)"].ok
    assert "sha256 mismatch" in by_name["ahus: weights sha256"].detail
    assert not by_name["ahus: interpreter"].ok  # the test venv has no torch
    assert not by_name["at least one engine ready"].ok and not usable


def test_cli_doctor_json_and_exit_code(tmp_path: Path, capsys) -> None:
    rc = main(["doctor", "--engines-config", str(tmp_path / "none.yml"), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1 and out["usable"] is False
    assert any(c["name"] == "engine registry" and not c["ok"] for c in out["checks"])
