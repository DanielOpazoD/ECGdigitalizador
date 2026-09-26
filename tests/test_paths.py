import json
from pathlib import Path

from ecg_photo.digitizers.ahus import DEFAULT_LAYOUTS
from ecg_photo.digitizers.base import load_engine_specs
from ecg_photo.ingest import load_supported_inputs
from ecg_photo.paths import REPO_CONFIG_DIR, config_file
from ecg_photo.store import load_execution_limits


def test_configs_found_from_any_directory(tmp_path: Path, monkeypatch) -> None:
    # every command used to fail with FileNotFoundError outside the repo root
    monkeypatch.chdir(tmp_path)
    assert "ahus" in load_engine_specs()
    assert load_execution_limits().max_active_jobs == 1
    assert load_supported_inputs().max_pdf_pages > 0
    assert DEFAULT_LAYOUTS.exists()


def test_config_dir_override(tmp_path: Path, monkeypatch) -> None:
    src = json.loads((REPO_CONFIG_DIR / "checkpoints.json").read_text(encoding="utf-8"))
    src["engines"] = [e for e in src["engines"] if e["engine_id"] == "ahus"]
    (tmp_path / "checkpoints.json").write_text(json.dumps(src), encoding="utf-8")
    monkeypatch.setenv("ECG_PHOTO_CONFIG_DIR", str(tmp_path))
    assert config_file("checkpoints.json") == tmp_path / "checkpoints.json"
    assert list(load_engine_specs()) == ["ahus"]
