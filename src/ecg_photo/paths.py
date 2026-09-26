"""Location of the repository's configuration files (`configs/`).

Paths used to be relative to the current directory, so any command run
outside the repository root failed with FileNotFoundError. They are now
resolved from the source tree (editable install); ECG_PHOTO_CONFIG_DIR
overrides it (e.g. an installed wheel, which does not ship `configs/`).
"""

import os
from pathlib import Path

REPO_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def config_dir() -> Path:
    env = os.environ.get("ECG_PHOTO_CONFIG_DIR")
    return Path(env) if env else REPO_CONFIG_DIR


def config_file(name: str) -> Path:
    return config_dir() / name
