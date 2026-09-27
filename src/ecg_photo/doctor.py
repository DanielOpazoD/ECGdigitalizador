"""`ecg-photo doctor`: is this installation ready to digitize?

Checks, in order: Python version, the versioned configuration files, the local
engine registry (configs/engines.local.yml), and for each registered engine
its checkout, the minimal patches, the weights against their pinned sha256
and that its interpreter can import torch. Nothing is modified. The
installation is usable when every required check passes and at least one
engine is ready.
"""

import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import yaml

from ecg_photo.digitizers.base import load_engine_specs, verify_weights
from ecg_photo.paths import config_dir, config_file

REQUIRED_CONFIGS = (
    "checkpoints.json",
    "default.yml",
    "supported_inputs.yml",
    "ahus_lead_layouts.yml",
)
# engine id in engines.local.yml -> (id in checkpoints.json, patched file, markers)
ENGINES = {
    "ahus": ("ahus", "src/digitize.py", ("save_geometry_json",)),
    "ecg_digitiser": (
        "ecg_digitiser",
        "src/run/digitize.py",
        ("float(rot_angle)", "_geometry.json"),
    ),
}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def _python_check() -> Check:
    v = sys.version_info
    return Check("python >= 3.12", v >= (3, 12), f"{v.major}.{v.minor}.{v.micro}")


def _engine_checks(name: str, cfg: dict) -> list[Check]:
    spec_id, patch_file, markers = ENGINES[name]
    root = Path(cfg.get("root", ""))
    python = Path(cfg.get("python", ""))
    out = [Check(f"{name}: checkout", root.is_dir(), str(root), required=False)]
    if not root.is_dir():
        return out
    patched = root / patch_file
    text = patched.read_text(encoding="utf-8") if patched.exists() else ""
    missing = [m for m in markers if m not in text]
    out.append(
        Check(
            f"{name}: patches (patches/)",
            not missing,
            "applied" if not missing else f"missing in {patch_file}: {', '.join(missing)}",
            required=False,
        )
    )
    spec = load_engine_specs()[spec_id]
    if cfg.get("model_dir") or name == "ecg_digitiser":
        # only the weights of the configured model are used (and checked at
        # run time by the adapter); other models listed need not be present
        model_rel = str(cfg.get("model_dir", "models/M3")).rstrip("/") + "/"
        spec = replace(spec, weights=tuple(w for w in spec.weights if w.path.startswith(model_rel)))
    problems = verify_weights(root, spec)
    out.append(
        Check(
            f"{name}: weights sha256",
            not problems,
            "match configs/checkpoints.json" if not problems else "; ".join(problems),
            required=False,
        )
    )
    if not python.exists():
        out.append(Check(f"{name}: interpreter", False, f"not found: {python}", required=False))
        return out
    try:
        proc = subprocess.run(
            [str(python), "-c", "import torch; print(torch.__version__)"],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        ok = proc.returncode == 0
        detail = f"torch {proc.stdout.strip()}" if ok else proc.stderr.strip()[-300:]
    except (OSError, subprocess.TimeoutExpired) as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    out.append(Check(f"{name}: interpreter", ok, detail, required=False))
    return out


def run_checks(engines_config: Path | None = None) -> tuple[list[Check], bool]:
    """All checks and whether the installation can digitize."""
    checks = [_python_check()]
    for name in REQUIRED_CONFIGS:
        p = config_file(name)
        checks.append(Check(f"config {name}", p.exists(), str(p)))
    reg = Path(engines_config) if engines_config else config_file("engines.local.yml")
    checks.append(
        Check(
            "engine registry",
            reg.exists(),
            str(reg) if reg.exists() else f"{reg} missing: run ./install.sh",
        )
    )
    ready: list[str] = []
    if reg.exists():
        cfg = yaml.safe_load(reg.read_text(encoding="utf-8")) or {}
        for name in ENGINES:
            if not cfg.get(name):
                continue
            engine = _engine_checks(name, cfg[name])
            checks.extend(engine)
            if all(c.ok for c in engine):
                ready.append(name)
    checks.append(
        Check(
            "at least one engine ready",
            bool(ready),
            ", ".join(ready) if ready else "none: run ./install.sh",
        )
    )
    usable = all(c.ok for c in checks if c.required)
    return checks, usable


def report(checks: list[Check], usable: bool) -> str:
    lines = [f"configs: {config_dir()}"]
    for c in checks:
        mark = "OK   " if c.ok else ("FALLA" if c.required else "aviso")
        lines.append(f"[{mark}] {c.name}: {c.detail}")
    lines.append("listo para digitalizar" if usable else "NO está listo: vea las líneas FALLA")
    return "\n".join(lines)


def as_dict(checks: list[Check], usable: bool) -> dict:
    return {"usable": usable, "checks": [asdict(c) for c in checks]}
