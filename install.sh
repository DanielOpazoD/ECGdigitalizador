#!/usr/bin/env bash
# One-step installation of ecg-photo (Linux / macOS):
#   ./install.sh                  program + Ahus engine (CPU)
#   ./install.sh --with-digitiser also the secondary engine ECG-Digitiser
# Idempotent: re-running skips what is already in place and re-verifies the
# engine weights (sha256). Ends with `ecg-photo doctor`.
# Needs: python3.12, git, curl, uv (https://docs.astral.sh/uv/). Downloads
# about 2.5 GB (engine weights + CPU torch) the first time.
set -euo pipefail
cd "$(dirname "$0")"

WITH_DIGITISER=0
for arg in "$@"; do
  case "$arg" in
    --with-digitiser) WITH_DIGITISER=1 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

missing=()
for tool in python3.12 git curl uv; do
  command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "missing: ${missing[*]}" >&2
  echo "install them first (uv: curl -LsSf https://astral.sh/uv/install.sh | sh)" >&2
  exit 1
fi

echo "== 1/3 program environment (.venv, versions from requirements.lock)"
[ -x .venv/bin/python ] || python3.12 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -c requirements.lock -e ".[dev]"

echo "== 2/3 engines (external/, weights verified against configs/checkpoints.json)"
WITH_DIGITISER=$WITH_DIGITISER bash benchmarks/setup_engines.sh

echo "== 3/3 check"
.venv/bin/ecg-photo doctor
echo
echo "Ready. Start the review interface with:"
echo "  .venv/bin/ecg-photo serve --store ~/ecg-estudios   # then open http://127.0.0.1:8000/ui"
echo "User guide: docs/guia_uso.md"
