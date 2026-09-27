#!/usr/bin/env bash
# Reproducible CPU setup of the two pinned engines under external/ (ignored by git).
# Commits, patches and weight sha256 come from configs/checkpoints.json and patches/.
# Weights are fetched from media.githubusercontent.com (git lfs budget is not
# assumed) and verified; a mismatch aborts. Needs: git, curl, uv, python3.12.
# Engine environments are pinned (configs/engines/*-requirements.lock) and use
# the CPU build of torch (~1.6 GB instead of ~6 GB with CUDA); set
# TORCH_INDEX_URL to another PyTorch index (e.g. a CUDA one) to use a GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p external
AHUS_COMMIT=97a15087d4abcda843da8c58ee74b1d8f47e6f9a
DIG_COMMIT=e6f62aa776f105e4c7b04f21669da4d4f0df370b
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
TORCH_PINS="torch==2.14.0 torchvision==0.29.0"
M3=models/M3/nnUNet_results/Dataset500_Signals/nnUNetTrainer__nnUNetPlans__2d/fold_all

checkout() {  # repo_url dir commit
  if [ ! -d "external/$2/.git" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone -q --filter=blob:none --no-checkout "$1" "external/$2"
  fi
  GIT_LFS_SKIP_SMUDGE=1 git -C "external/$2" -c advice.detachedHead=false checkout -q "$3"
}

apply_patch() {  # dir patch
  if git -C "external/$1" apply --check "../../patches/$2" 2>/dev/null; then
    git -C "external/$1" apply "../../patches/$2"
  elif git -C "external/$1" apply --reverse --check "../../patches/$2" 2>/dev/null; then
    echo "patch $2 already applied"
  else
    echo "patch $2 does not apply" >&2; exit 1
  fi
}

fetch_weight() {  # owner/repo commit dir relpath sha256
  local dst="external/$3/$4"
  if [ -f "$dst" ] && [ "$(sha256sum "$dst" | cut -d' ' -f1)" = "$5" ]; then return; fi
  curl -sSL --fail --max-time 1800 -o "$dst" "https://media.githubusercontent.com/media/$1/$2/$4"
  local got; got="$(sha256sum "$dst" | cut -d' ' -f1)"
  if [ "$got" != "$5" ]; then echo "sha256 mismatch for $4: $got" >&2; exit 1; fi
}

# --- Ahus Open-ECG-Digitizer ---------------------------------------------
checkout https://github.com/Ahus-AIM/Open-ECG-Digitizer ahus "$AHUS_COMMIT"
apply_patch ahus ahus-0001-write-geometry.patch
fetch_weight Ahus-AIM/Open-ECG-Digitizer "$AHUS_COMMIT" ahus weights/unet_weights_07072025.pt \
  17fe7071ef270102631306127262fc08c250d79d4e3aeb572ab1719dd34d320b
fetch_weight Ahus-AIM/Open-ECG-Digitizer "$AHUS_COMMIT" ahus weights/lead_name_unet_weights_07072025.pt \
  840bd6bf2433ee6c22db67f57c861d9d427f29e10a32eeb334f0bcf061b175a2
[ -x external/.venv-ahus/bin/python ] || uv venv -q -p python3.12 external/.venv-ahus
# inference-only subset of external/ahus/requirements.txt (src/utils imports
# ray.tune), pinned to the versions the benchmarks ran with
VIRTUAL_ENV=external/.venv-ahus uv pip install -q $TORCH_PINS --index-url "$TORCH_INDEX_URL"
VIRTUAL_ENV=external/.venv-ahus uv pip install -q -r configs/engines/ahus-requirements.lock

# --- ECG-Digitiser (Krones), model M3 -------------------------------------
# secondary engine (benchmarks, comparison); skip with WITH_DIGITISER=0
if [ "${WITH_DIGITISER:-1}" = "1" ]; then
  checkout https://github.com/felixkrones/ECG-Digitiser ecg-digitiser "$DIG_COMMIT"
  apply_patch ecg-digitiser ecg-digitiser-0001-rotate-float-angle.patch
  apply_patch ecg-digitiser ecg-digitiser-0002-write-geometry.patch
  fetch_weight felixkrones/ECG-Digitiser "$DIG_COMMIT" ecg-digitiser "$M3/checkpoint_final.pth" \
    8e4bae0b568b91ee26bc29841ba2a1d9eb5571149f19a009459c85342375cffb
  fetch_weight felixkrones/ECG-Digitiser "$DIG_COMMIT" ecg-digitiser "$M3/checkpoint_best.pth" \
    60020c47840f95e0574952c161ae3d19f2f8aeb9a5dc440a01c6ea4b927f255b
  [ -x external/.venv-digitiser/bin/python ] || uv venv -q -p python3.12 external/.venv-digitiser
  VIRTUAL_ENV=external/.venv-digitiser uv pip install -q $TORCH_PINS --index-url "$TORCH_INDEX_URL"
  VIRTUAL_ENV=external/.venv-digitiser uv pip install -q -r configs/engines/ecg-digitiser-requirements.lock
  VIRTUAL_ENV=external/.venv-digitiser uv pip install -q --no-deps -e external/ecg-digitiser/nnUNet
fi

# worker/API engine registry (not versioned). Absolute paths: the engine
# subprocess runs with cwd=<engine root>, where a relative interpreter path
# would not resolve. An existing file is left as the user wrote it.
if [ ! -f configs/engines.local.yml ]; then
  cat > configs/engines.local.yml <<YML
ahus:
  root: $PWD/external/ahus
  python: $PWD/external/.venv-ahus/bin/python
  config: $PWD/external/ahus/src/config/inference_wrapper_george-moody-2024.yml
YML
  if [ "${WITH_DIGITISER:-1}" = "1" ]; then
    cat >> configs/engines.local.yml <<YML
ecg_digitiser:
  root: $PWD/external/ecg-digitiser
  python: $PWD/external/.venv-digitiser/bin/python
  model_dir: models/M3
YML
  fi
  echo "wrote configs/engines.local.yml"
fi

echo "engines ready: external/ahus (+.venv-ahus)$([ "${WITH_DIGITISER:-1}" = "1" ] && echo ', external/ecg-digitiser M3 (+.venv-digitiser)')"
