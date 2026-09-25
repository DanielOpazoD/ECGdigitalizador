#!/usr/bin/env bash
# Reproducible CPU setup of the two pinned engines under external/ (ignored by git).
# Commits, patches and weight sha256 come from configs/checkpoints.json and patches/.
# Weights are fetched from media.githubusercontent.com (git lfs budget is not
# assumed) and verified; a mismatch aborts. Needs: git, curl, uv, python3.12.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p external
AHUS_COMMIT=97a15087d4abcda843da8c58ee74b1d8f47e6f9a
DIG_COMMIT=e6f62aa776f105e4c7b04f21669da4d4f0df370b
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
# inference-only subset of external/ahus/requirements.txt (src/utils imports ray.tune)
VIRTUAL_ENV=external/.venv-ahus uv pip install -q torch torchvision numpy scipy matplotlib \
  pillow scikit-image tqdm yacs "ray[tune]" networkx scikit-learn opencv-python-headless \
  torch-tps pyyaml pandas

# --- ECG-Digitiser (Krones), model M3 -------------------------------------
checkout https://github.com/felixkrones/ECG-Digitiser ecg-digitiser "$DIG_COMMIT"
apply_patch ecg-digitiser ecg-digitiser-0001-rotate-float-angle.patch
apply_patch ecg-digitiser ecg-digitiser-0002-write-geometry.patch
fetch_weight felixkrones/ECG-Digitiser "$DIG_COMMIT" ecg-digitiser "$M3/checkpoint_final.pth" \
  8e4bae0b568b91ee26bc29841ba2a1d9eb5571149f19a009459c85342375cffb
fetch_weight felixkrones/ECG-Digitiser "$DIG_COMMIT" ecg-digitiser "$M3/checkpoint_best.pth" \
  60020c47840f95e0574952c161ae3d19f2f8aeb9a5dc440a01c6ea4b927f255b
[ -x external/.venv-digitiser/bin/python ] || uv venv -q -p python3.12 external/.venv-digitiser
VIRTUAL_ENV=external/.venv-digitiser uv pip install -q torch torchvision numpy scipy \
  scikit-image opencv-python-headless matplotlib tqdm wfdb pillow pandas \
  -e external/ecg-digitiser/nnUNet

echo "engines ready: external/ahus (+.venv-ahus), external/ecg-digitiser M3 (+.venv-digitiser)"
