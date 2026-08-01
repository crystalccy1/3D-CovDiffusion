#!/usr/bin/env bash
set -euo pipefail

# Prevent machine-local user packages from leaking into any validation command.
export PYTHONNOUSERSITE=1

usage() {
  echo "Usage: $0 [ENV_NAME] [--visualization]" >&2
}

env_name="3dcov-cu117"
install_visualization=false

if [[ $# -gt 0 && "$1" != "--visualization" ]]; then
  env_name="$1"
  shift
fi
if [[ $# -gt 0 && "$1" == "--visualization" ]]; then
  install_visualization=true
  shift
fi
if [[ $# -ne 0 ]]; then
  usage
  exit 2
fi

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "The locked environment targets Linux x86_64." >&2
  exit 2
fi

conda_exe="${CONDA_EXE:-conda}"
if ! command -v "$conda_exe" >/dev/null 2>&1; then
  echo "Conda was not found. Set CONDA_EXE or add conda to PATH." >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
lock_root="$repo_root/requirements"

if "$conda_exe" run -n "$env_name" python -c "import sys" >/dev/null 2>&1; then
  echo "Conda environment '$env_name' already exists; refusing to modify it." >&2
  exit 2
fi

echo "Creating fresh conda environment: $env_name"
"$conda_exe" env create --yes --name "$env_name" \
  --file "$repo_root/environment-linux-cu117.yml"

echo "Installing locked build tools"
"$conda_exe" run --no-capture-output -n "$env_name" \
  python -m pip install --require-hashes \
  -r "$lock_root/requirements-build-linux-py310.lock.txt"

echo "Installing locked CUDA 11.7 runtime"
"$conda_exe" run --no-capture-output -n "$env_name" \
  python -m pip install --no-build-isolation --require-hashes \
  -r "$lock_root/requirements-runtime-linux-py310-cu117.lock.txt"

if [[ "$install_visualization" == true ]]; then
  echo "Installing locked visualization dependencies"
  "$conda_exe" run --no-capture-output -n "$env_name" \
    python -m pip install --no-build-isolation --require-hashes \
    -r "$lock_root/requirements-visualization-linux-py310-cu117.lock.txt"
fi

"$conda_exe" run --no-capture-output -n "$env_name" python -m pip check
"$conda_exe" run --no-capture-output -n "$env_name" python -c \
  "import sys, torch; assert sys.version_info[:3] == (3, 10, 18), sys.version; assert torch.__version__ == '1.13.1+cu117', torch.__version__; assert torch.version.cuda == '11.7', torch.version.cuda; assert torch.cuda.is_available(), 'CUDA is not available'; print(f'validated python={sys.version.split()[0]} torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}')"

echo "Environment '$env_name' is ready."
echo "Run commands with: conda run --no-capture-output -n $env_name <command>"
