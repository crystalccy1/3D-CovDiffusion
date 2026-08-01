#!/usr/bin/env bash
set -euo pipefail

required_uv="0.10.0"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv $required_uv is required to regenerate the lock files." >&2
  exit 2
fi
actual_uv="$(uv --version | awk '{print $2}')"
if [[ "$actual_uv" != "$required_uv" ]]; then
  echo "Expected uv $required_uv, found $actual_uv." >&2
  exit 2
fi

requirements_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../requirements" && pwd)"
cd "$requirements_dir"

uv pip compile requirements-build.in \
  --output-file requirements-build-linux-py310.lock.txt \
  --python-version 3.10 \
  --python-platform x86_64-manylinux_2_31 \
  --generate-hashes --emit-index-url --emit-index-annotation

uv pip compile requirements-runtime.in \
  --constraints constraints-runtime-validated.txt \
  --output-file requirements-runtime-linux-py310-cu117.lock.txt \
  --python-version 3.10 \
  --python-platform x86_64-manylinux_2_31 \
  --generate-hashes --emit-index-url --emit-index-annotation

uv pip compile requirements-visualization.in \
  --constraints constraints-visualization-validated.txt \
  --output-file requirements-visualization-linux-py310-cu117.lock.txt \
  --python-version 3.10 \
  --python-platform x86_64-manylinux_2_31 \
  --generate-hashes --emit-index-url --emit-index-annotation
