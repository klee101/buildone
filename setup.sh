#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
root=$PWD
if [[ $# -gt 1 || ( $# -eq 1 && $1 != --qwen-views ) ]]; then
    echo "Usage: ./setup.sh [--qwen-views]" >&2
    [[ ${1:-} == --help && $# -eq 1 ]] && exit 0
    exit 2
fi
source src/env.sh
if [[ "$(uname -sm)" != "Linux x86_64" ]]; then
    echo "BuildOne requires Linux x86-64." >&2
    exit 1
fi
for tool in git patch; do
    command -v "$tool" >/dev/null || { echo "Install $tool and rerun setup." >&2; exit 1; }
done
if command -v uv >/dev/null; then
    uv_bin=$(command -v uv)
else
    uv_bin="$root/.tools/uv"
    if [[ ! -x "$uv_bin" ]]; then
        curl -fsSL https://astral.sh/uv/0.11.7/install.sh | UV_UNMANAGED_INSTALL="$root/.tools" sh
    fi
fi
git submodule update --init --recursive
# build123d resolves its package version from the release tag.
if ! git -C third_party/build123d show-ref --verify --quiet refs/tags/v0.11.1; then
    git -C third_party/build123d fetch --no-tags --depth 1 origin tag v0.11.1
fi
UV_PROJECT_ENVIRONMENT="$root/.venv" "$uv_bin" sync --python 3.11 --locked --link-mode copy

# Apply the merged SAM 3.1 checkpoint loader fix to the installed copy.
site_packages=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
sam_patch="$root/third_party/sam3-multiplex.patch"
if ! patch --silent --dry-run --force --reverse -p1 -d "$site_packages" < "$sam_patch" >/dev/null 2>&1; then
    patch --force -p1 -d "$site_packages" < "$sam_patch"
fi

.venv/bin/python - <<'PY'
import torch
from build123d import Box
from pxr import Usd
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
    raise SystemExit("A CUDA-visible NVIDIA GPU with BF16 support and a CUDA 12.9 compatible driver is required.")
print("CUDA device:", torch.cuda.get_device_name(0))
PY
.venv/bin/python -B src/weights.py "$@"

codex_bin=$(.venv/bin/python -c 'from codex_cli_bin import bundled_codex_path; print(bundled_codex_path())')
if [[ -n ${BUILDONE_CODEX_HOME:-} ]]; then
    codex_dir=$(.venv/bin/python -c 'import os; from pathlib import Path; print(Path(os.environ["BUILDONE_CODEX_HOME"]).expanduser().resolve())')
    export CODEX_HOME="$codex_dir"
    mkdir -p "$CODEX_HOME"
fi
if ! "$codex_bin" login status; then
    if [[ -n ${OPENAI_API_KEY:-} ]]; then
        printenv OPENAI_API_KEY | "$codex_bin" login --with-api-key
    elif [[ -t 0 ]]; then
        "$codex_bin" login --device-auth
    else
        echo "Rerun ./setup.sh in a terminal to sign in to Codex, or supply OPENAI_API_KEY." >&2
        exit 1
    fi
fi
echo "Ready. Run ./run.sh --input data/*.jpg${1:+ $1}"
