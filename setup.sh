#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
root=$PWD

git submodule update --init --recursive
UV_PROJECT_ENVIRONMENT="$root/.venv" uv sync --locked --link-mode copy

# Apply the merged SAM 3.1 checkpoint loader fix to the installed copy.
site_packages=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
sam_patch="$root/third_party/sam3-multiplex.patch"
if ! patch --silent --dry-run --force --reverse -p1 -d "$site_packages" < "$sam_patch" >/dev/null 2>&1; then
    patch --force -p1 -d "$site_packages" < "$sam_patch"
fi

mkdir -p output
