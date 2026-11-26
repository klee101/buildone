#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$root/src/env.sh"
if [[ ! -x "$root/.venv/bin/python" ]]; then
    echo "Run $root/setup.sh first." >&2
    exit 1
fi
export PATH="$root/.venv/bin:$PATH"
exec "$root/.venv/bin/python" -B "$root/src/run.py" "$@"
