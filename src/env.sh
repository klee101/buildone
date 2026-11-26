# Load optional shell settings for both setup.sh and run.sh.
buildone_env=${BUILDONE_ENV_FILE:-"$(dirname -- "${BASH_SOURCE[0]}")/../.env"}
if [[ -f "$buildone_env" || -n "${BUILDONE_ENV_FILE:-}" ]]; then
    source "$buildone_env"
fi
unset buildone_env
