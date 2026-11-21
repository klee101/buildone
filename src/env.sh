# BUILDONE_ENV_FILE=path/to/file source src/env.sh
if [ -n "${BUILDONE_ENV_FILE:-}" ]; then
    source "$BUILDONE_ENV_FILE"
fi
