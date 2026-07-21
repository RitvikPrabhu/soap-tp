#!/usr/bin/env bash

# Incrementally rebuild only the soap-tp pybind extensions. The installer
# records the native-library and compiler choices consumed here.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_CONFIG="${SOAP_TP_BUILD_CONFIG:-${ROOT}/build/bindings.env}"
FORCE=0

case "${1:-}" in
    "") ;;
    --force) FORCE=1 ;;
    -h|--help)
        echo "Usage: scripts/rebuild_bindings.sh [--force]"
        exit 0
        ;;
    *)
        echo "Usage: scripts/rebuild_bindings.sh [--force]" >&2
        exit 2
        ;;
esac

if [[ ! -f "${BUILD_CONFIG}" ]]; then
    echo "Build configuration not found: ${BUILD_CONFIG}" >&2
    echo "Run install.sh first, or set SOAP_TP_BUILD_CONFIG." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${BUILD_CONFIG}"

PYTHON="${PYTHON:-${SOAP_TP_PYTHON}}"
export ELPA_PROFILE="${SOAP_TP_PROFILE}"
export SLATE_PROFILE="${SOAP_TP_PROFILE}"
export SOAP_TP_BUILD_ELPA_BINDINGS=1
export SOAP_TP_BUILD_SLATE_BINDINGS=1
export ELPA_PREFIX SLATE_PREFIX CC CXX FC

BUILD_ARGS=(build_ext --inplace)
if [[ "${FORCE}" == "1" ]]; then
    BUILD_ARGS+=(--force)
fi

cd "${ROOT}"
"${PYTHON}" setup.py "${BUILD_ARGS[@]}"
