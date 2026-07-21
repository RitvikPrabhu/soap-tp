#!/usr/bin/env bash

# Bootstrap a complete development installation from a fresh clone. Native
# solver details remain in scripts/install.sh; this wrapper owns the virtual
# environment and Python dependencies. Native build scripts initialize only
# the submodules needed for libraries that are not supplied externally.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${1:-cpu}"

usage() {
    cat <<'EOF'
Usage: ./install.sh [cpu|cuda|rocm] [options]

Creates or reuses .venv, installs Python dependencies, builds the selected
native libraries, and installs soap-tp in editable mode.

Examples:
  ./install.sh cpu
  ./install.sh cuda
  ./install.sh rocm
  ./install.sh cpu --skip-elpa --skip-slate
  ./install.sh rocm --skip-slate

Options:
  --no-editable    Install a snapshot instead of linking the checkout.
  --skip-elpa      Reuse an existing ELPA installation.
  --skip-slate     Reuse an existing SLATE installation.
  -h, --help       Show this help.

Environment:
  PYTHON            Use this Python instead of creating .venv.
  BOOTSTRAP_PYTHON  Python used to create .venv (default: python3).
  SOAP_TP_VENV      Virtual-environment path (default: <repo>/.venv).
  TORCH_INDEX_URL   Override the automatically selected PyTorch wheel index.

Additional options are forwarded to scripts/install.sh and then pip.
System MPI/compiler prerequisites are listed when they are missing.
EOF
}

if [[ "${PROFILE}" == "-h" || "${PROFILE}" == "--help" ]]; then
    usage
    exit 0
fi

case "${PROFILE}" in
    cpu|cuda|rocm) ;;
    *) usage >&2; exit 2 ;;
esac
shift $(( $# > 0 ? 1 : 0 ))

EDITABLE=1
SKIP_ELPA=0
SKIP_SLATE=0
INSTALL_ARGS=()
for argument in "$@"; do
    case "${argument}" in
        -h|--help) usage; exit 0 ;;
        --no-editable) EDITABLE=0 ;;
        --editable) EDITABLE=1 ;;
        --skip-elpa) SKIP_ELPA=1; INSTALL_ARGS+=("${argument}") ;;
        --skip-slate) SKIP_SLATE=1; INSTALL_ARGS+=("${argument}") ;;
        *) INSTALL_ARGS+=("${argument}") ;;
    esac
done

missing_tools=()
required_tools=()
if [[ "${SKIP_ELPA}" == "0" ]]; then
    required_tools+=(git cmake make autoreconf automake)
fi
if [[ "${SKIP_SLATE}" == "0" ]]; then
    required_tools+=(git cmake)
fi
for tool in "${required_tools[@]}"; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        missing_tools+=("${tool}")
    fi
done
if [[ "${SKIP_ELPA}" == "0" ]] && \
   ! command -v libtoolize >/dev/null 2>&1 && \
   ! command -v glibtoolize >/dev/null 2>&1; then
    missing_tools+=("libtool")
fi

if [[ -n "${CC:-}" && -n "${CXX:-}" && -n "${FC:-}" ]]; then
    for compiler in "${CC}" "${CXX}" "${FC}"; do
        if ! command -v "${compiler}" >/dev/null 2>&1; then
            missing_tools+=("${compiler}")
        fi
    done
elif ! { command -v mpicc >/dev/null 2>&1 && \
         command -v mpicxx >/dev/null 2>&1 && \
         command -v mpifort >/dev/null 2>&1; } && \
     ! { command -v cc >/dev/null 2>&1 && \
         command -v CC >/dev/null 2>&1 && \
         command -v ftn >/dev/null 2>&1; }; then
    missing_tools+=("MPI C/C++/Fortran wrappers")
fi

if (( ${#missing_tools[@]} > 0 )); then
    echo "Missing native build prerequisites:" >&2
    for tool in "${missing_tools[@]}"; do
        echo "  - ${tool}" >&2
    done
    echo >&2
    case "$(uname -s)" in
        Darwin)
            echo "Install them with Homebrew:" >&2
            echo "  brew install cmake gcc open-mpi autoconf automake libtool" >&2
            ;;
        Linux)
            if command -v apt-get >/dev/null 2>&1; then
                echo "On Ubuntu/Debian, install them with:" >&2
                echo "  sudo apt-get update && sudo apt-get install -y \\" >&2
                echo "    build-essential cmake gfortran openmpi-bin \\" >&2
                echo "    libopenmpi-dev autoconf automake libtool python3-venv" >&2
            else
                echo "Install CMake, Autotools, Libtool, a C++17/Fortran" >&2
                echo "toolchain, and MPI development wrappers." >&2
            fi
            ;;
    esac
    exit 1
fi

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_CANDIDATE="${PYTHON}"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_CANDIDATE="${VIRTUAL_ENV}/bin/python"
else
    VENV="${SOAP_TP_VENV:-${ROOT}/.venv}"
    BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3}"
    if ! command -v "${BOOTSTRAP_PYTHON}" >/dev/null 2>&1; then
        echo "Python executable not found: ${BOOTSTRAP_PYTHON}" >&2
        exit 1
    fi
    if [[ ! -x "${VENV}/bin/python" ]]; then
        echo "Creating virtual environment in ${VENV}"
        "${BOOTSTRAP_PYTHON}" -m venv "${VENV}"
    fi
    PYTHON_CANDIDATE="${VENV}/bin/python"
fi

if ! PYTHON_BIN="$("${PYTHON_CANDIDATE}" -c \
    'import sys; print(sys.executable)' 2>/dev/null)"; then
    echo "Unable to run Python executable: ${PYTHON_CANDIDATE}" >&2
    exit 1
fi

if ! "${PYTHON_BIN}" -c \
    'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
    echo "soap-tp requires Python 3.10 or newer: ${PYTHON_BIN}" >&2
    exit 1
fi

if ! "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
    "${PYTHON_BIN}" -m ensurepip --upgrade
fi

PYTORCH_REQUIREMENT="torch>=2.6"

detect_toolkit_version() {
    local version=""

    case "${PROFILE}" in
        rocm)
            if command -v hipconfig >/dev/null 2>&1; then
                version="$(hipconfig --version 2>/dev/null | \
                    grep -Eo '[0-9]+[.][0-9]+([.][0-9]+)?' | head -n 1 || true)"
            fi
            if [[ -z "${version}" && -n "${ROCM_PATH:-}" ]]; then
                version="$(printf '%s\n' "${ROCM_PATH}" | \
                    grep -Eo '[0-9]+[.][0-9]+([.][0-9]+)?' | head -n 1 || true)"
            fi
            ;;
        cuda)
            if command -v nvcc >/dev/null 2>&1; then
                version="$(nvcc --version 2>/dev/null | sed -n \
                    's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
            fi
            ;;
    esac
    printf '%s\n' "${version}"
}

automatic_torch_index() {
    local major
    local minor
    local version

    version="$(detect_toolkit_version)"
    [[ -n "${version}" ]] || return 1
    major="${version%%.*}"
    minor="${version#*.}"
    minor="${minor%%.*}"
    case "${PROFILE}" in
        rocm) printf 'https://download.pytorch.org/whl/rocm%s.%s\n' "${major}" "${minor}" ;;
        cuda) printf 'https://download.pytorch.org/whl/cu%s%s\n' "${major}" "${minor}" ;;
    esac
}

pytorch_is_compatible() {
    "${PYTHON_BIN}" - <<'PY'
import inspect

try:
    import torch.distributed as dist
except ImportError:
    raise SystemExit(1)

raise SystemExit(
    "group_peer" not in inspect.signature(dist.P2POp).parameters
)
PY
}

if ! pytorch_is_compatible; then
    # Older virtualenvs can start with pip versions that reject normalized
    # dependency names on the PyTorch wheel index and silently backtrack to an
    # incompatible torch release.
    "${PYTHON_BIN}" -m pip install --upgrade pip

    if "${PYTHON_BIN}" -c "import torch" >/dev/null 2>&1; then
        TORCH_VERSION="$("${PYTHON_BIN}" -c \
            'import torch; print(torch.__version__)')"
        echo "Upgrading incompatible PyTorch ${TORCH_VERSION} for ${PROFILE}."
    else
        echo "Installing PyTorch for the ${PROFILE} profile."
    fi

    TORCH_ARGS=("${PYTORCH_REQUIREMENT}")
    if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
        TORCH_ARGS+=(--index-url "${TORCH_INDEX_URL}")
    elif [[ "${PROFILE}" == "cpu" && "$(uname -s)" == "Linux" ]]; then
        TORCH_ARGS+=(--index-url "https://download.pytorch.org/whl/cpu")
    elif [[ "${PROFILE}" != "cpu" ]] && \
         TORCH_INDEX_URL="$(automatic_torch_index)"; then
        echo "Using PyTorch index ${TORCH_INDEX_URL}"
        TORCH_ARGS+=(--index-url "${TORCH_INDEX_URL}")
    elif [[ "${PROFILE}" != "cpu" ]]; then
        echo "Unable to determine the active ${PROFILE} toolkit version." >&2
        echo "Load the toolkit or set TORCH_INDEX_URL explicitly." >&2
        exit 1
    fi
    "${PYTHON_BIN}" -m pip install --upgrade "${TORCH_ARGS[@]}"
else
    TORCH_VERSION="$("${PYTHON_BIN}" -c 'import torch; print(torch.__version__)')"
    echo "Reusing compatible PyTorch ${TORCH_VERSION} from ${PYTHON_BIN}"
fi

if ! pytorch_is_compatible; then
    TORCH_VERSION="$("${PYTHON_BIN}" -c \
        'import torch; print(torch.__version__)' 2>/dev/null || echo unavailable)"
    echo "soap-tp requires PyTorch 2.6 or newer; found ${TORCH_VERSION}." >&2
    exit 1
fi

"${PYTHON_BIN}" -m pip install \
    "setuptools>=68" \
    wheel \
    "pybind11>=2.11" \
    numpy \
    ninja \
    mpi4py \
    matplotlib

NATIVE_ARGS=("${PROFILE}")
if [[ "${EDITABLE}" == "1" ]]; then
    NATIVE_ARGS+=(--editable)
fi
NATIVE_ARGS+=("${INSTALL_ARGS[@]}")

PYTHON="${PYTHON_BIN}" "${ROOT}/scripts/install.sh" "${NATIVE_ARGS[@]}"

echo
echo "Installation complete."
if [[ "${PYTHON_BIN}" == "${ROOT}/.venv/bin/python" ]]; then
    echo "Activate it with: source .venv/bin/activate && source build/bindings.env"
else
    echo "Python: ${PYTHON_BIN}"
    echo "Native runtime environment: source build/bindings.env"
fi
