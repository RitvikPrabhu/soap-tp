"""Build the optional ELPA and SLATE Python extensions.

The all-in-one installer sets the binding flags and native install prefixes.
Keeping discovery here makes regular PEP 517 builds work without embedding
paths from any one workstation or cluster in the repository.
"""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import Extension, setup

try:
    import pybind11

    PYBIND11_INCLUDE = pybind11.get_include()
except ImportError:
    # PyTorch vendors pybind11 headers. The all-in-one installer deliberately
    # disables pip build isolation so it can use those headers on clusters
    # whose login nodes cannot reach PyPI.
    from torch.utils.cpp_extension import include_paths

    PYBIND11_INCLUDE = include_paths()[0]


TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


def enabled(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise RuntimeError(
        f"{name} must be one of: "
        f"{', '.join(sorted(TRUE_VALUES | FALSE_VALUES))}"
    )


def elpa_extension() -> Extension:
    prefix_value = os.environ.get("ELPA_PREFIX")
    if not prefix_value:
        raise RuntimeError(
            "ELPA_PREFIX is required when building the ELPA binding. "
            "Run ./scripts/install.sh <cpu|cuda|rocm> instead of invoking "
            "the native build directly."
        )

    prefix = Path(prefix_value).expanduser().resolve()
    include_candidates = sorted((prefix / "include").glob("elpa-*"))
    if len(include_candidates) != 1:
        found = ", ".join(str(path) for path in include_candidates) or "none"
        raise RuntimeError(
            f"Expected exactly one ELPA include directory under {prefix / 'include'}; "
            f"found {found}"
        )

    library_candidates = [prefix / "lib", prefix / "lib64"]
    library_dir = next(
        (
            path
            for path in library_candidates
            if path.is_dir() and any(path.glob("libelpa.*"))
        ),
        None,
    )
    if library_dir is None:
        raise RuntimeError(f"Could not find libelpa under {prefix}")

    # Use an absolute rpath because ELPA is an external HPC dependency rather
    # than part of the wheel. The install script prints this prefix so users
    # know which directory must remain available at runtime.
    return Extension(
        "soap_tp.elpa_bindings",
        sources=["src/soap_tp/csrc/elpa_bindings.cpp"],
        include_dirs=[PYBIND11_INCLUDE, str(include_candidates[0])],
        library_dirs=[str(library_dir)],
        libraries=["elpa"],
        language="c++",
        extra_compile_args=["-std=c++17"],
        extra_link_args=[f"-Wl,-rpath,{library_dir}"],
    )


def slate_extension() -> Extension:
    prefix_value = os.environ.get("SLATE_PREFIX")
    if not prefix_value:
        raise RuntimeError(
            "SLATE_PREFIX is required when building the SLATE binding. "
            "Run ./scripts/install.sh <cpu|cuda|rocm> instead of invoking "
            "the native build directly."
        )

    prefix = Path(prefix_value).expanduser().resolve()
    include_dir = prefix / "include"
    if not (include_dir / "slate/slate.hh").is_file():
        raise RuntimeError(f"Could not find slate/slate.hh under {include_dir}")

    library_candidates = [prefix / "lib", prefix / "lib64"]
    library_dir = next(
        (
            path
            for path in library_candidates
            if path.is_dir() and any(path.glob("libslate.*"))
        ),
        None,
    )
    if library_dir is None:
        raise RuntimeError(f"Could not find libslate under {prefix}")

    profile = os.environ.get("SLATE_PROFILE", "cpu").lower()
    if profile not in {"cpu", "cuda", "rocm"}:
        raise RuntimeError("SLATE_PROFILE must be one of: cpu, cuda, rocm")

    define_macros = []
    if profile == "cuda":
        define_macros.append(("SOAP_TP_SLATE_WITH_CUDA", "1"))
    elif profile == "rocm":
        define_macros.append(("SOAP_TP_SLATE_WITH_ROCM", "1"))

    # Link SLATE's BLAS++ and LAPACK++ dependencies explicitly so the extension
    # resolves the same native stack selected by SLATE_PREFIX.
    return Extension(
        "soap_tp.slate_bindings",
        sources=["src/soap_tp/csrc/slate_bindings.cpp"],
        include_dirs=[PYBIND11_INCLUDE, str(include_dir)],
        library_dirs=[str(library_dir)],
        libraries=["slate", "lapackpp", "blaspp"],
        define_macros=define_macros,
        language="c++",
        extra_compile_args=["-std=c++17"],
        extra_link_args=[f"-Wl,-rpath,{library_dir}"],
    )


build_elpa_binding = enabled(
    "SOAP_TP_BUILD_ELPA_BINDINGS",
    default="ELPA_PREFIX" in os.environ,
)
build_slate_binding = enabled(
    "SOAP_TP_BUILD_SLATE_BINDINGS",
    default="SLATE_PREFIX" in os.environ,
)

extensions = []
if build_elpa_binding:
    extensions.append(elpa_extension())
if build_slate_binding:
    extensions.append(slate_extension())

setup(ext_modules=extensions)
