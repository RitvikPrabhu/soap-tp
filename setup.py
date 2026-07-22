"""Build the optional ELPA and SLATE Python extensions."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup

try:
    import pybind11

    PYBIND11_INCLUDE = pybind11.get_include()
except ImportError:
    # PyTorch vendors pybind11 headers, which remain a useful fallback for
    # environments where the standalone Python package is unavailable.
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


def slate_dependency_include_dirs(slate_library: Path) -> list[str]:
    """Find the transitive public headers required by SLATE's headers."""
    candidates: list[Path] = []

    def add_candidate(path: Path) -> None:
        path = path.expanduser()
        if path.is_dir() and path not in candidates:
            candidates.append(path)

    add_candidate(slate_library.parent.parent / "include")

    for variable in (
        "CPATH",
        "CPLUS_INCLUDE_PATH",
        "CMAKE_PREFIX_PATH",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
    ):
        for value in os.environ.get(variable, "").split(os.pathsep):
            if not value:
                continue
            path = Path(value)
            add_candidate(path)
            add_candidate(path / "include")
            for parent in tuple(path.parents)[:2]:
                add_candidate(parent / "include")

    dependency_tool = shutil.which("otool" if sys.platform == "darwin" else "ldd")
    if dependency_tool:
        if Path(dependency_tool).name == "otool":
            command = [dependency_tool, "-L", str(slate_library)]
        else:
            command = [dependency_tool, str(slate_library)]
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            for token in result.stdout.replace("=>", " ").split():
                dependency = Path(token.rstrip(":"))
                if not dependency.is_absolute() or not dependency.exists():
                    continue
                for parent in tuple(dependency.parents)[:4]:
                    add_candidate(parent / "include")

    include_dirs: list[str] = []
    missing = []
    for header in ("blas.hh", "lapack.hh"):
        include_dir = next(
            (candidate for candidate in candidates if (candidate / header).is_file()),
            None,
        )
        if include_dir is None:
            missing.append(header)
        elif str(include_dir) not in include_dirs:
            include_dirs.append(str(include_dir))

    if missing:
        raise RuntimeError(
            "The selected SLATE installation does not expose its required "
            f"public headers: {', '.join(missing)}"
        )
    return include_dirs


def elpa_extension() -> Extension:
    prefix_value = os.environ.get("ELPA_PREFIX")
    if not prefix_value:
        raise RuntimeError(
            "ELPA_PREFIX is required when building the ELPA binding. "
            "Build ELPA or select an existing installation first."
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
            "Build SLATE or select an existing installation first."
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

    slate_library = next(
        (
            path
            for path in sorted(library_dir.glob("libslate.*"))
            if path.is_file() and (".so" in path.name or path.suffix == ".dylib")
        ),
        None,
    )
    if slate_library is None:
        raise RuntimeError(f"Could not find a shared libslate under {prefix}")
    dependency_include_dirs = [
        path
        for path in slate_dependency_include_dirs(slate_library)
        if path != str(include_dir)
    ]

    profile = os.environ.get("SLATE_PROFILE", "cpu").lower()
    if profile not in {"cpu", "cuda", "rocm"}:
        raise RuntimeError("SLATE_PROFILE must be one of: cpu, cuda, rocm")

    define_macros = []
    if profile == "cuda":
        define_macros.append(("SOAP_TP_SLATE_WITH_CUDA", "1"))
    elif profile == "rocm":
        define_macros.append(("SOAP_TP_SLATE_WITH_ROCM", "1"))

    # The binding calls SLATE, so it links only libslate. Its resolved dynamic
    # dependencies also identify the public BLAS++ and LAPACK++ header roots.
    return Extension(
        "soap_tp.slate_bindings",
        sources=["src/soap_tp/csrc/slate_bindings.cpp"],
        include_dirs=[
            PYBIND11_INCLUDE,
            str(include_dir),
            *dependency_include_dirs,
        ],
        library_dirs=[str(library_dir)],
        libraries=["slate"],
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
