"""Tests for the ELPA Python binding."""

import ctypes
import importlib.util
import os
from pathlib import Path
import socket
import sys
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from mpi4py import MPI
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[2]
PROFILE = os.environ.get("ELPA_PROFILE", "cpu")
PREFIX = ROOT / "build" / "elpa-install" / PROFILE
ELPA_OK = 0


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _load_binding(path):
    name = Path(path).name.split(".", 1)[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_elpa(path):
    elpa = ctypes.CDLL(path)
    error_pointer = ctypes.POINTER(ctypes.c_int)
    elpa.elpa_init.argtypes = [ctypes.c_int]
    elpa.elpa_init.restype = ctypes.c_int
    elpa.elpa_allocate.argtypes = [error_pointer]
    elpa.elpa_allocate.restype = ctypes.c_void_p
    elpa.elpa_set_integer.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int,
        error_pointer,
    ]
    elpa.elpa_setup.argtypes = [ctypes.c_void_p]
    elpa.elpa_setup.restype = ctypes.c_int
    elpa.elpa_deallocate.argtypes = [ctypes.c_void_p, error_pointer]
    elpa.elpa_uninit.argtypes = [error_pointer]
    return elpa


def _run_eigenvector_test(rank, world_size, port, binding_path, library_path):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    binding = _load_binding(binding_path)
    elpa = _load_elpa(library_path)
    error = ctypes.c_int()
    assert elpa.elpa_init(20260202) == ELPA_OK
    handle = elpa.elpa_allocate(ctypes.byref(error))
    assert error.value == ELPA_OK

    size = 7 + rank
    generator = torch.Generator().manual_seed(rank)
    random = torch.randn(size, size, generator=generator)
    matrix = random + random.T
    a = torch.empty(size, size).T
    a.copy_(matrix)
    eigenvectors = torch.empty_like(a)
    eigenvalues = torch.empty(size)

    def set_integer(name, value):
        elpa.elpa_set_integer(
            handle, name.encode(), value, ctypes.byref(error)
        )
        assert error.value == ELPA_OK

    try:
        for name, value in (
            ("na", size),
            ("nev", size),
            ("local_nrows", size),
            ("local_ncols", size),
            ("nblk", 1),
            ("mpi_comm_parent", MPI.COMM_SELF.py2f()),
            ("process_row", 0),
            ("process_col", 0),
        ):
            set_integer(name, value)
        assert elpa.elpa_setup(handle) == ELPA_OK
        assert binding.elpa_eigenvectors_float(
            handle,
            a.data_ptr(),
            eigenvalues.data_ptr(),
            eigenvectors.data_ptr(),
        ) == ELPA_OK

        expected_values, expected_vectors = torch.linalg.eigh(matrix)
        torch.testing.assert_close(eigenvalues, expected_values, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            torch.abs(eigenvectors.T @ expected_vectors),
            torch.eye(size),
            atol=1e-3,
            rtol=1e-3,
        )
        dist.barrier()
    finally:
        elpa.elpa_deallocate(handle, ctypes.byref(error))
        elpa.elpa_uninit(ctypes.byref(error))
        dist.destroy_process_group()


class TestElpaBinding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        include_dir = next((PREFIX / "include").glob("elpa-*"))
        library_dir = PREFIX / "lib"
        os.environ["CXX"] = "mpicxx"
        os.environ["PATH"] = f"{Path(sys.executable).parent}:{os.environ['PATH']}"
        os.environ["TORCH_EXTENSIONS_DIR"] = str(ROOT / "build/torch-extensions")
        cls.binding = load(
            name=f"_soap_tp_elpa_{PROFILE}_test",
            sources=[str(ROOT / "src/soap_tp/csrc/elpa_bindings.cpp")],
            extra_include_paths=[str(include_dir)],
            extra_cflags=["-O0"],
            extra_ldflags=[
                f"-L{library_dir}",
                "-lelpa",
                f"-Wl,-rpath,{library_dir}",
            ],
        )
        cls.library_path = next(
            str(path)
            for pattern in ("libelpa.*.dylib", "libelpa.so.*", "libelpa.so")
            for path in library_dir.glob(pattern)
        )

    def test_binding_loads(self):
        self.assertIn(
            self.binding.compiled_gpu_backend(),
            {"none", "cuda", "rocm", "sycl"},
        )

    def test_eigenvectors_float_matches_torch_eigh(self):
        for world_size in (1, 2, 3, 4):
            with self.subTest(world_size=world_size):
                mp.spawn(
                    _run_eigenvector_test,
                    args=(
                        world_size,
                        _free_port(),
                        self.binding.__file__,
                        self.library_path,
                    ),
                    nprocs=world_size,
                    join=True,
                )


if __name__ == "__main__":
    unittest.main()
