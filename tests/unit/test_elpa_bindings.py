"""Tests for the ELPA Python binding."""

import ctypes
import importlib.util
import os
from pathlib import Path
import shutil
import socket
import subprocess
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

BLOCK_BOUNDARY_CASES = (
    {
        "name": "singleton_unit_block",
        "size": 1,
        "nev": 1,
        "nblk": 1,
        "matrix": "singleton",
    },
    {
        "name": "block_larger_than_matrix",
        "size": 2,
        "nev": 2,
        "nblk": 3,
        "matrix": "random",
    },
    {
        "name": "block_equals_matrix",
        "size": 3,
        "nev": 3,
        "nblk": 3,
        "matrix": "random",
    },
    {
        "name": "partial_boundary_block",
        "size": 4,
        "nev": 4,
        "nblk": 3,
        "matrix": "random",
    },
    {
        "name": "exact_multiple_of_block",
        "size": 6,
        "nev": 6,
        "nblk": 3,
        "matrix": "random",
    },
    {
        "name": "unit_blocks_known_order",
        "size": 4,
        "nev": 4,
        "nblk": 1,
        "matrix": "diagonal",
    },
)

REPEATED_EIGENVALUE_CASE = (
    {
        "name": "repeated_eigenvalues",
        "size": 4,
        "nev": 4,
        "nblk": 2,
        "matrix": "repeated",
    },
)

PARTIAL_EIGENVECTOR_CASE = (
    {
        "name": "partial_eigenvectors",
        "size": 5,
        "nev": 2,
        "nblk": 2,
        "matrix": "random",
    },
)

MULTIRANK_CASES = (
    {
        "name": "equal_shards_2x2_grid",
        "size": 8,
        "nev": 8,
        "nblk": 2,
        "process_grid": (2, 2),
    },
    {
        "name": "uneven_columns_1x4_grid",
        "size": 7,
        "nev": 5,
        "nblk": 2,
        "process_grid": (1, 4),
    },
    {
        "name": "uneven_rows_4x1_grid",
        "size": 7,
        "nev": 5,
        "nblk": 2,
        "process_grid": (4, 1),
    },
    {
        "name": "uneven_shards_2x2_grid",
        "size": 9,
        "nev": 9,
        "nblk": 2,
        "process_grid": (2, 2),
    },
)

MULTIRANK_WORKER = "SOAP_TP_ELPA_MULTIRANK_WORKER"
MULTIRANK_BINDING_PATH = "SOAP_TP_ELPA_BINDING_PATH"
MULTIRANK_LIBRARY_PATH = "SOAP_TP_ELPA_LIBRARY_PATH"


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


def _set_integer(elpa, handle, error, name, value):
    elpa.elpa_set_integer(
        handle, name.encode(), value, ctypes.byref(error)
    )
    assert error.value == ELPA_OK


def _make_symmetric_matrix(case):
    size = case["size"]
    if case["matrix"] == "singleton":
        matrix = torch.tensor([[2.5]], dtype=torch.float32)
    elif case["matrix"] == "diagonal":
        diagonal = torch.tensor([4.0, -2.0, 1.0, 3.0])
        matrix = torch.diag(diagonal)
    elif case["matrix"] == "repeated":
        matrix = torch.tensor(
            [
                [2.0, 1.0, 0.0, 0.0],
                [1.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 1.0],
                [0.0, 0.0, 1.0, 2.0],
            ],
            dtype=torch.float32,
        )
    else:
        generator = torch.Generator().manual_seed(1000 + size)
        random = torch.randn(size, size, generator=generator)
        matrix = random + random.T
    return matrix


def _assert_matches_torch_eigh(
    matrix,
    eigenvalues,
    eigenvectors,
    requested_count,
    case_name,
    atol,
    rtol,
):
    expected_values, expected_vectors = torch.linalg.eigh(matrix)
    torch.testing.assert_close(
        eigenvalues,
        expected_values,
        atol=atol,
        rtol=rtol,
        msg=case_name,
    )

    requested_vectors = eigenvectors[:, :requested_count]
    group_start = 0
    while group_start < requested_count:
        group_end = group_start + 1
        while group_end < requested_count and torch.isclose(
            expected_values[group_end],
            expected_values[group_start],
            atol=1e-5,
            rtol=1e-5,
        ):
            group_end += 1

        actual_subspace = requested_vectors[:, group_start:group_end]
        expected_subspace = expected_vectors[:, group_start:group_end]
        torch.testing.assert_close(
            actual_subspace @ actual_subspace.T,
            expected_subspace @ expected_subspace.T,
            atol=atol,
            rtol=rtol,
            msg=case_name,
        )
        group_start = group_end

    torch.testing.assert_close(
        matrix @ requested_vectors,
        requested_vectors * eigenvalues[:requested_count],
        atol=atol,
        rtol=rtol,
        msg=case_name,
    )
    torch.testing.assert_close(
        requested_vectors.T @ requested_vectors,
        torch.eye(requested_count),
        atol=atol,
        rtol=rtol,
        msg=case_name,
    )


def _call_eigenvectors(
    binding,
    elpa,
    case,
    communicator,
    process_row,
    process_col,
    a,
    eigenvalues,
    eigenvectors,
):
    error = ctypes.c_int()
    initialized = False
    handle = None
    try:
        assert elpa.elpa_init(20260202) == ELPA_OK
        initialized = True
        handle = elpa.elpa_allocate(ctypes.byref(error))
        assert error.value == ELPA_OK
        assert handle is not None

        for name, value in (
            ("na", case["size"]),
            ("nev", case["nev"]),
            ("local_nrows", a.shape[0]),
            ("local_ncols", a.shape[1]),
            ("nblk", case["nblk"]),
            ("mpi_comm_parent", communicator.py2f()),
            ("process_row", process_row),
            ("process_col", process_col),
        ):
            _set_integer(elpa, handle, error, name, value)
        assert elpa.elpa_setup(handle) == ELPA_OK

        result = binding.elpa_eigenvectors_float(
            handle,
            a.data_ptr(),
            eigenvalues.data_ptr(),
            eigenvectors.data_ptr(),
        )
        assert result == ELPA_OK, (
            f"{case['name']}: {binding.elpa_error_string(result)}"
        )
    finally:
        if handle is not None:
            elpa.elpa_deallocate(handle, ctypes.byref(error))
        if initialized:
            elpa.elpa_uninit(ctypes.byref(error))


def _solve_eigenvector_case(binding, elpa, case):
    size = case["size"]
    matrix = _make_symmetric_matrix(case)
    a = torch.empty(size, size, dtype=torch.float32).T
    a.copy_(matrix)
    eigenvectors = torch.empty_like(a)
    eigenvectors.fill_(torch.nan)
    eigenvalues = torch.full((size,), torch.nan, dtype=torch.float32)

    _call_eigenvectors(
        binding,
        elpa,
        case,
        MPI.COMM_SELF,
        0,
        0,
        a,
        eigenvalues,
        eigenvectors,
    )

    _assert_matches_torch_eigh(
        matrix,
        eigenvalues,
        eigenvectors,
        case["nev"],
        case["name"],
        atol=1e-3,
        rtol=1e-3,
    )


def _run_eigenvector_cases(
    rank,
    world_size,
    port,
    binding_path,
    library_path,
    cases,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    try:
        binding = _load_binding(binding_path)
        elpa = _load_elpa(library_path)
        for case in cases[rank::world_size]:
            _solve_eigenvector_case(binding, elpa, case)
    finally:
        dist.destroy_process_group()


def _owned_global_indices(size, block_size, process, process_count):
    return [
        index
        for index in range(size)
        if (index // block_size) % process_count == process
    ]


def _solve_multirank_case(binding, elpa, case, rank, world_size):
    process_rows, process_cols = case["process_grid"]
    assert process_rows * process_cols == world_size
    process_row = rank % process_rows
    process_col = rank // process_rows

    size = case["size"]
    matrix = torch.empty((size, size), dtype=torch.float32)
    if rank == 0:
        generator = torch.Generator().manual_seed(2000 + size)
        random = torch.randn(size, size, generator=generator)
        matrix.copy_(random + random.T)
    dist.broadcast(matrix, src=0)

    global_rows = _owned_global_indices(
        size, case["nblk"], process_row, process_rows
    )
    global_cols = _owned_global_indices(
        size, case["nblk"], process_col, process_cols
    )
    row_indices = torch.tensor(global_rows, dtype=torch.long)
    col_indices = torch.tensor(global_cols, dtype=torch.long)
    local_values = matrix.index_select(0, row_indices).index_select(
        1, col_indices
    )

    a = torch.empty(
        (len(global_cols), len(global_rows)), dtype=torch.float32
    ).T
    a.copy_(local_values)
    eigenvectors = torch.empty_like(a)
    eigenvectors.fill_(torch.nan)
    eigenvalues = torch.full((size,), torch.nan, dtype=torch.float32)

    _call_eigenvectors(
        binding,
        elpa,
        case,
        MPI.COMM_WORLD,
        process_row,
        process_col,
        a,
        eigenvalues,
        eigenvectors,
    )

    local_result = (
        global_rows,
        global_cols,
        eigenvectors.contiguous(),
    )
    all_results = [None] * world_size
    dist.all_gather_object(all_results, local_result)

    global_vectors = torch.full(
        (size, size), torch.nan, dtype=torch.float32
    )
    for shard_rows, shard_cols, shard_vectors in all_results:
        shard_row_indices = torch.tensor(shard_rows, dtype=torch.long)
        shard_col_indices = torch.tensor(shard_cols, dtype=torch.long)
        global_vectors[
            shard_row_indices[:, None], shard_col_indices
        ] = shard_vectors

    requested_vectors = global_vectors[:, : case["nev"]]
    assert torch.isfinite(requested_vectors).all(), case["name"]
    _assert_matches_torch_eigh(
        matrix,
        eigenvalues,
        global_vectors,
        case["nev"],
        case["name"],
        atol=2e-3,
        rtol=2e-3,
    )


def _run_multirank_worker(binding_path, library_path):
    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()
    assert world_size == 4
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    try:
        binding = _load_binding(binding_path)
        elpa = _load_elpa(library_path)
        for case in MULTIRANK_CASES:
            _solve_multirank_case(binding, elpa, case, rank, world_size)
    finally:
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

    # Tests: the extension compiles, links against ELPA, and reports its backend.
    # Expected: the reported backend is one supported by this binding.
    def test_binding_loads(self):
        self.assertIn(
            self.binding.compiled_gpu_backend(),
            {"none", "cuda", "rocm", "sycl"},
        )

    def _run_cases(self, cases, world_size):
        mp.spawn(
            _run_eigenvector_cases,
            args=(
                world_size,
                _free_port(),
                self.binding.__file__,
                self.library_path,
                cases,
            ),
            nprocs=world_size,
            join=True,
        )

    # Tests: each raw-pointer argument enforces the documented integer-address API.
    # Expected: a non-convertible value raises TypeError before native code runs.
    def test_eigenvectors_float_rejects_non_integer_addresses(self):
        valid_arguments = {"handle": 1, "a": 1, "ev": 1, "q": 1}
        for argument in valid_arguments:
            with self.subTest(argument=argument):
                arguments = valid_arguments.copy()
                arguments[argument] = object()
                with self.assertRaises(TypeError):
                    self.binding.elpa_eigenvectors_float(**arguments)

    # Tests: single-rank solves at singleton, equal, partial, and oversized blocks.
    # Expected: eigenvalues and eigenspaces match torch.linalg.eigh.
    def test_eigenvectors_float_handles_block_boundaries(self):
        self._run_cases(BLOCK_BOUNDARY_CASES, world_size=4)

    # Tests: a symmetric matrix whose eigenvalues have multidimensional eigenspaces.
    # Expected: ELPA and torch.linalg.eigh produce the same spectral projectors.
    def test_eigenvectors_float_handles_repeated_eigenvalues(self):
        self._run_cases(REPEATED_EIGENVALUE_CASE, world_size=1)

    # Tests: a collective four-rank solve under 1x4, 2x2, and 4x1 ELPA grids.
    # Expected: gathered global eigenpairs match torch.linalg.eigh.
    def test_eigenvectors_float_multirank_block_cyclic(self):
        mpiexec = shutil.which("mpiexec")
        if mpiexec is None:
            self.skipTest("mpiexec is required for the multirank ELPA test")

        environment = os.environ.copy()
        environment.update(
            {
                MULTIRANK_WORKER: "1",
                MULTIRANK_BINDING_PATH: self.binding.__file__,
                MULTIRANK_LIBRARY_PATH: self.library_path,
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(_free_port()),
                "OMP_NUM_THREADS": "1",
            }
        )
        completed = subprocess.run(
            [
                mpiexec,
                "--oversubscribe",
                "-n",
                "4",
                sys.executable,
                str(Path(__file__).resolve()),
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=(
                f"multirank worker failed\nstdout:\n{completed.stdout}"
                f"\nstderr:\n{completed.stderr}"
            ),
        )

    # Tests: nev smaller than na while eigenvalue storage remains length na.
    # Expected: all values and requested eigenspaces match torch.linalg.eigh.
    def test_eigenvectors_float_returns_requested_eigenvectors(self):
        self._run_cases(PARTIAL_EIGENVECTOR_CASE, world_size=1)


if __name__ == "__main__":
    if os.environ.get(MULTIRANK_WORKER) == "1":
        _run_multirank_worker(
            os.environ[MULTIRANK_BINDING_PATH],
            os.environ[MULTIRANK_LIBRARY_PATH],
        )
    else:
        unittest.main()
