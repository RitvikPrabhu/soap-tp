"""Tests for the SLATE Python binding."""

import importlib.util
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import unittest

from mpi4py import MPI
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[2]
PROFILE = os.environ.get("SLATE_PROFILE", "cpu")
PREFIX = Path(
    os.environ.get(
        "SLATE_PREFIX",
        ROOT / "build" / "slate-install" / PROFILE,
    )
)
MULTIRANK_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "4"))

MULTIRANK_WORKER = "SOAP_TP_SLATE_MULTIRANK_WORKER"
BINDING_PATH = "SOAP_TP_SLATE_BINDING_PATH"

# These cases cover the meaningful relationships between the matrix dimension,
# block size, and local leading dimension without involving MPI distribution.
SINGLE_RANK_CASES = (
    ("singleton_unit_block", 1, 1, 0),
    ("block_larger_than_matrix", 1, 2, 0),
    ("block_equals_matrix", 2, 2, 0),
    ("partial_block_with_padded_lda", 3, 2, 2),
    ("exact_block_multiple", 4, 2, 0),
)


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _init_process_group(rank, world_size, port, backend):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def _device_for_profile():
    if PROFILE == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"{PROFILE} requires one visible GPU per MPI rank")
    return torch.device("cuda:0")


def _backend_for_device(device):
    return "nccl" if device.type == "cuda" else "gloo"


def _extension_from_path(path):
    name = Path(path).name.split(".", 1)[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _block_cyclic_indices(size, block_size, process, process_count):
    # SOAP-TP assigns 2D block-cyclic tiles using row-major process coordinates.
    return [
        index
        for index in range(size)
        if (index // block_size) % process_count == process
    ]


def _pack_local_matrix(matrix, rows, columns, lda_padding):
    # Transposing the allocation gives SLATE column-major local storage without
    # moving the logical matrix between ranks or devices.
    lda = max(1, len(rows) + lda_padding)
    local = torch.full(
        (max(1, len(columns)), lda),
        torch.nan,
        dtype=torch.float32,
        device=matrix.device,
    ).T
    if rows and columns:
        row_index = torch.tensor(rows, device=matrix.device)
        column_index = torch.tensor(columns, device=matrix.device)
        local[: len(rows), : len(columns)].copy_(
            matrix.index_select(0, row_index).index_select(1, column_index)
        )
    return local


def _reference_problem(size, device):
    index = torch.arange(size, dtype=torch.float32)
    distance = (index[:, None] - index[None, :]).abs()
    matrix = 0.25 / (distance + 1.0)
    matrix.diagonal().add_(size + index)
    orthogonal = torch.eye(size)
    return matrix.to(device), orthogonal.to(device)


def _torch_qr_reference(matrix, orthogonal):
    # This is the relevant portion of get_orthogonal_matrix_QR: order the old
    # basis by its estimated eigenvalues, then perform one power iteration + QR.
    estimated_eigenvalues = torch.diag(
        orthogonal.T @ matrix @ orthogonal
    )
    sort_index = torch.argsort(estimated_eigenvalues, descending=True)
    sorted_orthogonal = orthogonal[:, sort_index]
    expected, _ = torch.linalg.qr(matrix @ sorted_orthogonal)
    return sorted_orthogonal, expected


def _assert_matches_torch(case_name, actual, expected):
    # Householder QR may independently flip each output column. Align only
    # those signs before comparing values; a permutation or rotation still fails.
    assert torch.isfinite(actual).all(), f"{case_name}: non-finite Q"
    alignment = torch.diag(expected.T @ actual)
    signs = torch.where(alignment < 0, -1.0, 1.0)
    torch.testing.assert_close(
        actual * signs,
        expected,
        atol=3e-3,
        rtol=3e-3,
        msg=case_name,
    )
    torch.testing.assert_close(
        actual.T @ actual,
        torch.eye(actual.size(0)),
        atol=2e-3,
        rtol=2e-3,
        msg=case_name,
    )


def _check_power_iteration_case(
    binding,
    rank,
    process_grid,
    case,
    device,
):
    name, size, block_size, lda_padding = case
    case_name = (
        f"{name}: n={size}, block={block_size}, grid={process_grid}"
    )
    process_rows, process_cols = process_grid
    process_row = rank // process_cols
    process_col = rank % process_cols
    rows = _block_cyclic_indices(
        size, block_size, process_row, process_rows
    )
    columns = _block_cyclic_indices(
        size, block_size, process_col, process_cols
    )

    matrix, orthogonal = _reference_problem(size, device)
    sorted_orthogonal, expected = _torch_qr_reference(matrix, orthogonal)

    # A is declared lower-triangular storage. NaNs in the upper triangle make
    # an accidental upper-triangle read visible in the reconstructed result.
    stored_matrix = matrix.clone()
    upper = torch.triu(torch.ones_like(matrix, dtype=torch.bool), diagonal=1)
    stored_matrix.masked_fill_(upper, torch.nan)

    a = _pack_local_matrix(
        stored_matrix, rows, columns, lda_padding
    )
    q = _pack_local_matrix(
        sorted_orthogonal, rows, columns, lda_padding
    )
    work = torch.full_like(q, torch.nan, memory_format=torch.preserve_format)
    a_before = a.clone(memory_format=torch.preserve_format)

    if rank == 0 and os.environ.get(MULTIRANK_WORKER) == "1":
        print(f"SLATE start: {case_name}", flush=True)
    binding.slate_power_iteration_qr_float(
        a.data_ptr(),
        q.data_ptr(),
        work.data_ptr(),
        size,
        a.stride(1),
        block_size,
        process_rows,
        process_cols,
    )
    if rank == 0 and os.environ.get(MULTIRANK_WORKER) == "1":
        print(f"SLATE done: {case_name}", flush=True)
    actual = torch.zeros((size, size), dtype=torch.float32, device=device)
    owners = torch.zeros_like(actual)
    if rows and columns:
        row_index = torch.tensor(rows, device=device)
        column_index = torch.tensor(columns, device=device)
        actual[row_index[:, None], column_index] = q[
            : len(rows), : len(columns)
        ]
        owners[row_index[:, None], column_index] = 1

    a_unchanged = torch.all(
        (a == a_before) | (torch.isnan(a) & torch.isnan(a_before))
    ).to(dtype=torch.float32)
    dist.all_reduce(actual)
    dist.all_reduce(owners)
    dist.all_reduce(a_unchanged, op=dist.ReduceOp.MIN)
    if rank == 0 and os.environ.get(MULTIRANK_WORKER) == "1":
        print(f"Torch collectives done: {case_name}", flush=True)

    assert a_unchanged.item() == 1, f"{case_name}: A was modified"
    torch.testing.assert_close(owners, torch.ones_like(owners), msg=case_name)
    _assert_matches_torch(case_name, actual.cpu(), expected.cpu())


def _single_rank_worker(rank, world_size, port, binding_path):
    assert rank == MPI.COMM_WORLD.Get_rank() == 0
    assert world_size == MPI.COMM_WORLD.Get_size() == 1
    device = _device_for_profile()
    if device.type == "cuda":
        torch.cuda.set_device(device)
    _init_process_group(rank, world_size, port, _backend_for_device(device))

    try:
        binding = _extension_from_path(binding_path)
        for case in SINGLE_RANK_CASES:
            _check_power_iteration_case(
                binding, rank, (1, 1), case, device
            )
    finally:
        dist.destroy_process_group()


def _multirank_worker(binding_path, port):
    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()
    assert world_size == MULTIRANK_WORLD_SIZE > 1
    device = _device_for_profile()
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if rank == 0:
        print("Torch process-group start", flush=True)
    _init_process_group(rank, world_size, port, _backend_for_device(device))
    if rank == 0:
        print("Torch process-group ready", flush=True)

    cases = (
        # More ranks than tiles forces ranks with no local rows or columns.
        ("empty_local_owners", 1, 1, 0),
        # Every process-grid orientation receives equally sized local shards.
        ("equal_shards", 2 * world_size, 2, 0),
        # The final partial block produces uneven shards and padded local LDAs.
        ("uneven_shards", 2 * world_size + 1, 2, 1),
    )
    process_grids = [
        (rows, world_size // rows)
        for rows in range(1, world_size + 1)
        if world_size % rows == 0
    ]

    try:
        binding = _extension_from_path(binding_path)
        for case in cases:
            for process_grid in process_grids:
                _check_power_iteration_case(
                    binding,
                    rank,
                    process_grid,
                    case,
                    device,
                )
    finally:
        if rank == 0:
            print("Torch process-group destroy start", flush=True)
        dist.destroy_process_group()
        if rank == 0:
            print("Torch process-group destroyed", flush=True)


class TestSlateBinding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if PROFILE not in {"cpu", "cuda", "rocm"}:
            raise RuntimeError("SLATE_PROFILE must be cpu, cuda, or rocm")

        include_dir = PREFIX / "include"
        library_dir = next(
            (path for path in (PREFIX / "lib", PREFIX / "lib64") if path.is_dir()),
            None,
        )
        if not (include_dir / "slate/slate.hh").is_file() or library_dir is None:
            raise RuntimeError(f"SLATE is not installed under {PREFIX}")

        os.environ["CXX"] = "mpicxx"
        os.environ["PATH"] = f"{Path(sys.executable).parent}:{os.environ['PATH']}"
        os.environ["TORCH_EXTENSIONS_DIR"] = str(ROOT / "build/torch-extensions")

        extra_cflags = ["-O0"]
        if PROFILE == "cuda":
            extra_cflags.append("-DSOAP_TP_SLATE_WITH_CUDA=1")
        elif PROFILE == "rocm":
            extra_cflags.append("-DSOAP_TP_SLATE_WITH_ROCM=1")

        cls.binding = load(
            name=f"_soap_tp_slate_{PROFILE}_test",
            sources=[str(ROOT / "src/soap_tp/csrc/slate_bindings.cpp")],
            extra_include_paths=[str(include_dir)],
            extra_cflags=extra_cflags,
            extra_ldflags=[
                f"-L{library_dir}",
                "-lslate",
                "-llapackpp",
                "-lblaspp",
                f"-Wl,-rpath,{library_dir}",
            ],
        )

    def test_compiled_backend_matches_profile(self):
        # Detects a build that selected a different SLATE backend than requested.
        expected = {"cpu": "none", "cuda": "cuda", "rocm": "rocm"}[PROFILE]
        self.assertEqual(self.binding.compiled_gpu_backend(), expected)

    def test_single_rank_matches_torch_power_iteration_qr(self):
        # Compares SLATE with the sorted Torch reference at singleton, partial,
        # exact, oversized-block, and padded-leading-dimension boundaries.
        mp.spawn(
            _single_rank_worker,
            args=(1, _free_port(), self.binding.__file__),
            nprocs=1,
            join=True,
        )

    def test_multirank_row_major_result_matches_torch(self):
        # Exercises every legal process-grid orientation for the multirank world
        # and catches rank transposition, missing shards, and duplicates.
        self.assertGreater(MULTIRANK_WORLD_SIZE, 1)
        mpiexec = shutil.which("mpiexec")
        self.assertIsNotNone(mpiexec, "mpiexec is required")

        environment = os.environ.copy()
        environment.update(
            {
                MULTIRANK_WORKER: "1",
                BINDING_PATH: self.binding.__file__,
                "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else "lo",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(_free_port()),
                "MKL_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        command = [
            mpiexec,
            "--oversubscribe",
            "-n",
            str(MULTIRANK_WORLD_SIZE),
            sys.executable,
            str(Path(__file__).resolve()),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=240,
            )
        except subprocess.TimeoutExpired as error:
            self.fail(
                "MPI worker timed out\n"
                f"stdout:\n{error.stdout or ''}\n"
                f"stderr:\n{error.stderr or ''}"
            )
        self.assertEqual(
            completed.returncode,
            0,
            msg=(
                f"MPI worker failed\nstdout:\n{completed.stdout}"
                f"\nstderr:\n{completed.stderr}"
            ),
        )


if __name__ == "__main__":
    if os.environ.get(MULTIRANK_WORKER) == "1":
        _multirank_worker(
            os.environ[BINDING_PATH],
            int(os.environ["MASTER_PORT"]),
        )
    else:
        unittest.main()
