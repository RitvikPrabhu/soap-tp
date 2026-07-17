"""Unit tests for the distributed SLATE power-iteration QR binding."""

import importlib.util
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

from mpi4py import MPI
import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[2]
PROFILE = os.environ.get("SLATE_PROFILE", "cpu")
PREFIX = Path(
    os.environ.get(
        "SLATE_PREFIX",
        ROOT / "build" / "slate-install" / PROFILE,
    )
)
MULTIRANK_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "8"))

WORKER_MODE = "SOAP_TP_SLATE_WORKER_MODE"
WORKER_BINDING = "SOAP_TP_SLATE_WORKER_BINDING"
WORKER_WORLD_SIZE = "SOAP_TP_SLATE_WORKER_WORLD_SIZE"

# These cases exercise block size 1, n < block, n == block, n == block + 1,
# exact block multiples, partial boundary blocks, and padded local leading
# dimensions without adding MPI ownership as a second variable.
SINGLE_RANK_CASES = (
    ("singleton", 1, 1, 0),
    ("unit_tiles", 3, 1, 0),
    ("block_larger_than_matrix", 2, 3, 0),
    ("block_equals_matrix", 3, 3, 0),
    ("one_past_block", 4, 3, 0),
    ("exact_block_multiple", 6, 3, 0),
    ("partial_block_and_padded_lda", 5, 2, 2),
)


def _load_extension(path):
    name = Path(path).name.split(".", 1)[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _device_for_profile():
    if PROFILE == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"{PROFILE} requires exactly one visible GPU per MPI rank"
        )
    return torch.device("cuda:0")


def _owned_indices(size, block_size, process, process_count):
    # This is the row-major 2D block-cyclic ownership definition used by the
    # caller, independent of SLATE's internal column-major rank convention.
    return [
        index
        for index in range(size)
        if (index // block_size) % process_count == process
    ]


def _pack_column_major(matrix, rows, columns, lda_padding):
    # A transposed allocation has stride (1, lda), matching the documented
    # ScaLAPACK-style local buffer without copying data between ranks.
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
    # Increasing diagonal values make the estimated-eigenvalue ordering
    # nontrivial, while the positive diagonal dominance keeps QR full rank.
    index = torch.arange(size, dtype=torch.float32, device=device)
    distance = (index[:, None] - index[None, :]).abs()
    matrix = 0.25 / (distance + 1.0)
    matrix.diagonal().add_(size + index)

    # Adjacent Givens rotations provide a deterministic non-identity old basis
    # without using the QR routine that serves as the numerical oracle.
    orthogonal = torch.eye(size, dtype=torch.float32, device=device)
    for column in range(size - 1):
        angle = 0.19 * (column + 1)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        rotation = torch.eye(size, dtype=torch.float32, device=device)
        rotation[column, column] = cosine
        rotation[column, column + 1] = -sine
        rotation[column + 1, column] = sine
        rotation[column + 1, column + 1] = cosine
        orthogonal = orthogonal @ rotation
    return matrix, orthogonal


def _torch_reference(matrix, orthogonal):
    # This is the independent Torch behavior required by
    # get_orthogonal_matrix_QR: sort the old basis, take one power iteration,
    # then compute the reduced QR factor.
    estimated = torch.diag(orthogonal.T @ matrix @ orthogonal)
    order = torch.argsort(estimated, descending=True)
    sorted_orthogonal = orthogonal[:, order]
    expected, _ = torch.linalg.qr(matrix @ sorted_orthogonal)
    return sorted_orthogonal, expected


def _assert_same_q(actual, expected, case_name):
    # A full-rank real QR factor is unique up to independent column signs.
    # Align only those signs; permutations and arbitrary rotations still fail.
    assert torch.isfinite(actual).all(), f"{case_name}: Q is not finite"
    alignment = torch.diag(expected.T @ actual)
    assert torch.all(alignment.abs() > 0.9), (
        f"{case_name}: SLATE returned different QR columns"
    )
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
        msg=f"{case_name}: Q is not orthogonal",
    )


def _run_case(binding, rank, process_grid, case, device):
    name, size, block_size, lda_padding = case
    process_rows, process_columns = process_grid
    process_row = rank // process_columns
    process_column = rank % process_columns
    rows = _owned_indices(size, block_size, process_row, process_rows)
    columns = _owned_indices(
        size,
        block_size,
        process_column,
        process_columns,
    )
    case_name = (
        f"{name}: n={size}, block={block_size}, "
        f"grid={process_rows}x{process_columns}"
    )

    matrix, orthogonal = _reference_problem(size, device)
    sorted_orthogonal, expected = _torch_reference(matrix, orthogonal)

    # Only the lower triangle belongs to the symmetric input contract. NaNs in
    # the upper triangle turn any accidental upper-triangle read into a failure.
    stored_matrix = matrix.clone()
    upper = torch.triu(
        torch.ones_like(stored_matrix, dtype=torch.bool),
        diagonal=1,
    )
    stored_matrix.masked_fill_(upper, torch.nan)

    a = _pack_column_major(stored_matrix, rows, columns, lda_padding)
    q = _pack_column_major(sorted_orthogonal, rows, columns, lda_padding)
    work = torch.full_like(q, torch.nan, memory_format=torch.preserve_format)
    a_before = a.clone(memory_format=torch.preserve_format)

    logical_q = torch.zeros_like(q, dtype=torch.bool)
    if rows and columns:
        logical_q[: len(rows), : len(columns)] = True

    if rank == 0:
        print(f"start {case_name}", flush=True)
    binding.slate_power_iteration_qr_float(
        a.data_ptr(),
        q.data_ptr(),
        work.data_ptr(),
        size,
        a.stride(1),
        block_size,
        process_rows,
        process_columns,
    )

    # Reconstruct Q according to row-major ownership. A rank transposition,
    # omitted shard, or duplicated shard changes either Q or the owner counts.
    actual = torch.zeros((size, size), dtype=torch.float32, device=device)
    owners = torch.zeros_like(actual)
    if rows and columns:
        row_index = torch.tensor(rows, device=device)
        column_index = torch.tensor(columns, device=device)
        actual[row_index[:, None], column_index] = q[
            : len(rows), : len(columns)
        ]
        owners[row_index[:, None], column_index] = 1

    # Verification uses portable host MPI buffers, so GPU tests do not require
    # a CUDA-aware MPI installation. The binding itself still receives and
    # operates on the original device pointers.
    actual = actual.cpu()
    owners = owners.cpu()
    a_unchanged = torch.tensor(
        [
            torch.all(
                (a == a_before)
                | (torch.isnan(a) & torch.isnan(a_before))
            ).item()
        ],
        dtype=torch.int32,
    )
    q_padding_unchanged = torch.tensor(
        [torch.isnan(q[~logical_q]).all().item()],
        dtype=torch.int32,
    )
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, actual.numpy(), op=MPI.SUM)
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, owners.numpy(), op=MPI.SUM)
    MPI.COMM_WORLD.Allreduce(
        MPI.IN_PLACE,
        a_unchanged.numpy(),
        op=MPI.MIN,
    )
    MPI.COMM_WORLD.Allreduce(
        MPI.IN_PLACE,
        q_padding_unchanged.numpy(),
        op=MPI.MIN,
    )

    assert a_unchanged.item() == 1, f"{case_name}: A was modified"
    assert q_padding_unchanged.item() == 1, (
        f"{case_name}: Q padding was modified"
    )
    torch.testing.assert_close(
        owners,
        torch.ones_like(owners),
        msg=f"{case_name}: each entry must have exactly one owner",
    )
    _assert_same_q(actual, expected.cpu(), case_name)
    if rank == 0:
        print(f"pass {case_name}", flush=True)


def _worker():
    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()
    expected_world_size = int(os.environ[WORKER_WORLD_SIZE])
    assert world_size == expected_world_size

    device = _device_for_profile()
    if device.type == "cuda":
        torch.cuda.set_device(device)
    binding = _load_extension(os.environ[WORKER_BINDING])
    mode = os.environ[WORKER_MODE]
    if mode == "single":
        assert world_size == 1
        for case in SINGLE_RANK_CASES:
            _run_case(binding, rank, (1, 1), case, device)
        return

    assert mode == "multirank"
    assert world_size > 1
    process_grids = [
        (process_rows, world_size // process_rows)
        for process_rows in range(1, world_size + 1)
        if world_size % process_rows == 0
    ]
    cases = (
        # A single tile leaves most ranks with no local rows or columns.
        ("empty_local_shards", 1, 1, 0),
        # Eight tile rows and columns divide evenly over an eight-rank grid.
        ("equal_shards", 2 * world_size, 2, 0),
        # A ninth partial tile creates uneven ownership and padded LDAs.
        ("uneven_partial_shards", 2 * world_size + 1, 2, 1),
    )
    for case in cases:
        for process_grid in process_grids:
            _run_case(binding, rank, process_grid, case, device)

    # Repeating a prior decomposition after all communicator reorderings
    # catches stale SLATE state and premature communicator cleanup.
    repeat_grid = process_grids[len(process_grids) // 2]
    _run_case(
        binding,
        rank,
        repeat_grid,
        ("repeated_after_grid_changes", 2 * world_size, 2, 0),
        device,
    )


def _output_text(output):
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode(errors="replace")
    return output


class TestSlateBindings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if PROFILE not in {"cpu", "cuda", "rocm"}:
            raise RuntimeError("SLATE_PROFILE must be cpu, cuda, or rocm")

        include_directory = PREFIX / "include"
        library_directory = next(
            (
                path
                for path in (PREFIX / "lib", PREFIX / "lib64")
                if path.is_dir()
            ),
            None,
        )
        if (
            not (include_directory / "slate/slate.hh").is_file()
            or library_directory is None
        ):
            raise RuntimeError(f"SLATE is not installed under {PREFIX}")

        os.environ["CXX"] = "mpicxx"
        os.environ["PATH"] = (
            f"{Path(sys.executable).parent}:{os.environ['PATH']}"
        )
        os.environ["TORCH_EXTENSIONS_DIR"] = str(
            ROOT / "build/torch-extensions"
        )

        compile_flags = ["-O0"]
        if PROFILE == "cuda":
            compile_flags.append("-DSOAP_TP_SLATE_WITH_CUDA=1")
        elif PROFILE == "rocm":
            compile_flags.append("-DSOAP_TP_SLATE_WITH_ROCM=1")

        cls.binding = load(
            name=f"_soap_tp_slate_{PROFILE}_test",
            sources=[str(ROOT / "src/soap_tp/csrc/slate_bindings.cpp")],
            extra_include_paths=[str(include_directory)],
            extra_cflags=compile_flags,
            extra_ldflags=[
                f"-L{library_directory}",
                "-lslate",
                "-llapackpp",
                "-lblaspp",
                f"-Wl,-rpath,{library_directory}",
            ],
        )

    def _run_worker(self, mode, world_size):
        mpiexec = shutil.which("mpiexec")
        self.assertIsNotNone(mpiexec, "mpiexec is required")

        environment = os.environ.copy()
        environment.update(
            {
                WORKER_MODE: mode,
                WORKER_BINDING: self.binding.__file__,
                WORKER_WORLD_SIZE: str(world_size),
                "MKL_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        command = [
            mpiexec,
            "--oversubscribe",
            "-n",
            str(world_size),
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
                f"{mode} MPI worker timed out\n"
                f"stdout:\n{_output_text(error.stdout)}\n"
                f"stderr:\n{_output_text(error.stderr)}"
            )
        self.assertEqual(
            completed.returncode,
            0,
            msg=(
                f"{mode} MPI worker failed\nstdout:\n{completed.stdout}"
                f"\nstderr:\n{completed.stderr}"
            ),
        )

    def test_compiled_backend_matches_requested_profile(self):
        # This catches an extension compiled for host memory while the test
        # expects device pointers, or vice versa.
        expected = {"cpu": "none", "cuda": "cuda", "rocm": "rocm"}[PROFILE]
        self.assertEqual(self.binding.compiled_gpu_backend(), expected)

    def test_rejects_non_integer_pointer_arguments(self):
        # The public pybind API accepts raw addresses as integers. A Python
        # object must fail conversion before native code can dereference it.
        valid = {
            "a": 1,
            "q": 1,
            "work": 1,
            "n": 1,
            "lda": 1,
            "block_size": 1,
            "process_rows": 1,
            "process_cols": 1,
        }
        for pointer in ("a", "q", "work"):
            with self.subTest(pointer=pointer):
                arguments = valid.copy()
                arguments[pointer] = object()
                with self.assertRaises(TypeError):
                    self.binding.slate_power_iteration_qr_float(**arguments)

    def test_single_rank_matches_torch_across_block_boundaries(self):
        # One rank isolates numerical behavior and covers unit, oversized,
        # exact, partial, and padded block-storage boundaries.
        self._run_worker("single", 1)

    def test_eight_ranks_match_torch_for_every_row_major_grid(self):
        # Eight ranks exercise 1x8, 2x4, 4x2, and 8x1 row-major grids. The
        # cases include equal, uneven, partial, and empty local ownership.
        self.assertEqual(
            MULTIRANK_WORLD_SIZE,
            8,
            "the multirank SLATE contract test requires eight ranks",
        )
        self._run_worker("multirank", MULTIRANK_WORLD_SIZE)


if __name__ == "__main__":
    if WORKER_MODE in os.environ:
        _worker()
    else:
        unittest.main()
