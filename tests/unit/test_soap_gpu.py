"""Eight-GPU version of ``test_soap.py``, launched directly with ``srun``.

For every runnable shape and TP layout, this test compares six distributed
``soap_step`` updates with the original SOAP implementation, records the
relative errors, and writes the same CSV-and-plot output as the CPU test.
The refresh interval deliberately exercises the distributed SLATE QR path.

The original implementation keeps a full parameter and full SOAP state on
every GPU.  Before starting a configuration, the test estimates its peak HBM
requirement from the matrix dimensions and current process grid.  A
configuration that cannot fit is reported and recorded as skipped instead of
failing with an out-of-memory error.
"""

from __future__ import annotations

import csv
import gc
import math
import os
from pathlib import Path
import runpy
import socket
import time
from datetime import timedelta
from typing import Any

from mpi4py import MPI
import torch
import torch.distributed as dist

from soap_tp import soap_step
from soap_tp.ops._utils import local_2d_block_cyclic_shape


ROOT = Path(__file__).resolve().parents[2]
OriginalSOAP = runpy.run_path(ROOT / "tests" / "reference" / "soap.py")["SOAP"]

SHAPES = (
    (8, 8),
    (8, 16),
    (16, 8),
    (16, 16),
    (16, 24),
    (24, 16),
    (32, 48),
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (3072, 3072),
    (8192, 3072),
    (11008, 4096),
    (18000, 18000),
    (32768, 8192),
    (49152, 12288),
    (65536, 16384),
)

EXPECTED_WORLD_SIZE = 8
GRADIENT_COUNT = 6
BASIS_REFRESH_INTERVAL = 2
EXPECTED_QR_CALLS = 2 * ((GRADIENT_COUNT - 1) // BASIS_REFRESH_INTERVAL)
MAX_RELATIVE_ERROR = 5e-3
LAYOUTS = ("row", "column")
MAX_BLOCK_SIZE = int(os.environ.get("SOAP_TP_GPU_BLOCK_SIZE", "256"))
MEMORY_USE_FRACTION = float(
    os.environ.get("SOAP_TP_GPU_MEMORY_USE_FRACTION", "0.80")
)
MEMORY_SAFETY_FACTOR = float(
    os.environ.get("SOAP_TP_GPU_MEMORY_SAFETY_FACTOR", "1.25")
)
OUTPUT_DIRECTORY = Path(
    os.environ.get("SOAP_TP_GPU_OUTPUT", "outputs/gpu_soap_test")
)

RESULT_FIELDS = (
    "status",
    "reason",
    "ranks",
    "rows",
    "columns",
    "shard_layout",
    "block_size",
    "gradients",
    "qr_calls",
    "expected_qr_calls",
    "seconds",
    "peak_gpu_gib",
    "estimated_peak_gpu_gib",
    "free_gpu_gib",
    "max_relative_l2_error",
)
ERROR_FIELDS = (
    "ranks",
    "rows",
    "columns",
    "shard_layout",
    "gradient",
    "relative_l2_error",
    "difference_percent",
)


class _CountingSlateBinding:
    """Forward SLATE calls while counting calls to its native QR routine."""

    def __init__(self, binding: Any) -> None:
        self._binding = binding
        self.qr_calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._binding, name)

    def slate_qr_float(self, *args: Any, **kwargs: Any) -> Any:
        self.qr_calls += 1
        return self._binding.slate_qr_float(*args, **kwargs)


def _block_size(
    shape: tuple[int, int],
    process_grid: tuple[int, int],
) -> int:
    # ELPA requires every rank to own at least one block row and block column
    # for both square preconditioners.
    ownership_limit = min(shape) // max(process_grid)
    return max(1, min(MAX_BLOCK_SIZE, ownership_limit))


def _initialize_torch_distributed(
    rank: int,
    world_size: int,
) -> None:
    if rank == 0:
        address = os.environ.get("MASTER_ADDR") or socket.gethostname()
        if "MASTER_PORT" in os.environ:
            port = int(os.environ["MASTER_PORT"])
        else:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("", 0))
                port = int(sock.getsockname()[1])
        rendezvous = address, port
    else:
        rendezvous = None
    address, port = MPI.COMM_WORLD.bcast(rendezvous, root=0)

    os.environ["MASTER_ADDR"] = address
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # Slurm exposes one accelerator to each task, so its index in that task's
    # visibility namespace is always zero.
    os.environ["LOCAL_RANK"] = "0"

    timeout_minutes = int(os.environ.get("SOAP_TP_GPU_TIMEOUT_MINUTES", "120"))
    dist.init_process_group(
        "nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=timeout_minutes),
    )


def _make_gradient(
    shape: tuple[int, int],
    gradient_index: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if gradient_index == 0:
        gradient = torch.zeros(shape, dtype=torch.float32, device=device)
        diagonal_size = min(shape)
        diagonal_indices = torch.arange(diagonal_size, device=device)
        gradient[diagonal_indices, diagonal_indices] = torch.linspace(
            2.0,
            0.5,
            diagonal_size,
            dtype=torch.float32,
            device=device,
        )
        return gradient

    return torch.randn(
        shape,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )


def _all_ranks_true(value: bool, device: torch.device) -> bool:
    result = torch.tensor(int(value), dtype=torch.int32, device=device)
    dist.all_reduce(result, op=dist.ReduceOp.MIN)
    return bool(result.item())


def _memory_feasibility(
    shape: tuple[int, int],
    shard_dim: int,
    block_size: int,
    process_grid: tuple[int, int],
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[str | None, float, float]:
    if shape[shard_dim] % world_size:
        reason = (
            f"global dimension {shard_dim}={shape[shard_dim]} is not "
            f"divisible by {world_size} TP ranks"
        )
        return reason, 0.0, 0.0

    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    rows, columns = shape
    matrix_elements = rows * columns
    left_square = rows * rows
    right_square = columns * columns
    reference_elements = (
        10 * matrix_elements
        + 2 * (left_square + right_square)
        + 4 * max(left_square, right_square)
    )

    local_elements = []
    for matrix_shape in (shape, (rows, rows), (columns, columns)):
        local_rows, local_columns = local_2d_block_cyclic_shape(
            matrix_shape,
            block_size,
            process_grid,
            rank,
        )
        local_elements.append(max(1, local_rows) * max(1, local_columns))
    local_matrix, local_left, local_right = local_elements
    distributed_elements = 5 * (local_left + local_right) + 6 * local_matrix
    required = math.ceil(
        4
        * (reference_elements + distributed_elements)
        * MEMORY_SAFETY_FACTOR
    )
    free, _total = torch.cuda.mem_get_info(device)
    required_max = torch.tensor(required, dtype=torch.int64, device=device)
    free_min = torch.tensor(free, dtype=torch.int64, device=device)
    dist.all_reduce(required_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(free_min, op=dist.ReduceOp.MIN)

    required_bytes = int(required_max.item())
    free_bytes = int(free_min.item())
    usable_bytes = int(free_bytes * MEMORY_USE_FRACTION)
    required_gib = required_bytes / 2**30
    free_gib = free_bytes / 2**30
    if required_bytes > usable_bytes:
        reason = (
            "configuration is not possible with the available GPU memory: "
            f"estimated peak {required_gib:.2f} GiB per rank, "
            f"minimum free {free_gib:.2f} GiB per rank "
            f"({MEMORY_USE_FRACTION:.0%} usable)"
        )
        return reason, required_gib, free_gib
    return None, required_gib, free_gib


def _relative_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    difference_squared = (actual - expected).square().sum(dtype=torch.float64)
    expected_squared = expected.square().sum(dtype=torch.float64)
    dist.all_reduce(difference_squared, op=dist.ReduceOp.SUM)
    dist.all_reduce(expected_squared, op=dist.ReduceOp.SUM)
    denominator = max(float(expected_squared.item()), 1e-60)
    return math.sqrt(float(difference_squared.item()) / denominator)


def _result_row(
    world_size: int,
    shape: tuple[int, int],
    layout: str,
    block_size: int,
) -> dict[str, object]:
    return {
        "status": "pending",
        "reason": "not started",
        "ranks": world_size,
        "rows": shape[0],
        "columns": shape[1],
        "shard_layout": layout,
        "block_size": block_size,
        "gradients": "",
        "qr_calls": "",
        "expected_qr_calls": EXPECTED_QR_CALLS,
        "seconds": "",
        "peak_gpu_gib": "",
        "estimated_peak_gpu_gib": "",
        "free_gpu_gib": "",
        "max_relative_l2_error": "",
    }


def _run_case(
    shape: tuple[int, int],
    shard_dim: int,
    case_index: int,
    process_grid: tuple[int, int],
    rank: int,
    world_size: int,
    device: torch.device,
    elpa_binding: Any,
    slate_binding: Any,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    layout = LAYOUTS[shard_dim]
    block_size = _block_size(shape, process_grid)
    result = _result_row(world_size, shape, layout, block_size)
    skip_reason, estimated_gib, free_gib = _memory_feasibility(
        shape,
        shard_dim,
        block_size,
        process_grid,
        rank,
        world_size,
        device,
    )
    result.update(
        estimated_peak_gpu_gib=estimated_gib,
        free_gpu_gib=free_gib,
    )
    if skip_reason is not None:
        result.update(status="skipped", reason=skip_reason)
        if rank == 0:
            print(
                f"SKIP shape={shape} layout={layout}: {skip_reason}",
                flush=True,
            )
        return result, []

    if rank == 0:
        print(
            f"START shape={shape} layout={layout} block={block_size} "
            f"gradients={GRADIENT_COUNT} qr_expected={EXPECTED_QR_CALLS} "
            f"estimated_peak_gpu_gib={estimated_gib:.2f}",
            flush=True,
        )

    torch.cuda.reset_peak_memory_stats(device)
    dist.barrier()
    started = time.perf_counter()

    parameter: torch.nn.Parameter | None = None
    original: Any | None = None
    state: dict[str, object] = {}
    try:
        try:
            parameter = torch.nn.Parameter(
                torch.zeros(shape, dtype=torch.float32, device=device)
            )
            original = OriginalSOAP(
                [parameter],
                lr=1.0,
                betas=(0.8, 0.9),
                shampoo_beta=0.75,
                eps=1e-6,
                weight_decay=0.0,
                precondition_frequency=BASIS_REFRESH_INTERVAL,
                max_precond_dim=max(shape),
            )
        except torch.OutOfMemoryError:
            setup_succeeded = False
        else:
            setup_succeeded = True

        if not _all_ranks_true(setup_succeeded, device):
            reason = "runtime GPU OOM while allocating the full reference parameter"
            result.update(status="skipped", reason=reason)
            if rank == 0:
                print(f"SKIP shape={shape} layout={layout}: {reason}", flush=True)
            return result, []

        assert parameter is not None
        assert original is not None
        counted_slate = _CountingSlateBinding(slate_binding)
        errors: list[float] = []
        error_records: list[dict[str, object]] = []
        generator = torch.Generator(device=device)
        generator.manual_seed(90210 + case_index)

        for gradient_index in range(GRADIENT_COUNT):
            try:
                gradient = _make_gradient(
                    shape,
                    gradient_index,
                    device,
                    generator,
                )
                parameter.grad = gradient
                before = parameter.detach().clone()
                original.step()
            except torch.OutOfMemoryError:
                reference_succeeded = False
            else:
                reference_succeeded = True

            if not _all_ranks_true(reference_succeeded, device):
                reason = (
                    "runtime GPU OOM in the full reference SOAP calculation at "
                    f"gradient {gradient_index + 1}"
                )
                result.update(
                    status="skipped",
                    reason=reason,
                    gradients=gradient_index,
                    qr_calls=counted_slate.qr_calls,
                )
                if rank == 0:
                    print(
                        f"SKIP shape={shape} layout={layout}: {reason}",
                        flush=True,
                    )
                return result, []

            expected = before.sub_(parameter.detach())
            shard_size = shape[shard_dim] // world_size
            shard_start = rank * shard_size
            gradient_shard = gradient.narrow(
                shard_dim,
                shard_start,
                shard_size,
            ).contiguous()
            expected_shard = expected.narrow(
                shard_dim,
                shard_start,
                shard_size,
            )

            local_update = soap_step(
                gradient_shard,
                state,
                global_shape=shape,
                shard_dim=shard_dim,
                block_size=block_size,
                process_grid_shape=process_grid,
                preconditioner_beta=0.75,
                beta1=0.8,
                beta2=0.9,
                eps=1e-6,
                basis_refresh_interval=BASIS_REFRESH_INTERVAL,
                elpa_binding=elpa_binding,
                slate_binding=counted_slate,
            )
            error = _relative_error(local_update, expected_shard)
            errors.append(error)
            if rank == 0:
                error_records.append(
                    {
                        "ranks": world_size,
                        "rows": shape[0],
                        "columns": shape[1],
                        "shard_layout": layout,
                        "gradient": gradient_index + 1,
                        "relative_l2_error": error,
                        "difference_percent": error * 100,
                    }
                )

            parameter.grad = None
            del gradient, before, expected, gradient_shard, expected_shard
            del local_update

        torch.cuda.synchronize(device)
        dist.barrier()
        elapsed = time.perf_counter() - started
        peak_bytes = torch.cuda.max_memory_allocated(device)
        elapsed_max = torch.tensor(elapsed, dtype=torch.float64, device=device)
        peak_max = torch.tensor(peak_bytes, dtype=torch.int64, device=device)
        dist.all_reduce(elapsed_max, op=dist.ReduceOp.MAX)
        dist.all_reduce(peak_max, op=dist.ReduceOp.MAX)

        qr_was_exercised = _all_ranks_true(
            counted_slate.qr_calls == EXPECTED_QR_CALLS,
            device,
        )
        max_error = max(errors)
        reason = ""
        if not qr_was_exercised:
            reason = (
                f"expected {EXPECTED_QR_CALLS} native SLATE QR calls per rank, "
                f"observed {counted_slate.qr_calls} on rank {rank}"
            )
        elif max_error >= MAX_RELATIVE_ERROR:
            reason = (
                f"maximum relative error {max_error:.6e} is not below "
                f"{MAX_RELATIVE_ERROR:.6e}"
            )

        result.update(
            status="passed" if not reason else "failed",
            reason=reason,
            gradients=GRADIENT_COUNT,
            qr_calls=counted_slate.qr_calls,
            seconds=float(elapsed_max.item()),
            peak_gpu_gib=float(peak_max.item()) / 2**30,
            max_relative_l2_error=max_error,
        )
        if rank == 0:
            print(
                f"{result['status'].upper()} shape={shape} layout={layout} "
                f"qr_calls={counted_slate.qr_calls} "
                f"max_error={max_error:.3e} "
                f"seconds={result['seconds']:.3f} "
                f"peak_gpu_gib={result['peak_gpu_gib']:.3f}"
                + (f" reason={reason}" if reason else ""),
                flush=True,
            )
        return result, error_records
    finally:
        if parameter is not None:
            parameter.grad = None
        state.clear()
        parameter = None
        original = None
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()


def _write_plot(
    records: list[dict[str, object]],
    image_path: Path,
    world_size: int,
) -> None:
    """Write the same error plot as test_soap.py, labeled for eight ranks."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FuncFormatter

    shapes: list[tuple[int, int]] = []
    errors: dict[str, dict[tuple[int, int], float]] = {
        "row": {},
        "column": {},
    }
    for record in records:
        shape = (int(record["rows"]), int(record["columns"]))
        if shape not in shapes:
            shapes.append(shape)
        layout = record["shard_layout"]
        error_percent = float(record["relative_l2_error"]) * 100
        errors[layout][shape] = max(
            error_percent,
            errors[layout].get(shape, 0.0),
        )

    labels = [f"{rows}×{columns}" for rows, columns in shapes]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for layout, label in (
        ("row", f"Rows sharded across {world_size} ranks"),
        ("column", f"Columns sharded across {world_size} ranks"),
    ):
        axis.plot(
            labels,
            [errors[layout].get(shape, math.nan) for shape in shapes],
            marker="o",
            label=label,
        )
    axis.set_title(
        f"Merged {world_size}-rank soap_step vs. original soap.py (GPU)"
    )
    axis.set_xlabel("Matrix shape (rows × columns)")
    axis.set_ylabel("Maximum relative update difference (%)")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.5f}%"))
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    temporary_path = image_path.with_name(
        f".{image_path.stem}.tmp{image_path.suffix}"
    )
    try:
        figure.savefig(temporary_path, dpi=160)
        os.replace(temporary_path, image_path)
    finally:
        plt.close(figure)
        temporary_path.unlink(missing_ok=True)


def _checkpoint(
    results: list[dict[str, object]],
    error_records: list[dict[str, object]],
    world_size: int,
    *,
    refresh_plot: bool,
) -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    results_path = OUTPUT_DIRECTORY / "results.csv"
    errors_path = OUTPUT_DIRECTORY / "errors.csv"
    for path, fields, records in (
        (results_path, RESULT_FIELDS, results),
        (errors_path, ERROR_FIELDS, error_records),
    ):
        temporary_path = path.with_name(f".{path.name}.tmp")
        try:
            with temporary_path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(records)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    plot_path = OUTPUT_DIRECTORY / "errors.png"
    if error_records and refresh_plot:
        _write_plot(error_records, plot_path, world_size)
    elif not error_records:
        plot_path.unlink(missing_ok=True)

    status_counts = {
        status: sum(result["status"] == status for result in results)
        for status in ("pending", "running", "passed", "failed", "skipped")
    }
    plot_status = str(plot_path) if plot_path.exists() else "not available yet"
    print(
        "CHECKPOINT "
        + " ".join(f"{status}={count}" for status, count in status_counts.items())
        + f" error_points={len(error_records)} plot={plot_status}",
        flush=True,
    )


def main() -> None:
    from soap_tp import elpa_bindings, slate_bindings

    if not 0.0 < MEMORY_USE_FRACTION <= 1.0:
        raise ValueError("SOAP_TP_GPU_MEMORY_USE_FRACTION must be in (0, 1]")
    if MEMORY_SAFETY_FACTOR < 1.0:
        raise ValueError("SOAP_TP_GPU_MEMORY_SAFETY_FACTOR must be at least 1")

    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()
    if world_size != EXPECTED_WORLD_SIZE:
        raise RuntimeError(
            f"test_soap_gpu.py requires {EXPECTED_WORLD_SIZE} MPI ranks; "
            f"got {world_size}. Launch it with the provided srun command."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("no GPU is visible to this MPI rank")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "each MPI rank must see exactly one GPU; use "
            "--gpus-per-task=1 --gpu-bind=closest"
        )

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    if torch.version.hip is not None:
        expected_backend = "rocm"
    elif torch.version.cuda is not None:
        expected_backend = "cuda"
    else:
        raise RuntimeError("test_soap_gpu.py requires a CUDA or ROCm PyTorch build")
    actual_backends = {
        "ELPA": elpa_bindings.compiled_gpu_backend(),
        "SLATE": slate_bindings.compiled_gpu_backend(),
    }
    for library, actual_backend in actual_backends.items():
        if actual_backend != expected_backend:
            raise RuntimeError(
                f"{library} binding reports {actual_backend!r}; "
                f"expected {expected_backend!r}"
            )

    _initialize_torch_distributed(rank, world_size)
    try:
        grid_rows = math.isqrt(world_size)
        while world_size % grid_rows:
            grid_rows -= 1
        process_grid = (grid_rows, world_size // grid_rows)
        if rank == 0:
            print(
                f"GPU SOAP comparison: ranks={world_size} grid={process_grid} "
                f"backend={expected_backend} device={torch.cuda.get_device_name(0)} "
                f"memory_fraction={MEMORY_USE_FRACTION:.0%} "
                f"memory_safety_factor={MEMORY_SAFETY_FACTOR:.2f}",
                flush=True,
            )

        attempts = [
            (case_index, shape, shard_dim)
            for case_index, shape in enumerate(SHAPES)
            for shard_dim in (0, 1)
        ]
        results = [
            _result_row(
                world_size,
                shape,
                LAYOUTS[shard_dim],
                _block_size(shape, process_grid),
            )
            for _, shape, shard_dim in attempts
        ]
        error_records: list[dict[str, object]] = []
        if rank == 0:
            _checkpoint(
                results,
                error_records,
                world_size,
                refresh_plot=False,
            )
        dist.barrier()

        for attempt_index, (case_index, shape, shard_dim) in enumerate(attempts):
            results[attempt_index].update(
                status="running",
                reason="attempt started but has not completed",
            )
            if rank == 0:
                _checkpoint(
                    results,
                    error_records,
                    world_size,
                    refresh_plot=False,
                )
            dist.barrier()

            result, case_errors = _run_case(
                shape,
                shard_dim,
                case_index,
                process_grid,
                rank,
                world_size,
                device,
                elpa_bindings,
                slate_bindings,
            )
            results[attempt_index] = result
            error_records.extend(case_errors)
            if rank == 0:
                _checkpoint(
                    results,
                    error_records,
                    world_size,
                    refresh_plot=bool(case_errors),
                )
            dist.barrier()

        failed = [result for result in results if result["status"] == "failed"]
        skipped = [result for result in results if result["status"] == "skipped"]
        passed = [result for result in results if result["status"] == "passed"]
        if rank == 0:
            print(
                f"SUMMARY passed={len(passed)} skipped={len(skipped)} "
                f"failed={len(failed)}",
                flush=True,
            )
            print(f"Results: {OUTPUT_DIRECTORY / 'results.csv'}", flush=True)
            print(f"Errors: {OUTPUT_DIRECTORY / 'errors.csv'}", flush=True)
            if error_records:
                print(f"Plot: {OUTPUT_DIRECTORY / 'errors.png'}", flush=True)

        if failed:
            descriptions = "; ".join(
                f"({result['rows']}, {result['columns']}) "
                f"{result['shard_layout']}: {result['reason']}"
                for result in failed
            )
            raise AssertionError(f"GPU SOAP comparison failures: {descriptions}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
