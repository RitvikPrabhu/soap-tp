import csv
import json
import math
import os
from pathlib import Path
import runpy
import signal
import shutil
import socket
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist

from soap_tp import soap_step

try:
    from mpi4py import MPI
    from soap_tp import elpa_bindings, slate_bindings
except ImportError as error:
    DISTRIBUTED_IMPORT_ERROR = error
    MPI = None
    elpa_bindings = None
    slate_bindings = None
else:
    DISTRIBUTED_IMPORT_ERROR = None


ROOT = Path(__file__).resolve().parents[2]

SEED = 42
RTOL = 1e-5
MPI_WORKER = "SOAP_TP_TEST_MPI_WORKER"
MPI_RANKS = int(os.environ.get("SOAP_TP_TEST_MPI_RANKS", "4"))
MPI_WORKER_SHAPE = "SOAP_TP_TEST_MPI_SHAPE"
MPI_WORKER_SHARD_DIM = "SOAP_TP_TEST_MPI_SHARD_DIM"

COMPARISON_ITERATIONS = 10
COMPARISON_BLOCK_SIZE = 2
PRECONDITIONER_BETA = 0.75
BETA1 = 0.8
BETA2 = 0.9
EPS = 1e-6
BASIS_REFRESH_INTERVAL = 2
COMPARISON_SHAPES = (
    # (8, 8),
    (20, 20),
    # (12, 12),
    # (12, 20),
    # (20, 12),
)


def _free_port(address="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((address, 0))
        return sock.getsockname()[1]


def _process_grid_shape(world_size):
    process_rows = math.isqrt(world_size)
    while world_size % process_rows:
        process_rows -= 1
    return process_rows, world_size // process_rows


def _initialize_distributed(device="cpu"):
    """Initialize Torch WORLD with the rank layout of MPI_COMM_WORLD.

    MPI itself is initialized by mpi4py after mpiexec launches the workers.
    soap_step uses this default Torch group, while ELPA and SLATE use the
    matching MPI communicator.
    """

    if DISTRIBUTED_IMPORT_ERROR is not None:
        raise RuntimeError(
            "mpi4py and the native ELPA/SLATE bindings are required."
        ) from DISTRIBUTED_IMPORT_ERROR
    if not MPI.Is_initialized() or MPI.Is_finalized():
        raise RuntimeError("MPI must be initialized and active before soap_step.")

    device = torch.device(device)
    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()

    if dist.is_initialized():
        torch_world = dist.get_rank(), dist.get_world_size()
        mpi_world = rank, world_size
        if torch_world != mpi_world:
            raise RuntimeError(
                "Torch and MPI worlds must have identical ranks and sizes: "
                f"Torch {torch_world}, MPI {mpi_world}."
            )
        return rank, world_size, _process_grid_shape(world_size)

    if device.type == "cuda":
        if device.index is None:
            raise ValueError("A concrete CUDA device is required for each MPI rank.")
        torch.cuda.set_device(device)
        backend = "nccl"
    elif device.type == "cpu":
        backend = "gloo"
    else:
        raise ValueError(f"Unsupported distributed test device: {device}.")

    if rank == 0:
        address = os.environ.get("MASTER_ADDR", "127.0.0.1")
        if "MASTER_PORT" in os.environ:
            port = int(os.environ["MASTER_PORT"])
        else:
            port = _free_port(address)
        rendezvous = address, port
    else:
        rendezvous = None

    address, port = MPI.COMM_WORLD.bcast(rendezvous, root=0)
    os.environ["MASTER_ADDR"] = address
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(
        backend,
        rank=rank,
        world_size=world_size,
    )
    try:
        torch_world = dist.get_rank(), dist.get_world_size()
        for name, binding in (
            ("ELPA", elpa_bindings),
            ("SLATE", slate_bindings),
        ):
            mpi_world = tuple(binding.mpi_world_rank_and_size())
            if torch_world != mpi_world:
                raise RuntimeError(
                    f"{name} uses MPI world {mpi_world}, but Torch uses {torch_world}."
                )
    except BaseException:
        dist.destroy_process_group()
        raise

    return rank, world_size, _process_grid_shape(world_size)


def _destroy_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# Make a tensor of shape (m, n) using normal distribution, mean 0 and std 2
def _make_gradient(
    m,
    n,
    dtype=torch.float32,
    seed=SEED,
    *,
    generator=None,
):
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return torch.normal(
        mean=0,
        std=2,
        size=(m, n),
        generator=generator,
        dtype=dtype,
    )


# def _make_gradient(
#     m,
#     n,
#     dtype=torch.float32,
#     seed=SEED,
#     *,
#     generator=None,
# ):
#     if generator is None:
#         generator = torch.Generator()
#         generator.manual_seed(seed)

#     return torch.randint(
#         low=-3,
#         high=4,  # Produces -3, -2, ..., 3
#         size=(m, n),
#         generator=generator,
#         dtype=torch.int64,
#     ).to(dtype)


def _normalize_shape(shape):
    try:
        dimensions = tuple(shape)
    except TypeError as error:
        raise ValueError("shape must contain exactly two positive integers.") from error
    if len(dimensions) != 2 or any(
        isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
        for dimension in dimensions
    ):
        raise ValueError("shape must contain exactly two positive integers.")
    return dimensions


def _shard_tensor(tensor, shard_dim, rank, world_size):
    if shard_dim not in (0, 1):
        raise ValueError(f"shard_dim must be 0 or 1, got {shard_dim}.")
    sharded_size = tensor.size(shard_dim)
    if sharded_size % world_size:
        raise ValueError(
            f"dimension {shard_dim} with size {sharded_size} must be divisible "
            f"by world_size={world_size}."
        )
    shard_size = sharded_size // world_size
    return tensor.narrow(shard_dim, rank * shard_size, shard_size).contiguous()


def _col_shard_gradient(gradient, rank, world_size):
    return _shard_tensor(gradient, 1, rank, world_size)


def _row_shard_gradient(gradient, rank, world_size):
    return _shard_tensor(gradient, 0, rank, world_size)


def _reconstruct_sharded_tensor(sharded_tensors, shard_dim):
    if shard_dim not in (0, 1):
        raise ValueError(f"shard_dim must be 0 or 1, got {shard_dim}.")
    if not sharded_tensors:
        raise ValueError("sharded_tensors must not be empty.")
    return torch.cat(tuple(sharded_tensors), dim=shard_dim)


def _reconstruct_sharded_gradient(
    sharded_gradient,
    shard_dim,
    rank=None,
    world_size=None,
):
    del rank
    if world_size is not None and len(sharded_gradient) != world_size:
        raise ValueError(
            f"expected {world_size} gradient shards, got {len(sharded_gradient)}."
        )
    return _reconstruct_sharded_tensor(sharded_gradient, shard_dim)


def _compare_tensor_output(
    actual,
    expected,
    *,
    output_name,
    reference_name,
    rtol=RTOL,
):
    actual = actual.detach()
    expected = expected.detach()

    if actual.shape != expected.shape:
        raise AssertionError(
            f"{output_name} shape does not match {reference_name}: "
            f"{tuple(actual.shape)} != {tuple(expected.shape)}."
        )
    if actual.device != expected.device:
        raise AssertionError(
            f"{output_name} device does not match {reference_name}: "
            f"{actual.device} != {expected.device}."
        )
    if actual.dtype != expected.dtype:
        raise AssertionError(
            f"{output_name} dtype does not match {reference_name}: "
            f"{actual.dtype} != {expected.dtype}."
        )
    if rtol < 0:
        raise ValueError("rtol must be nonnegative.")

    (
        relative_l2_error,
        difference_norm,
        reference_norm,
        reference_value,
        actual_value,
        max_index,
        opposite_signs,
        total_elements,
    ) = _relative_comparison_values(
        actual,
        expected,
    )
    if relative_l2_error > rtol:
        raise AssertionError(
            f"{output_name} does not match {reference_name}: "
            f"relative L2 error {relative_l2_error:.6e} exceeds "
            f"rtol={rtol:.6e} "
            f"(||TP-ref||₂={difference_norm:.6e}, "
            f"||ref||₂={reference_norm:.6e}); "
            f"opposite signs={opposite_signs}/{total_elements}; "
            f"largest residual is at index {max_index} "
            f"(ref={reference_value:+.6e}, TP={actual_value:+.6e})."
        )


def _relative_comparison_values(actual, expected):
    actual_flat = actual.detach().to(torch.float64).reshape(-1)
    expected_flat = expected.detach().to(torch.float64).reshape(-1)
    difference = actual_flat - expected_flat
    absolute_difference = difference.abs()
    difference_norm = float(torch.linalg.vector_norm(difference))
    reference_norm = float(torch.linalg.vector_norm(expected_flat))

    tensors_are_finite = bool(
        torch.isfinite(actual_flat).all() and torch.isfinite(expected_flat).all()
    )
    if not tensors_are_finite:
        relative_l2_error = math.inf
    elif reference_norm == 0:
        relative_l2_error = 0.0 if difference_norm == 0 else math.inf
    else:
        relative_l2_error = difference_norm / reference_norm

    opposite_signs = int(
        (
            ((expected_flat > 0) & (actual_flat < 0))
            | ((expected_flat < 0) & (actual_flat > 0))
        )
        .sum()
        .item()
    )

    if not absolute_difference.numel():
        return (
            relative_l2_error,
            difference_norm,
            reference_norm,
            math.nan,
            math.nan,
            (),
            opposite_signs,
            0,
        )

    diagnostic_difference = torch.where(
        torch.isfinite(absolute_difference),
        absolute_difference,
        torch.full_like(absolute_difference, torch.inf),
    )
    max_flat_index = int(diagnostic_difference.argmax().item())
    max_index = tuple(
        int(index.item())
        for index in torch.unravel_index(
            torch.tensor(max_flat_index, device=actual.device),
            actual.shape,
        )
    )
    return (
        relative_l2_error,
        difference_norm,
        reference_norm,
        float(expected_flat[max_flat_index]),
        float(actual_flat[max_flat_index]),
        max_index,
        opposite_signs,
        actual.numel(),
    )


def _compare_parameter_output(
    parameter_output,
    torch_parameter,
    *,
    rtol=RTOL,
):
    """Assert that a SOAP-TP parameter matches its PyTorch reference."""
    return _compare_tensor_output(
        parameter_output,
        torch_parameter,
        output_name="SOAP-TP parameter output",
        reference_name="the PyTorch parameter",
        rtol=rtol,
    )


def _compare_soap_step_output(
    soap_step_output,
    reference_output,
    *,
    rtol=RTOL,
):
    """Assert that a reconstructed SOAP-TP update matches the reference update."""
    return _compare_tensor_output(
        soap_step_output,
        reference_output,
        output_name="SOAP-TP soap_step output",
        reference_name="the reference SOAP step output",
        rtol=rtol,
    )


class _CountingSlateBinding:
    def __init__(self, binding):
        self._binding = binding
        self.qr_calls = 0

    def __getattr__(self, name):
        return getattr(self._binding, name)

    def slate_qr_float(self, *args, **kwargs):
        self.qr_calls += 1
        return self._binding.slate_qr_float(*args, **kwargs)


def _make_reference_optimizer(
    parameter,
    *,
    preconditioner_beta,
    beta1,
    beta2,
    eps,
    basis_refresh_interval,
):
    reference_soap = runpy.run_path(str(ROOT / "tests" / "reference" / "soap.py"))[
        "SOAP"
    ]
    return reference_soap(
        [parameter],
        lr=1.0,
        betas=(beta1, beta2),
        shampoo_beta=preconditioner_beta,
        eps=eps,
        weight_decay=0.0,
        precondition_frequency=basis_refresh_interval,
        max_precond_dim=max(parameter.shape),
        merge_dims=False,
        precondition_1d=False,
        normalize_grads=False,
        correct_bias=True,
    )


def _run_reference_soap_step(parameter, optimizer, gradient):
    parameter.grad = gradient.clone()
    parameter_before = parameter.detach().clone()
    try:
        optimizer.step()
        return parameter_before - parameter.detach()
    finally:
        parameter.grad = None


def _local_shard_shape(shape, shard_dim, world_size):
    local_shape = list(shape)
    local_shape[shard_dim] //= world_size
    return tuple(local_shape)


def _scatter_rank_zero_tensor(
    full_tensor,
    shape,
    shard_dim,
    rank,
    world_size,
):
    local = torch.empty(
        _local_shard_shape(shape, shard_dim, world_size),
        dtype=torch.float32,
    )
    scatter_list = None
    if rank == 0:
        scatter_list = [
            _shard_tensor(full_tensor, shard_dim, source_rank, world_size)
            for source_rank in range(world_size)
        ]
    dist.scatter(local, scatter_list=scatter_list, src=0)
    return local


def _gather_rank_zero_tensor(local_tensor, shard_dim, rank, world_size):
    gathered = (
        [torch.empty_like(local_tensor) for _ in range(world_size)]
        if rank == 0
        else None
    )
    dist.gather(local_tensor.contiguous(), gather_list=gathered, dst=0)
    if rank == 0:
        return _reconstruct_sharded_tensor(gathered, shard_dim)
    return None


def _synchronize_rank_zero_error(error, *, rank, context):
    """Propagate rank-zero failures; tensor data moves through scatter/gather."""
    message = None
    if rank == 0 and error is not None:
        message = f"{context}: {type(error).__name__}: {error}"
    payload = [message]
    dist.broadcast_object_list(payload, src=0)
    if payload[0] is not None:
        if rank == 0:
            raise AssertionError(payload[0]) from error
        raise AssertionError(payload[0])


def _validate_comparison_case(
    shape,
    shard_dim,
    iterations,
    block_size,
    basis_refresh_interval,
    world_size,
    process_grid,
):
    shape = _normalize_shape(shape)
    if shard_dim not in (0, 1):
        raise ValueError(f"shard_dim must be 0 or 1, got {shard_dim}.")
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if basis_refresh_interval <= 0:
        raise ValueError("basis_refresh_interval must be positive.")
    if world_size <= 1:
        raise ValueError("the SOAP comparison requires multiple MPI ranks.")
    if shape[shard_dim] % world_size:
        raise ValueError(
            f"shape[{shard_dim}]={shape[shard_dim]} must be divisible by "
            f"world_size={world_size}."
        )

    required_blocks = max(process_grid)
    for dimension in shape:
        available_blocks = math.ceil(dimension / block_size)
        if available_blocks < required_blocks:
            raise ValueError(
                "ELPA requires every rank to own a block row and column; "
                f"dimension {dimension} with block_size={block_size} provides "
                f"{available_blocks} blocks, but {required_blocks} are required."
            )
    return shape


def _run_soap_comparison(
    shape,
    shard_dim,
    *,
    iterations=COMPARISON_ITERATIONS,
    seed=SEED,
    block_size=COMPARISON_BLOCK_SIZE,
    basis_refresh_interval=BASIS_REFRESH_INTERVAL,
    rtol=RTOL,
    device="cpu",
):
    """Compare original SOAP on rank zero with distributed SOAP on every rank."""
    distributed_was_initialized = dist.is_initialized()
    rank, world_size, process_grid = _initialize_distributed(device=device)
    try:
        shape = _validate_comparison_case(
            shape,
            shard_dim,
            iterations,
            block_size,
            basis_refresh_interval,
            world_size,
            process_grid,
        )
        if world_size != MPI_RANKS:
            raise AssertionError(
                f"SOAP comparison requires {MPI_RANKS} ranks, got {world_size}."
            )

        # native_backends = {
        #     "ELPA": elpa_bindings.compiled_gpu_backend(),
        #     "SLATE": slate_bindings.compiled_gpu_backend(),
        # }
        # if set(native_backends.values()) != {"none"}:
        #     raise RuntimeError(
        #         "The CPU SOAP comparison requires CPU native bindings; got "
        #         f"{native_backends}."
        #     )

        initial_parameter = None
        reference_parameter = None
        reference_optimizer = None
        gradient_generator = None
        setup_error = None
        if rank == 0:
            try:
                parameter_generator = torch.Generator().manual_seed(seed)
                gradient_generator = torch.Generator().manual_seed(seed + 1)
                initial_parameter = torch.normal(
                    mean=0,
                    std=1,
                    size=shape,
                    generator=parameter_generator,
                    dtype=torch.float32,
                )
                reference_parameter = torch.nn.Parameter(initial_parameter.clone())
                reference_optimizer = _make_reference_optimizer(
                    reference_parameter,
                    preconditioner_beta=PRECONDITIONER_BETA,
                    beta1=BETA1,
                    beta2=BETA2,
                    eps=EPS,
                    basis_refresh_interval=basis_refresh_interval,
                )
            except Exception as error:
                setup_error = error
        _synchronize_rank_zero_error(
            setup_error,
            rank=rank,
            context=f"shape={shape}, shard_dim={shard_dim}, reference setup",
        )

        local_parameter = _scatter_rank_zero_tensor(
            initial_parameter,
            shape,
            shard_dim,
            rank,
            world_size,
        )
        initial_parameter_output = _gather_rank_zero_tensor(
            local_parameter,
            shard_dim,
            rank,
            world_size,
        )
        initial_comparison_error = None
        if rank == 0:
            try:
                _compare_tensor_output(
                    reference_parameter.detach(),
                    initial_parameter,
                    output_name="Reference initial parameter",
                    reference_name="the generated initial parameter",
                    rtol=0,
                )
                _compare_tensor_output(
                    initial_parameter_output,
                    initial_parameter,
                    output_name="Reconstructed SOAP-TP initial parameter",
                    reference_name="the generated initial parameter",
                    rtol=0,
                )
                _compare_parameter_output(
                    initial_parameter_output,
                    reference_parameter,
                    rtol=rtol,
                )
            except Exception as error:
                initial_comparison_error = error
        _synchronize_rank_zero_error(
            initial_comparison_error,
            rank=rank,
            context=f"shape={shape}, shard_dim={shard_dim}, initial parameter",
        )

        distributed_state = {}
        counted_slate = _CountingSlateBinding(slate_bindings)
        comparison_failures = []

        for iteration in range(iterations):
            full_gradient = None
            gradient_error = None
            if rank == 0:
                try:
                    full_gradient = _make_gradient(
                        *shape,
                        generator=gradient_generator,
                    )
                except Exception as error:
                    gradient_error = error
            _synchronize_rank_zero_error(
                gradient_error,
                rank=rank,
                context=(
                    f"shape={shape}, shard_dim={shard_dim}, "
                    f"iteration={iteration + 1}, gradient generation"
                ),
            )
            local_gradient = _scatter_rank_zero_tensor(
                full_gradient,
                shape,
                shard_dim,
                rank,
                world_size,
            )

            reference_update = None
            reference_error = None
            if rank == 0:
                try:
                    reference_update = _run_reference_soap_step(
                        reference_parameter,
                        reference_optimizer,
                        full_gradient,
                    )
                except Exception as error:
                    reference_error = error
            _synchronize_rank_zero_error(
                reference_error,
                rank=rank,
                context=(
                    f"shape={shape}, shard_dim={shard_dim}, "
                    f"iteration={iteration + 1}, reference SOAP"
                ),
            )

            local_update = soap_step(
                local_gradient,
                distributed_state,
                global_shape=shape,
                shard_dim=shard_dim,
                block_size=block_size,
                process_grid_shape=process_grid,
                preconditioner_beta=PRECONDITIONER_BETA,
                beta1=BETA1,
                beta2=BETA2,
                eps=EPS,
                basis_refresh_interval=basis_refresh_interval,
                elpa_binding=elpa_bindings,
                slate_binding=counted_slate,
            )
            full_update = _gather_rank_zero_tensor(
                local_update,
                shard_dim,
                rank,
                world_size,
            )

            local_parameter.sub_(local_update)
            full_parameter = _gather_rank_zero_tensor(
                local_parameter,
                shard_dim,
                rank,
                world_size,
            )

            comparison_runtime_error = None
            if rank == 0:
                try:
                    (
                        soap_relative_l2_error,
                        _,
                        _,
                        soap_reference_value,
                        soap_tp_value,
                        soap_max_index,
                        soap_opposite_signs,
                        soap_total_elements,
                    ) = _relative_comparison_values(
                        full_update,
                        reference_update,
                    )
                    (
                        parameter_relative_l2_error,
                        _,
                        _,
                        parameter_reference_value,
                        parameter_tp_value,
                        parameter_max_index,
                        parameter_opposite_signs,
                        parameter_total_elements,
                    ) = _relative_comparison_values(
                        full_parameter,
                        reference_parameter,
                    )
                    print(
                        f"{shape[0]}x{shape[1]} dim{shard_dim} | "
                        f"step {iteration} | "
                        f"SOAP relL2={soap_relative_l2_error:.2e} "
                        f"(ref={soap_reference_value:+.3e}, "
                        f"TP={soap_tp_value:+.3e}, at={soap_max_index}) "
                        f"opposite={soap_opposite_signs}/{soap_total_elements} | "
                        f"PARAM relL2={parameter_relative_l2_error:.2e} "
                        f"(ref={parameter_reference_value:+.3e}, "
                        f"TP={parameter_tp_value:+.3e}, "
                        f"at={parameter_max_index}) "
                        f"opposite={parameter_opposite_signs}/"
                        f"{parameter_total_elements}",
                        flush=True,
                    )
                    if iteration == 0:
                        try:
                            _compare_tensor_output(
                                full_update,
                                torch.zeros_like(full_update),
                                output_name="First SOAP-TP output",
                                reference_name="an exactly zero tensor",
                                rtol=0,
                            )
                        except AssertionError as error:
                            comparison_failures.append((iteration, "SOAP", str(error)))
                    try:
                        _compare_soap_step_output(
                            full_update,
                            reference_update,
                            rtol=rtol,
                        )
                    except AssertionError as error:
                        comparison_failures.append((iteration, "SOAP", str(error)))
                    try:
                        _compare_parameter_output(
                            full_parameter,
                            reference_parameter,
                            rtol=rtol,
                        )
                    except AssertionError as error:
                        comparison_failures.append((iteration, "PARAM", str(error)))
                except Exception as error:
                    comparison_runtime_error = error
            _synchronize_rank_zero_error(
                comparison_runtime_error,
                rank=rank,
                context=(
                    f"shape={shape}, shard_dim={shard_dim}, "
                    f"iteration={iteration + 1}, comparison diagnostics"
                ),
            )

        local_counts = torch.tensor(
            [int(distributed_state["step"]), counted_slate.qr_calls],
            dtype=torch.int64,
        )
        gathered_counts = (
            [torch.empty_like(local_counts) for _ in range(world_size)]
            if rank == 0
            else None
        )
        dist.gather(local_counts, gather_list=gathered_counts, dst=0)

        final_error = None
        result = None
        if rank == 0:
            try:
                expected_step = iterations - 1
                expected_qr_calls = 2 * (expected_step // basis_refresh_interval)
                actual_counts = [tuple(count.tolist()) for count in gathered_counts]
                expected_counts = [
                    (expected_step, expected_qr_calls) for _ in range(world_size)
                ]
                if actual_counts != expected_counts:
                    raise AssertionError(
                        f"expected per-rank (step, QR calls) "
                        f"{expected_counts}, got {actual_counts}."
                    )
                reference_step = reference_optimizer.state[reference_parameter]["step"]
                if reference_step != expected_step:
                    raise AssertionError(
                        f"reference optimizer step is {reference_step}, "
                        f"expected {expected_step}."
                    )
                if comparison_failures:
                    soap_failure_steps = [
                        iteration
                        for iteration, output, _ in comparison_failures
                        if output == "SOAP"
                    ]
                    parameter_failure_steps = [
                        iteration
                        for iteration, output, _ in comparison_failures
                        if output == "PARAM"
                    ]
                    first_iteration, first_output, first_message = comparison_failures[
                        0
                    ]
                    raise AssertionError(
                        f"comparisons failed after all {iterations} iterations: "
                        f"SOAP steps={soap_failure_steps}, "
                        f"PARAM steps={parameter_failure_steps}; "
                        f"first failure at step {first_iteration} "
                        f"({first_output}): {first_message}"
                    )
                result = {
                    "shape": shape,
                    "shard_dim": shard_dim,
                    "iterations": iterations,
                    "qr_calls": expected_qr_calls,
                }
                print(
                    "SOAP comparison passed: "
                    f"shape={shape}, shard_dim={shard_dim}, "
                    f"iterations={iterations}, qr_calls={expected_qr_calls}, "
                    f"rtol={rtol:.1e}",
                    flush=True,
                )
            except Exception as error:
                final_error = error
        _synchronize_rank_zero_error(
            final_error,
            rank=rank,
            context=f"shape={shape}, shard_dim={shard_dim}, final state",
        )
        return result
    finally:
        if not distributed_was_initialized:
            _destroy_distributed()


def _worker_case_from_environment():
    try:
        shape = json.loads(os.environ[MPI_WORKER_SHAPE])
        shard_dim = int(os.environ[MPI_WORKER_SHARD_DIM])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("invalid SOAP MPI worker case environment") from error
    return _normalize_shape(shape), shard_dim


def _run_mpi_worker(shape, shard_dim, timeout=240):
    mpiexec = shutil.which("mpiexec")
    if mpiexec is None:
        pytest.skip("mpiexec is required for the SOAP comparison")

    environment = os.environ.copy()
    environment.update(
        {
            MPI_WORKER: "1",
            MPI_WORKER_SHAPE: json.dumps(list(shape), separators=(",", ":")),
            MPI_WORKER_SHARD_DIM: str(shard_dim),
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
        "--bind-to",
        "none",
        "-n",
        str(MPI_RANKS),
        sys.executable,
        str(Path(__file__).resolve()),
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        raise AssertionError(
            f"SOAP MPI worker timed out for shape={shape}, "
            f"shard_dim={shard_dim}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        ) from error

    if stdout:
        print(stdout.strip(), flush=True)
    if process.returncode != 0:
        raise AssertionError(
            f"SOAP MPI worker failed for shape={shape}, shard_dim={shard_dim}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )


@pytest.mark.skipif(
    DISTRIBUTED_IMPORT_ERROR is not None,
    reason="SOAP comparison requires mpi4py and native ELPA/SLATE bindings",
)
@pytest.mark.parametrize(
    ("shape", "shard_dim"),
    [
        pytest.param(
            shape,
            shard_dim,
            id=f"{shape[0]}x{shape[1]}-dim{shard_dim}",
        )
        for shape in COMPARISON_SHAPES
        for shard_dim in (0, 1)
    ],
)
def test_reference_matches_distributed_soap(shape, shard_dim):
    assert MPI.COMM_WORLD.Get_size() == 1, "run the pytest controller without mpiexec"
    _run_mpi_worker(shape, shard_dim)


if __name__ == "__main__" and os.environ.get(MPI_WORKER) == "1":
    worker_shape, worker_shard_dim = _worker_case_from_environment()
    _run_soap_comparison(worker_shape, worker_shard_dim)
