#!/usr/bin/env python3
"""Time SOAP-TP on one GPU per MPI rank."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import statistics
import time


TIMING_PREFIX = "SOAP_TIMING "


def _process_grid_shape(world_size: int) -> tuple[int, int]:
    process_rows = math.isqrt(world_size)
    while world_size % process_rows:
        process_rows -= 1
    return process_rows, world_size // process_rows


def _free_port(address: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((address, 0))
        return sock.getsockname()[1]


def _emit(rank: int, event: str, **values: object) -> None:
    if rank == 0:
        print(
            TIMING_PREFIX
            + json.dumps({"event": event, **values}, sort_keys=True),
            flush=True,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--shard-dim", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--basis-implementation",
        choices=("elpa", "eigh"),
        default="elpa",
    )
    parser.add_argument("--basis-refresh-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.size <= 0:
        parser.error("--size must be positive")
    if args.world_size <= 0:
        parser.error("--world-size must be positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.block_size <= 0:
        parser.error("--block-size must be positive")
    if args.basis_refresh_interval <= 0:
        parser.error("--basis-refresh-interval must be positive")
    return args


def main() -> int:
    args = _parse_args()

    from mpi4py import MPI
    import torch
    import torch.distributed as dist

    from soap_tp import elpa_bindings, slate_bindings, soap_step

    if not MPI.Is_initialized() or MPI.Is_finalized():
        raise RuntimeError("MPI must be initialized and active")

    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()
    if world_size != args.world_size:
        raise RuntimeError(
            f"expected {args.world_size} MPI ranks, got {world_size}"
        )
    if args.size % world_size:
        raise ValueError(
            f"matrix size {args.size} must be divisible by {world_size} ranks"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see a CUDA/ROCm GPU")

    visible_devices = torch.cuda.device_count()
    local_rank = int(os.environ.get("SLURM_LOCALID", rank))
    device_index = 0 if visible_devices == 1 else local_rank % visible_devices
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)

    process_grid_shape = _process_grid_shape(world_size)
    tile_count = math.ceil(args.size / args.block_size)
    if tile_count < max(process_grid_shape):
        raise ValueError(
            f"size={args.size} and block_size={args.block_size} provide only "
            f"{tile_count} tiles per dimension, but process grid "
            f"{process_grid_shape[0]}x{process_grid_shape[1]} requires at least "
            f"{max(process_grid_shape)}"
        )

    rendezvous = ("127.0.0.1", _free_port()) if rank == 0 else None
    address, port = MPI.COMM_WORLD.bcast(rendezvous, root=0)
    os.environ.update(
        MASTER_ADDR=address,
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    dist.init_process_group(
        "nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )

    try:
        local_shape = (
            (args.size // world_size, args.size)
            if args.shard_dim == 0
            else (args.size, args.size // world_size)
        )
        generator = torch.Generator(device=device).manual_seed(args.seed + rank)
        gradient = torch.normal(
            mean=0.0,
            std=2.0,
            size=local_shape,
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        state: dict[str, object] = {}

        def timed_step() -> float:
            dist.barrier()
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            soap_step(
                gradient,
                state,
                global_shape=(args.size, args.size),
                shard_dim=args.shard_dim,
                block_size=args.block_size,
                process_grid_shape=process_grid_shape,
                basis_refresh_interval=args.basis_refresh_interval,
                basis_implementation=args.basis_implementation,
                elpa_binding=elpa_bindings,
                slate_binding=slate_bindings,
            )
            torch.cuda.synchronize(device)
            elapsed = torch.tensor(
                time.perf_counter() - start,
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            return elapsed.item()

        _emit(
            rank,
            "configuration",
            timer="synchronized_max_rank_wall_clock",
            matrix_size=args.size,
            ranks=world_size,
            process_grid=list(process_grid_shape),
            local_shape=list(local_shape),
            block_size=args.block_size,
            shard_dim=args.shard_dim,
            basis_implementation=args.basis_implementation,
            basis_refresh_interval=args.basis_refresh_interval,
            measured_steps=args.steps,
            tensor_profile=False,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        )

        initialization_seconds = timed_step()
        _emit(
            rank,
            "initialization",
            step=0,
            seconds=initialization_seconds,
        )

        step_seconds = []
        for step in range(1, args.steps + 1):
            elapsed = timed_step()
            step_seconds.append(elapsed)
            _emit(rank, "step", step=step, seconds=elapsed)

        _emit(
            rank,
            "summary",
            matrix_size=args.size,
            ranks=world_size,
            initialization_seconds=initialization_seconds,
            measured_steps=args.steps,
            seconds_per_step=statistics.mean(step_seconds),
            median_seconds_per_step=statistics.median(step_seconds),
            min_seconds_per_step=min(step_seconds),
            max_seconds_per_step=max(step_seconds),
        )
    finally:
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
