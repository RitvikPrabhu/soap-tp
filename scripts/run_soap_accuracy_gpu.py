#!/usr/bin/env python3
"""Run the stock SOAP accuracy comparison on one GPU per MPI rank."""

import argparse
import importlib.util
from pathlib import Path
import sys

import torch
import torch.distributed as torch_dist


ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "tests" / "unit" / "test_soap.py"


def _scaled_block_size(size, process_grid, max_block_size):
    """Choose the largest capped block that keeps every grid axis occupied."""
    required_blocks = max(process_grid)
    if size < required_blocks:
        raise ValueError(
            f"size={size} is too small for a process grid requiring "
            f"{required_blocks} blocks per dimension"
        )
    return min(size // required_blocks, max_block_size)


def _load_test_module():
    spec = importlib.util.spec_from_file_location(
        "soap_tp_frontier_accuracy_test",
        TEST_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {TEST_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _GpuCollectiveProxy:
    """Move the stock test's final CPU counter gather through NCCL/RCCL."""

    def __init__(self, distributed, device):
        self._distributed = distributed
        self._device = torch.device(device)

    def __getattr__(self, name):
        return getattr(self._distributed, name)

    def gather(
        self,
        tensor,
        gather_list=None,
        dst=0,
        group=None,
        async_op=False,
    ):
        backend = self._distributed.get_backend(group)
        if tensor.device.type != "cpu" or backend != "nccl":
            return self._distributed.gather(
                tensor,
                gather_list=gather_list,
                dst=dst,
                group=group,
                async_op=async_op,
            )
        if async_op:
            raise RuntimeError("the GPU accuracy counter gather must be synchronous")

        device_tensor = tensor.to(self._device)
        gathered = [
            torch.empty_like(device_tensor)
            for _ in range(self._distributed.get_world_size(group))
        ]
        self._distributed.all_gather(gathered, device_tensor, group=group)
        if self._distributed.get_rank(group) == dst:
            if gather_list is None or len(gather_list) != len(gathered):
                raise ValueError("rank dst requires one counter buffer per rank")
            for destination, source in zip(gather_list, gathered):
                destination.copy_(source.cpu())
        return None


def _install_gpu_collectives(test_module, device):
    device = torch.device(device)

    def scatter_rank_zero_tensor(
        full_tensor,
        shape,
        shard_dim,
        rank,
        world_size,
    ):
        if rank == 0:
            distributed_tensor = full_tensor.to(device)
        else:
            distributed_tensor = torch.empty(
                shape,
                dtype=torch.float32,
                device=device,
            )
        torch_dist.broadcast(distributed_tensor, src=0)
        return test_module._shard_tensor(
            distributed_tensor,
            shard_dim,
            rank,
            world_size,
        )

    def gather_rank_zero_tensor(local_tensor, shard_dim, rank, world_size):
        gathered = [torch.empty_like(local_tensor) for _ in range(world_size)]
        torch_dist.all_gather(gathered, local_tensor.contiguous())
        if rank == 0:
            return test_module._reconstruct_sharded_tensor(
                gathered,
                shard_dim,
            ).cpu()
        return None

    def synchronize_rank_zero_error(error, *, rank, context):
        message = None
        if rank == 0 and error is not None:
            message = f"{context}: {type(error).__name__}: {error}"
        payload = [message]
        torch_dist.broadcast_object_list(payload, src=0, device=device)
        if payload[0] is not None:
            if rank == 0:
                raise AssertionError(payload[0]) from error
            raise AssertionError(payload[0])

    test_module._scatter_rank_zero_tensor = scatter_rank_zero_tensor
    test_module._gather_rank_zero_tensor = gather_rank_zero_tensor
    test_module._synchronize_rank_zero_error = synchronize_rank_zero_error
    test_module.dist = _GpuCollectiveProxy(torch_dist, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=(8, 16, 24))
    parser.add_argument(
        "--max-block-size",
        type=int,
        default=256,
        help=(
            "cap for grid-scaled block sizes (default: 256); on a 2x4 "
            "grid each size uses min(size/4, cap)"
        ),
    )
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--shard-dim",
        type=int,
        choices=(0, 1),
        required=True,
        help="gradient shard dimension: 0 for rows or 1 for columns",
    )
    parser.add_argument(
        "--basis-implementations",
        nargs="+",
        choices=("eigh", "elpa"),
        default=("eigh", "elpa"),
        help="initial basis implementations to compare (default: eigh elpa)",
    )
    args = parser.parse_args()

    if any(size <= 0 for size in args.sizes):
        parser.error("all sizes must be positive")
    if args.max_block_size <= 0:
        parser.error("--max-block-size must be positive")
    if args.rtol < 0:
        parser.error("--rtol must be nonnegative")
    if len(set(args.basis_implementations)) != len(args.basis_implementations):
        parser.error("--basis-implementations must not contain duplicates")

    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        parser.error("--device must name a concrete CUDA/ROCm device")
    torch.cuda.set_device(device)

    test_module = _load_test_module()
    _install_gpu_collectives(test_module, device)
    rank, world_size, process_grid = test_module._initialize_distributed(
        device=device,
    )
    failures = []
    try:
        if world_size != test_module.MPI_RANKS:
            raise RuntimeError(
                f"expected {test_module.MPI_RANKS} ranks, got {world_size}"
            )
        block_sizes = {
            size: _scaled_block_size(
                size,
                process_grid,
                args.max_block_size,
            )
            for size in args.sizes
        }
        if rank == 0:
            block_size_summary = ", ".join(
                f"{size}:{block_sizes[size]}" for size in args.sizes
            )
            print(
                f"SOAP GPU accuracy: ranks={world_size}, "
                f"process_grid={process_grid[0]}x{process_grid[1]}, "
                f"sizes={args.sizes}, block_sizes={{{block_size_summary}}}, "
                f"shard_dim={args.shard_dim}, "
                f"basis_implementations={args.basis_implementations}",
                flush=True,
            )

        for size in args.sizes:
            block_size = block_sizes[size]
            tile_count = (size + block_size - 1) // block_size
            if rank == 0:
                print(
                    f"SOAP GPU case: size={size}x{size}, "
                    f"block_size={block_size}, "
                    f"tile_grid={tile_count}x{tile_count}",
                    flush=True,
                )
            for basis_implementation in args.basis_implementations:
                if rank == 0:
                    print(
                        f"SOAP GPU run: size={size}x{size}, "
                        f"dim{args.shard_dim}, basis={basis_implementation}",
                        flush=True,
                    )
                try:
                    test_module._run_soap_comparison(
                        (size, size),
                        args.shard_dim,
                        block_size=block_size,
                        basis_implementation=basis_implementation,
                        rtol=args.rtol,
                        device=device,
                    )
                except AssertionError as error:
                    error_message = str(error)
                    failures.append(
                        (
                            size,
                            args.shard_dim,
                            basis_implementation,
                            block_size,
                            error_message,
                        )
                    )
                    if rank == 0:
                        print(
                            f"SOAP comparison failed: {size}x{size} "
                            f"dim{args.shard_dim}, "
                            f"basis={basis_implementation}, "
                            f"block_size={block_size}; "
                            f"continuing:\n{error_message}",
                            flush=True,
                        )
                finally:
                    torch.cuda.empty_cache()
    finally:
        test_module._destroy_distributed()

    if failures:
        failed_cases = ", ".join(
            f"{size}x{size} dim{shard_dim} basis={basis_implementation} "
            f"block={block_size}"
            for size, shard_dim, basis_implementation, block_size, _ in failures
        )
        if rank == 0:
            print(
                f"SOAP accuracy completed with {len(failures)} failing "
                f"case(s): {failed_cases}",
                flush=True,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
