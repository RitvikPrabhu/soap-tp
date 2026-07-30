#!/usr/bin/env python3
"""Run SOAP experiments on one node and directly write three PNG plots."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import runpy
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "SOAP_PLOT_RESULT "

# These match tests/unit/test_soap.py.
SEED = 42
LR = 1.0
PRECONDITIONER_BETA = 0.75
BETA1 = 0.8
BETA2 = 0.9
EPS = 1e-6

MPI = torch = dist = soap_step = elpa_bindings = slate_bindings = None


def parse_list(text, name):
    try:
        values = [int(value) for value in text.split(",")]
    except ValueError as error:
        raise ValueError(f"{name} must be comma-separated integers") from error
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive integers")
    if values != sorted(set(values)):
        raise ValueError(f"{name} must be strictly increasing")
    return values


def make_grid(ranks):
    rows = math.isqrt(ranks)
    while ranks % rows:
        rows -= 1
    return rows, ranks // rows


def check_case(n, ranks, block_size):
    grid = make_grid(ranks)
    if n % ranks:
        raise ValueError(f"N={n} must be divisible by ranks={ranks}")
    if math.ceil(n / block_size) < max(grid):
        raise ValueError(f"N={n} is too small for block={block_size}, grid={grid}")
    return grid


def make_weak_size(base, ranks, block_size):
    minimum = block_size * max(make_grid(ranks))
    target = max(minimum, round(base * ranks ** (1.0 / 3.0)))
    return math.ceil(target / ranks) * ranks


def load_worker_modules():
    global MPI, torch, dist, soap_step, elpa_bindings, slate_bindings

    from mpi4py import MPI as mpi
    import torch as torch_module
    import torch.distributed as torch_dist

    sys.path.insert(0, str(ROOT / "src"))
    from soap_tp import elpa_bindings as elpa
    from soap_tp import slate_bindings as slate
    from soap_tp import soap_step as step

    MPI, torch, dist = mpi, torch_module, torch_dist
    soap_step, elpa_bindings, slate_bindings = step, elpa, slate


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def initialize_distributed(device_name):
    rank = MPI.COMM_WORLD.Get_rank()
    ranks = MPI.COMM_WORLD.Get_size()
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot see a CUDA/ROCm GPU")
        visible = torch.cuda.device_count()
        local_rank = int(os.environ.get("SLURM_LOCALID", rank))
        index = 0 if visible == 1 else local_rank % visible
        torch.cuda.set_device(index)
        device = torch.device("cuda", index)
    else:
        device = torch.device("cpu")

    rendezvous = ("127.0.0.1", free_port()) if rank == 0 else None
    address, port = MPI.COMM_WORLD.bcast(rendezvous, root=0)
    os.environ.update(
        MASTER_ADDR=address,
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(ranks),
    )
    dist.init_process_group(
        "nccl" if device.type == "cuda" else "gloo",
        init_method="env://",
        rank=rank,
        world_size=ranks,
    )
    return rank, ranks, device


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_gradient(m, n, generator, device):
    """The normal distribution used by _make_gradient in test_soap.py."""
    return torch.normal(
        mean=0.0,
        std=2.0,
        size=(m, n),
        generator=generator,
        dtype=torch.float32,
        device=device,
    )


def scatter(full, n, shard_dim, rank, ranks, device):
    shape = (n // ranks, n) if shard_dim == 0 else (n, n // ranks)
    local = torch.empty(shape, dtype=torch.float32, device=device)
    shards = None
    if rank == 0:
        width = n // ranks
        shards = [
            full.narrow(shard_dim, source * width, width).contiguous()
            for source in range(ranks)
        ]
    dist.scatter(local, scatter_list=shards, src=0)
    return local


def make_reference_optimizer(parameter, n, refresh):
    optimizer = runpy.run_path(str(ROOT / "tests/reference/soap.py"))["SOAP"]
    return optimizer(
        [parameter],
        lr=LR,
        betas=(BETA1, BETA2),
        shampoo_beta=PRECONDITIONER_BETA,
        eps=EPS,
        weight_decay=0.0,
        precondition_frequency=refresh,
        max_precond_dim=n,
        merge_dims=False,
        precondition_1d=False,
        normalize_grads=False,
        correct_bias=True,
    )


def apply_reference_step(parameter, optimizer, gradient):
    parameter.grad = gradient
    optimizer.step()
    parameter.grad = None


def tp_step(gradient, state, args, grid):
    return soap_step(
        gradient,
        state,
        global_shape=(args.n, args.n),
        shard_dim=args.shard_dim,
        block_size=args.block_size,
        process_grid_shape=grid,
        preconditioner_beta=PRECONDITIONER_BETA,
        beta1=BETA1,
        beta2=BETA2,
        eps=EPS,
        basis_refresh_interval=args.refresh_interval,
        elpa_binding=elpa_bindings,
        slate_binding=slate_bindings,
    )


def compare_tensors(actual, reference, device):
    actual = actual.detach().to(torch.float64).reshape(-1)
    reference = reference.detach().to(torch.float64).reshape(-1)
    difference = actual - reference
    opposite = (
        ((reference > 0) & (actual < 0)) | ((reference < 0) & (actual > 0))
    ).sum()
    values = torch.stack(
        [
            torch.dot(difference, difference),
            torch.dot(reference, reference),
            opposite.to(torch.float64),
            torch.tensor(actual.numel(), dtype=torch.float64, device=device),
        ]
    )
    dist.all_reduce(values)
    difference_norm = math.sqrt(values[0].item())
    reference_norm = math.sqrt(values[1].item())
    relative_l2 = (
        difference_norm / reference_norm
        if reference_norm
        else (0.0 if difference_norm == 0 else math.inf)
    )
    return relative_l2, (values[2] / values[3]).item()


def correctness_worker(args, rank, ranks, device):
    grid = check_case(args.n, ranks, args.block_size)
    full_parameter = reference_parameter = optimizer = gradient_generator = None
    if rank == 0:
        parameter_generator = torch.Generator(device=device).manual_seed(args.seed)
        gradient_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
        full_parameter = torch.normal(
            mean=0.0,
            std=1.0,
            size=(args.n, args.n),
            generator=parameter_generator,
            dtype=torch.float32,
            device=device,
        )
        reference_parameter = torch.nn.Parameter(full_parameter.clone())
        optimizer = make_reference_optimizer(
            reference_parameter, args.n, args.refresh_interval
        )

    parameter = scatter(
        full_parameter, args.n, args.shard_dim, rank, ranks, device
    )
    state = {}
    result = {
        "experiment": "correctness",
        "implementation": "tp_vs_original",
        "n": args.n,
        "ranks": ranks,
        "iterations": [],
        "update_relative_l2": [],
        "parameter_relative_l2": [],
        "update_opposite_fraction": [],
        "parameter_opposite_fraction": [],
    }

    for call in range(args.iterations + 1):
        full_gradient = (
            make_gradient(args.n, args.n, gradient_generator, device)
            if rank == 0
            else None
        )
        gradient = scatter(
            full_gradient, args.n, args.shard_dim, rank, ranks, device
        )
        if rank == 0:
            before = reference_parameter.detach().clone()
            apply_reference_step(reference_parameter, optimizer, full_gradient)
            reference_update = before - reference_parameter.detach()
        else:
            reference_update = None

        update = tp_step(gradient, state, args, grid)
        parameter.sub_(update, alpha=LR)
        reference_update = scatter(
            reference_update, args.n, args.shard_dim, rank, ranks, device
        )
        reference_value = scatter(
            reference_parameter.detach() if rank == 0 else None,
            args.n,
            args.shard_dim,
            rank,
            ranks,
            device,
        )
        update_error, update_sign = compare_tensors(update, reference_update, device)
        parameter_error, parameter_sign = compare_tensors(
            parameter, reference_value, device
        )
        if call:
            result["iterations"].append(call)
            result["update_relative_l2"].append(update_error)
            result["parameter_relative_l2"].append(parameter_error)
            result["update_opposite_fraction"].append(update_sign)
            result["parameter_opposite_fraction"].append(parameter_sign)
    return result if rank == 0 else None


def max_rank_time(seconds, device):
    value = torch.tensor(seconds, dtype=torch.float64, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value.item()


def tp_timing_worker(args, rank, ranks, device):
    grid = check_case(args.n, ranks, args.block_size)
    shape = (args.n // ranks, args.n) if args.shard_dim == 0 else (
        args.n,
        args.n // ranks,
    )
    parameter = torch.zeros(shape, dtype=torch.float32, device=device)
    generator = (
        torch.Generator(device=device).manual_seed(args.seed) if rank == 0 else None
    )
    state = {}

    full_gradient = (
        make_gradient(args.n, args.n, generator, device) if rank == 0 else None
    )
    gradient = scatter(
        full_gradient, args.n, args.shard_dim, rank, ranks, device
    )
    parameter.sub_(tp_step(gradient, state, args, grid), alpha=LR)
    synchronize(device)

    total = 0.0
    for _ in range(args.iterations):
        full_gradient = (
            make_gradient(args.n, args.n, generator, device) if rank == 0 else None
        )
        gradient = scatter(
            full_gradient, args.n, args.shard_dim, rank, ranks, device
        )
        dist.barrier()
        synchronize(device)
        start = time.perf_counter()
        parameter.sub_(tp_step(gradient, state, args, grid), alpha=LR)
        synchronize(device)
        total += max_rank_time(time.perf_counter() - start, device)
    if rank:
        return None
    return {
        "experiment": args.experiment,
        "implementation": "tp_soap",
        "n": args.n,
        "ranks": ranks,
        "repeat": args.repeat,
        "seconds_per_step": total / args.iterations,
    }


def reference_timing_worker(args, rank, ranks, device):
    if ranks != 1:
        raise ValueError("original SOAP timing uses one rank")
    parameter = torch.nn.Parameter(
        torch.zeros((args.n, args.n), dtype=torch.float32, device=device)
    )
    optimizer = make_reference_optimizer(parameter, args.n, args.refresh_interval)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    gradient = make_gradient(args.n, args.n, generator, device)
    apply_reference_step(parameter, optimizer, gradient)
    synchronize(device)

    total = 0.0
    for _ in range(args.iterations):
        gradient = make_gradient(args.n, args.n, generator, device)
        synchronize(device)
        start = time.perf_counter()
        apply_reference_step(parameter, optimizer, gradient)
        synchronize(device)
        total += time.perf_counter() - start
    return {
        "experiment": args.experiment,
        "implementation": "original_soap",
        "n": args.n,
        "ranks": 1,
        "repeat": args.repeat,
        "seconds_per_step": total / args.iterations,
    }


def worker_main(args):
    load_worker_modules()
    rank, ranks, device = initialize_distributed(args.device)
    if ranks != args.world_size:
        raise RuntimeError(f"expected {args.world_size} ranks, got {ranks}")
    try:
        if args.worker == "correctness":
            result = correctness_worker(args, rank, ranks, device)
        elif args.implementation == "tp_soap":
            result = tp_timing_worker(args, rank, ranks, device)
        else:
            result = reference_timing_worker(args, rank, ranks, device)
        if rank == 0:
            print(RESULT_PREFIX + json.dumps(result), flush=True)
    finally:
        dist.destroy_process_group()
    return 0


def make_worker_command(args, worker, experiment, implementation, n, ranks, repeat):
    python = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        worker,
        "--experiment",
        experiment,
        "--implementation",
        implementation,
        "--n",
        str(n),
        "--world-size",
        str(ranks),
        "--repeat",
        str(repeat),
        "--iterations",
        str(args.iterations),
        "--block-size",
        str(args.block_size),
        "--shard-dim",
        str(args.shard_dim),
        "--refresh-interval",
        str(args.refresh_interval),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    if os.environ.get("SLURM_JOB_ID"):
        return [
            "srun",
            "-N1",
            "-n",
            str(ranks),
            f"--ntasks-per-node={ranks}",
            "--gpus-per-task=1",
            "--gpu-bind=closest",
            *python,
        ]
    return [
        "mpiexec",
        "--oversubscribe",
        "--bind-to",
        "none",
        "-n",
        str(ranks),
        *python,
    ]


def run_worker(args, worker, experiment, implementation, n, ranks, repeat=0):
    command = make_worker_command(
        args, worker, experiment, implementation, n, ranks, repeat
    )
    print(
        f"{experiment}: {implementation}, N={n}, ranks={ranks}, "
        f"repeat={repeat + 1}",
        flush=True,
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )
    result = None
    for line in completed.stdout.splitlines():
        position = line.find(RESULT_PREFIX)
        if position >= 0:
            result = json.loads(line[position + len(RESULT_PREFIX) :])
        elif line.strip():
            print(line)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode or result is None:
        raise RuntimeError(f"worker failed: {' '.join(command)}")
    return result


def timing_mean(results, experiment, implementation, n, ranks):
    values = [
        result["seconds_per_step"]
        for result in results
        if result["experiment"] == experiment
        and result["implementation"] == implementation
        and result["n"] == n
        and result["ranks"] == ranks
    ]
    return sum(values) / len(values)


def set_rank_axis(axis, ranks):
    axis.set_xticks(ranks)
    if len(ranks) > 1 and ranks[-1] / ranks[0] >= 4:
        axis.set_xscale("log", base=2)


def plot_correctness(results, output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    data = [result for result in results if result["experiment"] == "correctness"]
    figure, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    all_update_errors, all_parameter_errors = [], []
    for result in data:
        label = f"N={result['n']}"
        axes[0].plot(
            result["iterations"], result["update_relative_l2"], marker="o", label=label
        )
        axes[1].plot(
            result["iterations"],
            result["parameter_relative_l2"],
            marker="o",
            label=label,
        )
        axes[2].plot(
            result["iterations"],
            result["update_opposite_fraction"],
            label=f"{label} update",
        )
        axes[2].plot(
            result["iterations"],
            result["parameter_opposite_fraction"],
            linestyle="--",
            label=f"{label} parameter",
        )
        all_update_errors += result["update_relative_l2"]
        all_parameter_errors += result["parameter_relative_l2"]

    if all(value > 0 for value in all_update_errors):
        axes[0].set_yscale("log")
    if all(value > 0 for value in all_parameter_errors):
        axes[1].set_yscale("log")
    axes[2].yaxis.set_major_formatter(PercentFormatter(1.0))
    titles = ("SOAP update relative L2", "Parameter relative L2", "Opposite signs")
    for axis, title in zip(axes, titles):
        axis.set_title(title)
        axis.set_xlabel("Update")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize="small")
    figure.suptitle(f"TP SOAP vs original SOAP ({data[0]['ranks']} ranks)")
    figure.tight_layout()
    path = output / "correctness.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_strong(results, sizes, ranks, output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for n in sizes:
        times = [
            1000 * timing_mean(results, "strong", "tp_soap", n, rank)
            for rank in ranks
        ]
        reference = 1000 * timing_mean(
            results, "strong", "original_soap", n, 1
        )
        line = axes[0].plot(ranks, times, marker="o", label=f"TP N={n}")[0]
        axes[0].scatter(
            [1], [reference], marker="x", color=line.get_color(), label=f"Original N={n}"
        )
        axes[1].plot(
            ranks,
            [times[0] / value for value in times],
            marker="o",
            label=f"N={n}",
        )
    axes[1].plot(
        ranks, [rank / ranks[0] for rank in ranks], "k--", label="Ideal"
    )
    axes[0].set_yscale("log")
    axes[0].set_title("Latency")
    axes[1].set_title("TP speedup from 1 GPU")
    axes[0].set_ylabel("Mean ms / update (refreshes included)")
    axes[1].set_ylabel("Speedup from TP on 1 GPU")
    for axis in axes:
        set_rank_axis(axis, ranks)
        axis.set_xlabel("Ranks / GPUs")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize="small")
    figure.suptitle("SOAP strong scaling")
    figure.tight_layout()
    path = output / "strong_scaling.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_weak(results, cases, output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    ranks = [rank for rank, _ in cases]
    sizes = [n for _, n in cases]
    tp_times = [
        1000 * timing_mean(results, "weak", "tp_soap", n, rank)
        for rank, n in cases
    ]
    reference_times = [
        1000 * timing_mean(results, "weak", "original_soap", n, 1)
        for n in sizes
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    axes[0].plot(ranks, tp_times, marker="o", label="TP SOAP")
    axes[0].axhline(tp_times[0], color="black", linestyle="--", label="Ideal")
    axes[1].plot(sizes, tp_times, marker="o", label="TP SOAP")
    axes[1].plot(sizes, reference_times, marker="x", label="Original SOAP")
    axes[0].set_title("TP compute weak scaling")
    axes[1].set_title("Same matrix sizes")
    axes[0].set_xlabel("Ranks / GPUs")
    axes[1].set_xlabel("Square matrix dimension N")
    for axis in axes:
        axis.set_ylabel("Mean ms / update (refreshes included)")
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    set_rank_axis(axes[0], ranks)
    figure.suptitle(r"Weak scaling: $N(P) \approx N(1)P^{1/3}$")
    figure.tight_layout()
    path = output / "weak_scaling.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def write_log(results, path):
    fields = [
        "experiment",
        "implementation",
        "n",
        "ranks",
        "repeat",
        "iteration",
        "update_relative_l2",
        "parameter_relative_l2",
        "update_opposite_fraction",
        "parameter_opposite_fraction",
        "seconds_per_step",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            if result["experiment"] != "correctness":
                writer.writerow(result)
                continue
            for index, iteration in enumerate(result["iterations"]):
                writer.writerow(
                    {
                        "experiment": "correctness",
                        "implementation": result["implementation"],
                        "n": result["n"],
                        "ranks": result["ranks"],
                        "iteration": iteration,
                        "update_relative_l2": result["update_relative_l2"][index],
                        "parameter_relative_l2": result["parameter_relative_l2"][
                            index
                        ],
                        "update_opposite_fraction": result[
                            "update_opposite_fraction"
                        ][index],
                        "parameter_opposite_fraction": result[
                            "parameter_opposite_fraction"
                        ][index],
                    }
                )


def driver_main(args):
    sizes = parse_list(args.sizes, "--sizes")
    ranks = parse_list(args.ranks, "--ranks")
    if ranks[0] != 1:
        raise ValueError("--ranks must start with 1 for the speedup baseline")
    if min(
        args.iterations,
        args.repeats,
        args.block_size,
        args.refresh_interval,
        args.weak_base_size,
    ) <= 0:
        raise ValueError("iterations, repeats, block size, and refresh must be positive")
    for n in sizes:
        for rank in ranks:
            check_case(n, rank, args.block_size)

    weak_cases = [
        (rank, make_weak_size(args.weak_base_size, rank, args.block_size))
        for rank in ranks
    ]
    weak_sizes = [n for _, n in weak_cases]
    if weak_sizes != sorted(set(weak_sizes)):
        raise ValueError("increase --weak-base-size so weak sizes are increasing")

    results = []
    for n in sizes:
        results.append(
            run_worker(args, "correctness", "correctness", "both", n, ranks[-1])
        )
    for n in sizes:
        for rank in ranks:
            for repeat in range(args.repeats):
                results.append(
                    run_worker(
                        args, "timing", "strong", "tp_soap", n, rank, repeat
                    )
                )
        for repeat in range(args.repeats):
            results.append(
                run_worker(
                    args, "timing", "strong", "original_soap", n, 1, repeat
                )
            )
    for rank, n in weak_cases:
        for repeat in range(args.repeats):
            results.append(
                run_worker(args, "timing", "weak", "tp_soap", n, rank, repeat)
            )
    for n in weak_sizes:
        for repeat in range(args.repeats):
            results.append(
                run_worker(
                    args, "timing", "weak", "original_soap", n, 1, repeat
                )
            )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_correctness(results, output),
        plot_strong(results, sizes, ranks, output),
        plot_weak(results, weak_cases, output),
    ]
    if args.log_file:
        log = args.log_file if args.log_file.is_absolute() else output / args.log_file
        write_log(results, log)
        print(f"wrote {log}")
    for path in paths:
        print(f"wrote {path}")
    return 0


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="512,1024,2048")
    parser.add_argument("--ranks", default="1,2,4,8")
    parser.add_argument("--weak-base-size", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--shard-dim", type=int, choices=(0, 1), default=1)
    parser.add_argument("--refresh-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("soap_plots"))
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--worker", choices=("correctness", "timing"), help=argparse.SUPPRESS)
    parser.add_argument("--experiment", choices=("correctness", "strong", "weak"), help=argparse.SUPPRESS)
    parser.add_argument("--implementation", choices=("both", "tp_soap", "original_soap"), help=argparse.SUPPRESS)
    parser.add_argument("--n", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--world-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def main():
    args = make_parser().parse_args()
    return worker_main(args) if args.worker else driver_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
