import csv
import math
import os
from pathlib import Path
import runpy
import shutil
import socket
import subprocess
import sys
import unittest

import torch
import torch.distributed as dist

from soap_tp import soap_step

ROOT = Path(__file__).resolve().parents[2]
MPI_WORKER = "SOAP_TP_ORIGINAL_SOAP_MPI_WORKER"
MPI_RANKS = 4
MAX_RELATIVE_ERROR = 5e-3

OriginalSOAP = runpy.run_path(ROOT / "tests" / "reference" / "soap.py")["SOAP"]
write_plot = runpy.run_path(ROOT / "scripts" / "plot_soap_comparison.py")["write_plot"]

try:
    from mpi4py import MPI
    from soap_tp import elpa_bindings, slate_bindings  # noqa: F401
except ImportError:
    NATIVE_BINDINGS_AVAILABLE = False
else:
    NATIVE_BINDINGS_AVAILABLE = True


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


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_original_soap_comparison():
    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()
    if world_size < 2:
        raise AssertionError("the original SOAP comparison requires multiple ranks")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in os.environ:
        port = MPI.COMM_WORLD.bcast(_free_port() if rank == 0 else None, root=0)
        os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        records = []
        maxima = {0: [], 1: []}
        grid_rows = math.isqrt(world_size)
        while world_size % grid_rows:
            grid_rows -= 1
        process_grid = (grid_rows, world_size // grid_rows)

        for case, shape in enumerate(SHAPES):
            diagonal_gradient = torch.zeros(shape)
            diagonal = torch.linspace(2.0, 0.5, min(shape))
            diagonal_gradient[range(min(shape)), range(min(shape))] = diagonal
            generator = torch.Generator().manual_seed(90210 + case)
            gradients = [diagonal_gradient]
            gradients += [torch.randn(shape, generator=generator) for _ in range(5)]

            for shard_dim in (0, 1):
                parameter = torch.nn.Parameter(torch.zeros(shape))
                original = OriginalSOAP(
                    [parameter],
                    lr=1.0,
                    betas=(0.8, 0.9),
                    shampoo_beta=0.75,
                    eps=1e-6,
                    weight_decay=0.0,
                    precondition_frequency=2,
                )
                state = {}
                errors = []

                for step, gradient in enumerate(gradients, start=1):
                    parameter.grad = gradient
                    before = parameter.detach().clone()
                    original.step()
                    expected = before - parameter.detach()

                    shard = gradient.chunk(world_size, dim=shard_dim)[rank]
                    local = soap_step(
                        shard.contiguous(),
                        state,
                        global_shape=shape,
                        shard_dim=shard_dim,
                        block_size=2,
                        process_grid_shape=process_grid,
                        preconditioner_beta=0.75,
                        beta1=0.8,
                        beta2=0.9,
                        eps=1e-6,
                        basis_refresh_interval=2,
                    )
                    parts = [torch.empty_like(local) for _ in range(world_size)]
                    dist.all_gather(parts, local)
                    merged = torch.cat(parts, dim=shard_dim)
                    difference = (merged - expected).norm()
                    error = float(difference / expected.norm().clamp_min(1e-30))
                    errors.append(error)
                    if rank == 0:
                        layout = ("row", "column")[shard_dim]
                        records.append(
                            (world_size, *shape, layout, step, error, error * 100)
                        )

                maxima[shard_dim].append(max(errors))

        worst = max(map(max, maxima.values()))
        if rank == 0:
            output = Path(
                os.environ.get(
                    "SOAP_TP_ORACLE_OUTPUT",
                    "outputs/original_soap_comparison",
                )
            )
            output.mkdir(parents=True, exist_ok=True)
            with (output / "errors.csv").open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    (
                        "ranks",
                        "rows",
                        "columns",
                        "shard_layout",
                        "gradient",
                        "relative_l2_error",
                        "difference_percent",
                    )
                )
                writer.writerows(records)
            write_plot(output / "errors.csv", output / "errors.png")
            print(
                f"original SOAP comparison: ranks={world_size}, "
                f"comparisons={len(records)}, "
                f"row_max={max(maxima[0]):.3e}, column_max={max(maxima[1]):.3e}",
                flush=True,
            )

        if worst >= MAX_RELATIVE_ERROR:
            raise AssertionError(
                f"original SOAP comparison relative error {worst} "
                f">= {MAX_RELATIVE_ERROR}"
            )
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(
    NATIVE_BINDINGS_AVAILABLE,
    "the original SOAP comparison requires installed ELPA and SLATE bindings",
)
class TestOriginalSoapComparison(unittest.TestCase):
    def test_row_and_column_shards_match_original_soap(self):
        if MPI.COMM_WORLD.Get_size() > 1:
            _run_original_soap_comparison()
            return

        mpiexec = shutil.which("mpiexec")
        if mpiexec is None:
            self.skipTest("mpiexec is required for the original SOAP comparison")
        environment = os.environ.copy()
        environment.update(
            {
                MPI_WORKER: "1",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(_free_port()),
                "MKL_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            }
        )
        completed = subprocess.run(
            [
                mpiexec,
                "--oversubscribe",
                "--bind-to",
                "none",
                "-n",
                str(MPI_RANKS),
                sys.executable,
                str(Path(__file__).resolve()),
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if completed.stdout:
            print(completed.stdout.strip(), flush=True)
        self.assertEqual(
            completed.returncode,
            0,
            msg=(
                "MPI comparison failed\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            ),
        )


if __name__ == "__main__":
    if os.environ.get(MPI_WORKER) == "1":
        _run_original_soap_comparison()
    else:
        unittest.main()
