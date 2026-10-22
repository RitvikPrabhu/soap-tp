"""Tiny CPU basis-sensitivity experiment, not distributed training validation.

PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python research/frontier_eigh/soap_precision.py
Uses the real Adam primitive and SOAP projection/epsilon conventions. All update
arithmetic is FP64 to isolate basis error; FP16/BF16 modes round inputs only.
Add --solver build/custom-eigh/eigh to use our GPU solver's exported tiny bases.
"""
import argparse
import csv
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

from soap_tp.ops.optimizer import adam_update

parser = argparse.ArgumentParser()
parser.add_argument("--solver", help="custom eigh executable, one rank/GCD per tiny solve")
args = parser.parse_args()
torch.set_num_threads(1)
writer = csv.DictWriter(sys.stdout, [
    "seed", "condition", "variant", "steps", "residual_max", "orthogonality_max",
    "negative_eigenvalues", "update_first_rel", "update_last_rel", "update_max_rel",
    "update_last_cosine",
], lineterminator="\n")
writer.writeheader()


def eigenbasis(a, variant):
    if args.solver:
        if variant == "signed_permuted":
            w, q = eigenbasis(a, "fp64")
            return w.flip(0), q.flip(1) * (1 - 2 * (torch.arange(len(w)) % 2))
        with tempfile.TemporaryDirectory(dir=".amp/in") as directory:
            source, output = Path(directory) / "input.bin", Path(directory) / "basis.bin"
            a.T.contiguous().numpy().tofile(source)
            precision = "fp64" if variant == "fp64" else "fp32"
            command = [args.solver, "--n", str(len(a)), "--block", "16", "--back-block", "32",
                       "--input", str(source), "--export-basis", str(output), "--matrix", "supplied",
                       "--precision", precision, "--tridiag", "elpa" if precision == "fp64" else "elpa32"]
            if variant in ("fp16_input", "bf16_input"):
                command += ["--quant", variant.split("_")[0]]
            if variant == "split_half":
                command += ["--half-bulk", "2"]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if result.returncode:
                raise RuntimeError(result.stdout)
            values = np.fromfile(output, dtype=np.float64)
            n = len(a)
            w = torch.from_numpy(values[:n].copy()).flip(0)
            q = torch.from_numpy(values[n:].reshape(n, n, order="F").copy()).flip(1)
            return w, q
    if variant == "fp16_input":
        a = a.half().float()
    elif variant == "bf16_input":
        a = a.bfloat16().float()
    elif variant == "fp32":
        a = a.float()
    w, q = torch.linalg.eigh(a)
    w, q = w.flip(0).double(), q.flip(1).double()
    if variant == "signed_permuted":
        w, q = w.flip(0), q.flip(1) * (1 - 2 * (torch.arange(a.shape[0]) % 2))
    return w, q


for seed in (42, 43, 44):
    for condition in (1e2, 1e6, 1e9):
        generator = torch.Generator().manual_seed(seed)
        matrices, factors = [], []
        for n in (96, 64):
            u, _ = torch.linalg.qr(torch.randn(n, n, generator=generator, dtype=torch.float64))
            spectrum = torch.logspace(0, -math.log10(condition), n, dtype=torch.float64)
            a = (u * spectrum) @ u.T
            # Existing SOAP preconditioners are FP32, even when ELPA solves FP64.
            matrices.append(((a + a.T) * 0.5).float().double())
            factors.append(u * spectrum.sqrt())
        gradients = [
            (factors[0] @ torch.randn(96, 64, generator=generator, dtype=torch.float64)
             @ factors[1].T).float().double()
            for _ in range(20)
        ]
        reference = []
        variants = ["fp64", "signed_permuted", "fp32", "fp16_input", "bf16_input"]
        if args.solver:
            variants.append("split_half")
        for variant in variants:
            pairs = [eigenbasis(a, variant) for a in matrices]
            left, right = [q for _, q in pairs]
            momentum = torch.zeros(96, 64, dtype=torch.float64)
            variance = torch.zeros_like(momentum)
            errors = []
            for step, gradient in enumerate(gradients, 1):
                projected = left.T @ gradient @ right
                coordinate = adam_update(projected, momentum, variance, step,
                                         eps=1e-8 / math.sqrt(1 - 0.999**step))
                update = left @ coordinate @ right.T
                if variant == "fp64":
                    reference.append(update)
                errors.append(float(torch.linalg.norm(update - reference[step - 1])
                                    / torch.linalg.norm(reference[step - 1])))
            residuals, orthogonalities = [], []
            for a, (w, q) in zip(matrices, pairs):
                residuals.append(float(torch.linalg.norm(a @ q - q * w) / torch.linalg.norm(a)))
                orthogonalities.append(float(torch.linalg.norm(q.T @ q - torch.eye(len(w)))))
            cosine = float((update * reference[-1]).sum()
                           / (torch.linalg.norm(update) * torch.linalg.norm(reference[-1])))
            writer.writerow(dict(seed=seed, condition=condition, variant=variant, steps=len(gradients),
                                 residual_max=max(residuals), orthogonality_max=max(orthogonalities),
                                 negative_eigenvalues=sum(int((w < 0).sum()) for w, _ in pairs),
                                 update_first_rel=errors[0], update_last_rel=errors[-1],
                                 update_max_rel=max(errors), update_last_cosine=cosine))
            if variant == "signed_permuted":
                assert max(errors) < 1e-7, "SOAP update must be invariant to basis signs/permutations"
