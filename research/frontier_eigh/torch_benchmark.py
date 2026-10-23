"""Independent same-matrix single-GCD reference/control, NOT a distributed algorithm.

Native torch eigh only; setup, exports and FP64 diagnostics are outside timings.
CPU is for tiny validation only. No distributed solver state is consumed.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import shlex
import sys
import time

import numpy as np
import torch


def random_matrix(n, block_columns=128):
    """C++ entry(i,j,n,"random"), uint64 overflow intentional; O(n*block) scratch."""
    a = np.empty((n, n), dtype=np.float64, order="F")
    i = np.arange(1, n + 1, dtype=np.uint64)[:, None]
    for start in range(0, n, block_columns):
        j = np.arange(start + 1, min(n, start + block_columns) + 1,
                      dtype=np.uint64)[None, :]
        x = (np.minimum(i, j) * np.uint64(0x9E3779B97F4A7C15)
             + np.maximum(i, j) * np.uint64(0xBF58476D1CE4E5B9))
        x = (x ^ (x >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> 27)) * np.uint64(0x94D049BB133111EB)
        x ^= x >> 31
        a[:, start:start + j.size] = ((x >> 11).astype(np.float64) * 2.0**-53 - .5) / math.sqrt(n)
    d = np.arange(n)
    a[d, d] += 3.0 + d / n
    return a


def diagnostics(a, rounded, w, q, reference):
    """All products/reductions FP64, normalized Frobenius errors explicitly named."""
    n = a.shape[0]
    w64, q64 = w.double(), q.double()
    qnorm = float(torch.linalg.vector_norm(q64))
    result = {"q_norm_fro": qnorm,
              "q_norm_relative": abs(qnorm - math.sqrt(n)) / math.sqrt(n)}
    qw = q64 * w64
    for name, matrix in (("original", a), ("rounded", rounded)):
        norm = float(torch.linalg.vector_norm(matrix))
        error = matrix @ q64
        error.sub_(qw)
        raw = float(torch.linalg.vector_norm(error))
        del error
        result[f"{name}_norm_fro"] = norm
        result[f"residual_{name}_raw"] = raw
        result[f"residual_{name}_normalized"] = raw / norm
    del qw
    gram = q64.T @ q64
    gram.diagonal().sub_(1)
    raw = float(torch.linalg.vector_norm(gram))
    del gram
    result.update(orthogonality_raw=raw, orthogonality_normalized=raw / math.sqrt(n))
    delta = w64 - reference
    result["eigenvalue_original_fp64_relative"] = float(
        torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference))
    result["eigenvalue_original_fp64_max_scaled"] = float(delta.abs().max() / reference.abs().max())
    delta_a = rounded - a
    result["input_rounding_relative"] = float(torch.linalg.vector_norm(delta_a) / torch.linalg.vector_norm(a))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=[8192, 16384])
    p.add_argument("--device", choices=["cpu", "rocm"], default="rocm")
    p.add_argument("--output", type=Path, default=Path("output/custom-eigh/torch-large"))
    p.add_argument("--block-columns", type=int, default=128)
    args = p.parse_args()
    if min(args.sizes) < 2 or args.block_columns < 1:
        p.error("sizes >= 2 and block-columns >= 1 required")
    if args.device == "cpu" and max(args.sizes) >= 512:
        p.error("CPU allowed only for tiny validation (n < 512)")
    if args.device == "rocm" and (not torch.version.hip or not torch.cuda.is_available()):
        p.error("an available ROCm torch build is required")
    if int(os.environ.get("SLURM_NTASKS", "1")) != 1:
        p.error("launch exactly one rank/GCD, not replicated references")
    device = "cuda" if args.device == "rocm" else "cpu"
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {"role": "single-GCD reference/control; not distributed algorithm",
                "command": shlex.join([sys.executable, *sys.argv]),
                "torch": torch.__version__, "numpy": np.__version__,
                "hip": torch.version.hip, "cuda_build": torch.version.cuda,
                "device": device, "host": platform.node(),
                "torch_config": torch.__config__.show(),
                "linalg_backend": str(torch.backends.cuda.preferred_linalg_library()),
                "environment": {k: v for k, v in os.environ.items() if k.startswith(
                    ("SLURM_", "ROCR_", "HIP_", "CUDA_", "OMP_", "OPENBLAS_", "LOADEDMODULES"))},
                "reference_format": "native FP64 ascending values; largest min(8,n) eigenvectors, n by k column-major; no header",
                "normalization": "residual / ||A||F; orthogonality / sqrt(n); q norm error / sqrt(n)",
                "timing": "first/cold + two warm eigh calls per dtype; synchronization around call; diagnostics excluded",
                "memory": "peak torch allocated bytes, not total HIP/rocSOLVER device usage",
                "results": []}
    if device == "cuda":
        metadata["gpu"] = torch.cuda.get_device_name(0)
    try:
        torch.linalg.eigh(torch.eye(2, dtype=torch.float16, device=device))
        metadata["native_half_eigh"] = "supported by this backend (not benchmarked)"
    except (RuntimeError, NotImplementedError) as exc:
        metadata["native_half_eigh"] = f"unsupported; no promotion: {exc}"
    print("TORCH_BACKEND " + json.dumps({k: metadata[k] for k in
          ("role", "torch", "hip", "native_half_eigh")}), flush=True)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    csv_path = args.output / "metrics.csv"
    with csv_path.open("w", newline="") as stream:
        writer = None
        for n in args.sizes:
            host = random_matrix(n, args.block_columns)
            a = torch.from_numpy(host).to(device)
            del host
            reference = None
            for dtype in (torch.float64, torch.float32):
                rounded_native = a.to(dtype)
                sync()
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                times = []
                for repeat in range(3):
                    sync()
                    begin = time.perf_counter()
                    w, q = torch.linalg.eigh(rounded_native, UPLO="L")
                    sync()
                    elapsed = time.perf_counter() - begin
                    times.append(elapsed)
                    print(f"TORCH_TIMING n={n} dtype={dtype} run={repeat} phase={'cold' if repeat == 0 else 'warm'} seconds={elapsed:.9g} role=single-GCD", flush=True)
                    if repeat < 2:
                        del w, q
                solver_peak = torch.cuda.max_memory_allocated() if device == "cuda" else 0
                if dtype == torch.float64:
                    reference = w.clone()
                    prefix = args.output / f"n{n}"
                    w.cpu().numpy().tofile(str(prefix) + ".values.bin")
                    q[:, -min(8, n):].T.contiguous().cpu().numpy().tofile(str(prefix) + ".top.bin")
                rounded = rounded_native.double()
                row = {"n": n, "dtype": str(dtype), "cold_seconds": times[0],
                       "warm1_seconds": times[1], "warm2_seconds": times[2],
                       "solver_peak_torch_bytes": solver_peak,
                       **diagnostics(a, rounded, w, q, reference)}
                sync()
                row["total_peak_torch_bytes"] = torch.cuda.max_memory_allocated() if device == "cuda" else 0
                print("TORCH_RESULT " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                metadata["results"].append(row)
                (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
                del rounded, rounded_native, w, q
            del a, reference
            if device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
