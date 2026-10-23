"""Same-input tiny correctness checks; no timing claims and no job submission.

Inputs: native FP64 column-major, exactly representable FP32 symmetric matrices.
Exports: n FP64 eigenvalues followed by n*n FP64 column-major eigenvectors.
Example: .venv/bin/python research/frontier_eigh/torch_compare.py --device cpu
Use --run-custom only with an appropriate module environment and allocation.
--launcher 'srun -N1 -n8 --ntasks-per-node=8 -c7 --gpus-per-task=1 --gpu-bind=closest'
--pr 2 --pc 4 runs distributed solves. Without it, existing exports are analyzed.
half and split_half mean FP16 bulk GEMMs with FP32 state, NOT an FP16 eigensolver.
Input-only rounding is deliberately not used as a substitute for FP16 computation.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import torch


def metrics(a, w, q, wr, qr, groups):
    order = torch.argsort(w)
    w, q = w[order], q[:, order]
    n = len(w)
    scale = torch.linalg.vector_norm(a)
    result = {
        "residual": float(torch.linalg.vector_norm(a @ q - q * w) / scale),
        "orthogonality": float(torch.linalg.vector_norm(q.T @ q - torch.eye(n, dtype=q.dtype)) / math.sqrt(n)),
        "q_norm_relative": float(abs(torch.linalg.vector_norm(q) - math.sqrt(n)) / math.sqrt(n)),
        "eigenvalue_relative": float(torch.linalg.vector_norm(w - wr) / torch.linalg.vector_norm(wr)),
        # Absolute error scaled by spectral radius, not unstable division by tiny eigenvalues.
        "eigenvalue_max_scaled": float(torch.max(abs(w - wr)) / torch.max(abs(wr))),
    }
    errors = []
    for lo, hi in groups:
        # QR removes basis scaling, checked separately by orthogonality/norm metrics.
        v = torch.linalg.qr(q[:, lo:hi]).Q
        r = qr[:, lo:hi]
        errors.append(float(torch.linalg.vector_norm(v - r @ (r.T @ v)) / math.sqrt(hi - lo)))
    result["eigenspace_max"] = max(errors, default=0.0)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=Path("output/custom-eigh/torch-loop2"))
    p.add_argument("--n", type=int, default=97)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--scales", type=float, nargs="+", default=[1.0])
    p.add_argument("--cases", nargs="+", choices=["spd", "indefinite", "clustered", "illconditioned"],
                   default=["spd", "indefinite", "clustered", "illconditioned"])
    p.add_argument("--device", choices=["cpu", "rocm"], default="cpu")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--run-custom", action="store_true")
    p.add_argument("--binary", type=Path, default=Path("build/custom-eigh/pool-eigh"))
    p.add_argument("--launcher", default="")
    p.add_argument("--pr", type=int, default=1)
    p.add_argument("--pc", type=int, default=1)
    p.add_argument("--panel", type=int, default=1)
    p.add_argument("--half-panel", type=int, choices=[0, 1], default=0)
    p.add_argument("--panel-solution", type=int, default=0)
    p.add_argument("--solver", choices=["custom", "jacobi"], default="custom")
    p.add_argument("--jacobi-pivot64", type=int, choices=[0, 1], default=0)
    p.add_argument("--jacobi-local", choices=["dc", "jacobi"], default="dc")
    p.add_argument("--variants", nargs="+", choices=["fp64", "fp32", "half", "split_half"],
                   default=["fp64", "fp32", "half", "split_half"])
    args = p.parse_args()
    if args.panel_solution and (args.variants != ["fp32"] or args.half_panel or args.solver != "custom" or args.panel <= 1):
        p.error("panel-solution requires native fp32 only, custom solver and panel>1")
    if not 2 <= args.n < 512:
        p.error("only tiny correctness matrices 2 <= n < 512 are permitted")
    if any(not math.isfinite(scale) or scale <= 0 for scale in args.scales):
        p.error("scales must be positive and finite")
    if args.pr < 1 or args.pc < 1 or (args.run_custom and args.pr * args.pc > 1 and not args.launcher):
        p.error("positive process grid and an explicit launcher for distributed runs required")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = {"torch_version": torch.__version__, "torch_path": torch.__file__,
                "hip": torch.version.hip, "cuda_build": torch.version.cuda,
                "gpu_available": torch.cuda.is_available(), "requested_backend": args.device,
                "python": sys.executable, "numpy": np.__version__,
                "custom_solver": args.solver, "custom_panel": args.panel, "jacobi_pivot64": args.jacobi_pivot64,
                "half_panel": args.half_panel,
                "panel_solution": args.panel_solution,
                "scales": args.scales, "jacobi_local": args.jacobi_local,
                "note": "correctness only; torch FP16 eigh is probed, never silently promoted"}
    device = "cpu"
    if args.device == "rocm":
        if not torch.version.hip or not torch.cuda.is_available():
            p.error("ROCm requested but this torch is not an available HIP build")
        device = "cuda"  # PyTorch uses the cuda API name for HIP too.
        evidence["gpu_name"] = torch.cuda.get_device_name(0)
        try:
            torch.linalg.eigh(torch.eye(2, device=device, dtype=torch.float16))
            evidence["fp16_eigh"] = "supported"
        except (RuntimeError, NotImplementedError) as exc:
            evidence["fp16_eigh"] = str(exc)
    if args.binary.exists():
        evidence["custom_binary"] = str(args.binary.resolve())
        evidence["custom_sha256"] = hashlib.sha256(args.binary.read_bytes()).hexdigest()
    (args.output / "backend.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence), flush=True)
    rows, details, failures, missing_exports, reference_failures = [], [], [], [], []
    eps = torch.finfo(torch.float32).eps
    n = args.n
    for seed in args.seeds:
        gen = torch.Generator().manual_seed(seed)
        u = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64)).Q
        for case, scale in itertools.product(args.cases, args.scales):
            stem = f"{case}-s{seed}-n{n}" + (f"-scale{scale:g}" if scale != 1 else "")
            spectrum = torch.linspace(0.1, 1, n, dtype=torch.float64)
            if case == "indefinite":
                spectrum = torch.linspace(-1, 1, n, dtype=torch.float64)
            elif case == "clustered":
                # Repeated groups plus a tight pair; boundaries are inferred after FP32 rounding.
                spectrum = (torch.arange(n, dtype=torch.float64) // 4 + 1) / math.ceil(n / 4)
                spectrum[:2] = spectrum[0] + torch.tensor([0., 1e-7], dtype=torch.float64)
            elif case == "illconditioned":
                spectrum = torch.logspace(-7, 0, n, dtype=torch.float64)
            a = (u * spectrum) @ u.T
            a = ((a + a.T) * (scale / 2)).float().double()
            input_path = args.output / f"{stem}.input.bin"
            a.T.contiguous().numpy().tofile(input_path)
            if args.prepare_only:
                continue
            wr, qr = torch.linalg.eigh(a)  # CPU FP64 reference on exactly the FP32 input.
            torch_error = None
            try:
                wf, qf = torch.linalg.eigh(a.float().to(device))
                wf, qf = wf.cpu().double(), qf.cpu().double()
            except torch.linalg.LinAlgError as exc:
                torch_error = str(exc)
                reference_failures.append({"case": stem, "error": torch_error})
                print(f"TORCH_REFERENCE_FAILED {stem}: {torch_error}", flush=True)
            radius = float(torch.max(abs(wr)))
            # Never cut a repeated/unresolved subspace. The same groups apply to all solvers.
            cuts = [0] + [i for i in range(1, n) if float(wr[i] - wr[i - 1]) > 64 * eps * radius] + [n]
            groups = [(lo, hi) for lo, hi in zip(cuts[:-1], cuts[1:]) if hi - lo < n]
            gaps = [float(wr[i] - wr[i - 1]) for i in cuts[1:-1]]
            baseline = metrics(a, wf, qf, wr, qr, groups) if torch_error is None else {}
            # Bounded torch-matching policy: 8x measured torch error, precision floor,
            # and an independent hard ceiling. No universal loose 1e-4 residual gate.
            floor = 4 * eps * math.sqrt(n)
            cap = 64 * eps * math.sqrt(n)
            # If FP32 torch cannot solve the input, enforce only the precision floors;
            # retain the FP64 reference and explicitly report the missing comparison.
            limits = {k: min(cap, max(floor, 8 * baseline.get(k, 0)))
                      for k in metrics(a, wr, qr, wr, qr, groups)}
            limits["q_norm_relative"] = min(8 * eps, max(2 * eps, 4 * baseline.get("q_norm_relative", 0)))
            space_floor = min(0.05, 4 * eps * radius / min(gaps)) if gaps else floor
            limits["eigenspace_max"] = min(0.05, max(space_floor, 8 * baseline.get("eigenspace_max", 0)))
            details.append({"case": stem, "groups": groups, "gap_min": min(gaps, default=None),
                            "limits": limits, "torch_fp32_error": torch_error,
                            "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest()})
            candidates = [("torch_fp64_cpu", wr, qr)]
            if torch_error is None:
                candidates.append((f"torch_fp32_{args.device}", wf, qf))
            for variant in args.variants:
                export = args.output / f"{stem}.{variant}.basis.bin"
                if args.run_custom:
                    export.unlink(missing_ok=True)  # A failed solve must not reuse a stale export.
                    extra = ["--half-bulk", {"half": "1", "split_half": "2"}[variant]] if variant in ("half", "split_half") else []
                    extra += ["--device-vectors", "1"] if args.solver == "custom" else ["--solver", "jacobi", "--jacobi-pivot64", str(args.jacobi_pivot64), "--jacobi-local", args.jacobi_local]
                    if args.half_panel and variant != "fp64":
                        extra += ["--half-panel", "1"]
                    if args.panel_solution:
                        extra += ["--panel-solution", str(args.panel_solution)]
                    cmd = shlex.split(args.launcher) + [str(args.binary.resolve()), "--n", str(n),
                        "--block", "16", "--back-block", "32", "--pr", str(args.pr), "--pc", str(args.pc),
                        "--precision", "fp64" if variant == "fp64" else "fp32", "--tridiag",
                        "elpa" if variant == "fp64" else "elpa32", "--input", str(input_path.resolve()),
                        "--export-basis", str(export.resolve()),
                        "--gpu-mpi", "1", "--rccl", "1", "--async-alloc", "1",
                        "--panel", str(args.panel)] + extra
                    print(shlex.join(cmd), flush=True)
                    try:
                        run = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                        (args.output / f"{stem}.{variant}.log").write_text(shlex.join(cmd) + "\n" + run.stdout + run.stderr)
                        if run.returncode:
                            failures.append(f"{stem}/{variant}: custom exit {run.returncode}")
                            continue
                    except subprocess.TimeoutExpired:
                        failures.append(f"{stem}/{variant}: timeout")
                        continue
                if not export.exists():
                    missing_exports.append(str(export))
                    continue
                values = np.fromfile(export, dtype=np.float64)
                if values.size != n + n * n or not np.isfinite(values).all():
                    failures.append(f"{export}: malformed/nonfinite export")
                    continue
                candidates.append((f"custom_{variant}", torch.from_numpy(values[:n].copy()),
                                   torch.from_numpy(values[n:].reshape(n, n, order="F").copy())))
            for name, w, q in candidates:
                m = metrics(a, w, q, wr, qr, groups)
                passed = all(math.isfinite(v) and v <= limits[k] for k, v in m.items())
                rows.append({"case": stem, "solver": name, **m, "accepted": passed})
                if not passed:
                    failures.append(f"{stem}/{name}: " + ", ".join(k for k, v in m.items() if not math.isfinite(v) or v > limits[k]))
    if rows:
        with (args.output / "metrics.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    (args.output / "acceptance.json").write_text(json.dumps({"policy": "8x measured FP32 torch, 4eps*sqrt(n) floor, 64eps*sqrt(n) hard cap; relative Q norm min(8eps,max(2eps,4x torch)); eigenspaces gap-aware floor, 0.05 cap", "cases": details, "failures": failures,
        "missing_exports_not_evaluated": missing_exports,
        "torch_reference_failures": reference_failures}, indent=2) + "\n")
    if missing_exports:
        print(f"NOTE: {len(missing_exports)} absent custom exports were NOT evaluated", file=sys.stderr)
    for row in rows:
        print(json.dumps(row))
    if reference_failures:
        print(f"NOTE: {len(reference_failures)} FP32 torch reference failures; those custom cases use precision floors, not a measured torch match", file=sys.stderr)
    if failures:
        print("FAILED: " + "; ".join(failures), file=sys.stderr)
        return 1
    return int(bool(reference_failures))


if __name__ == "__main__":
    raise SystemExit(main())
