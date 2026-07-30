#!/usr/bin/env python3
"""Compare rank-zero SOAP_PROFILE JSON lines from reference and SOAP-TP.

Expected line format (other text in the log is ignored):

    SOAP_PROFILE {"impl":"reference","step":1,"stage":"gradient",
                  "shape":[20,20],"values":[...]}

``impl`` may also be ``ref``, ``src``, or ``source``.  Tensor data may instead
be nested under ``"tensor": {"shape": ..., "values": ...}``.

Eigenvector columns are sign-ambiguous.  This script aligns SOAP-TP basis
columns to the reference and applies the corresponding coordinate signs to
projected gradients, momentum, and projected/coordinate updates.  It does not
alter physical-space tensors or variance. Comparisons use relative L2 error;
strict elementwise opposite signs are reported separately.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


PREFIX = "SOAP_PROFILE "

# Canonical stage ordering. Unknown stages are printed afterward in name order.
STAGE_ORDER = (
    "gradient",
    "left_preconditioner",
    "right_preconditioner",
    "left_basis",
    "right_basis",
    "projected_gradient",
    "momentum_after_adam",
    "variance_after_adam",
    "denominator",
    "coordinate_update",
    "backrotated_update",
    "returned_update",
    "left_preconditioner_after_step",
    "right_preconditioner_after_step",
    "left_basis_after_step",
    "right_basis_after_step",
    "momentum_after_step",
    "variance_after_step",
    "parameter_after_step",
)
STAGE_INDEX = {name: index for index, name in enumerate(STAGE_ORDER)}

# Accept the short names proposed while the profiler instrumentation was being
# added, plus a few explicit/reference-internal spellings.
ALIASES = {
    "input.gradient": "gradient",
    "preconditioner.ema.0": "left_preconditioner",
    "preconditioner.ema.1": "right_preconditioner",
    "basis.initial.output.0": "left_basis",
    "basis.initial.output.1": "right_basis",
    "adam.m": "momentum_after_adam",
    "momentum": "momentum_after_adam",
    "adam.v": "variance_after_adam",
    "variance": "variance_after_adam",
    "projected_update": "coordinate_update",
    "soap_update": "returned_update",
    "left_preconditioner_next": "left_preconditioner_after_step",
    "right_preconditioner_next": "right_preconditioner_after_step",
    "left_basis_next": "left_basis_after_step",
    "right_basis_next": "right_basis_after_step",
    "momentum_next": "momentum_after_step",
    "variance_next": "variance_after_step",
    "parameter.after": "parameter_after_step",
}

IMPL_ALIASES = {
    "ref": "reference",
    "reference": "reference",
    "tp": "tp",
    "src": "tp",
    "source": "tp",
    "soap-tp": "tp",
    "soap_tp": "tp",
}

CURRENT_LEFT_BASIS = ("left_basis",)
CURRENT_RIGHT_BASIS = ("right_basis",)
NEXT_LEFT_BASIS = ("left_basis_after_step",)
NEXT_RIGHT_BASIS = ("right_basis_after_step",)

SIGNED_COORDINATE_STAGES = {
    "projected_gradient",
    "momentum_after_adam",
    "coordinate_update",
}
SIGNED_NEXT_COORDINATE_STAGES = {
    "momentum_after_step",
}


@dataclass(frozen=True)
class Event:
    impl: str
    step: int
    stage: str
    tensor: torch.Tensor
    source: str
    line_number: int


@dataclass(frozen=True)
class BasisAlignment:
    signs: torch.Tensor
    flipped: int
    minimum_absolute_dot: float
    maximum_off_diagonal_overlap: float


@dataclass(frozen=True)
class Comparison:
    relative_l2_error: float
    difference_norm: float
    reference_norm: float
    opposite_signs: int
    total_elements: int
    max_residual_index: tuple[int, ...]
    reference_value: float
    tp_value: float
    identical: bool


def _canonical_stage(stage: str) -> str:
    stage = stage.strip()
    if stage in ALIASES:
        return ALIASES[stage]

    # Tolerate the explicit names used by the source/ref profiler drafts.
    replacements = {
        "_next": "_after_step",
        ".next": "_after_step",
    }
    for suffix, replacement in replacements.items():
        if stage.endswith(suffix):
            stage = stage[: -len(suffix)] + replacement
    return ALIASES.get(stage, stage)


def _extract_tensor(payload: dict, source: str, line_number: int) -> torch.Tensor:
    tensor_payload = payload.get("tensor")
    if isinstance(tensor_payload, dict):
        shape = tensor_payload.get("shape")
        values = tensor_payload.get("values")
    else:
        shape = payload.get("shape")
        values = payload.get("values")

    if shape is None or values is None:
        raise ValueError(
            f"{source}:{line_number}: profile event needs shape and values"
        )

    shape_tuple = tuple(int(dimension) for dimension in shape)
    tensor = torch.tensor(values, dtype=torch.float64).reshape(-1)
    expected = math.prod(shape_tuple)
    if tensor.numel() != expected:
        raise ValueError(
            f"{source}:{line_number}: shape {shape_tuple} needs {expected} "
            f"values, found {tensor.numel()}"
        )
    return tensor.reshape(shape_tuple)


def _events_from_lines(lines: Iterable[str], source: str) -> list[Event]:
    events = []
    decoder = json.JSONDecoder()
    for line_number, line in enumerate(lines, 1):
        prefix_index = line.find(PREFIX)
        if prefix_index < 0:
            continue
        encoded = line[prefix_index + len(PREFIX) :].lstrip()
        try:
            payload, _ = decoder.raw_decode(encoded)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{source}:{line_number}: invalid SOAP_PROFILE JSON: {error}"
            ) from error

        raw_impl = str(payload.get("impl", "")).lower()
        if raw_impl not in IMPL_ALIASES:
            raise ValueError(
                f"{source}:{line_number}: unknown impl {raw_impl!r}"
            )
        if "step" not in payload:
            raise ValueError(f"{source}:{line_number}: profile event needs step")
        if "stage" not in payload:
            raise ValueError(f"{source}:{line_number}: profile event needs stage")

        events.append(
            Event(
                impl=IMPL_ALIASES[raw_impl],
                step=int(payload["step"]),
                stage=_canonical_stage(str(payload["stage"])),
                tensor=_extract_tensor(payload, source, line_number),
                source=source,
                line_number=line_number,
            )
        )
    return events


def _read_events(paths: list[Path]) -> list[Event]:
    if not paths:
        return _events_from_lines(sys.stdin, "<stdin>")

    events = []
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as handle:
            events.extend(_events_from_lines(handle, str(path)))
    return events


def _split_profile_runs(events: list[Event]) -> list[list[Event]]:
    """Split logs containing multiple parameterized SOAP comparisons."""
    runs: list[list[Event]] = []
    current: list[Event] = []
    seen: set[tuple[str, int, str]] = set()
    implementations: set[str] = set()

    for event in events:
        key = (event.impl, event.step, event.stage)
        starts_another_run = (
            event.stage == "gradient"
            and key in seen
            and implementations == {"reference", "tp"}
        )
        if starts_another_run:
            runs.append(current)
            current = []
            seen = set()
            implementations = set()

        current.append(event)
        seen.add(key)
        implementations.add(event.impl)

    if current:
        runs.append(current)
    return runs


def _profile_run_description(events: list[Event]) -> str:
    gradient = next(
        (
            event
            for event in events
            if event.impl == "reference" and event.stage == "gradient"
        ),
        events[0],
    )
    shape = "x".join(str(dimension) for dimension in gradient.tensor.shape)
    first = events[0]
    last = events[-1]
    if first.source == last.source:
        location = f"{first.source}:{first.line_number}-{last.line_number}"
    else:
        location = (
            f"{first.source}:{first.line_number} through "
            f"{last.source}:{last.line_number}"
        )
    return f"shape={shape}, {location}"


def _index_events(events: list[Event]) -> dict[str, dict[tuple[int, str], Event]]:
    indexed: dict[str, dict[tuple[int, str], Event]] = {
        "reference": {},
        "tp": {},
    }
    for event in events:
        key = (event.step, event.stage)
        previous = indexed[event.impl].get(key)
        if previous is not None:
            if torch.equal(previous.tensor, event.tensor):
                continue
            raise ValueError(
                f"duplicate {event.impl} step={event.step} stage={event.stage}: "
                f"{previous.source}:{previous.line_number} and "
                f"{event.source}:{event.line_number}"
            )
        indexed[event.impl][key] = event
    return indexed


def _basis_alignment(
    reference: torch.Tensor,
    tp: torch.Tensor,
) -> BasisAlignment:
    if reference.ndim != 2 or tp.ndim != 2:
        raise ValueError("basis tensors must be matrices")
    if reference.shape != tp.shape:
        raise ValueError(
            f"basis shape mismatch: reference={tuple(reference.shape)}, "
            f"TP={tuple(tp.shape)}"
        )

    dots = (reference * tp).sum(dim=0)
    signs = torch.where(dots < 0, -torch.ones_like(dots), torch.ones_like(dots))
    overlap = reference.T @ tp
    if overlap.numel() <= 1:
        maximum_off_diagonal = 0.0
    else:
        diagonal_mask = torch.eye(
            overlap.shape[0], dtype=torch.bool, device=overlap.device
        )
        maximum_off_diagonal = float(overlap.masked_fill(diagonal_mask, 0).abs().max())

    return BasisAlignment(
        signs=signs,
        flipped=int((signs < 0).sum()),
        minimum_absolute_dot=float(dots.abs().min()),
        maximum_off_diagonal_overlap=maximum_off_diagonal,
    )


def _find_alignment(
    indexed: dict[str, dict[tuple[int, str], Event]],
    step: int,
    candidate_stages: tuple[str, ...],
) -> BasisAlignment | None:
    for stage in candidate_stages:
        ref_event = indexed["reference"].get((step, stage))
        tp_event = indexed["tp"].get((step, stage))
        if ref_event is not None and tp_event is not None:
            return _basis_alignment(ref_event.tensor, tp_event.tensor)
    return None


def _find_current_alignment(
    indexed: dict[str, dict[tuple[int, str], Event]],
    step: int,
    current_stages: tuple[str, ...],
    after_stages: tuple[str, ...],
) -> BasisAlignment | None:
    """Find the basis active during ``step``, even if it was emitted earlier."""
    direct = _find_alignment(indexed, step, current_stages)
    if direct is not None:
        return direct

    # A post-step basis from a prior step supersedes that step's input basis.
    candidates: list[tuple[int, int, str]] = []
    for candidate_step in range(step):
        for stage in current_stages:
            if (
                (candidate_step, stage) in indexed["reference"]
                and (candidate_step, stage) in indexed["tp"]
            ):
                candidates.append((candidate_step, 0, stage))
        for stage in after_stages:
            if (
                (candidate_step, stage) in indexed["reference"]
                and (candidate_step, stage) in indexed["tp"]
            ):
                candidates.append((candidate_step, 1, stage))
    if not candidates:
        return None
    candidate_step, _, stage = max(candidates)
    return _find_alignment(indexed, candidate_step, (stage,))


def _align_tp_tensor(
    stage: str,
    tensor: torch.Tensor,
    left: BasisAlignment | None,
    right: BasisAlignment | None,
    left_next: BasisAlignment | None,
    right_next: BasisAlignment | None,
) -> torch.Tensor:
    if stage == "left_basis" and left is not None:
        return tensor * left.signs.unsqueeze(0)
    if stage == "right_basis" and right is not None:
        return tensor * right.signs.unsqueeze(0)
    if stage == "left_basis_after_step" and left_next is not None:
        return tensor * left_next.signs.unsqueeze(0)
    if stage == "right_basis_after_step" and right_next is not None:
        return tensor * right_next.signs.unsqueeze(0)

    if stage in SIGNED_COORDINATE_STAGES:
        if left is None or right is None:
            raise ValueError(
                f"cannot sign-align {stage}: current left/right bases missing"
            )
        if tensor.ndim != 2:
            raise ValueError(f"{stage} must be a matrix for basis sign alignment")
        return tensor * left.signs[:, None] * right.signs[None, :]

    if stage in SIGNED_NEXT_COORDINATE_STAGES:
        if left_next is None or right_next is None:
            raise ValueError(
                f"cannot sign-align {stage}: post-step left/right bases missing"
            )
        if tensor.ndim != 2:
            raise ValueError(f"{stage} must be a matrix for basis sign alignment")
        return tensor * left_next.signs[:, None] * right_next.signs[None, :]

    # Variance is invariant to eigenvector signs. Preconditioners, gradients,
    # parameters, and backrotated/returned updates live in physical space.
    return tensor


def _unravel(flat_index: int, shape: torch.Size) -> tuple[int, ...]:
    if not shape:
        return ()
    result = []
    remaining = flat_index
    for dimension in reversed(shape):
        result.append(remaining % dimension)
        remaining //= dimension
    return tuple(reversed(result))


def _compare(
    reference: torch.Tensor,
    tp: torch.Tensor,
) -> Comparison:
    if reference.shape != tp.shape:
        raise ValueError(
            f"shape mismatch: reference={tuple(reference.shape)}, "
            f"TP={tuple(tp.shape)}"
        )
    ref_flat = reference.detach().to(torch.float64).reshape(-1)
    tp_flat = tp.detach().to(torch.float64).reshape(-1)
    residual = tp_flat - ref_flat
    absolute_residual = residual.abs()
    difference_norm = float(torch.linalg.vector_norm(residual))
    reference_norm = float(torch.linalg.vector_norm(ref_flat))

    tensors_are_finite = bool(
        torch.isfinite(ref_flat).all() and torch.isfinite(tp_flat).all()
    )
    if not tensors_are_finite:
        relative_l2_error = math.inf
    elif reference_norm == 0:
        relative_l2_error = 0.0 if difference_norm == 0 else math.inf
    else:
        relative_l2_error = difference_norm / reference_norm

    opposite_signs = int(
        (
            ((ref_flat > 0) & (tp_flat < 0))
            | ((ref_flat < 0) & (tp_flat > 0))
        )
        .sum()
        .item()
    )
    diagnostic = torch.where(
        torch.isfinite(absolute_residual),
        absolute_residual,
        torch.full_like(absolute_residual, torch.inf),
    )
    max_flat_index = int(diagnostic.argmax()) if diagnostic.numel() else 0
    return Comparison(
        relative_l2_error=relative_l2_error,
        difference_norm=difference_norm,
        reference_norm=reference_norm,
        opposite_signs=opposite_signs,
        total_elements=absolute_residual.numel(),
        max_residual_index=_unravel(max_flat_index, reference.shape),
        reference_value=(
            float(ref_flat[max_flat_index]) if ref_flat.numel() else math.nan
        ),
        tp_value=float(tp_flat[max_flat_index]) if tp_flat.numel() else math.nan,
        identical=torch.equal(reference, tp),
    )


def _stage_sort_key(key: tuple[int, str]) -> tuple[int, int, str]:
    step, stage = key
    return step, STAGE_INDEX.get(stage, len(STAGE_ORDER)), stage


def compare_profiles(
    indexed: dict[str, dict[tuple[int, str], Event]],
    rtol: float,
) -> int:
    reference_keys = set(indexed["reference"])
    tp_keys = set(indexed["tp"])
    common_keys = sorted(reference_keys & tp_keys, key=_stage_sort_key)
    only_reference = sorted(reference_keys - tp_keys, key=_stage_sort_key)
    only_tp = sorted(tp_keys - reference_keys, key=_stage_sort_key)
    if not common_keys:
        raise ValueError("no matching reference/TP profile stages found")

    alignments = {}
    for step, _ in common_keys:
        if step in alignments:
            continue
        alignments[step] = (
            _find_current_alignment(
                indexed,
                step,
                CURRENT_LEFT_BASIS,
                NEXT_LEFT_BASIS,
            ),
            _find_current_alignment(
                indexed,
                step,
                CURRENT_RIGHT_BASIS,
                NEXT_RIGHT_BASIS,
            ),
            _find_alignment(indexed, step, NEXT_LEFT_BASIS),
            _find_alignment(indexed, step, NEXT_RIGHT_BASIS),
        )

    print(f"relative L2 tolerance: {rtol:.3e}")
    print(
        f"{'step':>4}  {'stage':<34} {'status':<4} "
        f"{'relL2':>11} {'opposite':>13}  {'index':<14} "
        f"{'reference':>14} {'TP(aligned)':>14}"
    )

    first_nonzero = None
    first_failure = None
    failed_stage_count = 0
    for key in common_keys:
        step, stage = key
        ref_event = indexed["reference"][key]
        tp_event = indexed["tp"][key]
        left, right, left_next, right_next = alignments[step]
        aligned_tp = _align_tp_tensor(
            stage,
            tp_event.tensor,
            left,
            right,
            left_next,
            right_next,
        )
        comparison = _compare(ref_event.tensor, aligned_tp)
        failed = comparison.relative_l2_error > rtol
        status = "FAIL" if failed else "ok"
        print(
            f"{step:4d}  {stage:<34} {status:<4} "
            f"{comparison.relative_l2_error:11.3e} "
            f"{comparison.opposite_signs:6d}/{comparison.total_elements:<6d}  "
            f"{str(comparison.max_residual_index):<14} "
            f"{comparison.reference_value:+14.6e} "
            f"{comparison.tp_value:+14.6e}"
        )
        if not comparison.identical and first_nonzero is None:
            first_nonzero = (step, stage, comparison.relative_l2_error)
        if failed:
            failed_stage_count += 1
            if first_failure is None:
                first_failure = (step, stage, comparison)

    print()
    for step in sorted(alignments):
        left, right, left_next, right_next = alignments[step]
        descriptions = []
        for name, alignment in (
            ("QL", left),
            ("QR", right),
            ("QL-after", left_next),
            ("QR-after", right_next),
        ):
            if alignment is not None:
                descriptions.append(
                    f"{name}: flips={alignment.flipped}/{alignment.signs.numel()}, "
                    f"min|diag overlap|={alignment.minimum_absolute_dot:.6f}, "
                    f"max|offdiag overlap|="
                    f"{alignment.maximum_off_diagonal_overlap:.3e}"
                )
        if descriptions:
            print(f"step {step} basis alignment:")
            for description in descriptions:
                print(f"  {description}")

    print()
    if first_nonzero is None:
        print("first non-bit-identical stage: none")
    else:
        step, stage, relative_l2_error = first_nonzero
        print(
            f"first non-bit-identical stage: step {step} {stage} "
            f"(relL2={relative_l2_error:.3e})"
        )
    if first_failure is None:
        print(f"first stage above {rtol:.3e}: none")
    else:
        step, stage, comparison = first_failure
        print(
            f"first stage above {rtol:.3e}: step {step} {stage} "
            f"(relL2={comparison.relative_l2_error:.3e}, "
            f"opposite={comparison.opposite_signs}/"
            f"{comparison.total_elements})"
        )
    print(f"stages above tolerance: {failed_stage_count}/{len(common_keys)}")

    if only_reference:
        print(
            "reference-only stages: "
            + ", ".join(f"step {step} {stage}" for step, stage in only_reference)
        )
    if only_tp:
        print(
            "TP-only stages: "
            + ", ".join(f"step {step} {stage}" for step, stage in only_tp)
        )

    return int(first_failure is not None)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare reference and SOAP-TP SOAP_PROFILE JSON lines."
    )
    parser.add_argument(
        "logs",
        nargs="*",
        type=Path,
        help="one or more logs; omit to read stdin",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
        help="relative L2 tolerance (default: 1e-6)",
    )
    args = parser.parse_args()
    if args.rtol < 0:
        parser.error("--rtol must be nonnegative")

    try:
        events = _read_events(args.logs)
        if not events:
            raise ValueError(f"no {PREFIX.strip()} lines found")
        runs = _split_profile_runs(events)
        exit_status = 0
        for run_index, run in enumerate(runs, 1):
            if len(runs) > 1:
                if run_index > 1:
                    print()
                print(
                    f"=== profile run {run_index}/{len(runs)} "
                    f"({_profile_run_description(run)}) ==="
                )
            exit_status |= compare_profiles(_index_events(run), args.rtol)
        return exit_status
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
