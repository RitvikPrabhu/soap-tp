#!/usr/bin/env python3
"""Plot maximum, first-iteration, and final-iteration SOAP errors."""

import argparse
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


RESULT_PATTERN = re.compile(
    r"(?P<rows>\d+)x(?P<columns>\d+)\s+"
    r"dim(?P<shard_dim>[01])\s+"
    r"basis=(?P<implementation>eigh|elpa)\s+\|\s+"
    r"step\s+(?P<iteration>\d+)\s+\|.*?"
    r"SOAP relL2=(?P<error>[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:e[+-]?\d+)?)",
    re.IGNORECASE,
)

IMPLEMENTATIONS = (
    ("eigh", "torch.linalg.eigh"),
    ("elpa", "ELPA"),
)
SHARD_TITLES = {
    0: "Row-sharded gradients (dim 0)",
    1: "Column-sharded gradients (dim 1)",
}
METRICS = (
    ("maximum", "Maximum", "tab:red"),
    ("first", "Step 1", "tab:blue"),
    ("last", "Final step", "tab:green"),
)
LINESTYLES = {
    "eigh": "--",
    "elpa": "-",
}
FIRST_ITERATION = 1


def _parse_error_history(lines):
    error_history = {
        shard_dim: {implementation: {} for implementation, _ in IMPLEMENTATIONS}
        for shard_dim in SHARD_TITLES
    }
    for line in lines:
        match = RESULT_PATTERN.search(line)
        if match is None:
            continue

        rows = int(match.group("rows"))
        columns = int(match.group("columns"))
        if rows != columns:
            continue

        shard_dim = int(match.group("shard_dim"))
        implementation = match.group("implementation").lower()
        iteration = int(match.group("iteration"))
        error = float(match.group("error"))
        size_history = error_history[shard_dim][implementation].setdefault(
            rows,
            {},
        )
        size_history[iteration] = error
    return error_history


def _common_sizes(error_history, shard_dim):
    series_sizes = {
        implementation: set(error_history[shard_dim][implementation])
        for implementation, _ in IMPLEMENTATIONS
    }
    missing = [
        implementation
        for implementation, sizes in series_sizes.items()
        if not sizes
    ]
    if missing:
        raise ValueError(
            f"missing dim{shard_dim} SOAP accuracy results for "
            + ", ".join(missing)
        )

    distinct_size_sets = {frozenset(sizes) for sizes in series_sizes.values()}
    if len(distinct_size_sets) != 1:
        details = ", ".join(
            f"{implementation}={sorted(sizes)}"
            for implementation, sizes in series_sizes.items()
        )
        raise ValueError(
            f"dim{shard_dim} SOAP accuracy series use different sizes: "
            f"{details}"
        )
    return sorted(next(iter(distinct_size_sets)))


def _error_for_metric(
    history,
    metric,
    *,
    shard_dim,
    implementation,
    size,
):
    if metric == "maximum":
        return max(history.values())
    if metric == "first":
        if FIRST_ITERATION not in history:
            raise ValueError(
                f"missing dim{shard_dim} {implementation} step "
                f"{FIRST_ITERATION} result for size {size}"
            )
        return history[FIRST_ITERATION]
    if metric == "last":
        return history[max(history)]
    raise ValueError(f"unknown SOAP accuracy metric {metric!r}")


def _create_figure(error_history, shard_dim):
    sizes = _common_sizes(error_history, shard_dim)
    figure, axis = plt.subplots(figsize=(11, 6))
    for metric, metric_label, color in METRICS:
        for implementation, label in IMPLEMENTATIONS:
            errors = [
                _error_for_metric(
                    error_history[shard_dim][implementation][size],
                    metric,
                    shard_dim=shard_dim,
                    implementation=implementation,
                    size=size,
                )
                for size in sizes
            ]
            axis.plot(
                sizes,
                errors,
                color=color,
                linestyle=LINESTYLES[implementation],
                marker="o",
                label=f"{metric_label} — {label}",
            )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks(
        sizes,
        [f"{size}x{size}" for size in sizes],
        rotation=45,
    )
    axis.set_xlabel("Matrix size")
    axis.set_ylabel("SOAP relative L2 error")
    axis.set_title(SHARD_TITLES[shard_dim])
    axis.grid(alpha=0.3)
    axis.legend(
        ncols=2,
        fontsize="small",
    )
    figure.tight_layout()
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument(
        "--shard-dim",
        type=int,
        choices=tuple(SHARD_TITLES),
        required=True,
        help="0 for the row-sharded graph or 1 for the column-sharded graph",
    )
    args = parser.parse_args()

    error_history = _parse_error_history(args.log.read_text().splitlines())
    output = args.output or args.log.with_suffix(".png")

    try:
        figure = _create_figure(error_history, args.shard_dim)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    figure.savefig(output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
