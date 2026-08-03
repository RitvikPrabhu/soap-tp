#!/usr/bin/env python3
"""Plot maximum SOAP relative L2 error by matrix size from a test log."""

import argparse
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


RESULT_PATTERN = re.compile(
    r"(?P<rows>\d+)x(?P<columns>\d+).*?"
    r"SOAP relL2=(?P<error>[+-]?[0-9.]+(?:e[+-]?\d+)?)",
    re.IGNORECASE,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()

    max_errors = {}
    for line in args.log.read_text().splitlines():
        match = RESULT_PATTERN.search(line)
        if match is None:
            continue

        rows = int(match.group("rows"))
        columns = int(match.group("columns"))
        if rows != columns:
            continue

        error = float(match.group("error"))
        max_errors[rows] = max(error, max_errors.get(rows, 0.0))

    if not max_errors:
        raise SystemExit("No square SOAP accuracy results found in the log.")

    sizes = sorted(max_errors)
    errors = [max_errors[size] for size in sizes]
    output = args.output or args.log.with_suffix(".png")

    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(sizes, errors, marker="o")
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks(sizes, [f"{size}x{size}" for size in sizes], rotation=45)
    axis.set_xlabel("Matrix size")
    axis.set_ylabel("Maximum SOAP relative L2 error")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
