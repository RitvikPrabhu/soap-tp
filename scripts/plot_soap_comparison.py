#!/usr/bin/env python3
"""Plot the SOAP comparison results."""

import argparse
import csv
from pathlib import Path


def _load_max_errors(csv_path):
    with Path(csv_path).open(newline="") as file:
        records = list(csv.DictReader(file))

    shapes = []
    errors = {"row": {}, "column": {}}
    for record in records:
        shape = (int(record["rows"]), int(record["columns"]))
        if shape not in shapes:
            shapes.append(shape)
        layout = record["shard_layout"]
        error_percent = float(record["relative_l2_error"]) * 100
        errors[layout][shape] = max(
            error_percent,
            errors[layout].get(shape, 0.0),
        )
    return shapes, errors


def write_plot(csv_path, image_path):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FuncFormatter

    shapes, errors = _load_max_errors(csv_path)
    labels = [f"{rows}×{columns}" for rows, columns in shapes]

    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(
        labels,
        [errors["row"][shape] for shape in shapes],
        marker="o",
        label="Rows sharded across 4 ranks",
    )
    axis.plot(
        labels,
        [errors["column"][shape] for shape in shapes],
        marker="o",
        label="Columns sharded across 4 ranks",
    )
    axis.set_title("Merged 4-rank soap_step vs. original soap.py (full matrix)")
    axis.set_xlabel("Matrix shape (rows × columns)")
    axis.set_ylabel("Maximum relative update difference (%)")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.5f}%"))
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(image_path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    arguments = parser.parse_args()
    write_plot(
        arguments.csv,
        arguments.output or arguments.csv.with_suffix(".png"),
    )


if __name__ == "__main__":
    main()
