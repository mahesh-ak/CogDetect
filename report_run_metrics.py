#!/usr/bin/env python3
"""Print mean and sample standard deviation of canonical run metrics."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ("BCPrec", "BCRec", "BCF")
VARIANT_ORDER = {"language_tokens": 0, "no_language_tokens": 1}
HEADERS = ("Model", "Proportion", "Input variant", "Runs", *METRICS)
COLUMN_WIDTHS = (10, 12, 20, 6, 19, 19, 19)


def _sort_key(group: tuple[str, str, str]) -> tuple[float, int, str]:
    model, proportion, variant = group
    proportion_number = int(proportion.removeprefix("prop_"))
    displayed_proportion = 12.5 if proportion_number == 125 else proportion_number
    return (
        displayed_proportion,
        VARIANT_ORDER.get(variant, 2),
        model,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report mean ± sample standard deviation across run metrics."
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    grouped: dict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
    for path in sorted(args.runs_dir.rglob("test_metrics.json")):
        relative = path.relative_to(args.runs_dir)
        if len(relative.parts) < 4:
            continue
        model, proportion = relative.parts[:2]
        variant = relative.parts[-2]
        if not proportion.startswith("prop_"):
            continue

        with path.open(encoding="utf-8") as stream:
            metrics = json.load(stream)
        missing = [metric for metric in METRICS if metric not in metrics]
        if missing:
            raise ValueError(f"{path} is missing required metrics: {', '.join(missing)}")
        for metric in METRICS:
            value = metrics[metric]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{path} has an invalid {metric} value: {value!r}")
        grouped[(model, proportion, variant)].append(metrics)

    if not grouped:
        raise FileNotFoundError(
            f"No canonical test_metrics.json files found under {args.runs_dir}"
        )

    rows = []
    for (model, proportion, variant), runs in sorted(
        grouped.items(), key=lambda item: _sort_key(item[0])
    ):
        values = {
            metric: [float(run[metric]) for run in runs] for metric in METRICS
        }
        rendered = []
        for metric in METRICS:
            mean = statistics.mean(values[metric])
            std = statistics.stdev(values[metric]) if len(runs) > 1 else None
            rendered.append(
                f"{mean:.4f} ± {std:.4f}"
                if std is not None
                else f"{mean:.4f} ± —"
            )
        label = {
            "language_tokens": "Language tokens",
            "no_language_tokens": "Plain ASJP",
        }.get(variant, variant)
        proportion_label = (
            "12.5"
            if proportion == "prop_125"
            else proportion.removeprefix("prop_")
        )
        rows.append((model, proportion_label, label, str(len(runs)), *rendered))

    def format_row(row: tuple[str, ...]) -> str:
        return "  ".join(
            f"{value:<{width}}" if index < 3 else f"{value:>{width}}"
            for index, (value, width) in enumerate(zip(row, COLUMN_WIDTHS))
        )

    print(format_row(HEADERS))
    print("  ".join("-" * width for width in COLUMN_WIDTHS))
    for row in rows:
        print(format_row(row))


if __name__ == "__main__":
    main()
