"""Plot workload-aware rowwise covariance speed benchmark results.

Run:
    python examples/plot_speed_comparison.py --input results/speed.csv --output results/speed.png
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import OrderedDict
from pathlib import Path


def _read(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results/speed_comparison.png")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plotting the speed benchmark requires matplotlib") from exc

    rows = [
        row for row in _read(Path(args.input))
        if row.get("status", "ok") == "ok" and row.get("median_seconds", "")
    ]
    grouped: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for row in rows:
        grouped.setdefault(row.get("workload_label") or row.get("workload", "workload"), []).append(row)
    if not grouped:
        raise ValueError("no successful timing rows were found")

    fig, axes = plt.subplots(
        len(grouped), 1,
        figsize=(10.5, max(4.2, 0.34 * len(rows) + 2.2 * len(grouped))),
        squeeze=False,
    )
    for ax, (label, workload_rows) in zip(axes[:, 0], grouped.items()):
        workload_rows = sorted(workload_rows, key=lambda row: float(row["median_seconds"]))
        names = [row["method"] for row in workload_rows]
        values = [float(row["median_seconds"]) for row in workload_rows]
        positions = list(range(len(names)))
        ax.barh(positions, values)
        ax.set_yticks(positions, labels=names)
        ax.invert_yaxis()
        ax.set_xscale("log" if max(values) / max(min(values), 1e-12) > 30 else "linear")
        ax.set_xlabel("Median complete-fit seconds")
        ax.set_title(label)
        ax.grid(axis="x", alpha=0.25)
        for pos, value in zip(positions, values):
            ax.text(value, pos, f" {value:.4g}s", va="center", fontsize=8)

    fig.suptitle("Rowwise covariance and scatter estimator speed by workload", fontsize=14)
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved plot to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
