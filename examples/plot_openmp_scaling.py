#!/usr/bin/env python
"""Plot estimator- and kernel-level OpenMP scaling results."""
from __future__ import annotations

import argparse
import csv
from collections import OrderedDict, defaultdict
from pathlib import Path


def read_rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV from benchmarks/openmp_scaling.py")
    parser.add_argument("--output", required=True, help="PNG output path")
    parser.add_argument(
        "--metric",
        default="speedup_vs_1",
        choices=["speedup_vs_1", "median_seconds"],
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plotting OpenMP scaling requires matplotlib") from exc

    rows = read_rows(Path(args.input))
    scopes: OrderedDict[str, dict[str, list[tuple[int, float]]]] = OrderedDict()
    for row in rows:
        scope = row.get("scope", "estimator")
        grouped = scopes.setdefault(scope, defaultdict(list))
        grouped[row["method"]].append((int(row["threads"]), float(row[args.metric])))

    fig, axes = plt.subplots(len(scopes), 1, figsize=(9.2, 4.2 * len(scopes)), squeeze=False)
    for ax, (scope, grouped) in zip(axes[:, 0], scopes.items()):
        for method, values in grouped.items():
            values = sorted(values)
            ax.plot(
                [value[0] for value in values],
                [value[1] for value in values],
                marker="o",
                label=method,
            )
        ax.set_xlabel("OpenMP threads")
        ax.set_ylabel(
            "Speedup vs 1 thread" if args.metric == "speedup_vs_1" else "Median seconds"
        )
        ax.set_title(scope.capitalize())
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("OpenMP scaling for all threaded native workloads", fontsize=14)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved plot to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
