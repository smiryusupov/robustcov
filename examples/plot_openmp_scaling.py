#!/usr/bin/env python
"""Plot OpenMP scaling results from benchmarks/openmp_scaling.py."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV from benchmarks/openmp_scaling.py")
    ap.add_argument("--output", required=True, help="PNG output path")
    ap.add_argument("--metric", default="speedup_vs_1", choices=["speedup_vs_1", "median_seconds"])
    args = ap.parse_args()

    rows = read_rows(Path(args.input))
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        grouped[r["method"]].append((int(r["threads"]), float(r[args.metric])))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for method, vals in grouped.items():
        vals = sorted(vals)
        xs = [v[0] for v in vals]
        ys = [v[1] for v in vals]
        ax.plot(xs, ys, marker="o", label=method)
    ax.set_xlabel("OpenMP threads")
    ax.set_ylabel("Speedup vs 1 thread" if args.metric == "speedup_vs_1" else "Median seconds")
    ax.set_title("OpenMP scaling")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved plot to {out}")


if __name__ == "__main__":
    main()
