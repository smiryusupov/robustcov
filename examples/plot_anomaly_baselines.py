from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--metric", default="f1", choices=["f1", "precision", "recall", "roc_auc", "seconds"])
    args = ap.parse_args()
    rows = read_rows(args.input)
    rows = [r for r in rows if r.get("method")]
    rows.sort(key=lambda r: float(r.get(args.metric, 0) or 0), reverse=args.metric != "seconds")
    methods = [r["method"].replace("sklearn ", "sklearn\n").replace("robustcov ", "robustcov\n") for r in rows]
    values = [float(r.get(args.metric, 0) or 0) for r in rows]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(range(len(methods)), values)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel(args.metric)
    ax.set_title(f"Anomaly baseline comparison by {args.metric}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"saved plot to {out}")


if __name__ == "__main__":
    main()
