"""Separate a low-rank signal from sparse gross cell corruption."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from robustcov import PrincipalComponentPursuit


def _relative_frobenius(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.linalg.norm(estimate - truth, ord="fro")
        / np.linalg.norm(truth, ord="fro")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        default="results/use_cases/principal_component_pursuit",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n_samples, n_features, rank = 60, 45, 3
    left, _ = np.linalg.qr(rng.normal(size=(n_samples, rank)))
    right, _ = np.linalg.qr(rng.normal(size=(n_features, rank)))
    truth = left @ np.diag([20.0, 14.0, 9.0]) @ right.T
    sparse_truth = np.zeros_like(truth)
    indices = rng.choice(
        truth.size,
        size=int(0.04 * truth.size),
        replace=False,
    )
    sparse_truth.flat[indices] = (
        rng.choice([-1.0, 1.0], size=indices.size)
        * rng.uniform(6.0, 12.0, size=indices.size)
    )
    observed = truth + sparse_truth

    left_svd, singular_values, right_svd = np.linalg.svd(
        observed,
        full_matrices=False,
    )
    truncated = (
        left_svd[:, :rank] * singular_values[:rank]
    ) @ right_svd[:rank]
    estimator = PrincipalComponentPursuit(tol=1e-7).fit(observed)

    rows = [
        {
            "method": "Rank-3 truncated SVD",
            "low_rank_relative_error": _relative_frobenius(truncated, truth),
            "estimated_rank": rank,
            "estimated_sparse_fraction": 0.0,
        },
        {
            "method": "Principal Component Pursuit",
            "low_rank_relative_error": _relative_frobenius(
                estimator.low_rank_, truth
            ),
            "estimated_rank": estimator.rank_,
            "estimated_sparse_fraction": estimator.sparse_fraction_,
        },
    ]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (outdir / "convergence.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        records = estimator.history_records()
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print("method,low_rank_relative_error,estimated_rank,estimated_sparse_fraction")
    for row in rows:
        print(
            f"{row['method']},{row['low_rank_relative_error']:.8f},"
            f"{row['estimated_rank']},{row['estimated_sparse_fraction']:.6f}"
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"saved,{outdir}")
        return

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.8), constrained_layout=True)
    panels = [
        (observed, "Observed"),
        (truth, "True low rank"),
        (estimator.low_rank_, "Recovered low rank"),
        (estimator.sparse_, "Recovered sparse"),
    ]
    limit = max(float(np.max(np.abs(observed))), 1.0)
    for axis, (matrix, title) in zip(axes, panels, strict=True):
        image = axis.imshow(matrix, aspect="auto", vmin=-limit, vmax=limit)
        axis.set_title(title)
        axis.set_xlabel("feature")
        axis.set_ylabel("row")
    fig.colorbar(image, ax=axes, shrink=0.78)
    fig.savefig(outdir / "decomposition.png", dpi=160)
    plt.close(fig)

    records = estimator.history_records()
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.semilogy(
        [record["iteration"] for record in records],
        [record["relative_residual"] for record in records],
        marker="o",
    )
    axis.axhline(estimator.tol, linestyle="--", label="stopping tolerance")
    axis.set_xlabel("IALM iteration")
    axis.set_ylabel("relative reconstruction residual")
    axis.set_title("Principal Component Pursuit convergence")
    axis.legend()
    fig.tight_layout()
    fig.savefig(outdir / "convergence.png", dpi=160)
    plt.close(fig)
    print(f"saved,{outdir}")


if __name__ == "__main__":
    main()
