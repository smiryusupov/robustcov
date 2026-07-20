"""Compare empirical and spectral-filter covariance under a row attack."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from robustcov.experimental import SpectralFilteringCovariance


def _relative_frobenius(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.linalg.norm(estimate - truth, ord="fro")
        / np.linalg.norm(truth, ord="fro")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        default="results/use_cases/adversarial_covariance_filtering",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n, p = 900, 10
    Q, _ = np.linalg.qr(rng.normal(size=(p, p)))
    truth = Q @ np.diag(np.geomspace(4.0, 0.6, p)) @ Q.T
    X = rng.multivariate_normal(np.zeros(p), truth, size=n)
    outlier_mask = np.zeros(n, dtype=bool)
    indices = rng.choice(n, size=90, replace=False)
    outlier_mask[indices] = True
    direction = rng.normal(size=p)
    direction /= np.linalg.norm(direction)
    X[indices] = (
        rng.choice([-1.0, 1.0], size=indices.size)[:, None]
        * 11.0
        * direction
        + rng.normal(scale=0.35, size=(indices.size, p))
    )

    empirical = np.cov(X, rowvar=False, bias=True)
    filtered = SpectralFilteringCovariance(
        contamination=0.1,
        power_iterations=15,
        random_state=0,
    ).fit(X)
    rows = [
        {
            "method": "Empirical covariance",
            "relative_frobenius_error": _relative_frobenius(empirical, truth),
            "removed_rows": 0,
            "attack_row_recall": 0.0,
        },
        {
            "method": "Spectral filtering",
            "relative_frobenius_error": _relative_frobenius(
                filtered.covariance_, truth
            ),
            "removed_rows": filtered.n_removed_,
            "attack_row_recall": float(
                np.mean(~filtered.support_[outlier_mask])
            ),
        },
    ]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (outdir / "filter_history.csv").open("w", newline="", encoding="utf-8") as handle:
        records = filtered.history_records()
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"saved,{outdir}")
        return

    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.bar(
        [row["method"] for row in rows],
        [row["relative_frobenius_error"] for row in rows],
    )
    axis.set_ylabel("relative Frobenius covariance error")
    axis.set_title("Adversarial row contamination")
    fig.tight_layout()
    fig.savefig(outdir / "covariance_error.png", dpi=160)
    plt.close(fig)

    records = filtered.history_records()
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.plot(
        [record["iteration"] for record in records],
        [record["operator_eigenvalue"] for record in records],
        marker="o",
        label="observed lifted eigenvalue",
    )
    axis.plot(
        [record["iteration"] for record in records],
        [record["operator_threshold"] for record in records],
        linestyle="--",
        label="filter tolerance",
    )
    axis.set_xlabel("filter iteration")
    axis.set_ylabel("quadratic operator eigenvalue")
    axis.set_title("Spectral filtering diagnostic")
    axis.legend()
    fig.tight_layout()
    fig.savefig(outdir / "filter_diagnostic.png", dpi=160)
    plt.close(fig)
    print(f"saved,{outdir}")


if __name__ == "__main__":
    main()
