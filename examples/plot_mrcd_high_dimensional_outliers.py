# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Estimate covariance and detect row outliers when the feature count exceeds n.

The example mimics a spectroscopy or embedding table with 80 observations and
120 correlated features.  A minority of rows are displaced in a low-variance
direction.  Ordinary sample covariance is singular, so the non-robust baseline
uses Ledoit-Wolf shrinkage.  MRCD combines subset trimming with a condition-
number-calibrated target weight.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from sklearn.covariance import LedoitWolf

import robustcov as rc


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    ranks = rankdata(scores, method="average")
    n_positive = int(labels.sum())
    n_negative = labels.size - n_positive
    return float(
        (ranks[labels].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def _squared_distances(X: np.ndarray, location: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    centered = X - location
    precision = np.linalg.inv(covariance)
    return np.einsum("ij,jk,ik->i", centered, precision, centered)


def _subspace_error(covariance: np.ndarray, basis: np.ndarray) -> float:
    values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    estimated = vectors[:, np.argsort(values)[::-1][: basis.shape[1]]]
    return float(np.linalg.norm(estimated @ estimated.T - basis @ basis.T, ord="fro"))


def make_data(seed: int = 14):
    rng = np.random.default_rng(seed)
    n_samples = 80
    n_features = 120
    rank = 6
    noise_scale = 0.45

    raw_basis = rng.normal(size=(n_features, rank))
    basis, _ = np.linalg.qr(raw_basis)
    factor_variances = np.array([7.0, 5.0, 3.5, 2.4, 1.7, 1.2])
    loadings = basis * np.sqrt(factor_variances)

    factors = rng.normal(size=(n_samples, rank))
    clean = factors @ loadings.T + rng.normal(
        scale=noise_scale, size=(n_samples, n_features)
    )
    observed = clean.copy()

    sparse_direction = np.zeros(n_features)
    sparse_columns = rng.choice(n_features, size=18, replace=False)
    sparse_direction[sparse_columns] = rng.normal(size=sparse_columns.size)
    sparse_direction -= basis @ (basis.T @ sparse_direction)
    sparse_direction /= np.linalg.norm(sparse_direction)

    outlier_indices = np.arange(12)
    observed[outlier_indices] += 7.5 * sparse_direction
    observed[outlier_indices] += (
        rng.normal(loc=2.0, scale=0.4, size=(outlier_indices.size, 1))
        * basis[:, [0]].T
    )

    labels = np.zeros(n_samples, dtype=bool)
    labels[outlier_indices] = True
    true_covariance = loadings @ loadings.T + noise_scale**2 * np.eye(n_features)
    return observed, clean, labels, true_covariance, basis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results/use_cases/mrcd_high_dimensional_outliers")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X, clean, labels, true_covariance, true_basis = make_data()

    mrcd = rc.MRCD(
        contamination=0.20,
        target="identity",
        max_condition_number=50,
        quality="fast",
        n_init=18,
        n_best=5,
        max_iter=50,
        random_state=0,
    ).fit(X)

    ledoit = LedoitWolf().fit(X)
    ledoit_distances = _squared_distances(X, ledoit.location_, ledoit.covariance_)

    mrcd_auc = _auc(labels, mrcd.distances_)
    ledoit_auc = _auc(labels, ledoit_distances)
    mrcd_covariance_error = np.linalg.norm(
        mrcd.covariance_ - true_covariance, ord="fro"
    ) / np.linalg.norm(true_covariance, ord="fro")
    ledoit_covariance_error = np.linalg.norm(
        ledoit.covariance_ - true_covariance, ord="fro"
    ) / np.linalg.norm(true_covariance, ord="fro")
    mrcd_subspace_error = _subspace_error(mrcd.covariance_, true_basis)
    ledoit_subspace_error = _subspace_error(ledoit.covariance_, true_basis)

    print("MRCD for high-dimensional row contamination")
    print("===========================================")
    print(f"observations / features: {X.shape[0]} / {X.shape[1]}")
    print(f"outlying rows: {labels.sum()}")
    print(f"raw support size: {mrcd.h_}")
    print(f"outlying rows retained in support: {np.count_nonzero(mrcd.support_ & labels)}")
    print(f"automatic target weight rho: {mrcd.regularization_:.4f}")
    print(f"target-relative condition number: {mrcd.standardized_condition_number_:.2f}")
    print(f"outlier AUROC, Ledoit-Wolf / MRCD: {ledoit_auc:.3f} / {mrcd_auc:.3f}")
    print(
        "relative covariance error, Ledoit-Wolf / MRCD: "
        f"{ledoit_covariance_error:.3f} / {mrcd_covariance_error:.3f}"
    )
    print(
        "six-factor subspace error, Ledoit-Wolf / MRCD: "
        f"{ledoit_subspace_error:.3f} / {mrcd_subspace_error:.3f}"
    )

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.0), sharex=True)
    index = np.arange(X.shape[0])
    for ax, distances, title in [
        (axes[0], ledoit_distances, "Ledoit-Wolf distances"),
        (axes[1], mrcd.distances_, "MRCD distances"),
    ]:
        ax.scatter(index[~labels], distances[~labels], s=24, alpha=0.75, label="central row")
        ax.scatter(index[labels], distances[labels], s=36, marker="x", label="contaminated row")
        ax.set_yscale("log")
        ax.set_ylabel("squared distance")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=2)
    axes[1].set_xlabel("row index")
    fig.tight_layout()
    fig.savefig(outdir / "distance_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    true_values = np.linalg.eigvalsh(true_covariance)[::-1]
    ledoit_values = np.linalg.eigvalsh(ledoit.covariance_)[::-1]
    mrcd_values = np.linalg.eigvalsh(mrcd.covariance_)[::-1]
    shown = np.arange(1, 31)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(shown, true_values[:30], marker="o", label="clean population")
    ax.plot(shown, ledoit_values[:30], marker="s", label="Ledoit-Wolf")
    ax.plot(shown, mrcd_values[:30], marker="^", label="MRCD")
    ax.set_yscale("log")
    ax.set_xlabel("ordered eigenvalue")
    ax.set_ylabel("eigenvalue")
    ax.set_title("Covariance spectrum under row contamination")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "covariance_spectrum.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.scatter(
        ledoit_distances[~labels],
        mrcd.distances_[~labels],
        s=26,
        alpha=0.72,
        label="central row",
    )
    ax.scatter(
        ledoit_distances[labels],
        mrcd.distances_[labels],
        s=42,
        marker="x",
        label="contaminated row",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Ledoit-Wolf squared distance")
    ax.set_ylabel("MRCD squared distance")
    ax.set_title("Rows masked by a non-robust shrinkage fit")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "distance_crossplot.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    (outdir / "metrics.csv").write_text(
        "metric,ledoit_wolf,mrcd\n"
        f"outlier_auc,{ledoit_auc:.8f},{mrcd_auc:.8f}\n"
        f"relative_covariance_error,{ledoit_covariance_error:.8f},{mrcd_covariance_error:.8f}\n"
        f"subspace_error,{ledoit_subspace_error:.8f},{mrcd_subspace_error:.8f}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
