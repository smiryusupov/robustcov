# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Extract robust yield-curve factors and diagnose unusual curve moves.

The data are synthetic daily yield changes generated from level, slope, and
curvature factors.  A few days contain large but structurally familiar factor
moves, while others contain maturity-specific quote dislocations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import robustcov as rc


def _nelson_siegel_loadings(maturities: np.ndarray, tau: float = 2.5) -> np.ndarray:
    x = maturities / tau
    slope = (1.0 - np.exp(-x)) / x
    curvature = slope - np.exp(-x)
    return np.column_stack([np.ones_like(maturities), slope, curvature])


def _empirical_pca(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    location = X.mean(axis=0)
    covariance = np.cov(X, rowvar=False)
    values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    order = np.argsort(values)[::-1]
    return location, values[order[:n_components]], vectors[:, order[:n_components]].T


def _projection_error(components: np.ndarray, true_basis: np.ndarray) -> float:
    return float(np.linalg.norm(components.T @ components - true_basis @ true_basis.T, ord="fro"))


def make_data(seed: int = 7):
    rng = np.random.default_rng(seed)
    maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30], dtype=float)
    raw_loadings = _nelson_siegel_loadings(maturities)
    true_basis, _ = np.linalg.qr(raw_loadings)
    true_basis = true_basis[:, :3]

    n_days = 520
    factor_scale = np.array([3.0, 1.7, 0.9])
    factors = rng.standard_t(df=7, size=(n_days, 3)) * factor_scale
    clean = factors @ true_basis.T + rng.normal(scale=0.18, size=(n_days, maturities.size))

    observed = clean.copy()
    event = np.full(n_days, "ordinary", dtype=object)

    familiar_days = np.arange(420, 432)
    observed[familiar_days] += np.array([9.0, -5.0, 2.0]) @ true_basis.T
    event[familiar_days] = "large factor move"

    quote_days = np.arange(460, 472)
    for row, maturity_index in zip(quote_days, [1, 3, 5, 8] * 3):
        observed[row, maturity_index] += rng.choice([-1.0, 1.0]) * rng.uniform(8.0, 11.0)
    event[quote_days] = "maturity-specific dislocation"

    mixed_days = np.arange(495, 501)
    observed[mixed_days] += np.array([-8.0, 5.0, -2.5]) @ true_basis.T
    observed[mixed_days, 7] += 7.0
    event[mixed_days] = "mixed stress"
    return maturities, observed, clean, true_basis, event


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results/use_cases/robust_pca_yield_curve")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    maturities, X, clean, true_basis, event = make_data()
    rpca = rc.RobustPCA(
        n_components=3,
        estimator=rc.FastMCD(
            contamination=0.08,
            quality="fast",
            random_state=0,
            n_jobs=2,
        ),
    ).fit(X)
    empirical_location, empirical_values, empirical_components = _empirical_pca(X, 3)

    robust_error = _projection_error(rpca.components_, true_basis)
    empirical_error = _projection_error(empirical_components, true_basis)
    score_distance = rpca.score_distances(X)
    orthogonal_distance = rpca.orthogonal_distances(X)

    ordinary = event == "ordinary"
    factor_move = event == "large factor move"
    quote = event == "maturity-specific dislocation"
    mixed = event == "mixed stress"

    print("Robust yield-curve factor extraction")
    print("====================================")
    print(f"observations: {X.shape[0]}")
    print(f"maturities: {X.shape[1]}")
    print(f"retained components: {rpca.n_components_}")
    print(f"robust explained variance: {rpca.explained_variance_ratio_.sum():.3f}")
    print(f"robust subspace error: {robust_error:.3f}")
    print(f"empirical subspace error: {empirical_error:.3f}")
    print(f"median score distance, ordinary days: {np.median(score_distance[ordinary]):.3f}")
    print(f"median score distance, large factor moves: {np.median(score_distance[factor_move]):.3f}")
    print(f"median orthogonal distance, ordinary days: {np.median(orthogonal_distance[ordinary]):.3f}")
    print(f"median orthogonal distance, quote dislocations: {np.median(orthogonal_distance[quote]):.3f}")
    print(f"median score/orthogonal distance, mixed stress: {np.median(score_distance[mixed]):.3f} / {np.median(orthogonal_distance[mixed]):.3f}")

    robust_loadings = rpca.components_.copy()
    empirical_loadings = empirical_components.copy()
    for index in range(3):
        if np.dot(robust_loadings[index], true_basis[:, index]) < 0:
            robust_loadings[index] *= -1
        if np.dot(empirical_loadings[index], true_basis[:, index]) < 0:
            empirical_loadings[index] *= -1

    fig = plt.figure(figsize=(9, 8.2))
    labels = ["Level-like", "Slope-like", "Curvature-like"]
    for index, label in enumerate(labels, start=1):
        ax = fig.add_subplot(3, 1, index)
        ax.plot(maturities, empirical_loadings[index - 1], marker="o", label="empirical PCA")
        ax.plot(maturities, robust_loadings[index - 1], marker="s", label="RobustPCA")
        ax.plot(maturities, true_basis[:, index - 1], linestyle="--", label="clean factor subspace")
        ax.set_ylabel(label)
        ax.set_xscale("log")
        ax.grid(alpha=0.2)
        if index == 1:
            ax.legend(ncol=3)
    ax.set_xlabel("maturity (years, log scale)")
    fig.suptitle("Yield-curve component loadings")
    fig.tight_layout()
    fig.savefig(outdir / "factor_loadings.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    scores = rpca.transform(X)
    fig = plt.figure(figsize=(10, 6.5))
    for index in range(3):
        ax = fig.add_subplot(3, 1, index + 1)
        ax.plot(scores[:, index], linewidth=0.9)
        ax.scatter(np.where(factor_move)[0], scores[factor_move, index], s=24, label="large factor move")
        ax.scatter(np.where(quote)[0], scores[quote, index], s=24, marker="x", label="quote dislocation")
        ax.set_ylabel(f"PC{index + 1}")
        if index == 0:
            ax.legend(ncol=2)
    ax.set_xlabel("trading day")
    fig.suptitle("Robust yield-curve factor scores")
    fig.tight_layout()
    fig.savefig(outdir / "factor_scores.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(7.5, 5.3))
    ax = fig.add_subplot(111)
    for mask, label in [
        (ordinary, "ordinary"),
        (factor_move, "large factor move"),
        (quote, "maturity-specific dislocation"),
        (mixed, "mixed stress"),
    ]:
        ax.scatter(score_distance[mask], orthogonal_distance[mask], s=20, alpha=0.72, label=label)
    ax.set_xlabel("score distance")
    ax.set_ylabel("orthogonal distance")
    ax.set_title("Yield-curve outlier map")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "outlier_map.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    reconstructed = rpca.reconstruct(clean)
    clean_rmse = np.sqrt(np.mean((clean - reconstructed) ** 2, axis=1))
    empirical_reconstructed = ((clean - empirical_location) @ empirical_components.T) @ empirical_components + empirical_location
    empirical_rmse = np.sqrt(np.mean((clean - empirical_reconstructed) ** 2, axis=1))
    (outdir / "metrics.csv").write_text(
        "metric,empirical,robust\n"
        f"subspace_error,{empirical_error:.8f},{robust_error:.8f}\n"
        f"clean_reconstruction_rmse,{np.median(empirical_rmse):.8f},{np.median(clean_rmse):.8f}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
