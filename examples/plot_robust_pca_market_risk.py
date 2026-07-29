# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Separate systemic and idiosyncratic market shocks with robust PCA.

The synthetic cross-asset return panel contains heavy-tailed common factors,
large systemic factor moves, and isolated instrument dislocations.  Robust PCA
provides a compact factor representation and keeps score and orthogonal distance
separate so the two event types remain interpretable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import robustcov as rc


ASSETS = [
    "US Equity", "Europe", "EM Equity", "Technology", "Banks", "2Y Treasury",
    "10Y Treasury", "IG Credit", "HY Credit", "Gold", "Oil", "USD",
]
FACTORS = ["Growth", "Duration", "Credit", "Inflation"]


def _factor_loadings() -> np.ndarray:
    return np.array([
        [1.00, -0.20, 0.25, 0.05],
        [0.90, -0.15, 0.20, 0.05],
        [1.10, -0.25, 0.45, 0.15],
        [1.20, -0.30, 0.10, 0.05],
        [0.95, -0.40, 0.45, 0.00],
        [-0.25, 0.80, -0.05, -0.20],
        [-0.45, 1.10, -0.10, -0.25],
        [0.35, -0.25, 0.85, -0.05],
        [0.70, -0.30, 1.10, 0.10],
        [-0.10, 0.35, -0.05, 0.75],
        [0.45, -0.20, 0.15, 1.10],
        [-0.35, -0.05, -0.10, -0.45],
    ], dtype=float)


def _empirical_pca(X: np.ndarray, n_components: int):
    location = X.mean(axis=0)
    covariance = np.cov(X, rowvar=False)
    values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    order = np.argsort(values)[::-1]
    values = values[order]
    components = vectors[:, order[:n_components]].T
    return location, values, components


def _projection_error(components: np.ndarray, true_basis: np.ndarray) -> float:
    return float(np.linalg.norm(components.T @ components - true_basis @ true_basis.T, ord="fro"))


def make_data(seed: int = 19):
    rng = np.random.default_rng(seed)
    loadings = _factor_loadings()
    true_basis, _ = np.linalg.qr(loadings)
    true_basis = true_basis[:, :4]

    n_days = 650
    factor_scales = np.array([1.5, 1.0, 0.9, 0.75])
    factors = rng.standard_t(df=5, size=(n_days, 4)) * factor_scales
    clean = factors @ true_basis.T + rng.standard_t(df=8, size=(n_days, len(ASSETS))) * 0.18
    observed = clean.copy()
    event = np.full(n_days, "ordinary", dtype=object)

    systemic_days = np.arange(520, 536)
    systemic_factor = np.array([-7.5, 4.5, 5.5, 2.0])
    observed[systemic_days] += systemic_factor @ true_basis.T
    event[systemic_days] = "systemic shock"

    idiosyncratic_days = np.arange(570, 590)
    affected_assets = [3, 8, 10, 4, 11] * 4
    for day, asset in zip(idiosyncratic_days, affected_assets):
        observed[day, asset] += rng.choice([-1.0, 1.0]) * rng.uniform(7.0, 10.0)
    event[idiosyncratic_days] = "idiosyncratic dislocation"

    mixed_days = np.arange(620, 628)
    observed[mixed_days] += np.array([-6.0, 3.0, 4.0, 2.5]) @ true_basis.T
    observed[mixed_days, 8] -= 7.0
    event[mixed_days] = "mixed shock"
    return observed, clean, true_basis, event


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results/use_cases/robust_pca_market_risk")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X, clean, true_basis, event = make_data()
    rpca = rc.RobustPCA(
        n_components=4,
        estimator=rc.RegularizedCauchy(
            alpha=0.08,
            max_iter=120,
            tol=1e-5,
            warn_on_nonconvergence=False,
        ),
    ).fit(X)
    empirical_location, empirical_values, empirical_components = _empirical_pca(X, 4)

    robust_error = _projection_error(rpca.components_, true_basis)
    empirical_error = _projection_error(empirical_components, true_basis)
    score_distance = rpca.score_distances(X)
    orthogonal_distance = rpca.orthogonal_distances(X)

    ordinary = event == "ordinary"
    systemic = event == "systemic shock"
    idiosyncratic = event == "idiosyncratic dislocation"
    mixed = event == "mixed shock"

    print("Cross-asset market-risk decomposition with RobustPCA")
    print("====================================================")
    print(f"observations: {X.shape[0]}")
    print(f"assets: {X.shape[1]}")
    print(f"retained factors: {rpca.n_components_}")
    print(f"robust explained variance: {rpca.explained_variance_ratio_.sum():.3f}")
    print(f"robust subspace error: {robust_error:.3f}")
    print(f"empirical subspace error: {empirical_error:.3f}")
    print(f"median score distance, ordinary days: {np.median(score_distance[ordinary]):.3f}")
    print(f"median score distance, systemic shocks: {np.median(score_distance[systemic]):.3f}")
    print(f"median orthogonal distance, ordinary days: {np.median(orthogonal_distance[ordinary]):.3f}")
    print(f"median orthogonal distance, idiosyncratic dislocations: {np.median(orthogonal_distance[idiosyncratic]):.3f}")
    print(f"median score/orthogonal distance, mixed shocks: {np.median(score_distance[mixed]):.3f} / {np.median(orthogonal_distance[mixed]):.3f}")

    fig = plt.figure(figsize=(10.5, 5.4))
    ax = fig.add_subplot(111)
    image = ax.imshow(rpca.components_, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(ASSETS)))
    ax.set_xticklabels(ASSETS, rotation=45, ha="right")
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels([f"Robust PC{i}" for i in range(1, 5)])
    ax.set_title("Robust cross-asset factor loadings")
    fig.colorbar(image, ax=ax, label="loading")
    fig.tight_layout()
    fig.savefig(outdir / "asset_loadings.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    robust_ratio = rpca.all_eigenvalues_ / np.sum(rpca.all_eigenvalues_)
    empirical_ratio = empirical_values / np.sum(empirical_values)
    fig = plt.figure(figsize=(7.5, 4.8))
    ax = fig.add_subplot(111)
    positions = np.arange(1, 7)
    width = 0.36
    ax.bar(positions - width / 2, empirical_ratio[:6], width=width, label="empirical PCA")
    ax.bar(positions + width / 2, robust_ratio[:6], width=width, label="RobustPCA")
    ax.set_xlabel("component")
    ax.set_ylabel("explained variance ratio")
    ax.set_title("Common-factor concentration under contaminated returns")
    ax.set_xticks(positions)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "explained_variance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(7.6, 5.4))
    ax = fig.add_subplot(111)
    for mask, label in [
        (ordinary, "ordinary"),
        (systemic, "systemic shock"),
        (idiosyncratic, "idiosyncratic dislocation"),
        (mixed, "mixed shock"),
    ]:
        ax.scatter(score_distance[mask], orthogonal_distance[mask], s=22, alpha=0.72, label=label)
    ax.set_xlabel("score distance")
    ax.set_ylabel("orthogonal distance")
    ax.set_title("Systemic versus idiosyncratic market events")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "outlier_map.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    robust_residual = np.abs(clean - rpca.reconstruct(clean)).mean(axis=0)
    empirical_reconstructed = ((clean - empirical_location) @ empirical_components.T) @ empirical_components + empirical_location
    empirical_residual = np.abs(clean - empirical_reconstructed).mean(axis=0)
    fig = plt.figure(figsize=(10, 4.8))
    ax = fig.add_subplot(111)
    positions = np.arange(len(ASSETS))
    width = 0.36
    ax.bar(positions - width / 2, empirical_residual, width=width, label="empirical PCA")
    ax.bar(positions + width / 2, robust_residual, width=width, label="RobustPCA")
    ax.set_xticks(positions)
    ax.set_xticklabels(ASSETS, rotation=45, ha="right")
    ax.set_ylabel("mean absolute clean reconstruction residual")
    ax.set_title("Clean-data reconstruction by asset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "reconstruction_residual.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
