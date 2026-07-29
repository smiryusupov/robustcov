# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Monitor production embeddings with robust PCA.

The example is deterministic and uses synthetic embeddings so it can run in CI
and in the Sphinx gallery without downloading a model or dataset.  It mimics a
reference window contaminated by a few corrupted vectors, followed by production
batches containing ordinary traffic, semantic drift inside the learned subspace,
and genuinely out-of-subspace observations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import robustcov as rc


def _orthonormal_basis(rng: np.random.Generator, n_features: int, rank: int) -> np.ndarray:
    basis, _ = np.linalg.qr(rng.normal(size=(n_features, rank)))
    return basis[:, :rank]


def _empirical_pca(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    location = X.mean(axis=0)
    covariance = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    order = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, order[:n_components]].T
    return location, components


def _projection_error(components: np.ndarray, true_basis: np.ndarray) -> float:
    estimated = components.T @ components
    target = true_basis @ true_basis.T
    return float(np.linalg.norm(estimated - target, ord="fro"))


def make_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    n_features = 48
    rank = 6
    basis = _orthonormal_basis(rng, n_features, rank)
    scales = np.array([3.0, 2.4, 1.9, 1.5, 1.1, 0.8])

    def sample_clean(n: int) -> np.ndarray:
        latent = rng.normal(size=(n, rank)) * scales
        return latent @ basis.T + rng.normal(scale=0.10, size=(n, n_features))

    reference_clean = sample_clean(720)
    reference = reference_clean.copy()
    contaminated_rows = rng.choice(reference.shape[0], size=45, replace=False)
    corrupt_direction = _orthonormal_basis(rng, n_features, 1)[:, 0]
    corrupt_direction -= basis @ (basis.T @ corrupt_direction)
    corrupt_direction /= np.linalg.norm(corrupt_direction)
    reference[contaminated_rows] += rng.normal(8.0, 0.8, size=(45, 1)) * corrupt_direction

    batches = []
    batch_kind = []
    point_kind = []
    for batch in range(12):
        X = sample_clean(90)
        labels = np.full(X.shape[0], "baseline", dtype=object)
        if 5 <= batch <= 8:
            drift = 2.0 * (batch - 4)
            X += drift * basis[:, 0]
            labels[:] = "in-subspace drift"
            batch_kind.append("in-subspace drift")
        elif batch >= 9:
            X += 0.8 * basis[:, 1]
            ood_rows = rng.choice(X.shape[0], size=18, replace=False)
            X[ood_rows] += rng.normal(4.5, 0.4, size=(ood_rows.size, 1)) * corrupt_direction
            labels[:] = "shifted batch"
            labels[ood_rows] = "out-of-subspace"
            batch_kind.append("OOD mixture")
        else:
            batch_kind.append("baseline")
        batches.append(X)
        point_kind.extend(labels.tolist())

    production = np.vstack(batches)
    return reference, reference_clean, production, basis, np.asarray(batch_kind), np.asarray(point_kind)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results/use_cases/robust_pca_embedding_monitoring")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    reference, reference_clean, production, true_basis, batch_kind, point_kind = make_data()

    rpca = rc.RobustPCA(
        n_components=true_basis.shape[1],
        estimator=rc.RegularizedCauchy(
            alpha=0.12,
            max_iter=100,
            tol=1e-5,
            warn_on_nonconvergence=False,
        ),
    ).fit(reference)
    empirical_location, empirical_components = _empirical_pca(reference, true_basis.shape[1])

    robust_error = _projection_error(rpca.components_, true_basis)
    empirical_error = _projection_error(empirical_components, true_basis)

    score_distance = rpca.score_distances(production)
    orthogonal_distance = rpca.orthogonal_distances(production)
    score_by_batch = score_distance.reshape(12, -1)
    orthogonal_by_batch = orthogonal_distance.reshape(12, -1)
    median_score = np.median(score_by_batch, axis=1)
    median_orthogonal = np.median(orthogonal_by_batch, axis=1)
    tail_orthogonal = np.quantile(orthogonal_by_batch, 0.90, axis=1)

    empirical_scores = (reference_clean - empirical_location) @ empirical_components.T
    robust_scores = rpca.transform(reference_clean)
    empirical_cov_error = float(np.linalg.norm(np.cov(empirical_scores, rowvar=False) - np.diag(np.var(empirical_scores, axis=0)), ord="fro"))
    robust_cov_error = float(np.linalg.norm(np.cov(robust_scores, rowvar=False) - np.diag(np.var(robust_scores, axis=0)), ord="fro"))

    print("Production embedding monitoring with RobustPCA")
    print("==============================================")
    print(f"reference shape: {reference.shape}")
    print(f"production shape: {production.shape}")
    print(f"retained components: {rpca.n_components_}")
    print(f"robust subspace error: {robust_error:.3f}")
    print(f"empirical subspace error: {empirical_error:.3f}")
    print(f"robust/empirical error ratio: {robust_error / empirical_error:.3f}")
    print(f"clean-score covariance off-diagonal norm, robust: {robust_cov_error:.3f}")
    print(f"clean-score covariance off-diagonal norm, empirical: {empirical_cov_error:.3f}")
    print(f"median score distance, baseline batches: {np.median(median_score[:5]):.3f}")
    print(f"median score distance, drift batches: {np.median(median_score[5:9]):.3f}")
    print(f"90th-percentile orthogonal distance, baseline batches: {np.median(tail_orthogonal[:5]):.3f}")
    print(f"90th-percentile orthogonal distance, OOD-mixture batches: {np.median(tail_orthogonal[9:]):.3f}")
    ood = point_kind == "out-of-subspace"
    print(f"median orthogonal distance, OOD points: {np.median(orthogonal_distance[ood]):.3f}")
    print(f"median orthogonal distance, non-OOD points: {np.median(orthogonal_distance[~ood]):.3f}")

    fig = plt.figure(figsize=(9, 4.8))
    ax = fig.add_subplot(111)
    batches = np.arange(1, 13)
    ax.plot(batches, median_score, marker="o", label="median score distance")
    ax.plot(batches, tail_orthogonal, marker="s", label="90th-percentile orthogonal distance")
    ax.axvspan(5.5, 9.5, color="tab:blue", alpha=0.10, label="in-subspace drift")
    ax.axvspan(9.5, 12.5, color="tab:orange", alpha=0.10, label="OOD mixture")
    ax.set_xlabel("production batch")
    ax.set_ylabel("batch distance summary")
    ax.set_title("RobustPCA separates semantic drift from out-of-subspace traffic")
    ax.set_xticks(batches)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "batch_monitoring.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(7.5, 5.2))
    ax = fig.add_subplot(111)
    groups = [
        (point_kind == "baseline", "baseline"),
        (point_kind == "in-subspace drift", "in-subspace drift"),
        (point_kind == "shifted batch", "shifted batch"),
        (point_kind == "out-of-subspace", "out-of-subspace"),
    ]
    for mask, label in groups:
        if np.any(mask):
            ax.scatter(score_distance[mask], orthogonal_distance[mask], s=14, alpha=0.65, label=label)
    ax.set_xlabel("score distance")
    ax.set_ylabel("orthogonal distance")
    ax.set_title("Production embedding outlier map")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "outlier_map.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(6.5, 4.4))
    ax = fig.add_subplot(111)
    ax.bar(["Empirical PCA", "RobustPCA"], [empirical_error, robust_error])
    ax.set_ylabel("projection-matrix error")
    ax.set_title("Recovery of the uncontaminated embedding subspace")
    for index, value in enumerate([empirical_error, robust_error]):
        ax.text(index, value, f"{value:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(outdir / "subspace_recovery.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
