"""Density-power PCA under mixed rowwise and cellwise contamination.

The data follow a three-factor model with structured loading blocks.  A small
fraction of individual cells receive large errors and a separate group of rows
moves outside the clean factor subspace.  The example compares ordinary PCA,
scatter-based robust PCA, direct density-power PCA, and CellPCA.  It also shows
how the density-power tuning parameter trades clean-sample efficiency against
contamination resistance.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

import robustcov as rc


class EmpiricalScatter:
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.location_ = X.mean(axis=0)
        centered = X - self.location_
        self.covariance_ = centered.T @ centered / X.shape[0]
        return self


def projection_error(components: np.ndarray, truth: np.ndarray) -> float:
    estimated, _ = np.linalg.qr(np.asarray(components).T)
    reference, _ = np.linalg.qr(np.asarray(truth).T)
    return float(
        np.linalg.norm(
            estimated @ estimated.T - reference @ reference.T,
            ord="fro",
        )
        / np.sqrt(2.0 * truth.shape[0])
    )


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    ranks = rankdata(scores, method="average")
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    return float(
        (ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0)
        / (n_pos * n_neg)
    )


def make_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    n, p, q = 220, 30, 3

    loadings = np.zeros((p, q))
    loadings[0:8, 0] = np.linspace(1.0, 0.4, 8)
    loadings[9:18, 1] = np.linspace(0.9, 0.3, 9)
    loadings[19:28, 2] = np.linspace(1.0, 0.35, 9)
    loadings, _ = np.linalg.qr(loadings)

    scores = rng.standard_t(5, size=(n, q)) * np.array([2.5, 1.8, 1.2])
    clean = scores @ loadings.T + rng.normal(scale=0.16, size=(n, p))
    damaged = clean.copy()

    cell_labels = np.zeros_like(damaged, dtype=bool)
    selected = rng.choice(n * p, int(round(0.045 * n * p)), replace=False)
    row_index, column_index = np.unravel_index(selected, damaged.shape)
    damaged[row_index, column_index] += rng.normal(0.0, 8.0, selected.size)
    cell_labels[row_index, column_index] = True

    row_labels = np.zeros(n, dtype=bool)
    abnormal_rows = rng.choice(n, 18, replace=False)
    row_labels[abnormal_rows] = True
    orthogonal = rng.normal(size=(p, q))
    orthogonal -= loadings @ (loadings.T @ orthogonal)
    orthogonal, _ = np.linalg.qr(orthogonal)
    damaged[abnormal_rows] += (
        rng.normal(size=(abnormal_rows.size, q)) @ orthogonal[:, :q].T * 5.0
    )
    return clean, damaged, loadings.T, cell_labels, row_labels


def main() -> None:
    clean, X, truth, cell_labels, row_labels = make_data()
    q = truth.shape[0]

    empirical = rc.RobustPCA(
        n_components=q,
        estimator=EmpiricalScatter(),
    ).fit(X)
    cauchy = rc.RobustPCA(
        n_components=q,
        estimator=rc.RegularizedCauchy(alpha=0.10, max_iter=180),
    ).fit(X)
    dpd = rc.DensityPowerRobustPCA(
        n_components=q,
        alpha=0.30,
        max_iter=100,
        tol=1e-5,
    ).fit(X)
    cellpca = rc.CellPCA(
        n_components=q,
        max_iter=70,
        tol=1e-4,
    ).fit(X)

    models = {
        "Empirical PCA": empirical,
        "RobustPCA(Cauchy)": cauchy,
        "DensityPowerRobustPCA": dpd,
        "CellPCA": cellpca,
    }
    errors = {
        name: projection_error(model.components_, truth)
        for name, model in models.items()
    }

    result_dir = Path("results/use_cases/density_power_pca")
    result_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    names = list(errors)
    values = [errors[name] for name in names]
    positions = np.arange(len(names))
    ax.bar(positions, values)
    ax.set_xticks(positions, names, rotation=15, ha="right")
    ax.set_ylabel("Normalized projection error")
    ax.set_title("Recovery of the clean three-factor subspace")
    for index, value in enumerate(values):
        ax.text(index, value + 0.012, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(result_dir / "subspace_comparison.png", dpi=150)
    plt.close(fig)

    alphas = np.array([0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70])
    contaminated_errors = []
    clean_errors = []
    for alpha in alphas:
        location = "mean" if alpha == 0.0 else "geometric_median"
        init = "svd" if alpha == 0.0 else "winsorized_svd"
        contaminated_model = rc.DensityPowerRobustPCA(
            n_components=q,
            alpha=float(alpha),
            location=location,
            init=init,
            max_iter=100,
            tol=2e-4,
        ).fit(X)
        clean_model = rc.DensityPowerRobustPCA(
            n_components=q,
            alpha=float(alpha),
            location=location,
            init=init,
            max_iter=100,
            tol=2e-4,
        ).fit(clean)
        contaminated_errors.append(
            projection_error(contaminated_model.components_, truth)
        )
        clean_errors.append(projection_error(clean_model.components_, truth))

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(alphas, contaminated_errors, marker="o", label="contaminated sample")
    ax.plot(alphas, clean_errors, marker="s", linestyle="--", label="clean sample")
    ax.set_xlabel("Density-power tuning parameter alpha")
    ax.set_ylabel("Normalized projection error")
    ax.set_title("Robustness–efficiency tradeoff")
    ax.legend()
    fig.tight_layout()
    fig.savefig(result_dir / "alpha_tradeoff.png", dpi=150)
    plt.close(fig)

    cell_scores = 1.0 - dpd.weights_
    ordering = np.argsort(np.max(cell_scores, axis=1))[::-1][:70]
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    image = ax.imshow(cell_scores[ordering], aspect="auto", interpolation="nearest")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Rows ordered by largest cell downweighting")
    ax.set_title("Density-power cell downweighting")
    fig.colorbar(image, ax=ax, label="1 - DPD weight")
    fig.tight_layout()
    fig.savefig(result_dir / "cell_weight_map.png", dpi=150)
    plt.close(fig)

    combined_rows = row_labels | cell_labels.any(axis=1)
    rc.plot_robust_pca_outlier_map(
        dpd,
        X,
        labels=combined_rows,
        title="Density-power PCA outlier map",
        output_path=result_dir / "outlier_map.png",
        show=False,
    )

    metrics = {
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_components": q,
        "cell_contamination_fraction": float(cell_labels.mean()),
        "abnormal_row_fraction": float(row_labels.mean()),
        "empirical_subspace_error": errors["Empirical PCA"],
        "cauchy_subspace_error": errors["RobustPCA(Cauchy)"],
        "dpd_subspace_error": errors["DensityPowerRobustPCA"],
        "cellpca_subspace_error": errors["CellPCA"],
        "dpd_cell_auc": auc(cell_labels, cell_scores),
        "dpd_row_auc": auc(combined_rows, dpd.orthogonal_distances(X)),
        "dpd_iterations": dpd.n_iter_,
        "dpd_residual_scale": dpd.residual_scale_,
    }
    with (result_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics.items())

    print(f"data shape: {X.shape}")
    print(f"cell contamination: {100 * cell_labels.mean():.1f}%")
    print(f"abnormal complete rows: {row_labels.sum()}")
    print("subspace error, empirical / Cauchy / DPD / CellPCA:")
    print(
        f"{errors['Empirical PCA']:.3f} / {errors['RobustPCA(Cauchy)']:.3f} / "
        f"{errors['DensityPowerRobustPCA']:.3f} / {errors['CellPCA']:.3f}"
    )
    print(f"DPD cell-outlier AUROC: {metrics['dpd_cell_auc']:.3f}")
    print(f"DPD row-outlier AUROC: {metrics['dpd_row_auc']:.3f}")
    print(f"DPD iterations: {dpd.n_iter_}; converged: {dpd.converged_}")


if __name__ == "__main__":
    main()
