"""Recover a sparse cross-asset network when returns contain bad ticks.

The example compares an empirical graphical lasso with the same sparse solver
fed by CellMCD.  The data are synthetic so the conditional-dependence graph is
known and the example remains deterministic in documentation builds.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import robustcov as rc


def make_precision() -> tuple[np.ndarray, list[str]]:
    labels = [
        "Tech A", "Tech B", "Tech C",
        "Bank A", "Bank B", "Bank C",
        "Energy A", "Energy B", "Energy C",
        "Bonds", "FX", "Gold",
    ]
    p = len(labels)
    precision = np.eye(p) * 1.35
    for start in (0, 3, 6):
        for index in range(start, start + 2):
            precision[index, index + 1] = precision[index + 1, index] = -0.30
        precision[start, start + 2] = precision[start + 2, start] = -0.16
    for index in (0, 3, 6):
        precision[index, 9] = precision[9, index] = 0.12
    precision[9, 10] = precision[10, 9] = -0.20
    precision[10, 11] = precision[11, 10] = 0.14
    return precision, labels


def edge_f1(adjacency: np.ndarray, truth: np.ndarray) -> float:
    predicted = np.triu(adjacency, 1)
    expected = np.triu(truth, 1)
    tp = np.count_nonzero(predicted & expected)
    fp = np.count_nonzero(predicted & ~expected)
    fn = np.count_nonzero(~predicted & expected)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def partial_from_precision(precision: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.diag(precision))
    partial = -precision / np.outer(scale, scale)
    np.fill_diagonal(partial, 1.0)
    return partial


def main() -> None:
    rng = np.random.default_rng(20260717)
    truth_precision, labels = make_precision()
    truth_covariance = np.linalg.inv(truth_precision)
    n, p = 280, truth_precision.shape[0]

    gaussian = rng.multivariate_normal(np.zeros(p), truth_covariance, size=n)
    radial = np.sqrt(rng.chisquare(df=4, size=n) / 4.0)
    clean = gaussian / radial[:, None]
    observed = clean.copy()

    cell_mask = rng.random(observed.shape) < 0.055
    observed[cell_mask] += rng.choice([-1.0, 1.0], size=cell_mask.sum()) * rng.uniform(
        7.0, 12.0, size=cell_mask.sum()
    )
    missing_mask = rng.random(observed.shape) < 0.015
    observed[missing_mask] = np.nan

    medians = np.nanmedian(observed, axis=0)
    empirical_input = np.where(np.isnan(observed), medians, observed)
    empirical = rc.RobustGraphicalLasso(
        alpha="ebic",
        scatter_estimator="empirical",
        n_alphas=18,
        ebic_gamma=0.5,
        max_iter=500,
    ).fit(empirical_input)

    robust = rc.RobustGraphicalLasso(
        alpha="ebic",
        scatter_estimator=rc.CellMCD(
            alpha=0.75,
            max_iter=40,
            min_samples_per_feature=None,
        ),
        n_alphas=18,
        ebic_gamma=0.5,
        max_iter=500,
    ).fit(observed)

    truth_adjacency = np.abs(truth_precision) > 1e-12
    np.fill_diagonal(truth_adjacency, False)
    empirical_f1 = edge_f1(empirical.adjacency_, truth_adjacency)
    robust_f1 = edge_f1(robust.adjacency_, truth_adjacency)

    truth_partial = partial_from_precision(truth_precision)
    empirical_error = np.linalg.norm(empirical.partial_correlation_ - truth_partial, ord="fro")
    robust_error = np.linalg.norm(robust.partial_correlation_ - truth_partial, ord="fro")

    output = Path("results/use_cases/robust_graphical_lasso_market_network")
    output.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    matrices = [truth_partial, empirical.partial_correlation_, robust.partial_correlation_]
    titles = ["True partial correlations", "Empirical scatter", "CellMCD scatter"]
    for ax, matrix, title in zip(axes, matrices, titles):
        image = ax.imshow(matrix, vmin=-0.35, vmax=0.35, cmap="coolwarm")
        ax.set_title(title)
        ax.set_xticks(range(p), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(p), labels, fontsize=7)
    fig.colorbar(image, ax=axes, shrink=0.78, label="partial correlation")
    fig.savefig(output / "partial_correlation_comparison.png", dpi=160)
    plt.close(fig)

    rc.plot_partial_correlation_network(
        robust,
        feature_names=labels,
        min_abs_partial_correlation=0.04,
        title="Robust sparse market network",
        output_path=output / "robust_network.png",
        show=False,
    )

    fig = plt.figure(figsize=(6.5, 4.0))
    ax = fig.add_subplot(111)
    ax.plot(empirical.alphas_, empirical.ebic_scores_, marker="o", label="empirical")
    ax.plot(robust.alphas_, robust.ebic_scores_, marker="o", label="CellMCD")
    ax.axvline(empirical.alpha_, linestyle="--", alpha=0.7)
    ax.axvline(robust.alpha_, linestyle="--", alpha=0.7)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("graphical-lasso penalty")
    ax.set_ylabel("EBIC")
    ax.set_title("Penalty selection")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "ebic_path.png", dpi=160)
    plt.close(fig)

    metrics = np.array(
        [
            [empirical_f1, robust_f1],
            [empirical_error, robust_error],
            [empirical.n_edges_, robust.n_edges_],
            [empirical.alpha_, robust.alpha_],
        ]
    )
    np.savetxt(
        output / "metrics.csv",
        metrics,
        delimiter=",",
        header="empirical,robust",
        comments="",
    )

    print(f"data shape: {observed.shape}")
    print(f"corrupted cells: {cell_mask.mean():.1%}")
    print(f"rows containing a corrupted cell: {cell_mask.any(axis=1).mean():.1%}")
    print(f"missing cells: {missing_mask.mean():.1%}")
    print(f"selected penalty, empirical / robust: {empirical.alpha_:.4f} / {robust.alpha_:.4f}")
    print(f"selected edges, empirical / robust: {empirical.n_edges_} / {robust.n_edges_}")
    print(f"edge F1, empirical / robust: {empirical_f1:.3f} / {robust_f1:.3f}")
    print(
        "partial-correlation Frobenius error, empirical / robust: "
        f"{empirical_error:.3f} / {robust_error:.3f}"
    )
    print("strongest robust edges:")
    for left, right, value in robust.edge_list(labels)[:6]:
        print(f"  {left} -- {right}: {value:+.3f}")


if __name__ == "__main__":
    main()
