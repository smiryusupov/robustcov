"""Recover a sparse graph from heavy-tailed elliptical observations.

Spatial signs discard each observation's radial magnitude before estimating the
shape graph. The example compares empirical, Cauchy-scatter, and spatial-sign
graphical lasso fits and then checks how much each graph changes under pairwise
radial rescaling.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import robustcov as rc


def make_precision() -> tuple[np.ndarray, list[str]]:
    labels = [
        "Equity 1", "Equity 2", "Equity 3", "Equity 4", "Equity 5",
        "Rates 1", "Rates 2", "Rates 3", "Rates 4", "Rates 5",
        "FX 1", "FX 2", "Commodity", "Volatility",
    ]
    p = len(labels)
    precision = np.eye(p) * 1.25
    for start, stop in ((0, 5), (5, 10), (10, 14)):
        for index in range(start, stop - 1):
            precision[index, index + 1] = precision[index + 1, index] = -0.27
        if stop - start >= 3:
            precision[start, stop - 1] = precision[stop - 1, start] = -0.12
    precision[4, 5] = precision[5, 4] = -0.16
    precision[9, 10] = precision[10, 9] = -0.16
    return precision, labels


def partial_from_precision(precision: np.ndarray) -> np.ndarray:
    scales = np.sqrt(np.diag(precision))
    partial = -precision / np.outer(scales, scales)
    np.fill_diagonal(partial, 1.0)
    return partial


def graph_metrics(adjacency: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    predicted = np.triu(np.asarray(adjacency, dtype=bool), 1)
    expected = np.triu(np.asarray(truth, dtype=bool), 1)
    tp = np.count_nonzero(predicted & expected)
    fp = np.count_nonzero(predicted & ~expected)
    fn = np.count_nonzero(~predicted & expected)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return float(precision), float(recall), float(f1)


def main() -> None:
    rng = np.random.default_rng(20260718)
    truth_precision, labels = make_precision()
    truth_covariance = np.linalg.inv(truth_precision)
    truth_partial = partial_from_precision(truth_precision)
    truth_adjacency = np.abs(truth_precision) > 1e-12
    np.fill_diagonal(truth_adjacency, False)

    n, p = 220, truth_precision.shape[0]
    gaussian = rng.multivariate_normal(np.zeros(p), truth_covariance, size=n)
    radial = np.sqrt(rng.chisquare(df=1.5, size=n) / 1.5)
    observed = gaussian / radial[:, None]
    shock_rows = rng.choice(n, size=32, replace=False)
    observed[shock_rows] *= rng.uniform(5.0, 18.0, size=shock_rows.size)[:, None]

    common = dict(alpha=0.12, max_iter=900, edge_tolerance=1e-6)
    empirical = rc.RobustGraphicalLasso(
        scatter_estimator="empirical", **common
    ).fit(observed)
    cauchy = rc.RobustGraphicalLasso(
        scatter_estimator=rc.RegularizedCauchy(alpha=0.10, max_iter=250),
        **common,
    ).fit(observed)
    spatial = rc.SGLASSO(**common).fit(observed)

    models = [empirical, cauchy, spatial]
    names = ["Empirical", "Cauchy scatter", "Spatial sign"]
    metrics = [graph_metrics(model.adjacency_, truth_adjacency) for model in models]
    errors = [
        np.linalg.norm(model.partial_correlation_ - truth_partial, ord="fro")
        / np.linalg.norm(truth_partial, ord="fro")
        for model in models
    ]

    output = Path("results/use_cases/spatial_sign_graphical_lasso")
    output.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.0), constrained_layout=True)
    matrices = [truth_partial] + [model.partial_correlation_ for model in models]
    titles = ["True"] + names
    for axis, matrix, title in zip(axes, matrices, titles):
        image = axis.imshow(matrix, vmin=-0.32, vmax=0.32, cmap="coolwarm")
        axis.set_title(title)
        axis.set_xticks(range(p), labels, rotation=90, fontsize=6)
        axis.set_yticks(range(p), labels, fontsize=6)
    fig.colorbar(image, ax=axes, shrink=0.78, label="partial correlation")
    fig.savefig(output / "partial_correlation_comparison.png", dpi=160)
    plt.close(fig)

    rc.plot_partial_correlation_network(
        spatial,
        feature_names=labels,
        min_abs_partial_correlation=0.035,
        title="Spatial-sign conditional-association network",
        output_path=output / "spatial_sign_network.png",
        show=False,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
    axes[0].bar(names, [value[2] for value in metrics])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("edge F1")
    axes[0].set_title("Sparse graph recovery")
    axes[0].tick_params(axis="x", rotation=18)
    axes[1].bar(names, errors)
    axes[1].set_ylabel("relative partial-correlation error")
    axes[1].set_title("Estimated edge strengths")
    axes[1].tick_params(axis="x", rotation=18)
    fig.savefig(output / "graph_recovery.png", dpi=160)
    plt.close(fig)

    # A separate symmetric sample makes the center exactly zero. Multiplying
    # each +/- pair by an arbitrary radius leaves the spatial signs unchanged.
    half = rng.multivariate_normal(np.zeros(p), truth_covariance, size=100)
    paired = np.vstack([half, -half])
    pair_scales = np.exp(rng.normal(scale=1.5, size=half.shape[0]))
    rescaled = np.vstack([
        half * pair_scales[:, None],
        -half * pair_scales[:, None],
    ])
    stability_models = [
        (
            "Empirical",
            rc.RobustGraphicalLasso(alpha=0.12, scatter_estimator="empirical", max_iter=900),
        ),
        (
            "Cauchy scatter",
            rc.RobustGraphicalLasso(
                alpha=0.12,
                scatter_estimator=rc.RegularizedCauchy(alpha=0.10, max_iter=250),
                max_iter=900,
            ),
        ),
        ("Spatial sign", rc.SGLASSO(alpha=0.12, max_iter=900)),
    ]
    radial_changes = []
    for _, estimator in stability_models:
        first = estimator.fit(paired).partial_correlation_.copy()
        second = estimator.fit(rescaled).partial_correlation_.copy()
        radial_changes.append(
            np.linalg.norm(second - first, ord="fro")
            / max(np.linalg.norm(first, ord="fro"), np.finfo(float).eps)
        )

    fig = plt.figure(figsize=(6.8, 4.0))
    axis = fig.add_subplot(111)
    axis.bar([name for name, _ in stability_models], radial_changes)
    axis.set_yscale("log")
    axis.set_ylabel("relative graph change")
    axis.set_title("Sensitivity to observation-specific radial rescaling")
    axis.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(output / "radial_stability.png", dpi=160)
    plt.close(fig)

    table = np.column_stack(
        [
            [value[0] for value in metrics],
            [value[1] for value in metrics],
            [value[2] for value in metrics],
            errors,
            [model.n_edges_ for model in models],
            radial_changes,
        ]
    )
    np.savetxt(
        output / "metrics.csv",
        table,
        delimiter=",",
        header="edge_precision,edge_recall,edge_f1,partial_error,n_edges,radial_change",
        comments="",
    )

    print(f"data shape: {observed.shape}")
    print(f"Student-t degrees of freedom: 1.5")
    print(f"additional radial-shock rows: {shock_rows.size}")
    for name, model, values, error in zip(names, models, metrics, errors):
        print(
            f"{name}: edges={model.n_edges_}, precision={values[0]:.3f}, "
            f"recall={values[1]:.3f}, F1={values[2]:.3f}, "
            f"partial error={error:.3f}"
        )
    print("relative graph change after pairwise radial rescaling:")
    for (name, _), change in zip(stability_models, radial_changes):
        print(f"  {name}: {change:.3e}")
    print("strongest spatial-sign edges:")
    for left, right, value in spatial.edge_list(labels)[:6]:
        print(f"  {left} -- {right}: {value:+.3f}")


if __name__ == "__main__":
    main()
