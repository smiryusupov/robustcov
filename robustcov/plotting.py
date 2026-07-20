# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import chi2

try:  # plotting is an optional dependency
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - depends on optional installation
    _matplotlib_import_error = exc

    class _MissingMatplotlib:
        def __getattr__(self, name):
            raise ImportError(
                "Plotting requires matplotlib. Install it with "
                "`python -m pip install 'robustcov[plot]'`."
            ) from _matplotlib_import_error

    plt = _MissingMatplotlib()


def _maybe_save_show(fig, output_path=None, show=True):
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_mahalanobis_diagnostics(estimator, X=None, alpha=0.975, output_path=None, show=True):
    """Plot ordered Mahalanobis distances with a cutoff line.

    If X is provided, distances are recomputed on X; otherwise fitted distances are used.
    """
    if X is None:
        d2 = np.asarray(estimator.distances_)
    else:
        d2 = np.asarray(estimator.mahalanobis(X))
    d2 = np.sort(d2)
    n = d2.size
    cutoff = chi2.ppf(alpha, getattr(estimator, "n_features_in_", 1))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(1, n + 1), d2)
    ax.axhline(cutoff, linestyle='--')
    ax.set_xlabel('Ordered observation index')
    ax.set_ylabel('Squared robust Mahalanobis distance')
    ax.set_title(f'Mahalanobis diagnostics ({type(estimator).__name__})')
    ax.text(0.01, 0.98, f'cutoff={cutoff:.3f}', transform=ax.transAxes, va='top')
    if hasattr(estimator, 'radial_kurtosis_'):
        ax.text(0.01, 0.90, f'radial_kurtosis={estimator.radial_kurtosis_:.3f}', transform=ax.transAxes, va='top')
    _maybe_save_show(fig, output_path=output_path, show=show)
    return fig


def plot_mahalanobis_qq(estimator, X=None, output_path=None, show=True):
    """Chi-square QQ diagnostic for robust Mahalanobis distances."""
    if X is None:
        d2 = np.asarray(estimator.distances_)
    else:
        d2 = np.asarray(estimator.mahalanobis(X))
    d2 = np.sort(d2)
    n = d2.size
    p = getattr(estimator, 'n_features_in_', 1)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theo = chi2.ppf(probs, p)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(theo, d2, marker='o', linestyle='None', markersize=3)
    lo = min(theo.min(), d2.min())
    hi = max(theo.max(), d2.max())
    ax.plot([lo, hi], [lo, hi], linestyle='--')
    ax.set_xlabel(f'Chi-square({p}) quantiles')
    ax.set_ylabel('Ordered robust distances')
    ax.set_title(f'QQ diagnostic ({type(estimator).__name__})')
    _maybe_save_show(fig, output_path=output_path, show=show)
    return fig


def plot_distance_histogram(estimator, X=None, alpha=0.975, bins=30, output_path=None, show=True):
    """Histogram of robust Mahalanobis distances with cutoff line."""
    if X is None:
        d2 = np.asarray(estimator.distances_)
    else:
        d2 = np.asarray(estimator.mahalanobis(X))
    cutoff = chi2.ppf(alpha, getattr(estimator, 'n_features_in_', 1))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(d2, bins=bins)
    ax.axvline(cutoff, linestyle='--')
    ax.set_xlabel('Squared robust Mahalanobis distance')
    ax.set_ylabel('Count')
    ax.set_title(f'Distance histogram ({type(estimator).__name__})')
    _maybe_save_show(fig, output_path=output_path, show=show)
    return fig


def plot_covariance_heatmap(covariance, title='Covariance heatmap', output_path=None, show=True):
    """Heatmap for a covariance/scatter matrix."""
    cov = np.asarray(covariance)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cov, aspect='auto', interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel('Feature index')
    ax.set_ylabel('Feature index')
    fig.colorbar(im, ax=ax)
    _maybe_save_show(fig, output_path=output_path, show=show)
    return fig



def _distance_values(estimator, X=None):
    if X is None:
        if not hasattr(estimator, 'distances_'):
            raise RuntimeError('Estimator has no fitted distances_; pass X to recompute distances.')
        return np.asarray(estimator.distances_, dtype=float)
    return np.asarray(estimator.mahalanobis(X), dtype=float)


def plot_robust_distance_profile(estimator, X=None, alpha=0.975, sort=True, labels=None, title=None, output_path=None, show=True):
    """Profile/proline-style plot of robust squared Mahalanobis distances.

    This is useful for visually inspecting whether a few observations dominate the
    robust distance tail. If ``sort=True`` the observations are ordered by distance;
    otherwise the original row order is preserved.
    """
    d2 = _distance_values(estimator, X)
    n = d2.size
    order = np.argsort(d2) if sort else np.arange(n)
    y = d2[order]
    cutoff = chi2.ppf(alpha, getattr(estimator, 'n_features_in_', 1))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(1, n + 1), y, marker='o', markersize=3, linewidth=1)
    ax.axhline(cutoff, linestyle='--', label=f'chi2 cutoff {alpha:.3f}')
    if labels is not None:
        lab = np.asarray(labels)
        if lab.shape[0] == n:
            anomaly = lab.astype(bool)[order]
            if np.any(anomaly):
                idx = np.where(anomaly)[0]
                ax.scatter(idx + 1, y[idx], s=36, facecolors='none', edgecolors='black', label='known anomaly')
    ax.set_xlabel('Observation rank' if sort else 'Observation index')
    ax.set_ylabel('Squared robust Mahalanobis distance')
    ax.set_title(title or f'Robust distance profile ({type(estimator).__name__})')
    ax.legend()
    _maybe_save_show(fig, output_path=output_path, show=show)
    return fig


def plot_robust_distance_panel(estimator, X=None, alpha=0.975, labels=None, output_path=None, show=True):
    """Create a compact three-plot robust distance diagnostic panel.

    The panel contains a distance profile, a histogram, and a chi-square QQ plot.
    """
    d2 = _distance_values(estimator, X)
    d2_sorted = np.sort(d2)
    n = d2.size
    p = getattr(estimator, 'n_features_in_', 1)
    cutoff = chi2.ppf(alpha, p)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theo = chi2.ppf(probs, p)

    fig = plt.figure(figsize=(12, 3.8))
    ax1 = fig.add_subplot(131)
    ax1.plot(np.arange(1, n + 1), d2_sorted, marker='o', markersize=2, linewidth=1)
    ax1.axhline(cutoff, linestyle='--')
    ax1.set_title('Distance profile')
    ax1.set_xlabel('rank')
    ax1.set_ylabel('distance')

    ax2 = fig.add_subplot(132)
    ax2.hist(d2, bins=30)
    ax2.axvline(cutoff, linestyle='--')
    ax2.set_title('Distance histogram')
    ax2.set_xlabel('distance')

    ax3 = fig.add_subplot(133)
    ax3.plot(theo, d2_sorted, marker='o', linestyle='None', markersize=2)
    hi = max(float(np.nanmax(theo)), float(np.nanmax(d2_sorted)))
    ax3.plot([0, hi], [0, hi], linestyle='--')
    ax3.set_title('Chi-square QQ')
    ax3.set_xlabel(f'chi2({p})')
    ax3.set_ylabel('distance')

    fig.suptitle(f'Robust distance diagnostics ({type(estimator).__name__})')
    fig.tight_layout()
    _maybe_save_show(fig, output_path=output_path, show=show)
    return fig

def _read_csv_rows(csv_path):
    with open(csv_path, newline='') as f:
        return list(csv.DictReader(f))


def plot_benchmark_curve(csv_path, x_col, y_col, group_col='method', title='', output_path=None, show=True):
    """Plot a grouped line chart from a benchmark CSV file."""
    rows = _read_csv_rows(csv_path)
    groups = {}
    for row in rows:
        groups.setdefault(row[group_col], []).append(row)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for name, items in groups.items():
        xs = np.array([float(r[x_col]) for r in items])
        ys = np.array([float(r[y_col]) for r in items])
        order = np.argsort(xs)
        ax.plot(xs[order], ys[order], marker='o', label=name)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title or f'{y_col} vs {x_col}')
    ax.legend()
    _maybe_save_show(fig, output_path=output_path, show=show)
    return fig


def plot_benchmark_bars(csv_path, category_col, value_col, title='', output_path=None, show=True):
    """Plot a simple bar chart from a benchmark CSV file."""
    rows = _read_csv_rows(csv_path)
    cats = [row[category_col] for row in rows]
    vals = [float(row[value_col]) for row in rows]
    x = np.arange(len(cats))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha='right')
    ax.set_ylabel(value_col)
    ax.set_title(title or value_col)
    _maybe_save_show(fig, output_path=output_path, show=show)
    return fig


def _covariance_ellipse_points(location, covariance, n_std=2.0, n_points=200):
    loc = np.asarray(location, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    vals, vecs = np.linalg.eigh(cov[:2, :2])
    vals = np.maximum(vals, 0.0)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.linspace(0, 2 * np.pi, n_points)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    ellipse = vecs @ (np.sqrt(vals)[:, None] * circle) * n_std
    ellipse = ellipse + loc[:2, None]
    return ellipse[0], ellipse[1]


def plot_anomaly_scatter_2d(estimator, X, labels=None, alpha=0.975, title=None, output_path=None, show=True):
    """2D diagnostic scatter showing robust support/rejected points and covariance ellipse.

    Parameters
    ----------
    estimator:
        Fitted robustcov estimator.
    X:
        Input array. The first two columns are plotted.
    labels:
        Optional true labels. If provided, rejected points and known anomalies can be
        visually compared. Numeric labels are not interpreted beyond boolean conversion:
        nonzero/positive values are treated as anomalies.
    alpha:
        Chi-square cutoff level used for the covariance ellipse.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError('X must have at least two columns for a 2D scatter diagnostic')
    support = getattr(estimator, 'support_', None)
    if support is None or len(support) != X.shape[0]:
        d2 = estimator.mahalanobis(X)
        cutoff = chi2.ppf(alpha, getattr(estimator, 'n_features_in_', X.shape[1]))
        support = d2 <= cutoff
    support = np.asarray(support, dtype=bool)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[support, 0], X[support, 1], s=18, label='robust support')
    ax.scatter(X[~support, 0], X[~support, 1], s=28, marker='x', label='rejected')
    if labels is not None:
        y = np.asarray(labels)
        anomaly = y.astype(bool)
        if anomaly.shape[0] == X.shape[0] and np.any(anomaly):
            ax.scatter(X[anomaly, 0], X[anomaly, 1], s=60, facecolors='none', edgecolors='black', label='true anomaly')

    n_std = float(np.sqrt(chi2.ppf(alpha, 2)))
    try:
        ex, ey = _covariance_ellipse_points(estimator.location_, estimator.covariance_, n_std=n_std)
        ax.plot(ex, ey, linestyle='--', label=f'{alpha:.3f} ellipse')
    except Exception:
        pass
    ax.set_xlabel('feature 0')
    ax.set_ylabel('feature 1')
    ax.set_title(title or f'2D robust anomaly diagnostic ({type(estimator).__name__})')
    ax.legend()
    _maybe_save_show(fig, output_path=output_path, show=show)
    return fig


def plot_distance_scatter_2d(estimator, X, title=None, output_path=None, show=True):
    """2D scatter colored by robust Mahalanobis distance."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError('X must have at least two columns for a 2D scatter diagnostic')
    d2 = estimator.mahalanobis(X)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sc = ax.scatter(X[:, 0], X[:, 1], c=d2, s=22)
    ax.set_xlabel('feature 0')
    ax.set_ylabel('feature 1')
    ax.set_title(title or 'Robust distance-colored scatter')
    fig.colorbar(sc, ax=ax, label='squared robust Mahalanobis distance')
    _maybe_save_show(fig, output_path=output_path, show=show)
    return fig


def plot_speed_accuracy_pareto(csv_path, error_col='rel_fro_error', time_col='median_seconds', group_col='method', title='', output_path=None, show=True):
    """Plot speed-accuracy Pareto points from a benchmark CSV.

    The CSV must include one row per method with time and error columns.
    """
    rows = _read_csv_rows(csv_path)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for row in rows:
        try:
            x = float(row[time_col])
            y = float(row[error_col])
        except Exception:
            continue
        label = row.get(group_col, '')
        ax.scatter([x], [y], s=50)
        ax.annotate(label, (x, y), textcoords='offset points', xytext=(4, 4), fontsize=8)
    ax.set_xlabel(time_col)
    ax.set_ylabel(error_col)
    ax.set_title(title or 'Speed-accuracy Pareto plot')
    _maybe_save_show(fig, output_path=output_path, show=show)
    return fig



def plot_cluster_robust_distances(detector, X=None, labels=None, alpha=None, output_path=None, show=True):
    """Plot cluster-aware robust distances for a fitted multimodal detector.

    The left panel shows the first two feature dimensions colored by assigned
    cluster, with detected outliers highlighted. The right panel shows ordered
    cluster-conditional robust distances with the fitted threshold.
    """
    if X is None:
        if not hasattr(detector, "distances_"):
            raise RuntimeError("detector is not fitted and X was not provided")
        d = np.asarray(detector.distances_, dtype=float)
        clusters = np.asarray(detector.cluster_labels_, dtype=int)
        X_plot = None
        outliers = np.asarray(getattr(detector, "outlier_mask_", d > detector.threshold_), dtype=bool)
    else:
        X_plot = np.asarray(X, dtype=float)
        d = np.asarray(detector.decision_function(X_plot), dtype=float)
        clusters = detector._assign_clusters(X_plot)
        outliers = d > detector.threshold_

    fig = plt.figure(figsize=(11, 4))
    ax1 = fig.add_subplot(121)
    if X_plot is not None and X_plot.shape[1] >= 2:
        ax1.scatter(X_plot[:, 0], X_plot[:, 1], c=clusters, s=18, alpha=0.75)
        if np.any(outliers):
            ax1.scatter(X_plot[outliers, 0], X_plot[outliers, 1], s=60, facecolors='none', edgecolors='black', label='detected outlier')
            ax1.legend()
        ax1.set_xlabel('feature 0')
        ax1.set_ylabel('feature 1')
        ax1.set_title('Cluster assignment and outliers')
    else:
        counts = np.bincount(clusters, minlength=getattr(detector, 'n_clusters', int(np.max(clusters)) + 1))
        ax1.bar(np.arange(counts.size), counts)
        ax1.set_xlabel('cluster')
        ax1.set_ylabel('count')
        ax1.set_title('Cluster sizes')

    ax2 = fig.add_subplot(122)
    order = np.argsort(d)
    ax2.plot(np.arange(1, len(d) + 1), d[order], marker='o', markersize=3, linewidth=1)
    ax2.axhline(detector.threshold_, linestyle='--', label=f'threshold={detector.threshold_:.3g}')
    if labels is not None:
        lab = np.asarray(labels)
        if lab.shape[0] == len(d):
            idx = np.where(lab.astype(bool)[order])[0]
            if idx.size:
                ax2.scatter(idx + 1, d[order][idx], s=40, facecolors='none', edgecolors='black', label='known anomaly')
    ax2.set_xlabel('observation rank')
    ax2.set_ylabel('cluster-robust squared distance')
    ax2.set_title('Cluster robust distance profile')
    ax2.legend()
    fig.tight_layout()
    _maybe_save_show(fig, output_path=output_path, show=show)
    return fig


def plot_robust_pca_outlier_map(
    pca,
    X=None,
    labels=None,
    score_cutoff=None,
    orthogonal_cutoff=None,
    title=None,
    output_path=None,
    show=True,
):
    """Plot score distance against orthogonal distance for a fitted RobustPCA.

    Parameters
    ----------
    pca : RobustPCA
        Fitted robust PCA object.
    X : array-like, optional
        Observations to diagnose. If omitted, stored training distances are used.
    labels : array-like, optional
        Boolean or binary labels used only to highlight known observations.
    score_cutoff : float, optional
        Optional vertical reference line.
    orthogonal_cutoff : float, optional
        Optional horizontal reference line.
    """
    if X is None:
        if not hasattr(pca, "score_distances_") or not hasattr(
            pca, "orthogonal_distances_"
        ):
            raise RuntimeError(
                "training distances are unavailable; fit with store_scores=True or pass X"
            )
        score_distance = np.asarray(pca.score_distances_, dtype=float)
        orthogonal_distance = np.asarray(pca.orthogonal_distances_, dtype=float)
    else:
        score_distance = np.asarray(pca.score_distances(X), dtype=float)
        orthogonal_distance = np.asarray(pca.orthogonal_distances(X), dtype=float)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(score_distance, orthogonal_distance, s=24, alpha=0.75)

    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape != score_distance.shape:
            raise ValueError("labels must have one value per observation")
        highlighted = labels.astype(bool)
        if np.any(highlighted):
            ax.scatter(
                score_distance[highlighted],
                orthogonal_distance[highlighted],
                s=64,
                facecolors="none",
                edgecolors="black",
                label="highlighted",
            )
            ax.legend()

    if score_cutoff is not None:
        ax.axvline(float(score_cutoff), linestyle="--")
    if orthogonal_cutoff is not None:
        ax.axhline(float(orthogonal_cutoff), linestyle="--")

    ax.set_xlabel("Score distance")
    ax.set_ylabel("Orthogonal distance")
    ax.set_title(title or f"Robust PCA outlier map ({type(pca.estimator_).__name__})")
    _maybe_save_show(fig, output_path=output_path, show=show)
    return fig


def plot_subspace_monitor_history(
    monitor_or_results,
    *,
    metrics=None,
    normalize=True,
    title=None,
    output_path=None,
    show=True,
):
    """Plot retained :class:`SubspaceDriftResult` monitoring history.

    Parameters
    ----------
    monitor_or_results : RobustSubspaceMonitor or sequence of SubspaceDriftResult
        Fitted monitor with retained ``history_`` or an explicit result sequence.
    metrics : sequence of str, optional
        Aggregate metrics shown in the upper panel.  Defaults to location,
        covariance shape, maximum subspace angle, orthogonal-distance shift,
        and combined outlier fraction.
    normalize : bool, default=True
        Divide every metric by its calibrated threshold.  In normalized mode,
        the horizontal line at one marks the alarm boundary.
    title : str, optional
        Figure title.
    """
    if hasattr(monitor_or_results, "history_"):
        results = list(monitor_or_results.history_)
    else:
        results = list(monitor_or_results)
    ready = [result for result in results if getattr(result, "ready", False)]
    if not ready:
        raise ValueError("monitoring history contains no ready results")

    if metrics is None:
        metrics = (
            "location_shift",
            "shape_shift",
            "max_subspace_angle",
            "orthogonal_distance_shift",
            "combined_outlier_fraction",
        )
    metrics = tuple(metrics)
    valid = set(ready[0].metrics)
    unknown = sorted(set(metrics) - valid)
    if unknown:
        raise ValueError("unknown monitoring metrics: " + ", ".join(unknown))

    x = np.arange(1, len(ready) + 1)
    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(211)
    for name in metrics:
        values = np.asarray([result.metrics[name] for result in ready], dtype=float)
        if normalize:
            thresholds = np.asarray(
                [result.thresholds.get(name, np.nan) for result in ready],
                dtype=float,
            )
            values = np.divide(
                values,
                thresholds,
                out=np.full_like(values, np.nan),
                where=np.isfinite(thresholds) & (thresholds > 0.0),
            )
        ax1.plot(x, values, marker="o", markersize=3, label=name.replace("_", " "))
    if normalize:
        ax1.axhline(1.0, linestyle="--", label="calibrated threshold")
        ax1.set_ylabel("metric / threshold")
    else:
        ax1.set_ylabel("metric value")
    ax1.set_title(title or "Robust rolling subspace monitoring")
    ax1.legend(ncol=2, fontsize="small")

    ax2 = fig.add_subplot(212, sharex=ax1)
    combined = np.asarray(
        [result.combined_outlier_fraction for result in ready], dtype=float
    )
    batch = np.asarray(
        [result.batch_combined_outlier_fraction for result in ready], dtype=float
    )
    ax2.plot(x, combined, marker="o", label="rolling-window outlier fraction")
    ax2.plot(x, batch, marker="s", label="incoming-batch outlier fraction")
    alarms = np.asarray([result.alarm for result in ready], dtype=bool)
    if np.any(alarms):
        ymax = max(float(np.nanmax(combined)), float(np.nanmax(batch)), 1e-12)
        ax2.scatter(x[alarms], np.full(np.sum(alarms), ymax), marker="x", s=55, label="alarm")
    ax2.set_xlabel("ready monitoring update")
    ax2.set_ylabel("outlier fraction")
    ax2.legend(ncol=2, fontsize="small")
    fig.tight_layout()
    _maybe_save_show(fig, output_path=output_path, show=show)
    return fig


def plot_matrix_outlier_contributions(
    estimator,
    X,
    *,
    index=0,
    absolute=False,
    row_labels=None,
    column_labels=None,
    title=None,
    output_path=None,
    show=True,
):
    """Plot cell contributions to a matrix Mahalanobis distance.

    Parameters
    ----------
    estimator : MatrixMinimumCovarianceDeterminant
        Fitted matrix covariance estimator exposing ``cell_contributions``.
    X : array-like of shape (n_samples, n_rows, n_columns)
        Matrix-valued observations.
    index : int, default=0
        Observation displayed in the heatmap.
    absolute : bool, default=False
        Plot absolute magnitudes instead of signed quadratic contributions.
    row_labels, column_labels : sequence of str, optional
        Axis labels for the matrix dimensions.

    Notes
    -----
    Signed contributions sum exactly to the squared matrix Mahalanobis distance.
    They are quadratic-form contributions and should not be interpreted as
    Shapley values.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError("X must have shape (n_samples, n_rows, n_columns)")
    index = int(index)
    if index < 0:
        index += X.shape[0]
    if not (0 <= index < X.shape[0]):
        raise IndexError("index is outside the sample")
    contribution = np.asarray(
        estimator.cell_contributions(X[index : index + 1])[0], dtype=float
    )
    shown = np.abs(contribution) if absolute else contribution

    fig = plt.figure(figsize=(8, 4.8))
    ax = fig.add_subplot(111)
    image = ax.imshow(shown, aspect="auto", interpolation="nearest")
    fig.colorbar(image, ax=ax, label=("absolute contribution" if absolute else "signed contribution"))
    if row_labels is not None:
        if len(row_labels) != shown.shape[0]:
            raise ValueError("row_labels must match the number of matrix rows")
        ax.set_yticks(np.arange(shown.shape[0]), labels=row_labels)
    else:
        ax.set_ylabel("matrix row")
    if column_labels is not None:
        if len(column_labels) != shown.shape[1]:
            raise ValueError("column_labels must match the number of matrix columns")
        ax.set_xticks(np.arange(shown.shape[1]), labels=column_labels, rotation=45, ha="right")
    else:
        ax.set_xlabel("matrix column")
    distance = float(estimator.mahalanobis(X[index : index + 1])[0])
    ax.set_title(title or f"Matrix-distance contributions (distance²={distance:.3g})")
    fig.tight_layout()
    _maybe_save_show(fig, output_path=output_path, show=show)
    return fig


def plot_cellwise_residual_map(
    estimator,
    X=None,
    *,
    row_labels=None,
    column_labels=None,
    clip=None,
    title=None,
    output_path=None,
    show=True,
):
    """Plot conditional standardized residuals from a fitted CellMCD model.

    Parameters
    ----------
    estimator : CellwiseMinimumCovarianceDeterminant
        Fitted estimator exposing ``standardized_residuals_`` and
        ``cell_outlier_mask_``.
    X : array-like, optional
        New rows to diagnose.  When omitted, fitted-data diagnostics are used.
    row_labels, column_labels : sequence of str, optional
        Labels for observations and variables.
    clip : float, optional
        Symmetric display limit.  The default is at least the fitted cell cutoff
        and otherwise follows the 99th percentile of finite residual magnitudes.
    """
    if X is None:
        if not hasattr(estimator, "standardized_residuals_"):
            raise RuntimeError("Estimator is not fitted")
        residuals = np.asarray(estimator.standardized_residuals_, dtype=float)
        outliers = np.asarray(estimator.cell_outlier_mask_, dtype=bool)
        missing = np.asarray(estimator.missing_mask_, dtype=bool)
    else:
        diagnostics = estimator.cellwise_diagnostics(X)
        residuals = np.asarray(diagnostics["standardized_residuals"], dtype=float)
        outliers = np.asarray(diagnostics["cell_outlier_mask"], dtype=bool)
        missing = np.asarray(diagnostics["missing_mask"], dtype=bool)

    finite_abs = np.abs(residuals[np.isfinite(residuals)])
    if clip is None:
        empirical = float(np.quantile(finite_abs, 0.99)) if finite_abs.size else 1.0
        clip = max(float(getattr(estimator, "cell_cutoff_", 0.0)), empirical, 1.0)
    clip = float(clip)
    if not np.isfinite(clip) or clip <= 0.0:
        raise ValueError("clip must be positive and finite")

    shown = np.clip(residuals, -clip, clip)
    shown = np.ma.masked_invalid(shown)
    height = max(4.2, min(10.0, 2.5 + 0.16 * residuals.shape[0]))
    width = max(7.0, min(13.0, 4.5 + 0.55 * residuals.shape[1]))
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111)
    image = ax.imshow(
        shown,
        aspect="auto",
        interpolation="nearest",
        vmin=-clip,
        vmax=clip,
        cmap="coolwarm",
    )
    fig.colorbar(image, ax=ax, label="conditional standardized residual")

    outlier_rows, outlier_cols = np.where(outliers)
    if outlier_rows.size:
        ax.scatter(outlier_cols, outlier_rows, marker="s", facecolors="none", edgecolors="black", s=36, linewidths=0.8, label="flagged cell")
    missing_rows, missing_cols = np.where(missing)
    if missing_rows.size:
        ax.scatter(missing_cols, missing_rows, marker="x", s=18, linewidths=0.7, label="missing cell")

    if column_labels is not None:
        if len(column_labels) != residuals.shape[1]:
            raise ValueError("column_labels must match the number of features")
        ax.set_xticks(np.arange(residuals.shape[1]), labels=column_labels, rotation=45, ha="right")
    else:
        ax.set_xlabel("feature")
    if row_labels is not None:
        if len(row_labels) != residuals.shape[0]:
            raise ValueError("row_labels must match the number of observations")
        ax.set_yticks(np.arange(residuals.shape[0]), labels=row_labels)
    else:
        ax.set_ylabel("observation")
    ax.set_title(title or "Cellwise conditional residual map")
    if outlier_rows.size or missing_rows.size:
        ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    _maybe_save_show(fig, output_path=output_path, show=show)
    return fig


def plot_cellpca_outlier_map(
    pca,
    X=None,
    labels=None,
    case_cutoff=None,
    cell_cutoff=None,
    title=None,
    output_path=None,
    show=True,
):
    """Plot casewise deviation against the largest cellwise residual.

    Parameters
    ----------
    pca : CellwiseRobustPCA
        Fitted cellwise robust PCA model.
    X : array-like, optional
        New observations.  When omitted, fitted training diagnostics are used.
    labels : array-like, optional
        Boolean labels used only to highlight known observations.
    case_cutoff, cell_cutoff : float, optional
        Optional vertical and horizontal reference lines.
    """
    if X is None:
        if not hasattr(pca, "case_deviations_") or not hasattr(
            pca, "max_cell_residuals_"
        ):
            raise RuntimeError("CellPCA is not fitted")
        case_deviation = np.asarray(pca.case_deviations_, dtype=float)
        max_cell = np.asarray(pca.max_cell_residuals_, dtype=float)
    else:
        mapping = np.asarray(pca.outlier_map(X), dtype=float)
        case_deviation = mapping[:, 0]
        max_cell = mapping[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(case_deviation, max_cell, s=24, alpha=0.75)

    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape != case_deviation.shape:
            raise ValueError("labels must have one value per observation")
        highlighted = labels.astype(bool)
        if np.any(highlighted):
            ax.scatter(
                case_deviation[highlighted],
                max_cell[highlighted],
                s=64,
                facecolors="none",
                edgecolors="black",
                label="highlighted",
            )
            ax.legend()

    if case_cutoff is not None:
        ax.axvline(float(case_cutoff), linestyle="--")
    if cell_cutoff is not None:
        ax.axhline(float(cell_cutoff), linestyle="--")

    ax.set_xlabel("Casewise total deviation")
    ax.set_ylabel("Maximum absolute cell residual")
    ax.set_title(title or "CellPCA outlier map")
    _maybe_save_show(fig, output_path=output_path, show=show)
    return fig



def plot_sparse_cellpca_loadings(
    pca,
    feature_names=None,
    components=None,
    title=None,
    output_path=None,
    show=True,
):
    """Plot sparse robust PCA loadings as a component-by-feature heatmap.

    Parameters
    ----------
    pca : SparseCellwiseRobustPCA
        Fitted sparse cellwise robust PCA estimator.
    feature_names : sequence of str, optional
        Labels for the feature axis.
    components : sequence of int, optional
        Components to show.  The default displays every retained component.
    """
    if not hasattr(pca, "components_") or not hasattr(pca, "loading_support_"):
        raise RuntimeError("SparseCellPCA is not fitted")
    loadings = np.asarray(pca.components_, dtype=float)
    if components is None:
        selected = np.arange(loadings.shape[0])
    else:
        selected = np.asarray(components, dtype=int)
        if selected.ndim != 1 or selected.size == 0:
            raise ValueError("components must be a non-empty one-dimensional sequence")
        if np.any((selected < 0) | (selected >= loadings.shape[0])):
            raise ValueError("components contains an invalid component index")
    shown = loadings[selected]
    limit = max(float(np.max(np.abs(shown))), np.finfo(float).eps)
    width = max(7.0, min(15.0, 5.0 + 0.28 * shown.shape[1]))
    height = max(3.5, 2.2 + 0.7 * shown.shape[0])
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111)
    image = ax.imshow(
        shown,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    fig.colorbar(image, ax=ax, label="loading")
    ax.set_yticks(np.arange(selected.size), labels=[f"PC {i + 1}" for i in selected])
    if feature_names is not None:
        if len(feature_names) != shown.shape[1]:
            raise ValueError("feature_names must match the number of features")
        ax.set_xticks(
            np.arange(shown.shape[1]),
            labels=feature_names,
            rotation=60,
            ha="right",
        )
    else:
        ax.set_xlabel("feature")
    ax.set_title(title or "Sparse CellPCA loadings")
    fig.tight_layout()
    _maybe_save_show(fig, output_path=output_path, show=show)
    return fig

def plot_partial_correlation_network(
    estimator,
    feature_names=None,
    min_abs_partial_correlation=0.05,
    title=None,
    output_path=None,
    show=True,
):
    """Plot a fitted sparse partial-correlation graph on a circular layout.

    Edge width is proportional to absolute partial correlation. Solid and
    dashed edges represent positive and negative conditional associations.
    """
    if not hasattr(estimator, "partial_correlation_"):
        raise RuntimeError("estimator must be fitted before plotting")
    partial = np.asarray(estimator.partial_correlation_, dtype=float)
    adjacency = np.asarray(estimator.adjacency_, dtype=bool)
    p = partial.shape[0]
    if feature_names is None:
        labels = [str(index) for index in range(p)]
    else:
        if len(feature_names) != p:
            raise ValueError("feature_names must have one entry per feature")
        labels = [str(value) for value in feature_names]
    threshold = float(min_abs_partial_correlation)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("min_abs_partial_correlation must be non-negative")

    angles = np.linspace(0.0, 2.0 * np.pi, p, endpoint=False)
    positions = np.column_stack([np.cos(angles), np.sin(angles)])
    fig = plt.figure(figsize=(7.0, 7.0))
    ax = fig.add_subplot(111)

    for i in range(p):
        for j in range(i + 1, p):
            value = float(partial[i, j])
            if not adjacency[i, j] or abs(value) < threshold:
                continue
            ax.plot(
                positions[[i, j], 0],
                positions[[i, j], 1],
                linewidth=0.6 + 4.0 * abs(value),
                linestyle="-" if value >= 0.0 else "--",
                alpha=0.75,
            )

    ax.scatter(positions[:, 0], positions[:, 1], s=520, zorder=3)
    for (x_coord, y_coord), label in zip(positions, labels):
        ax.text(x_coord, y_coord, label, ha="center", va="center", fontsize=8, zorder=4)
    ax.set_aspect("equal")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.axis("off")
    ax.set_title(title or "Robust partial-correlation network")
    _maybe_save_show(fig, output_path=output_path, show=show)
    return fig


def plot_subspace_stability(
    stability,
    *,
    component=0,
    feature_names=None,
    title=None,
    output_path=None,
    show=True,
):
    """Plot bootstrap loading intervals and retained-subspace angle stability.

    Parameters
    ----------
    stability : SubspaceStability
        Fitted bootstrap stability analysis.
    component : int, default=0
        Component shown in the loading-interval panel.
    feature_names : sequence of str, optional
        Labels for the loading coordinates.
    """
    if not hasattr(stability, "loading_interval_"):
        raise RuntimeError("SubspaceStability is not fitted")
    component = int(component)
    if component < 0 or component >= stability.n_components_:
        raise IndexError("component index is out of range")

    reference = np.asarray(stability.components_[component], dtype=float)
    lower = np.asarray(stability.loading_interval_lower_[component], dtype=float)
    upper = np.asarray(stability.loading_interval_upper_[component], dtype=float)
    p = reference.size
    if feature_names is None:
        labels = [str(index) for index in range(p)]
    else:
        if len(feature_names) != p:
            raise ValueError("feature_names must have one entry per feature")
        labels = [str(value) for value in feature_names]

    fig = plt.figure(figsize=(11.0, 4.6))
    ax1 = fig.add_subplot(121)
    x = np.arange(p)
    ax1.vlines(x, lower, upper, linewidth=1.2)
    ax1.hlines(lower, x - 0.08, x + 0.08, linewidth=1.0)
    ax1.hlines(upper, x - 0.08, x + 0.08, linewidth=1.0)
    ax1.scatter(x, reference, s=28, zorder=3)
    ax1.axhline(0.0, linewidth=0.8, linestyle="--")
    ax1.set_xticks(x, labels=labels, rotation=45, ha="right")
    ax1.set_ylabel("loading")
    ax1.set_title(
        f"Component {component + 1}: "
        f"{100.0 * stability.confidence_level:.0f}% bootstrap intervals"
    )

    ax2 = fig.add_subplot(122)
    values = np.asarray(stability.max_principal_angle_degrees_, dtype=float)
    ax2.hist(values, bins=min(25, max(8, int(np.sqrt(values.size)))))
    median = float(np.median(values))
    upper_angle = float(stability.max_principal_angle_interval_degrees_[1])
    ax2.axvline(median, linestyle="--", label=f"median {median:.2f}°")
    ax2.axvline(upper_angle, linestyle=":", label=f"upper interval {upper_angle:.2f}°")
    ax2.set_xlabel("largest principal angle (degrees)")
    ax2.set_ylabel("bootstrap count")
    ax2.set_title("Retained-subspace variation")
    ax2.legend()

    fig.suptitle(title or "Bootstrap PCA subspace stability")
    fig.tight_layout()
    _maybe_save_show(fig, output_path=output_path, show=show)
    return fig


def plot_multilinear_residual_map(
    estimator,
    *,
    index=0,
    row_labels=None,
    column_labels=None,
    title=None,
    output_path=None,
    show=True,
):
    """Plot standardized cell residuals for one fitted matrix observation."""
    if not hasattr(estimator, "standardized_residuals_"):
        raise RuntimeError("estimator must be fitted before plotting")
    index = int(index)
    residuals = np.asarray(estimator.standardized_residuals_, dtype=float)
    if index < 0 or index >= residuals.shape[0]:
        raise IndexError("index is out of range")
    shown = residuals[index]
    limit = max(float(np.nanmax(np.abs(shown))), np.finfo(float).eps)
    fig = plt.figure(figsize=(7.2, 5.0))
    ax = fig.add_subplot(111)
    image = ax.imshow(
        shown,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    fig.colorbar(image, ax=ax, label="standardized residual")
    if row_labels is not None:
        if len(row_labels) != shown.shape[0]:
            raise ValueError("row_labels must match the number of matrix rows")
        ax.set_yticks(np.arange(shown.shape[0]), labels=row_labels)
    else:
        ax.set_ylabel("row mode")
    if column_labels is not None:
        if len(column_labels) != shown.shape[1]:
            raise ValueError("column_labels must match the number of matrix columns")
        ax.set_xticks(
            np.arange(shown.shape[1]), labels=column_labels, rotation=55, ha="right"
        )
    else:
        ax.set_xlabel("column mode")
    flagged = np.argwhere(estimator.cell_outlier_mask_[index])
    if flagged.size:
        ax.scatter(flagged[:, 1], flagged[:, 0], marker="s", facecolors="none", edgecolors="black")
    ax.set_title(title or f"Multilinear PCA residual map — observation {index}")
    fig.tight_layout()
    _maybe_save_show(fig, output_path=output_path, show=show)
    return fig


def plot_multilinear_outlier_map(
    estimator,
    *,
    title=None,
    output_path=None,
    show=True,
):
    """Plot casewise deviation against the largest absolute cell residual."""
    if not hasattr(estimator, "case_deviations_"):
        raise RuntimeError("estimator must be fitted before plotting")
    x = np.asarray(estimator.case_deviations_, dtype=float)
    y = np.asarray(estimator.max_cell_residuals_, dtype=float)
    fig = plt.figure(figsize=(6.4, 5.0))
    ax = fig.add_subplot(111)
    regular = ~np.asarray(estimator.case_outlier_mask_, dtype=bool)
    ax.scatter(x[regular], y[regular], s=28, alpha=0.75, label="regular-weight cases")
    if np.any(~regular):
        ax.scatter(x[~regular], y[~regular], s=38, marker="x", label="downweighted cases")
    ax.set_xlabel("casewise deviation")
    ax.set_ylabel("maximum absolute cell residual")
    ax.set_title(title or "Robust multilinear PCA outlier map")
    ax.legend()
    fig.tight_layout()
    _maybe_save_show(fig, output_path=output_path, show=show)
    return fig
