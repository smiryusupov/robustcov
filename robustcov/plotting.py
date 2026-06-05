from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2


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
