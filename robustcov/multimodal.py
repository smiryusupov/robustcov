# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy

import numpy as np
from scipy.spatial.distance import cdist

from ._utils import check_array
from .m_estimators import RegularizedCauchy


def _simple_kmeans(X: np.ndarray, n_clusters: int, random_state: int = 0, max_iter: int = 100, tol: float = 1e-4):
    """Small NumPy k-means implementation used to avoid a hard sklearn dependency."""
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")
    if n_clusters > n:
        raise ValueError("n_clusters cannot exceed n_samples")
    rng = np.random.default_rng(random_state)
    # k-means++ style initialization.
    centers = np.empty((n_clusters, p), dtype=float)
    first = int(rng.integers(0, n))
    centers[0] = X[first]
    closest = np.sum((X - centers[0]) ** 2, axis=1)
    for k in range(1, n_clusters):
        total = float(np.sum(closest))
        if not np.isfinite(total) or total <= 0:
            centers[k] = X[int(rng.integers(0, n))]
        else:
            probs = closest / total
            centers[k] = X[int(rng.choice(n, p=probs))]
        closest = np.minimum(closest, np.sum((X - centers[k]) ** 2, axis=1))

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        distances = cdist(X, centers, metric="sqeuclidean")
        new_labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for k in range(n_clusters):
            mask = new_labels == k
            if np.any(mask):
                new_centers[k] = np.mean(X[mask], axis=0)
            else:
                # Re-seed empty cluster with a high-error point.
                far = int(np.argmax(np.min(distances, axis=1)))
                new_centers[k] = X[far]
                new_labels[far] = k
        shift = float(np.max(np.sqrt(np.sum((new_centers - centers) ** 2, axis=1))))
        centers = new_centers
        labels = new_labels
        if shift <= tol:
            break
    return labels, centers


class ClusterRobustOutlierDetector:
    """Cluster-aware robust distance detector for multimodal data.

    A single robust covariance estimator is appropriate for one central elliptical
    cloud plus contamination. In multimodal data, valid minority clusters can look
    like outliers to a global estimator. This detector uses a simple two-stage
    diagnostic:

    1. cluster the observations;
    2. fit a robust scatter estimator inside each sufficiently large cluster;
    3. score each observation by its robust distance to its assigned cluster.

    This is intentionally *not* a full robust mixture model. It is a pragmatic
    cluster-then-robust-scatter diagnostic for datasets with several legitimate
    modes.
    """

    def __init__(
        self,
        base_estimator=None,
        n_clusters: int = 2,
        threshold="empirical",
        alpha: float = 0.975,
        contamination=None,
        min_cluster_size=None,
        random_state: int = 0,
        max_iter: int = 100,
        cluster_tol: float = 1e-4,
    ):
        self.base_estimator = base_estimator
        self.n_clusters = int(n_clusters)
        self.threshold = threshold
        self.alpha = float(alpha)
        self.contamination = contamination
        self.min_cluster_size = min_cluster_size
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)
        self.cluster_tol = float(cluster_tol)

    def _make_estimator(self):
        if self.base_estimator is None:
            return RegularizedCauchy(alpha=0.05, scale_correction="radial_median", warn_on_nonconvergence=False)
        return copy.deepcopy(self.base_estimator)

    def fit(self, X, y=None):
        X = check_array(X, allow_nan=False)
        self.n_samples_in_, self.n_features_in_ = X.shape
        if self.n_clusters > self.n_samples_in_:
            raise ValueError("n_clusters cannot exceed number of samples")
        min_cluster_size = self.min_cluster_size
        if min_cluster_size is None:
            min_cluster_size = max(self.n_features_in_ + 2, 8)
        self.min_cluster_size_ = int(min_cluster_size)

        labels, centers = _simple_kmeans(
            X,
            self.n_clusters,
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=self.cluster_tol,
        )
        self.cluster_labels_ = labels
        self.cluster_centers_ = centers
        self.cluster_sizes_ = np.bincount(labels, minlength=self.n_clusters)

        self.global_estimator_ = self._make_estimator().fit(X)
        self.cluster_estimators_ = []
        self.cluster_valid_ = np.zeros(self.n_clusters, dtype=bool)
        for k in range(self.n_clusters):
            mask = labels == k
            if int(np.sum(mask)) >= self.min_cluster_size_:
                est = self._make_estimator().fit(X[mask])
                self.cluster_valid_[k] = True
            else:
                est = self.global_estimator_
            self.cluster_estimators_.append(est)

        d = np.empty(self.n_samples_in_, dtype=float)
        for k in range(self.n_clusters):
            mask = labels == k
            if np.any(mask):
                d[mask] = np.asarray(self.cluster_estimators_[k].mahalanobis(X[mask]), dtype=float)
        self.distances_ = d
        self.score_ = d

        if self.contamination is not None:
            q = 1.0 - float(self.contamination)
            if not (0 < q < 1):
                raise ValueError("contamination must be in (0, 1)")
            self.threshold_ = float(np.quantile(d, q))
        elif self.threshold in {"empirical", "tail_calibrated"}:
            self.threshold_ = float(np.quantile(d, self.alpha))
        elif isinstance(self.threshold, (int, float)):
            self.threshold_ = float(self.threshold)
        else:
            raise ValueError("threshold must be 'empirical', 'tail_calibrated', or a number")
        self.labels_ = np.where(d <= self.threshold_, 1, -1)
        self.outlier_mask_ = self.labels_ == -1
        return self

    def _assign_clusters(self, X):
        X = check_array(X, allow_nan=False)
        return np.argmin(cdist(X, self.cluster_centers_, metric="sqeuclidean"), axis=1)

    def distances_to_clusters(self, X):
        """Return an ``(n_samples, n_clusters)`` matrix of robust distances."""
        X = check_array(X, allow_nan=False)
        out = np.empty((X.shape[0], self.n_clusters), dtype=float)
        for k, est in enumerate(self.cluster_estimators_):
            out[:, k] = np.asarray(est.mahalanobis(X), dtype=float)
        return out

    def decision_function(self, X):
        X = check_array(X, allow_nan=False)
        assigned = self._assign_clusters(X)
        d = np.empty(X.shape[0], dtype=float)
        for k in range(self.n_clusters):
            mask = assigned == k
            if np.any(mask):
                d[mask] = np.asarray(self.cluster_estimators_[k].mahalanobis(X[mask]), dtype=float)
        return d

    def predict(self, X):
        d = self.decision_function(X)
        return np.where(d <= self.threshold_, 1, -1)

    def fit_predict(self, X, y=None):
        return self.fit(X, y=y).labels_

    def summary(self) -> str:
        parts = [
            f"ClusterRobustOutlierDetector(n_clusters={self.n_clusters})",
            f"cluster_sizes={self.cluster_sizes_.tolist()}",
            f"valid_clusters={self.cluster_valid_.tolist()}",
            f"threshold={self.threshold_:.4g}",
            f"detected={int(np.sum(self.outlier_mask_))}",
        ]
        return "; ".join(parts)
