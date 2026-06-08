# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from ._utils import check_array


def _check_2d_array(X: np.ndarray, *, name: str = "X") -> np.ndarray:
    X = np.asarray(X, dtype=np.float64, order="C")
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if X.shape[0] < 1 or X.shape[1] < 1:
        raise ValueError(f"{name} must have at least 1 row and 1 column")
    if not np.isfinite(X).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return X


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be a square 2D array")
    return np.asarray(0.5 * (matrix + matrix.T), dtype=np.float64, order="C")


def _as_spd_precision(matrix: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    """Return a symmetric positive-semidefinite precision-like matrix.

    Robust scatter estimators can be nearly singular in small-sample or
    high-dimensional settings. Kernel distances only require a positive
    semidefinite quadratic form, so this helper clips tiny/negative numerical
    eigenvalues instead of failing on benign roundoff.
    """
    P = _symmetrize(matrix)
    if not np.isfinite(P).all():
        raise ValueError("precision contains NaN or infinity")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    if ridge > 0:
        P = P + float(ridge) * np.eye(P.shape[0])
    vals, vecs = np.linalg.eigh(P)
    vals = np.where(vals > 0.0, vals, 0.0)
    return np.asarray((vecs * vals) @ vecs.T, dtype=np.float64, order="C")


def pairwise_mahalanobis_squared(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    *,
    precision: np.ndarray,
    center: np.ndarray | None = None,
    ridge: float = 0.0,
) -> np.ndarray:
    """Pairwise squared Mahalanobis distances under a fixed precision matrix.

    Parameters
    ----------
    X, Y : array-like of shape (n_samples, n_features)
        Input matrices. If ``Y`` is omitted, distances are computed between rows
        of ``X``.
    precision : array-like of shape (n_features, n_features)
        Positive-semidefinite inverse scatter/covariance matrix defining the
        input metric.
    center : array-like of shape (n_features,), optional
        Optional translation applied to both ``X`` and ``Y`` before distances are
        computed. Pairwise translation cancels mathematically, but the argument is
        useful for consistency with fitted robust metrics and transformed inputs.
    ridge : float, default=0.0
        Optional diagonal ridge added to ``precision`` before the quadratic form.

    Returns
    -------
    distances : ndarray of shape (n_X, n_Y)
        Squared distances ``(x - y)^T precision (x - y)``.
    """
    X = _check_2d_array(X, name="X")
    Y_was_none = Y is None
    Y = X if Y is None else _check_2d_array(Y, name="Y")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of features")
    P = _as_spd_precision(precision, ridge=ridge)
    if P.shape != (X.shape[1], X.shape[1]):
        raise ValueError("precision shape must match the number of features")
    if center is not None:
        c = np.asarray(center, dtype=np.float64)
        if c.shape != (X.shape[1],):
            raise ValueError("center must have shape (n_features,)")
        X = X - c
        Y = Y - c
    XP = X @ P
    YP = XP if Y_was_none else Y @ P
    X_norm = np.einsum("ij,ij->i", XP, X)[:, None]
    Y_norm = X_norm.T if Y_was_none else np.einsum("ij,ij->i", YP, Y)[None, :]
    D2 = X_norm + Y_norm - 2.0 * (XP @ Y.T)
    return np.maximum(D2, 0.0)


class RobustInputMetric:
    """Fit and reuse a robust full-matrix input metric.

    This class is intentionally smaller than a GP/KRR model. It only learns a
    robust scatter estimate from feature vectors and exposes the corresponding
    precision matrix for kernel methods.

    Parameters
    ----------
    estimator : robustcov estimator, optional
        Any estimator with ``fit(X)`` and fitted ``covariance_`` or
        ``precision_`` attributes. If omitted, ``RegularizedCauchy(alpha=0.05)``
        is used because it is robust in heavy-tailed and small-sample regimes.
    copy_estimator : bool, default=True
        Whether to deep-copy ``estimator`` before fitting.
    ridge : float, default=1e-10
        Diagonal ridge used when constructing a numerically stable precision.
    use_precision : bool, default=True
        Prefer an estimator's fitted ``precision_`` attribute when available.
        If false, or if ``precision_`` is absent, the pseudoinverse of
        ``covariance_`` is used.
    """

    def __init__(self, estimator: Any | None = None, *, copy_estimator: bool = True, ridge: float = 1e-10, use_precision: bool = True):
        self.estimator = estimator
        self.copy_estimator = bool(copy_estimator)
        self.ridge = float(ridge)
        self.use_precision = bool(use_precision)

    def _default_estimator(self):
        from .m_estimators import RegularizedCauchy

        return RegularizedCauchy(alpha=0.05, scale_correction="radial_median", warn_on_nonconvergence=False)

    def fit(self, X, y=None):
        X = check_array(X, allow_nan=False)
        est = self._default_estimator() if self.estimator is None else self.estimator
        self.estimator_ = copy.deepcopy(est) if self.copy_estimator else est
        self.estimator_.fit(X)
        self.n_samples_in_, self.n_features_in_ = X.shape
        self.location_ = np.asarray(getattr(self.estimator_, "location_", np.zeros(X.shape[1])), dtype=np.float64)
        if self.location_.shape != (X.shape[1],):
            raise ValueError("fitted estimator has incompatible location_ shape")
        if self.use_precision and hasattr(self.estimator_, "precision_"):
            raw_precision = getattr(self.estimator_, "precision_")
            self.covariance_ = np.asarray(getattr(self.estimator_, "covariance_", np.linalg.pinv(raw_precision)), dtype=np.float64)
        elif hasattr(self.estimator_, "covariance_"):
            self.covariance_ = _symmetrize(getattr(self.estimator_, "covariance_"))
            raw_precision = np.linalg.pinv(self.covariance_)
        else:
            raise ValueError("estimator must expose fitted precision_ or covariance_")
        self.precision_ = _as_spd_precision(raw_precision, ridge=self.ridge)
        self.covariance_ = _symmetrize(self.covariance_)
        return self

    def transform(self, X):
        """Center inputs with the fitted robust location."""
        self._check_is_fitted()
        X = _check_2d_array(X, name="X")
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the fitted metric")
        return X - self.location_

    def squared_distance(self, X, Y=None):
        """Pairwise robust squared Mahalanobis distances."""
        self._check_is_fitted()
        return pairwise_mahalanobis_squared(X, Y, precision=self.precision_, center=self.location_)

    def kernel_matrix(self, X, Y=None, *, kind: str = "rbf", length_scale: float = 1.0, nu: float = 1.5):
        """Compute an RBF or Matérn kernel matrix from this robust metric."""
        self._check_is_fitted()
        from .kernels import robust_matern_kernel, robust_rbf_kernel

        kind = kind.lower()
        if kind in {"rbf", "gaussian", "squared_exponential"}:
            return robust_rbf_kernel(X, Y, precision=self.precision_, length_scale=length_scale, center=self.location_)
        if kind in {"matern", "matérn"}:
            return robust_matern_kernel(X, Y, precision=self.precision_, length_scale=length_scale, nu=nu, center=self.location_)
        raise ValueError("kind must be 'rbf' or 'matern'")

    def _check_is_fitted(self):
        if not hasattr(self, "precision_"):
            raise RuntimeError("RobustInputMetric is not fitted")


__all__ = ["RobustInputMetric", "pairwise_mahalanobis_squared"]
