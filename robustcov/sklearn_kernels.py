# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import numpy as np

from .kernels import robust_matern_kernel, robust_rbf_kernel
from .metrics import pairwise_mahalanobis_squared

try:  # optional dependency
    from sklearn.gaussian_process.kernels import Hyperparameter, Kernel
except Exception as exc:  # pragma: no cover - exercised only without sklearn
    Hyperparameter = None
    Kernel = object
    _SKLEARN_IMPORT_ERROR = exc
else:
    _SKLEARN_IMPORT_ERROR = None


class _BaseRobustMahalanobisKernel(Kernel):
    """Shared sklearn adapter for robust full-matrix metric kernels."""

    requires_vector_input = True

    def __init__(self, precision, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), center=None, ridge=0.0):
        if _SKLEARN_IMPORT_ERROR is not None:
            raise ImportError("scikit-learn is required for robustcov.sklearn_kernels") from _SKLEARN_IMPORT_ERROR
        self.precision = precision
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.center = center
        self.ridge = ridge

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    def _precision_array(self):
        precision = np.asarray(self.precision, dtype=np.float64)
        if precision.ndim != 2 or precision.shape[0] != precision.shape[1]:
            raise ValueError("precision must be a square 2D array")
        return precision

    def _validate_X(self, X, Y=None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinity")
        if Y is not None:
            Y = np.asarray(Y, dtype=np.float64)
            if Y.ndim != 2:
                raise ValueError("Y must be a 2D array")
            if not np.isfinite(Y).all():
                raise ValueError("Y contains NaN or infinity")
            if X.shape[1] != Y.shape[1]:
                raise ValueError("X and Y must have the same number of features")
        precision = self._precision_array()
        if precision.shape != (X.shape[1], X.shape[1]):
            raise ValueError("precision shape must match X.shape[1]")
        return X, Y, precision

    def diag(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        return np.ones(X.shape[0])

    def is_stationary(self):
        return True

    def __repr__(self):
        return f"{self.__class__.__name__}(length_scale={self.length_scale!r})"


class RobustMahalanobisRBF(_BaseRobustMahalanobisKernel):
    """Scikit-learn RBF kernel with a fixed robust Mahalanobis input metric.

    Pass this kernel to ``sklearn.gaussian_process.GaussianProcessRegressor`` as
    one building block. The robust precision matrix is fixed; sklearn may still
    optimize ``length_scale`` and any surrounding kernels such as
    ``ConstantKernel`` or ``WhiteKernel``.
    """

    def __call__(self, X, Y=None, eval_gradient=False):
        X, Y, precision = self._validate_X(X, Y)
        if eval_gradient and Y is not None:
            raise ValueError("Gradient can only be evaluated when Y is None")
        K = robust_rbf_kernel(X, Y, precision=precision, length_scale=self.length_scale, center=self.center, ridge=float(self.ridge))
        if not eval_gradient:
            return K
        if self.hyperparameter_length_scale.fixed:
            return K, np.empty((X.shape[0], X.shape[0], 0))
        D2 = pairwise_mahalanobis_squared(X, precision=precision, center=self.center, ridge=float(self.ridge))
        grad = K * D2 / (float(self.length_scale) ** 2)
        return K, grad[:, :, np.newaxis]


class RobustMahalanobisMatern(_BaseRobustMahalanobisKernel):
    """Scikit-learn Matérn kernel with a fixed robust Mahalanobis input metric."""

    def __init__(self, precision, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), nu=1.5, center=None, ridge=0.0):
        super().__init__(precision=precision, length_scale=length_scale, length_scale_bounds=length_scale_bounds, center=center, ridge=ridge)
        self.nu = nu

    def __call__(self, X, Y=None, eval_gradient=False):
        X, Y, precision = self._validate_X(X, Y)
        if eval_gradient and Y is not None:
            raise ValueError("Gradient can only be evaluated when Y is None")
        K = robust_matern_kernel(X, Y, precision=precision, length_scale=self.length_scale, nu=self.nu, center=self.center, ridge=float(self.ridge))
        if not eval_gradient:
            return K
        if self.hyperparameter_length_scale.fixed:
            return K, np.empty((X.shape[0], X.shape[0], 0))
        # Use a stable numerical derivative with respect to log(length_scale).
        # This keeps the adapter compact and also supports arbitrary positive nu.
        eps = 1e-5
        old = float(self.length_scale)
        K_plus = robust_matern_kernel(X, precision=precision, length_scale=old * math.exp(eps), nu=self.nu, center=self.center, ridge=float(self.ridge))
        K_minus = robust_matern_kernel(X, precision=precision, length_scale=old * math.exp(-eps), nu=self.nu, center=self.center, ridge=float(self.ridge))
        grad = (K_plus - K_minus) / (2.0 * eps)
        return K, grad[:, :, np.newaxis]


__all__ = ["RobustMahalanobisRBF", "RobustMahalanobisMatern"]
