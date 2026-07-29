# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Robust static factor models for heavy-tailed and contaminated data."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from ._estimator import EstimatorMixin
from ._utils import check_array
from .m_estimators import RegularizedCauchy


def _sample_pairs(n_samples: int, max_pairs: int, random_state: int | None):
    total = n_samples * (n_samples - 1) // 2
    if total <= max_pairs:
        return np.triu_indices(n_samples, k=1)
    rng = np.random.default_rng(random_state)
    selected: set[tuple[int, int]] = set()
    while len(selected) < max_pairs:
        left = rng.integers(0, n_samples, size=max_pairs)
        right = rng.integers(0, n_samples, size=max_pairs)
        for i, j in zip(left, right, strict=True):
            if i == j:
                continue
            if i > j:
                i, j = j, i
            selected.add((int(i), int(j)))
            if len(selected) == max_pairs:
                break
    pairs = np.asarray(sorted(selected), dtype=int)
    return pairs[:, 0], pairs[:, 1]


def spatial_kendall_matrix(
    X: np.ndarray,
    *,
    max_pairs: int = 20000,
    random_state: int | None = 0,
    block_size: int = 4096,
) -> np.ndarray:
    """Estimate the multivariate spatial Kendall matrix from pair differences."""

    X = check_array(X)
    if not isinstance(max_pairs, (int, np.integer)) or max_pairs < 1:
        raise ValueError("max_pairs must be a positive integer")
    if not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    left, right = _sample_pairs(X.shape[0], int(max_pairs), random_state)
    result = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
    used = 0
    tiny = np.finfo(np.float64).tiny
    for start in range(0, left.size, int(block_size)):
        stop = min(start + int(block_size), left.size)
        differences = X[left[start:stop]] - X[right[start:stop]]
        row_scale = np.max(np.abs(differences), axis=1)
        valid = row_scale > 0.0
        if not np.any(valid):
            continue
        normalized = differences[valid] / row_scale[valid, None]
        norms = np.linalg.norm(normalized, axis=1)
        directions = normalized / np.maximum(norms, tiny)[:, None]
        result += directions.T @ directions
        used += int(np.sum(valid))
    if used == 0:
        raise ValueError("spatial Kendall matrix is undefined for coincident observations")
    result /= used
    return 0.5 * (result + result.T)


def _orient_loadings(loadings: np.ndarray) -> np.ndarray:
    loadings = np.asarray(loadings, dtype=np.float64).copy()
    peaks = np.argmax(np.abs(loadings), axis=0)
    signs = np.sign(loadings[peaks, np.arange(loadings.shape[1])])
    signs[signs == 0.0] = 1.0
    return loadings * signs[None, :]


def _batched_huber_factorization(
    centered: np.ndarray,
    loadings: np.ndarray,
    *,
    delta: float,
    max_iter: int,
    tol: float,
    ridge: float,
):
    n_samples, n_features = centered.shape
    n_factors = loadings.shape[1]
    factors = centered @ loadings
    previous_objective = np.inf
    converged = False
    tiny = np.finfo(np.float64).tiny

    for iteration in range(1, max_iter + 1):
        residual = centered - factors @ loadings.T
        scale = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
        data_scale = max(float(np.median(np.abs(centered))), tiny)
        scale = max(scale, np.sqrt(np.finfo(np.float64).eps) * data_scale, tiny)
        cutoff = delta * scale
        absolute = np.abs(residual)
        weights = np.minimum(1.0, cutoff / np.maximum(absolute, tiny))

        factor_gram = np.einsum(
            "np,pk,pl->nkl", weights, loadings, loadings, optimize=True
        )
        factor_rhs = np.einsum(
            "np,np,pk->nk", weights, centered, loadings, optimize=True
        )
        factor_gram += ridge * np.eye(n_factors)[None, :, :]
        factors = np.linalg.solve(factor_gram, factor_rhs[..., None])[..., 0]

        loading_gram = np.einsum(
            "np,nk,nl->pkl", weights, factors, factors, optimize=True
        )
        loading_rhs = np.einsum(
            "np,np,nk->pk", weights, centered, factors, optimize=True
        )
        loading_gram += ridge * np.eye(n_factors)[None, :, :]
        loadings = np.linalg.solve(loading_gram, loading_rhs[..., None])[..., 0]

        loadings, triangular = np.linalg.qr(loadings, mode="reduced")
        factors = factors @ triangular.T
        residual = centered - factors @ loadings.T
        absolute = np.abs(residual)
        objective = float(
            np.sum(
                np.where(
                    absolute <= cutoff,
                    0.5 * absolute * absolute,
                    cutoff * (absolute - 0.5 * cutoff),
                )
            )
        )
        relative_change = abs(previous_objective - objective) / max(
            previous_objective if np.isfinite(previous_objective) else objective,
            tiny,
        )
        if relative_change <= tol:
            converged = True
            break
        previous_objective = objective

    return factors, loadings, converged, iteration, objective


class RobustFactorModel(EstimatorMixin):
    """Static robust factor model with Kendall or Huber estimation.

    Parameters
    ----------
    n_factors : int or 'auto', default='auto'
        Number of common factors.  Automatic selection uses the largest
        eigenvalue ratio of the spatial Kendall matrix.
    method : {'kendall', 'huber'}, default='kendall'
        ``'kendall'`` estimates a robust loading subspace from pair directions.
        ``'huber'`` initializes from that subspace and minimizes a cellwise Huber
        reconstruction loss by alternating weighted least squares.
    max_factors : int or None, default=None
        Maximum candidate factor count for automatic selection.
    max_pairs : int, default=20000
        Maximum pair count for the spatial Kendall matrix.
    huber_delta : float, default=1.5
        Huber cutoff multiplier.
    max_iter : int, default=100
        Maximum Huber alternating iterations.
    tol : float, default=1e-6
        Relative Huber objective tolerance.
    ridge : float, default=1e-8
        Relative numerical ridge for weighted normal equations.
    random_state : int or None, default=0
        Pair-subsampling seed.
    """

    def __init__(
        self,
        n_factors="auto",
        method="kendall",
        max_factors=None,
        max_pairs=20000,
        huber_delta=1.5,
        max_iter=100,
        tol=1e-6,
        ridge=1e-8,
        random_state=0,
    ):
        self.n_factors = n_factors
        self.method = method
        self.max_factors = max_factors
        self.max_pairs = max_pairs
        self.huber_delta = huber_delta
        self.max_iter = max_iter
        self.tol = tol
        self.ridge = ridge
        self.random_state = random_state

    def _resolve_factor_count(self, eigenvalues: np.ndarray, n_samples: int) -> int:
        upper = min(eigenvalues.size - 1, n_samples - 1)
        if self.max_factors is not None:
            if not isinstance(self.max_factors, (int, np.integer)) or self.max_factors < 1:
                raise ValueError("max_factors must be a positive integer or None")
            upper = min(upper, int(self.max_factors))
        else:
            upper = min(upper, 10)
        if upper < 1:
            raise ValueError("factor model needs at least two non-constant features")
        if self.n_factors == "auto":
            denominator = np.maximum(
                eigenvalues[1 : upper + 1], np.finfo(np.float64).tiny
            )
            ratios = eigenvalues[:upper] / denominator
            self.factor_number_ratios_ = ratios
            return int(np.argmax(ratios) + 1)
        if isinstance(self.n_factors, (int, np.integer)) and not isinstance(
            self.n_factors, (bool, np.bool_)
        ):
            count = int(self.n_factors)
            if not 1 <= count <= upper:
                raise ValueError(f"n_factors must be between 1 and {upper}")
            self.factor_number_ratios_ = None
            return count
        raise ValueError("n_factors must be a positive integer or 'auto'")

    def _validate_parameters(self):
        method = str(self.method).lower()
        if method not in {"kendall", "huber"}:
            raise ValueError("method must be 'kendall' or 'huber'")
        if not isinstance(self.max_pairs, (int, np.integer)) or self.max_pairs < 1:
            raise ValueError("max_pairs must be a positive integer")
        if not np.isscalar(self.huber_delta) or self.huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not np.isscalar(self.tol) or self.tol <= 0:
            raise ValueError("tol must be positive")
        if not np.isscalar(self.ridge) or self.ridge <= 0:
            raise ValueError("ridge must be positive")
        return method

    def fit(self, X, y=None):
        del y
        X = check_array(X)
        method = self._validate_parameters()
        n_samples, n_features = X.shape
        location = np.median(X, axis=0)
        centered = X - location
        kendall = spatial_kendall_matrix(
            X,
            max_pairs=int(self.max_pairs),
            random_state=self.random_state,
        )
        eigenvalues, eigenvectors = np.linalg.eigh(kendall)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        n_factors = self._resolve_factor_count(eigenvalues, n_samples)
        loadings = _orient_loadings(eigenvectors[:, :n_factors])

        if method == "huber":
            data_variance = float(np.mean(centered * centered))
            ridge_amount = max(
                float(self.ridge) * max(data_variance, np.finfo(np.float64).tiny),
                np.finfo(np.float64).tiny,
            )
            factors, loadings, converged, n_iter, objective = _batched_huber_factorization(
                centered,
                loadings,
                delta=float(self.huber_delta),
                max_iter=int(self.max_iter),
                tol=float(self.tol),
                ridge=ridge_amount,
            )
            loadings = _orient_loadings(loadings)
            # Refit factor signs after deterministic loading orientation.
            factors = self._transform_huber(
                centered,
                loadings,
                ridge_amount,
                delta=float(self.huber_delta),
            )
        else:
            factors = centered @ loadings
            converged = True
            n_iter = 1
            objective = float(np.sum((centered - factors @ loadings.T) ** 2))
            ridge_amount = 0.0

        common = factors @ loadings.T
        residual = centered - common
        factor_scatter = RegularizedCauchy(
            alpha=0.05,
            max_iter=300,
            tol=1e-7,
            warn_on_nonconvergence=False,
        ).fit(factors)
        factor_covariance = np.asarray(factor_scatter.covariance_, dtype=np.float64)
        residual_median_squared = np.median(residual * residual, axis=0)
        idiosyncratic_variance = residual_median_squared / chi2.ppf(0.5, 1)
        variance_scale = max(float(np.median(np.var(centered, axis=0))), np.finfo(np.float64).tiny)
        variance_floor = max(float(self.ridge) * variance_scale, np.finfo(np.float64).tiny)
        idiosyncratic_variance = np.maximum(idiosyncratic_variance, variance_floor)
        covariance = (
            loadings @ factor_covariance @ loadings.T
            + np.diag(idiosyncratic_variance)
        )
        covariance = 0.5 * (covariance + covariance.T)

        self.location_ = location
        self.mean_ = location
        self.kendall_matrix_ = kendall
        self.kendall_eigenvalues_ = eigenvalues
        self.loadings_ = loadings
        self.components_ = loadings.T
        self.factor_scores_ = factors
        self.factors_ = factors
        self.common_component_ = common + location
        self.idiosyncratic_ = residual
        self.factor_scatter_estimator_ = factor_scatter
        self.factor_covariance_ = factor_covariance
        self.idiosyncratic_variance_ = idiosyncratic_variance
        self.idiosyncratic_covariance_ = np.diag(idiosyncratic_variance)
        self.covariance_ = covariance
        self.precision_ = np.linalg.pinv(covariance, hermitian=True)
        self.n_factors_ = n_factors
        self.n_components_ = n_factors
        self.n_features_in_ = n_features
        self.n_samples_in_ = n_samples
        self.converged_ = bool(converged)
        self.n_iter_ = int(n_iter)
        self.objective_ = float(objective)
        self.ridge_amount_ = float(ridge_amount)
        self.idiosyncratic_variance_floor_ = float(variance_floor)
        return self

    @staticmethod
    def _transform_huber(centered, loadings, ridge_amount, delta=1.5, n_iter=8):
        factors = centered @ loadings
        tiny = np.finfo(np.float64).tiny
        identity = np.eye(loadings.shape[1])
        for _ in range(n_iter):
            residual = centered - factors @ loadings.T
            scale = 1.4826 * np.median(np.abs(residual), axis=1)
            row_reference = np.maximum(np.median(np.abs(centered), axis=1), tiny)
            scale = np.maximum(
                scale,
                np.sqrt(np.finfo(np.float64).eps) * row_reference,
            )
            cutoff = delta * scale[:, None]
            weights = np.minimum(1.0, cutoff / np.maximum(np.abs(residual), tiny))
            gram = np.einsum(
                "np,pk,pl->nkl", weights, loadings, loadings, optimize=True
            )
            rhs = np.einsum(
                "np,np,pk->nk", weights, centered, loadings, optimize=True
            )
            gram += ridge_amount * identity[None, :, :]
            factors = np.linalg.solve(gram, rhs[..., None])[..., 0]
        return factors

    def _check_fitted(self):
        if not hasattr(self, "loadings_"):
            raise AttributeError("RobustFactorModel is not fitted yet")

    def transform(self, X):
        self._check_fitted()
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has an incompatible number of features")
        centered = X - self.location_
        if str(self.method).lower() == "huber":
            return self._transform_huber(
                centered,
                self.loadings_,
                self.ridge_amount_,
                delta=float(self.huber_delta),
            )
        return centered @ self.loadings_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).factor_scores_

    def inverse_transform(self, factors):
        self._check_fitted()
        factors = np.asarray(factors, dtype=np.float64)
        if factors.ndim != 2 or factors.shape[1] != self.n_factors_:
            raise ValueError("factors has incompatible shape")
        if not np.all(np.isfinite(factors)):
            raise ValueError("factors must contain only finite values")
        return factors @ self.loadings_.T + self.location_

    def get_covariance(self):
        self._check_fitted()
        return self.covariance_.copy()

    def get_precision(self):
        self._check_fitted()
        return self.precision_.copy()
