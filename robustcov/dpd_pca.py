# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Density-power-divergence robust principal component analysis.

This module implements a package-native alternating-regression PCA estimator
based on the Gaussian density-power-divergence (DPD) loss.  It follows the
central rPCAdpd/rSVDdpd construction of Roy, Basu, and Ghosh: robust location
is estimated first, then a low-rank factorization is updated by alternating
DPD-weighted regressions and a DPD residual-scale update.

The reference implementation normalizes the alternating regression factors in
a particular sequence and offers additional tuning utilities.  ``robustcov``
uses block weighted-least-squares updates, QR reparameterization, and a final
SVD canonicalization.  It therefore optimizes the same Gaussian DPD working
loss but does not claim numerical identity with the authors' software.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._estimator import EstimatorMixin


_EPS = np.finfo(np.float64).eps


def _as_2d_finite_array(X: Any, *, name: str = "X", min_rows: int = 2) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64, order="C")
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if X.shape[0] < min_rows:
        raise ValueError(f"{name} must contain at least {min_rows} rows")
    if X.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two features")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} must contain only finite values")
    return X


def _geometric_median(X: np.ndarray, *, tol: float = 1e-8, max_iter: int = 200) -> np.ndarray:
    """Compute the Euclidean geometric median with Weiszfeld iterations."""
    current = np.median(X, axis=0).astype(np.float64, copy=True)
    for _ in range(max_iter):
        distances = np.linalg.norm(X - current, axis=1)
        closest = int(np.argmin(distances))
        if distances[closest] <= tol:
            return X[closest].copy()
        weights = 1.0 / np.maximum(distances, np.sqrt(_EPS))
        updated = np.sum(weights[:, None] * X, axis=0) / np.sum(weights)
        if np.linalg.norm(updated - current) <= tol * max(np.linalg.norm(current), 1.0):
            return updated
        current = updated
    return current


def _deterministic_svd_signs(U: np.ndarray, Vt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    U = np.asarray(U, dtype=np.float64).copy()
    Vt = np.asarray(Vt, dtype=np.float64).copy()
    if Vt.size == 0:
        return U, Vt
    rows = np.arange(Vt.shape[0])
    columns = np.argmax(np.abs(Vt), axis=1)
    signs = np.sign(Vt[rows, columns])
    signs[signs == 0.0] = 1.0
    Vt *= signs[:, None]
    U *= signs[None, :]
    return U, Vt


def _winsorized_matrix(X: np.ndarray, clip: float) -> np.ndarray:
    center = np.median(X, axis=0)
    scale = 1.482602218505602 * np.median(np.abs(X - center), axis=0)
    fallback = np.std(X, axis=0, ddof=1)
    scale = np.where(np.isfinite(scale) & (scale > np.sqrt(_EPS)), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > np.sqrt(_EPS)), scale, 1.0)
    lower = center - clip * scale
    upper = center + clip * scale
    return np.clip(X, lower, upper)


def _dpd_weights(residuals: np.ndarray, alpha: float, sigma2: float) -> np.ndarray:
    if alpha == 0.0:
        return np.ones_like(residuals, dtype=np.float64)
    exponent = -0.5 * alpha * residuals * residuals / max(sigma2, np.finfo(float).tiny)
    return np.exp(np.clip(exponent, -745.0, 0.0))


def _mad_variance(values: np.ndarray, floor: float) -> float:
    values = np.asarray(values, dtype=np.float64).ravel()
    center = float(np.median(values))
    scale = 1.482602218505602 * float(np.median(np.abs(values - center)))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.sqrt(np.mean((values - center) ** 2)))
    return max(scale * scale, floor)


def _dpd_scale_update(
    residuals: np.ndarray,
    alpha: float,
    sigma2: float,
    floor: float,
) -> float:
    """One Gaussian DPD scale fixed-point update.

    Differentiating the univariate Gaussian DPD objective yields

    sigma^2 = mean(w r^2) / [mean(w) - alpha/(1+alpha)^(3/2)].
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    if alpha == 0.0:
        return max(float(np.mean(residuals * residuals)), floor)
    weights = _dpd_weights(residuals, alpha, sigma2)
    correction = alpha / ((1.0 + alpha) ** 1.5)
    denominator = float(np.mean(weights) - correction)
    numerator = float(np.mean(weights * residuals * residuals))
    if not np.isfinite(denominator) or denominator <= 10.0 * _EPS:
        weighted_denominator = max(float(np.mean(weights)), 10.0 * _EPS)
        candidate = numerator / weighted_denominator
    else:
        candidate = numerator / denominator
    if not np.isfinite(candidate) or candidate <= floor:
        candidate = _mad_variance(residuals, floor)
    return max(float(candidate), floor)


def _dpd_objective(residuals: np.ndarray, alpha: float, sigma2: float) -> float:
    residuals = np.asarray(residuals, dtype=np.float64)
    sigma2 = max(float(sigma2), np.finfo(float).tiny)
    if alpha == 0.0:
        return float(0.5 * np.log(sigma2) + 0.5 * np.mean(residuals * residuals) / sigma2)
    weights = _dpd_weights(residuals, alpha, sigma2)
    first = 1.0 / np.sqrt(1.0 + alpha)
    second = ((1.0 + alpha) / alpha) * float(np.mean(weights))
    return float((2.0 * np.pi) ** (-0.5 * alpha) * sigma2 ** (-0.5 * alpha) * (first - second))


def _weighted_loading_update(
    X: np.ndarray,
    scores: np.ndarray,
    loadings: np.ndarray,
    alpha: float,
    sigma2: float,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    residuals = X - scores @ loadings.T
    weights = _dpd_weights(residuals, alpha, sigma2)
    q = scores.shape[1]
    identity = np.eye(q)
    updated = loadings.copy()
    for j in range(X.shape[1]):
        w = weights[:, j]
        gram = scores.T @ (w[:, None] * scores) + ridge * identity
        rhs = scores.T @ (w * X[:, j])
        try:
            updated[j] = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            updated[j] = np.linalg.pinv(gram) @ rhs

    qmat, rmat = np.linalg.qr(updated, mode="reduced")
    scores = scores @ rmat.T
    return scores, qmat


def _weighted_score_update(
    X: np.ndarray,
    scores: np.ndarray,
    loadings: np.ndarray,
    alpha: float,
    sigma2: float,
    ridge: float,
) -> np.ndarray:
    residuals = X - scores @ loadings.T
    weights = _dpd_weights(residuals, alpha, sigma2)
    q = loadings.shape[1]
    identity = np.eye(q)
    updated = scores.copy()
    for i in range(X.shape[0]):
        w = weights[i]
        gram = loadings.T @ (w[:, None] * loadings) + ridge * identity
        rhs = loadings.T @ (w * X[i])
        try:
            updated[i] = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            updated[i] = np.linalg.pinv(gram) @ rhs
    return updated


@dataclass
class DensityPowerRobustPCA(EstimatorMixin):
    """Direct robust PCA using a Gaussian density-power-divergence loss.

    The estimator fits a rank-``n_components`` low-rank model by alternating
    DPD-weighted regressions for the score and loading matrices.  The tuning
    parameter ``alpha`` provides a smooth efficiency--robustness tradeoff:
    ``alpha=0`` gives the Gaussian least-squares limit, while larger values
    downweight large reconstruction residuals more strongly.

    Parameters
    ----------
    n_components : int, default=2
        Rank of the fitted low-rank model.
    alpha : float, default=0.3
        Density-power-divergence tuning parameter in ``[0, 1]``.
    location : {"geometric_median", "coordinate_median", "mean"} or array-like,
        default="geometric_median"
        Center used before the low-rank fit.  The geometric median preserves
        orthogonal equivariance and follows the practical recommendation in
        the rPCAdpd paper.
    init : {"winsorized_svd", "svd"}, default="winsorized_svd"
        Initialization of the low-rank factors.
    winsorize : float, default=4.0
        Marginal clipping threshold used by ``winsorized_svd``.
    max_iter : int, default=100
        Maximum number of alternating-regression iterations.
    inner_iter : int, default=2
        Number of score/loading fixed-point sweeps per outer iteration.
    tol : float, default=1e-6
        Relative convergence tolerance for the fitted low-rank matrix and
        residual scale.
    ridge : float, default=1e-8
        Positive ridge added to the small weighted normal equations.
    min_scale : float, default=1e-8
        Lower bound for the residual standard deviation.
    whiten : bool, default=False
        Divide returned scores by the square roots of the fitted DPD
        eigenvalues.
    store_scores : bool, default=True
        Store training scores and diagnostic distances.

    Notes
    -----
    This is a package-native implementation of the Gaussian DPD alternating
    regression formulation.  It does not claim numerical identity with the
    authors' rSVDdpd software, whose normalization sequence and initialization
    differ.
    """

    n_components: int = 2
    alpha: float = 0.3
    location: Any = "geometric_median"
    init: str = "winsorized_svd"
    winsorize: float = 4.0
    max_iter: int = 100
    inner_iter: int = 2
    tol: float = 1e-6
    ridge: float = 1e-8
    min_scale: float = 1e-8
    whiten: bool = False
    store_scores: bool = True

    def _validate_parameters(self, n_samples: int, n_features: int) -> None:
        if isinstance(self.n_components, (bool, np.bool_)) or not isinstance(
            self.n_components, (int, np.integer)
        ):
            raise TypeError("n_components must be an integer")
        if not 1 <= int(self.n_components) <= min(n_samples, n_features):
            raise ValueError("n_components must be between 1 and min(n_samples, n_features)")
        if not np.isscalar(self.alpha) or not np.isfinite(self.alpha) or not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be a finite number in [0, 1]")
        if self.init not in {"winsorized_svd", "svd"}:
            raise ValueError("init must be 'winsorized_svd' or 'svd'")
        if not np.isscalar(self.winsorize) or not np.isfinite(self.winsorize) or self.winsorize <= 0.0:
            raise ValueError("winsorize must be positive and finite")
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not isinstance(self.inner_iter, (int, np.integer)) or self.inner_iter < 1:
            raise ValueError("inner_iter must be a positive integer")
        if not np.isscalar(self.tol) or not np.isfinite(self.tol) or self.tol <= 0.0:
            raise ValueError("tol must be positive and finite")
        if not np.isscalar(self.ridge) or not np.isfinite(self.ridge) or self.ridge <= 0.0:
            raise ValueError("ridge must be positive and finite")
        if not np.isscalar(self.min_scale) or not np.isfinite(self.min_scale) or self.min_scale <= 0.0:
            raise ValueError("min_scale must be positive and finite")
        if not isinstance(self.whiten, (bool, np.bool_)):
            raise TypeError("whiten must be a boolean")
        if not isinstance(self.store_scores, (bool, np.bool_)):
            raise TypeError("store_scores must be a boolean")

    def _estimate_location(self, X: np.ndarray) -> np.ndarray:
        if isinstance(self.location, str):
            if self.location == "geometric_median":
                return _geometric_median(X)
            if self.location == "coordinate_median":
                return np.median(X, axis=0)
            if self.location == "mean":
                return np.mean(X, axis=0)
            raise ValueError(
                "location must be 'geometric_median', 'coordinate_median', 'mean', or an array"
            )
        location = np.asarray(self.location, dtype=np.float64)
        if location.shape != (X.shape[1],):
            raise ValueError("array-like location must have shape (n_features,)")
        if not np.all(np.isfinite(location)):
            raise ValueError("array-like location must contain only finite values")
        return location.copy()

    def fit(self, X: Any, y: Any | None = None) -> "DensityPowerRobustPCA":
        """Fit the density-power-divergence low-rank model."""
        del y
        X = _as_2d_finite_array(X)
        self._validate_parameters(*X.shape)
        q = int(self.n_components)
        location = self._estimate_location(X)
        centered = X - location

        initial = (
            _winsorized_matrix(centered, float(self.winsorize))
            if self.init == "winsorized_svd"
            else centered
        )
        U0, singular0, Vt0 = np.linalg.svd(initial, full_matrices=False)
        scores = U0[:, :q] * singular0[:q]
        loadings = Vt0[:q].T

        residuals = centered - scores @ loadings.T
        scale_floor = float(self.min_scale) ** 2
        sigma2 = _mad_variance(residuals, scale_floor)
        objective_history = [_dpd_objective(residuals, float(self.alpha), sigma2)]
        converged = False

        for iteration in range(1, int(self.max_iter) + 1):
            previous_fit = scores @ loadings.T
            previous_sigma2 = sigma2
            for _ in range(int(self.inner_iter)):
                scores, loadings = _weighted_loading_update(
                    centered,
                    scores,
                    loadings,
                    float(self.alpha),
                    sigma2,
                    float(self.ridge),
                )
                scores = _weighted_score_update(
                    centered,
                    scores,
                    loadings,
                    float(self.alpha),
                    sigma2,
                    float(self.ridge),
                )

            fitted = scores @ loadings.T
            residuals = centered - fitted
            sigma2 = _dpd_scale_update(
                residuals,
                float(self.alpha),
                sigma2,
                scale_floor,
            )
            objective_history.append(_dpd_objective(residuals, float(self.alpha), sigma2))

            fit_change = np.linalg.norm(fitted - previous_fit) / max(
                np.linalg.norm(previous_fit), 1.0
            )
            scale_change = abs(sigma2 - previous_sigma2) / max(previous_sigma2, scale_floor)
            if max(fit_change, scale_change) <= float(self.tol):
                converged = True
                break

        fitted = scores @ loadings.T
        U, singular_values, Vt = np.linalg.svd(fitted, full_matrices=False)
        U = U[:, :q]
        singular_values = singular_values[:q]
        Vt = Vt[:q]
        U, Vt = _deterministic_svd_signs(U, Vt)
        canonical_scores = U * singular_values
        fitted = canonical_scores @ Vt
        residuals = centered - fitted
        weights = _dpd_weights(residuals, float(self.alpha), sigma2)

        eigenvalues = singular_values * singular_values / X.shape[0]
        model_total = float(np.sum(eigenvalues) + X.shape[1] * sigma2)
        retained_total = max(float(np.sum(eigenvalues)), np.finfo(float).tiny)

        self.location_ = location
        self.mean_ = location
        self.components_ = Vt
        self.singular_values_ = singular_values
        self.eigenvalues_ = eigenvalues
        self.explained_variance_ = eigenvalues.copy()
        self.explained_variance_ratio_ = eigenvalues / max(model_total, np.finfo(float).tiny)
        self.retained_variance_ratio_ = eigenvalues / retained_total
        self.noise_variance_ = float(sigma2)
        self.residual_scale_ = float(np.sqrt(sigma2))
        self.n_components_ = q
        self.n_samples_in_ = X.shape[0]
        self.n_features_in_ = X.shape[1]
        self.n_iter_ = iteration
        self.converged_ = bool(converged)
        self.objective_history_ = np.asarray(objective_history, dtype=np.float64)
        self.weights_ = weights
        self.cell_outlier_scores_ = 1.0 - weights
        self.row_outlier_scores_ = np.mean(1.0 - weights, axis=1)
        self.fitted_values_ = location + fitted
        self.residuals_ = residuals
        self.estimator_ = self

        if self.store_scores:
            self.scores_ = canonical_scores / np.sqrt(eigenvalues) if self.whiten else canonical_scores
            raw_score_distances = np.sqrt(
                np.sum((canonical_scores * canonical_scores) / np.maximum(eigenvalues, scale_floor), axis=1)
            )
            self.score_distances_ = raw_score_distances
            self.orthogonal_distances_ = np.linalg.norm(residuals, axis=1)
        else:
            for attribute in ("scores_", "score_distances_", "orthogonal_distances_"):
                if hasattr(self, attribute):
                    delattr(self, attribute)
        return self

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "components_"):
            raise AttributeError("DensityPowerRobustPCA is not fitted yet")

    def _check_X(self, X: Any, *, name: str = "X") -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64, order="C")
        if X.ndim != 2:
            raise ValueError(f"{name} must be a 2D array")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"{name} has {X.shape[1]} features, expected {self.n_features_in_}"
            )
        if not np.all(np.isfinite(X)):
            raise ValueError(f"{name} must contain only finite values")
        return X

    def _raw_scores(self, X: Any) -> np.ndarray:
        X = self._check_X(X)
        centered = X - self.location_
        scores = centered @ self.components_.T
        # A few DPD fixed-point updates make scoring resistant to isolated bad
        # cells in a new row while reducing to the ordinary projection at alpha=0.
        for _ in range(int(self.inner_iter)):
            scores = _weighted_score_update(
                centered,
                scores,
                self.components_.T,
                float(self.alpha),
                self.noise_variance_,
                float(self.ridge),
            )
        return scores

    def transform(self, X: Any) -> np.ndarray:
        """Return robust component scores for new observations."""
        scores = self._raw_scores(X)
        if self.whiten:
            scores = scores / np.sqrt(np.maximum(self.eigenvalues_, self.min_scale**2))
        return scores

    def fit_transform(self, X: Any, y: Any | None = None) -> np.ndarray:
        """Fit the model and return training scores."""
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, scores: Any) -> np.ndarray:
        """Reconstruct observations from component scores."""
        self._check_is_fitted()
        scores = np.asarray(scores, dtype=np.float64)
        if scores.ndim != 2 or scores.shape[1] != self.n_components_:
            raise ValueError("scores must have shape (n_samples, n_components)")
        if not np.all(np.isfinite(scores)):
            raise ValueError("scores must contain only finite values")
        raw = scores
        if self.whiten:
            raw = raw * np.sqrt(np.maximum(self.eigenvalues_, self.min_scale**2))
        return self.location_ + raw @ self.components_

    def reconstruct(self, X: Any) -> np.ndarray:
        """Project and reconstruct observations."""
        return self.inverse_transform(self.transform(X))

    def reconstruction_error(self, X: Any) -> np.ndarray:
        """Return squared reconstruction error for each observation."""
        X = self._check_X(X)
        residuals = X - self.reconstruct(X)
        return np.sum(residuals * residuals, axis=1)

    def cell_weights(self, X: Any) -> np.ndarray:
        """Return final Gaussian DPD weights for each observation-feature cell."""
        X = self._check_X(X)
        residuals = X - self.reconstruct(X)
        return _dpd_weights(residuals, float(self.alpha), self.noise_variance_)

    def score_distances(self, X: Any) -> np.ndarray:
        """Distance within the retained robust principal subspace."""
        scores = self._raw_scores(X)
        return np.sqrt(
            np.sum(
                scores * scores / np.maximum(self.eigenvalues_, self.min_scale**2),
                axis=1,
            )
        )

    def orthogonal_distances(self, X: Any) -> np.ndarray:
        """Euclidean distance from the fitted low-rank subspace."""
        X = self._check_X(X)
        return np.linalg.norm(X - self.reconstruct(X), axis=1)

    def outlier_map(self, X: Any) -> np.ndarray:
        """Return score and orthogonal distances as two columns."""
        return np.column_stack([self.score_distances(X), self.orthogonal_distances(X)])


DPDRobustPCA = DensityPowerRobustPCA
