# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Casewise- and cellwise-robust multilinear PCA for matrix samples.

The estimator preserves the two modes of each matrix-valued observation.  It
uses a Tucker-2 reconstruction with one orthonormal loading matrix per mode and
combines redescending cellwise and casewise weights in an IRLS/ALS fit.  The
implementation follows the robust multilinear PCA modeling strategy but uses a
package-specific robust HOSVD initialization and fixed MAD residual scales; it
does not claim numerical identity with the reference ROMPCA software.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._estimator import EstimatorMixin
from ._native import resolve_backend, weighted_tucker_scores_2d
from .cellpca import _wrapping_rho, _wrapping_weight

_EPS = np.finfo(np.float64).eps


def _as_tensor_sample(X: Any, *, min_samples: int = 1, name: str = "X") -> np.ndarray:
    X = np.asarray(X, dtype=np.float64, order="C")
    if X.ndim != 3:
        raise ValueError(f"{name} must have shape (n_samples, n_rows, n_columns)")
    if X.shape[0] < min_samples:
        raise ValueError(f"{name} must contain at least {min_samples} observation(s)")
    if X.shape[1] < 2 or X.shape[2] < 2:
        raise ValueError("matrix observations must have at least two rows and columns")
    if np.isinf(X).any():
        raise ValueError(f"{name} contains infinity")
    if np.any(np.all(~np.isfinite(X), axis=(1, 2))):
        raise ValueError("every matrix observation must contain at least one finite cell")
    return X


def _robust_tensor_center_scale(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(X, axis=0)
    if not np.isfinite(center).all():
        bad = np.argwhere(~np.isfinite(center))
        preview = ", ".join(f"({i},{j})" for i, j in bad[:8])
        raise ValueError(f"every matrix cell needs at least one finite value; failed: {preview}")
    mad = 1.482602218505602 * np.nanmedian(np.abs(X - center), axis=0)
    fallback = np.nanstd(X, axis=0, ddof=1)
    valid = mad[np.isfinite(mad) & (mad > np.finfo(np.float64).tiny)]
    if not valid.size:
        fallback_valid = fallback[
            np.isfinite(fallback) & (fallback > np.finfo(np.float64).tiny)
        ]
        valid = fallback_valid
    if not valid.size:
        raise ValueError("matrix sample must contain nonzero variation")
    reference = float(np.median(valid))
    floor = max(np.sqrt(_EPS) * reference, np.finfo(np.float64).tiny)
    scale = np.where(np.isfinite(mad) & (mad > floor), mad, fallback)
    scale = np.where(np.isfinite(scale) & (scale > floor), scale, floor)
    return center, scale


def _leading_eigenvectors(covariance: np.ndarray, rank: int) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    order = np.argsort(values)[::-1][:rank]
    components = vectors[:, order]
    for j in range(components.shape[1]):
        k = int(np.argmax(np.abs(components[:, j])))
        if components[k, j] < 0:
            components[:, j] *= -1.0
    return components


def _reconstruct(center, row_components, cores, column_components):
    return center + np.einsum(
        "au,nuv,bv->nab", row_components, cores, column_components, optimize=True
    )


def _update_center(X, weights, low_rank, previous):
    numerator = np.sum(weights * (X - low_rank), axis=0)
    denominator = np.sum(weights, axis=0)
    return np.divide(
        numerator,
        denominator,
        out=previous.copy(),
        where=denominator > np.sqrt(_EPS),
    )


def _update_row_components(X, weights, center, cores, column_components, previous, ridge):
    n, n_rows, n_columns = X.shape
    rank_rows = cores.shape[1]
    identity = np.eye(rank_rows)
    output = previous.copy()
    centered = X - center
    predictors = np.einsum("nuv,bv->nbu", cores, column_components, optimize=True)
    for a in range(n_rows):
        w = weights[:, a, :].reshape(-1)
        Z = predictors.reshape(n * n_columns, rank_rows)
        y = centered[:, a, :].reshape(-1)
        if not np.any(w > 0.0):
            continue
        gram = Z.T @ (w[:, None] * Z)
        gram_scale = max(
            float(np.trace(gram)) / rank_rows, np.finfo(np.float64).tiny
        )
        gram = gram + max(ridge * gram_scale, np.finfo(np.float64).tiny) * identity
        rhs = Z.T @ (w * y)
        output[a] = np.linalg.solve(gram, rhs)
    qmat, rmat = np.linalg.qr(output, mode="reduced")
    cores[:] = np.einsum("uv,nvw->nuw", rmat, cores, optimize=True)
    return qmat


def _update_column_components(X, weights, center, cores, row_components, previous, ridge):
    n, n_rows, n_columns = X.shape
    rank_columns = cores.shape[2]
    identity = np.eye(rank_columns)
    output = previous.copy()
    centered = X - center
    predictors = np.einsum("nuv,au->nav", cores, row_components, optimize=True)
    for b in range(n_columns):
        w = weights[:, :, b].reshape(-1)
        Z = predictors.reshape(n * n_rows, rank_columns)
        y = centered[:, :, b].reshape(-1)
        if not np.any(w > 0.0):
            continue
        gram = Z.T @ (w[:, None] * Z)
        gram_scale = max(
            float(np.trace(gram)) / rank_columns, np.finfo(np.float64).tiny
        )
        gram = gram + max(ridge * gram_scale, np.finfo(np.float64).tiny) * identity
        rhs = Z.T @ (w * y)
        output[b] = np.linalg.solve(gram, rhs)
    qmat, rmat = np.linalg.qr(output, mode="reduced")
    cores[:] = np.einsum("nuv,wv->nuw", cores, rmat, optimize=True)
    return qmat


def _case_deviations(z: np.ndarray, observed: np.ndarray, b: float, c: float) -> np.ndarray:
    losses = np.where(observed, 2.0 * _wrapping_rho(z, b, c), 0.0)
    counts = observed.sum(axis=(1, 2))
    return np.sqrt(
        np.divide(
            losses.sum(axis=(1, 2)),
            counts,
            out=np.zeros(z.shape[0], dtype=np.float64),
            where=counts > 0,
        )
    )


@dataclass
class RobustMultilinearPCA(EstimatorMixin):
    """Robust Tucker-2 dimensionality reduction for matrix-valued samples.

    Parameters
    ----------
    ranks : tuple of int, default=(2, 2)
        Retained row-mode and column-mode dimensions.
    max_iter : int, default=60
        Maximum IRLS/alternating-least-squares iterations.
    tol : float, default=1e-5
        Relative fitted-value convergence tolerance.
    cell_b, cell_c : float, default=1.5, 4.0
        Inner and outer cutoffs for cellwise redescending weights.
    case_b, case_c : float, default=1.5, 4.0
        Inner and outer cutoffs for casewise redescending weights.
    ridge : float, default=1e-8
        Numerical ridge for weighted normal equations. Loading updates scale
        the ridge relative to the weighted Gram matrix.
    weight_threshold : float, default=0.5
        Threshold used to report outlying cells and cases.
    backend : {"auto", "python", "cpp"}, default="auto"
        Backend for repeated weighted core-score solves. The native backend is
        optional and returns numerically equivalent results.
    store_scores : bool, default=True
        Store training core scores and full diagnostics.

    Notes
    -----
    This class implements the casewise/cellwise robust MPCA structure using a
    package-native robust HOSVD initialization and fixed MAD residual scales.
    It is not a reference-parity implementation of the published ROMPCA
    initialization, recentering, or automatic rank-selection procedures.
    """

    ranks: tuple[int, int] = (2, 2)
    max_iter: int = 60
    tol: float = 1e-5
    cell_b: float = 1.5
    cell_c: float = 4.0
    case_b: float = 1.5
    case_c: float = 4.0
    ridge: float = 1e-8
    weight_threshold: float = 0.5
    backend: str = "auto"
    store_scores: bool = True

    def _validate_parameters(self, shape: tuple[int, int, int]) -> None:
        n, r, c = shape
        if not isinstance(self.ranks, tuple) or len(self.ranks) != 2:
            raise TypeError("ranks must be a tuple (row_rank, column_rank)")
        q1, q2 = self.ranks
        if any(isinstance(q, (bool, np.bool_)) or not isinstance(q, (int, np.integer)) for q in self.ranks):
            raise TypeError("both ranks must be integers")
        if not (1 <= q1 < r and 1 <= q2 < c):
            raise ValueError("ranks must satisfy 1 <= row_rank < n_rows and 1 <= column_rank < n_columns")
        if n < 2:
            raise ValueError("at least two matrix observations are required")
        if int(self.max_iter) < 1:
            raise ValueError("max_iter must be at least one")
        if not np.isfinite(self.tol) or self.tol <= 0:
            raise ValueError("tol must be positive and finite")
        for prefix, b, cc in (("cell", self.cell_b, self.cell_c), ("case", self.case_b, self.case_c)):
            if not np.isfinite(b) or not np.isfinite(cc) or b <= 0 or cc <= b:
                raise ValueError(f"{prefix}_b and {prefix}_c must satisfy 0 < b < c")
        if not np.isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError("ridge must be positive and finite")
        if not 0 < self.weight_threshold < 1:
            raise ValueError("weight_threshold must be in (0, 1)")
        if not isinstance(self.store_scores, (bool, np.bool_)):
            raise TypeError("store_scores must be a boolean")

    def _initialize(self, X, observed):
        center, marginal_scale = _robust_tensor_center_scale(X)
        safe = np.where(observed, X, center)
        clipped = center + np.clip((safe - center) / marginal_scale, -4.0, 4.0) * marginal_scale
        centered = clipped - center
        n, r, c = X.shape
        row_cov = np.einsum("nac,nbc->ab", centered, centered, optimize=True) / max(n * c, 1)
        col_cov = np.einsum("nra,nrb->ab", centered, centered, optimize=True) / max(n * r, 1)
        U = _leading_eigenvectors(row_cov, self.ranks[0])
        V = _leading_eigenvectors(col_cov, self.ranks[1])
        weights = observed.astype(np.float64)
        cores = weighted_tucker_scores_2d(
            safe, weights, center, U, V, ridge=self.ridge, backend=self.backend_
        )
        fitted = _reconstruct(center, U, cores, V)
        residual = np.where(observed, X - fitted, np.nan)
        scales = 1.482602218505602 * np.nanmedian(
            np.abs(residual - np.nanmedian(residual, axis=0)), axis=0
        )
        fallback = np.nanstd(residual, axis=0, ddof=1)
        floor = max(
            np.sqrt(_EPS) * float(np.nanmedian(marginal_scale)),
            np.finfo(np.float64).tiny,
        )
        scales = np.where(np.isfinite(scales) & (scales > floor), scales, fallback)
        scales = np.where(np.isfinite(scales) & (scales > floor), scales, marginal_scale)
        z = np.where(observed, residual / scales, 0.0)
        deviations = _case_deviations(z, observed, self.cell_b, self.cell_c)
        positive = deviations[deviations > np.sqrt(_EPS)]
        case_scale = float(np.median(positive) / 0.6744897501960817) if positive.size else 1.0
        case_scale = max(case_scale, np.sqrt(_EPS))
        return center, U, V, cores, scales, case_scale

    def fit(self, X: Any, y=None):
        X = _as_tensor_sample(X, min_samples=2)
        self._validate_parameters(X.shape)
        self.backend_ = resolve_backend(self.backend)
        observed = np.isfinite(X)
        safe_for_solve = np.where(observed, X, 0.0)
        center, U, V, cores, residual_scales, case_scale = self._initialize(X, observed)
        previous_fitted = _reconstruct(center, U, cores, V)
        objective_history = []
        converged = False

        for iteration in range(1, int(self.max_iter) + 1):
            residual = np.where(observed, X - previous_fitted, 0.0)
            z = residual / residual_scales
            cell_weights = np.where(
                observed, _wrapping_weight(z, self.cell_b, self.cell_c), 0.0
            )
            deviations = _case_deviations(z, observed, self.cell_b, self.cell_c)
            case_weights = _wrapping_weight(
                deviations / case_scale, self.case_b, self.case_c
            )
            weights = cell_weights * case_weights[:, None, None]

            safe = np.where(observed, X, center)
            cores = weighted_tucker_scores_2d(
                safe, weights, center, U, V, ridge=self.ridge, backend=self.backend_
            )
            low_rank = np.einsum("au,nuv,bv->nab", U, cores, V, optimize=True)
            center = _update_center(safe, weights, low_rank, center)
            U = _update_row_components(safe, weights, center, cores, V, U, self.ridge)
            V = _update_column_components(safe, weights, center, cores, U, V, self.ridge)
            cores = weighted_tucker_scores_2d(
                safe, weights, center, U, V, ridge=self.ridge, backend=self.backend_
            )
            fitted = _reconstruct(center, U, cores, V)
            relative = np.linalg.norm(fitted - previous_fitted) / max(
                np.linalg.norm(previous_fitted),
                np.linalg.norm(fitted),
                np.finfo(np.float64).tiny,
            )
            robust_loss = np.where(
                observed,
                _wrapping_rho((X - fitted) / residual_scales, self.cell_b, self.cell_c),
                0.0,
            )
            objective_history.append(float(np.sum(case_weights[:, None, None] * robust_loss)))
            previous_fitted = fitted
            if relative <= self.tol:
                converged = True
                break

        # Canonical signs make deterministic comparisons easier.
        for u in range(U.shape[1]):
            idx = int(np.argmax(np.abs(U[:, u])))
            if U[idx, u] < 0:
                U[:, u] *= -1
                cores[:, u, :] *= -1
        for v in range(V.shape[1]):
            idx = int(np.argmax(np.abs(V[:, v])))
            if V[idx, v] < 0:
                V[:, v] *= -1
                cores[:, :, v] *= -1

        fitted = _reconstruct(center, U, cores, V)
        residual = np.where(observed, X - fitted, 0.0)
        standardized = residual / residual_scales
        cell_weights = np.where(
            observed, _wrapping_weight(standardized, self.cell_b, self.cell_c), 0.0
        )
        deviations = _case_deviations(standardized, observed, self.cell_b, self.cell_c)
        case_weights = _wrapping_weight(deviations / case_scale, self.case_b, self.case_c)
        missing = ~observed
        cell_outliers = observed & (cell_weights < self.weight_threshold)
        case_outliers = case_weights < self.weight_threshold
        imputed = np.where(missing, fitted, X)
        corrected = np.where(missing | cell_outliers, fitted, X)

        self.center_ = center
        self.location_ = center
        self.row_components_ = U
        self.column_components_ = V
        self.ranks_ = (U.shape[1], V.shape[1])
        self.matrix_shape_in_ = X.shape[1:]
        self.n_samples_in_ = X.shape[0]
        self.n_features_in_ = X.shape[1] * X.shape[2]
        self.residual_scales_ = residual_scales
        self.case_scale_ = case_scale
        self.cell_weights_ = cell_weights
        self.case_weights_ = case_weights
        self.cell_outlier_mask_ = cell_outliers
        self.case_outlier_mask_ = case_outliers
        self.missing_mask_ = missing
        self.standardized_residuals_ = standardized
        self.case_deviations_ = deviations
        self.max_cell_residuals_ = np.max(np.abs(standardized), axis=(1, 2))
        self.fitted_values_ = fitted
        self.imputed_data_ = imputed
        self.corrected_data_ = corrected
        self.objective_history_ = np.asarray(objective_history, dtype=np.float64)
        self.n_iter_ = iteration
        self.converged_ = converged
        core_variance = np.mean(cores**2, axis=0)
        total = float(np.mean(np.sum((imputed - center) ** 2, axis=(1, 2))))
        self.explained_variance_ = core_variance
        self.explained_variance_ratio_ = core_variance / max(
            total, np.finfo(np.float64).tiny
        )
        self.total_explained_variance_ratio_ = float(np.sum(self.explained_variance_ratio_))
        if self.store_scores:
            self.core_scores_ = cores
        return self

    def _check_is_fitted(self):
        if not hasattr(self, "row_components_"):
            raise AttributeError("RobustMultilinearPCA is not fitted")

    def _predict_diagnostics(self, X):
        self._check_is_fitted()
        X = _as_tensor_sample(X)
        if X.shape[1:] != self.matrix_shape_in_:
            raise ValueError(
                f"matrix observations must have shape {self.matrix_shape_in_}, got {X.shape[1:]}"
            )
        observed = np.isfinite(X)
        safe = np.where(observed, X, self.center_)
        weights = observed.astype(np.float64)
        cores = None
        fitted = None
        for _ in range(3):
            cores = weighted_tucker_scores_2d(
                safe,
                weights,
                self.center_,
                self.row_components_,
                self.column_components_,
                ridge=self.ridge,
                backend=self.backend_,
            )
            fitted = _reconstruct(
                self.center_, self.row_components_, cores, self.column_components_
            )
            z = np.where(observed, (X - fitted) / self.residual_scales_, 0.0)
            cell_weights = np.where(
                observed, _wrapping_weight(z, self.cell_b, self.cell_c), 0.0
            )
            deviations = _case_deviations(z, observed, self.cell_b, self.cell_c)
            case_weights = _wrapping_weight(
                deviations / self.case_scale_, self.case_b, self.case_c
            )
            weights = cell_weights * case_weights[:, None, None]
        missing = ~observed
        cell_outliers = observed & (cell_weights < self.weight_threshold)
        return {
            "core_scores": cores,
            "fitted_values": fitted,
            "standardized_residuals": z,
            "cell_weights": cell_weights,
            "case_weights": case_weights,
            "cell_outlier_mask": cell_outliers,
            "case_outlier_mask": case_weights < self.weight_threshold,
            "case_deviations": deviations,
            "max_cell_residuals": np.max(np.abs(z), axis=(1, 2)),
            "imputed_data": np.where(missing, fitted, X),
            "corrected_data": np.where(missing | cell_outliers, fitted, X),
            "missing_mask": missing,
        }

    def transform(self, X):
        """Return core tensors with shape ``(n_samples, row_rank, column_rank)``."""
        return self._predict_diagnostics(X)["core_scores"]

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        if hasattr(self, "core_scores_"):
            return self.core_scores_.copy()
        return self.transform(X)

    def inverse_transform(self, core_scores):
        self._check_is_fitted()
        core_scores = np.asarray(core_scores, dtype=np.float64)
        if core_scores.ndim != 3 or core_scores.shape[1:] != self.ranks_:
            raise ValueError(f"core_scores must have shape (n_samples, {self.ranks_[0]}, {self.ranks_[1]})")
        if not np.isfinite(core_scores).all():
            raise ValueError("core_scores must be finite")
        return _reconstruct(
            self.center_, self.row_components_, core_scores, self.column_components_
        )

    def reconstruct(self, X):
        return self._predict_diagnostics(X)["fitted_values"]

    def impute(self, X):
        return self._predict_diagnostics(X)["imputed_data"]

    def correct(self, X):
        return self._predict_diagnostics(X)["corrected_data"]

    def cellwise_diagnostics(self, X):
        return self._predict_diagnostics(X)

    def outlier_map(self, X=None):
        self._check_is_fitted()
        if X is None:
            return np.column_stack([self.case_deviations_, self.max_cell_residuals_])
        result = self._predict_diagnostics(X)
        return np.column_stack([result["case_deviations"], result["max_cell_residuals"]])


CasewiseCellwiseMultilinearPCA = RobustMultilinearPCA
CellwiseRobustMultilinearPCA = RobustMultilinearPCA

__all__ = [
    "RobustMultilinearPCA",
    "CasewiseCellwiseMultilinearPCA",
    "CellwiseRobustMultilinearPCA",
]
