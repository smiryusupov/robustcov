# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Sparse cellwise- and casewise-robust principal component analysis.

This module adds exact-zero elastic-net loadings to the package-native
:class:`~robustcov.cellpca.CellwiseRobustPCA` weighting model.  It is inspired
by SCRAMBLE's robust reconstruction objective with sparse loadings, but it uses
alternating weighted elastic-net regressions rather than Riemannian stochastic
gradient descent.  The public documentation therefore describes the estimator
as ``SparseCellPCA`` and does not claim numerical parity with SCRAMBLE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .cellpca import (
    CellwiseRobustPCA,
    _EPS,
    _as_matrix,
    _case_deviation,
    _deterministic_component_signs,
    _weighted_center,
    _weighted_scores,
    _wrapping_rho,
    _wrapping_weight,
)


def _soft_threshold(value: float, threshold: float) -> float:
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


def _weighted_elastic_net_loadings(
    X: np.ndarray,
    observed: np.ndarray,
    weights: np.ndarray,
    center: np.ndarray,
    scores: np.ndarray,
    previous: np.ndarray,
    alpha: np.ndarray,
    l1_ratio: float,
    ridge: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """Solve one weighted elastic-net regression per feature.

    The penalty is component specific: ``alpha[k]`` controls the loading of
    component ``k`` in every feature regression.  Coordinate descent gives
    exact zeros while keeping the weighted low-rank update inexpensive when the
    retained rank is small.
    """

    _, p = X.shape
    q = scores.shape[1]
    safe = np.where(observed, X, center)
    centered = safe - center
    loadings = previous.copy()
    l1 = alpha * l1_ratio
    l2 = alpha * (1.0 - l1_ratio)

    for j in range(p):
        w = np.asarray(weights[:, j], dtype=float)
        active = w > 0.0
        if int(np.count_nonzero(active)) < 2:
            continue

        T = scores[active]
        y = centered[active, j]
        w_active = w[active]
        weight_sum = max(float(np.sum(w_active)), np.sqrt(_EPS))
        beta = loadings[j].copy()
        residual = y - T @ beta

        for _ in range(max_iter):
            old = beta.copy()
            for k in range(q):
                column = T[:, k]
                residual += column * beta[k]
                curvature = float(np.dot(w_active, column * column) / weight_sum)
                curvature += float(l2[k]) + ridge
                correlation = float(np.dot(w_active, column * residual) / weight_sum)
                beta[k] = _soft_threshold(correlation, float(l1[k])) / max(
                    curvature, np.sqrt(_EPS)
                )
                residual -= column * beta[k]
            if np.linalg.norm(beta - old) <= tol * max(np.linalg.norm(old), 1.0):
                break
        loadings[j] = beta
    return loadings


def _normalize_loading_columns(
    loadings: np.ndarray,
    scores: np.ndarray,
    fallback: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize loading columns while preserving fitted values."""

    loadings = np.asarray(loadings, dtype=float).copy()
    scores = np.asarray(scores, dtype=float).copy()
    collapsed = np.zeros(loadings.shape[1], dtype=bool)
    for k in range(loadings.shape[1]):
        norm = float(np.linalg.norm(loadings[:, k]))
        if norm <= np.sqrt(_EPS):
            replacement = np.asarray(fallback[:, k], dtype=float)
            replacement_norm = float(np.linalg.norm(replacement))
            if replacement_norm <= np.sqrt(_EPS):
                replacement = np.zeros(loadings.shape[0], dtype=float)
                replacement[k % loadings.shape[0]] = 1.0
                replacement_norm = 1.0
            loadings[:, k] = replacement / replacement_norm
            scores[:, k] = 0.0
            collapsed[k] = True
        else:
            loadings[:, k] /= norm
            scores[:, k] *= norm
    return loadings, scores, collapsed


def _component_alpha(alpha: Any, n_components: int) -> np.ndarray:
    values = np.asarray(alpha, dtype=float)
    if values.ndim == 0:
        values = np.repeat(float(values), n_components)
    elif values.ndim == 1 and values.size == n_components:
        values = values.copy()
    else:
        raise ValueError("alpha must be a scalar or have one value per component")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("alpha values must be finite and non-negative")
    return values


@dataclass
class SparseCellwiseRobustPCA(CellwiseRobustPCA):
    """Cellwise-robust PCA with exact-zero elastic-net loadings.

    Parameters
    ----------
    alpha : float or array-like, default=0.05
        Elastic-net strength.  A scalar applies the same penalty to every
        component.  A one-dimensional array may specify one value per
        component.
    l1_ratio : float, default=1.0
        Elastic-net mixing parameter.  ``1`` gives a lasso loading penalty;
        ``0`` gives ridge shrinkage without exact zeros.
    loading_max_iter : int, default=100
        Maximum coordinate-descent sweeps for each feature regression and
        outer sparse-loading update.
    loading_tol : float, default=1e-7
        Coordinate-descent convergence tolerance.
    sparsity_threshold : float, default=1e-8
        Final absolute loading threshold.  Coefficients below this value are
        set exactly to zero before the final score fit.

    Notes
    -----
    The estimator minimizes a robust weighted reconstruction criterion with an
    elastic-net penalty on the loading matrix.  Robust cell and case weights
    are inherited from :class:`CellwiseRobustPCA`.  Loading updates are solved
    by alternating weighted coordinate descent.  Sparse loading columns are
    normalized but are not constrained to be mutually orthogonal; scores are
    obtained by weighted least squares.

    This differs computationally from SCRAMBLE, which optimizes a smoothed
    robust sparse objective on the Stiefel manifold using Riemannian stochastic
    gradient descent.  The shared modeling idea is cellwise robust
    reconstruction with sparse loadings, not reference-software equivalence.
    """

    alpha: Any = 0.05
    l1_ratio: float = 1.0
    loading_max_iter: int = 100
    loading_tol: float = 1e-7
    sparsity_threshold: float = 1e-8

    def _validate_parameters(self, n_samples: int, n_features: int) -> None:
        super()._validate_parameters(n_samples, n_features)
        _component_alpha(self.alpha, int(self.n_components))
        if not np.isfinite(self.l1_ratio) or not 0.0 <= float(self.l1_ratio) <= 1.0:
            raise ValueError("l1_ratio must be in [0, 1]")
        if int(self.loading_max_iter) < 1:
            raise ValueError("loading_max_iter must be at least 1")
        if not np.isfinite(self.loading_tol) or float(self.loading_tol) <= 0.0:
            raise ValueError("loading_tol must be positive and finite")
        if not np.isfinite(self.sparsity_threshold) or float(self.sparsity_threshold) < 0.0:
            raise ValueError("sparsity_threshold must be finite and non-negative")

    def fit(self, X: Any, y: Any | None = None) -> "SparseCellwiseRobustPCA":
        """Fit sparse robust loadings to incomplete or contaminated data."""

        del y
        X = _as_matrix(X, min_rows=3)
        n, p = X.shape
        self._validate_parameters(n, p)
        observed = np.isfinite(X)
        if np.any(observed.sum(axis=0) < 2):
            bad = np.flatnonzero(observed.sum(axis=0) < 2)
            raise ValueError(
                "each feature must contain at least two finite values; failed columns: "
                + ", ".join(map(str, bad.tolist()))
            )
        if np.any(observed.sum(axis=1) == 0):
            bad = np.flatnonzero(observed.sum(axis=1) == 0)
            raise ValueError(
                "every row must contain at least one finite value; failed rows: "
                + ", ".join(map(str, bad.tolist()))
            )

        # Dense robust initialization provides residual scales and a stable
        # starting subspace.  Sparse updates then optimize the package-specific
        # penalized objective.
        initial = CellwiseRobustPCA(
            n_components=int(self.n_components),
            max_iter=int(self.max_iter),
            tol=float(self.tol),
            cell_b=float(self.cell_b),
            cell_c=float(self.cell_c),
            case_b=float(self.case_b),
            case_c=float(self.case_c),
            ridge=float(self.ridge),
            weight_threshold=float(self.weight_threshold),
            store_scores=True,
        ).fit(X)

        center = initial.center_.copy()
        loadings = initial.loadings_.copy()
        scores = initial.scores_.copy()
        residual_scales = initial.residual_scales_.copy()
        case_scale = float(initial.case_scale_)
        alpha = _component_alpha(self.alpha, int(self.n_components))
        fitted = center + scores @ loadings.T
        previous_fitted = fitted.copy()
        objective_history: list[float] = []
        reconstruction_history: list[float] = []
        penalty_history: list[float] = []
        collapsed_any = np.zeros(int(self.n_components), dtype=bool)
        converged = False

        for iteration in range(1, int(self.max_iter) + 1):
            residuals_safe = np.where(observed, X - fitted, 0.0)
            standardized = residuals_safe / residual_scales
            cell_weights = np.where(
                observed,
                _wrapping_weight(standardized, float(self.cell_b), float(self.cell_c)),
                0.0,
            )
            case_deviation = _case_deviation(
                standardized,
                observed,
                residual_scales,
                float(self.cell_b),
                float(self.cell_c),
            )
            case_weights = _wrapping_weight(
                case_deviation / case_scale,
                float(self.case_b),
                float(self.case_c),
            )
            weights = cell_weights * case_weights[:, None]
            weights = np.where(observed, weights, 0.0)

            scores = _weighted_scores(
                X, observed, weights, center, loadings, float(self.ridge)
            )
            center = _weighted_center(X, observed, weights, scores, loadings, center)
            scores = _weighted_scores(
                X, observed, weights, center, loadings, float(self.ridge)
            )
            sparse = _weighted_elastic_net_loadings(
                X,
                observed,
                weights,
                center,
                scores,
                loadings,
                alpha,
                float(self.l1_ratio),
                float(self.ridge),
                int(self.loading_max_iter),
                float(self.loading_tol),
            )
            loadings, scores, collapsed = _normalize_loading_columns(
                sparse, scores, loadings
            )
            collapsed_any |= collapsed
            scores = _weighted_scores(
                X, observed, weights, center, loadings, float(self.ridge)
            )
            fitted = center + scores @ loadings.T

            residuals_safe = np.where(observed, X - fitted, 0.0)
            standardized = residuals_safe / residual_scales
            robust_loss = float(
                np.sum(
                    np.where(
                        observed,
                        _wrapping_rho(
                            standardized,
                            float(self.cell_b),
                            float(self.cell_c),
                        ),
                        0.0,
                    )
                )
            )
            l1_penalty = float(np.sum(alpha[None, :] * np.abs(loadings)))
            l2_penalty = float(np.sum(alpha[None, :] * loadings * loadings) / 2.0)
            penalty = float(self.l1_ratio) * l1_penalty + (
                1.0 - float(self.l1_ratio)
            ) * l2_penalty
            objective_history.append(robust_loss + penalty)
            reconstruction_history.append(robust_loss)
            penalty_history.append(penalty)

            denominator = max(float(np.linalg.norm(previous_fitted)), 1.0)
            relative_change = float(np.linalg.norm(fitted - previous_fitted) / denominator)
            if relative_change <= float(self.tol):
                converged = True
                break
            previous_fitted = fitted.copy()

        # Enforce the documented final threshold, then refit scores without
        # changing the selected loading support.
        loadings[np.abs(loadings) < float(self.sparsity_threshold)] = 0.0
        loadings, scores, collapsed = _normalize_loading_columns(
            loadings, scores, initial.loadings_
        )
        collapsed_any |= collapsed

        # Recompute robust weights and scores for the final sparse basis.
        for _ in range(30):
            fitted = center + scores @ loadings.T
            standardized = np.where(observed, (X - fitted) / residual_scales, 0.0)
            cell_weights = np.where(
                observed,
                _wrapping_weight(standardized, float(self.cell_b), float(self.cell_c)),
                0.0,
            )
            case_deviation = _case_deviation(
                standardized,
                observed,
                residual_scales,
                float(self.cell_b),
                float(self.cell_c),
            )
            case_weights = _wrapping_weight(
                case_deviation / case_scale,
                float(self.case_b),
                float(self.case_c),
            )
            # Once the sparse subspace is fixed, estimate each row's scores
            # from its reliable cells.  Case weights were used to protect the
            # loading fit; applying them again here can leave a broad row
            # outlier with too little information for stable missing-cell
            # prediction.  This mirrors ``CellPCA.transform``.
            new_scores = _weighted_scores(
                X, observed, cell_weights, center, loadings, float(self.ridge)
            )
            if np.linalg.norm(new_scores - scores) <= 1e-9 * max(
                np.linalg.norm(scores), 1.0
            ):
                scores = new_scores
                break
            scores = new_scores

        weight_sum = max(float(np.sum(case_weights)), np.sqrt(_EPS))
        score_mean = np.sum(case_weights[:, None] * scores, axis=0) / weight_sum
        center = center + score_mean @ loadings.T
        scores = scores - score_mean
        score_variance = np.sum(case_weights[:, None] * scores * scores, axis=0) / weight_sum
        order = np.argsort(score_variance)[::-1]
        score_variance = np.maximum(score_variance[order], 0.0)
        scores = scores[:, order]
        loadings = loadings[:, order]
        alpha = alpha[order]
        collapsed_any = collapsed_any[order]

        components = _deterministic_component_signs(loadings.T)
        signs = np.sign(np.sum(components * loadings.T, axis=1))
        signs[signs == 0.0] = 1.0
        scores = scores * signs
        loadings = components.T
        fitted = center + scores @ components

        residuals = np.where(observed, X - fitted, np.nan)
        standardized_residuals = np.where(
            observed, residuals / residual_scales, np.nan
        )
        cell_weights = np.where(
            observed,
            _wrapping_weight(
                np.nan_to_num(standardized_residuals, nan=0.0),
                float(self.cell_b),
                float(self.cell_c),
            ),
            0.0,
        )
        case_deviation = _case_deviation(
            np.nan_to_num(standardized_residuals, nan=0.0),
            observed,
            residual_scales,
            float(self.cell_b),
            float(self.cell_c),
        )
        case_weights = _wrapping_weight(
            case_deviation / case_scale,
            float(self.case_b),
            float(self.case_c),
        )

        safe = np.where(observed, X, center)
        total_variance = 0.0
        for j in range(p):
            w = cell_weights[:, j] * case_weights
            denom = float(np.sum(w))
            if denom > np.sqrt(_EPS):
                total_variance += float(np.sum(w * (safe[:, j] - center[j]) ** 2) / denom)
        total_variance = max(total_variance, float(np.sum(score_variance)), np.sqrt(_EPS))

        cell_outliers = observed & (cell_weights < float(self.weight_threshold))
        case_outliers = case_weights < float(self.weight_threshold)
        corrected = np.array(X, copy=True)
        replace = (~observed) | cell_outliers
        corrected[replace] = fitted[replace]
        imputed = np.array(X, copy=True)
        imputed[~observed] = fitted[~observed]

        self.center_ = center
        self.location_ = center
        self.mean_ = center
        self.components_ = components
        self.loadings_ = loadings
        self.sparse_components_ = components.copy()
        self.n_components_ = int(self.n_components)
        self.n_samples_in_ = n
        self.n_features_in_ = p
        self.residual_scales_ = residual_scales
        self.case_scale_ = case_scale
        self.explained_variance_ = score_variance
        self.eigenvalues_ = score_variance.copy()
        self.explained_variance_ratio_ = score_variance / total_variance
        self.noise_variance_ = float(
            np.nansum(residuals * residuals)
            / max(int(observed.sum()) - n * self.n_components_, 1)
        )
        self.fitted_values_ = fitted
        self.residuals_ = residuals
        self.standardized_residuals_ = standardized_residuals
        self.cell_weights_ = cell_weights
        self.case_weights_ = case_weights
        self.cell_outlier_mask_ = cell_outliers
        self.case_outlier_mask_ = case_outliers
        self.missing_mask_ = ~observed
        self.case_deviations_ = case_deviation
        self.max_cell_residuals_ = np.nanmax(np.abs(standardized_residuals), axis=1)
        self.corrected_data_ = corrected
        self.imputed_data_ = imputed
        self.objective_history_ = np.asarray(objective_history, dtype=float)
        self.reconstruction_loss_history_ = np.asarray(
            reconstruction_history, dtype=float
        )
        self.penalty_history_ = np.asarray(penalty_history, dtype=float)
        self.n_iter_ = iteration
        self.converged_ = converged
        self.alpha_ = alpha
        self.loading_support_ = components != 0.0
        self.n_nonzero_loadings_ = np.count_nonzero(components, axis=1)
        self.component_sparsity_ = 1.0 - self.n_nonzero_loadings_ / p
        self.sparsity_ = float(1.0 - np.count_nonzero(components) / components.size)
        self.component_gram_ = components @ components.T
        self.collapsed_components_ = collapsed_any
        self.feature_importances_ = np.max(np.abs(components), axis=0)
        self.cell_cutoff_ = float(self.cell_c)
        self.case_cutoff_ = float(self.case_c)

        if self.store_scores:
            self.scores_ = scores
        elif hasattr(self, "scores_"):
            delattr(self, "scores_")
        return self


SparseCellPCA = SparseCellwiseRobustPCA
SparseCasewiseCellwisePCA = SparseCellwiseRobustPCA


__all__ = [
    "SparseCellwiseRobustPCA",
    "SparseCellPCA",
    "SparseCasewiseCellwisePCA",
]
