# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Cellwise- and casewise-robust principal component analysis.

The estimator in this module follows the central idea of cellPCA: a low-rank
subspace is fitted by iteratively reweighted least squares with one weight for
each observed cell and a second weight for each row.  Missing cells receive
zero weight.  The reference algorithm starts from MacroPCA and uses fixed
M-scales.  ``robustcov`` uses a deterministic robust marginal/SVD start and
fixed MAD-type residual scales, so numerical equality with the reference R
implementation is not claimed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import chi2

from ._estimator import EstimatorMixin


_EPS = np.finfo(np.float64).eps


def _as_matrix(X: Any, *, name: str = "X", min_rows: int = 1) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64, order="C")
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if X.shape[0] < min_rows:
        raise ValueError(f"{name} must contain at least {min_rows} row(s)")
    if X.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two features")
    if np.isinf(X).any():
        raise ValueError(f"{name} contains infinity")
    return X


def _robust_center_scale(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(X, axis=0)
    if not np.isfinite(center).all():
        bad = np.flatnonzero(~np.isfinite(center))
        raise ValueError(
            "every feature must contain at least one finite value; failed columns: "
            + ", ".join(map(str, bad.tolist()))
        )
    mad = 1.482602218505602 * np.nanmedian(np.abs(X - center), axis=0)
    fallback = np.nanstd(X, axis=0, ddof=1)
    valid = mad[np.isfinite(mad) & (mad > 0.0)]
    reference = float(np.median(valid)) if valid.size else 1.0
    floor = max(np.sqrt(_EPS) * max(reference, 1.0), np.finfo(float).tiny)
    scale = np.where(np.isfinite(mad) & (mad > floor), mad, fallback)
    scale = np.where(np.isfinite(scale) & (scale > floor), scale, floor)
    return center, scale


def _wrapping_constants(b: float, c: float) -> tuple[float, float]:
    # The default b=1.5, c=4 reproduces the usual wrapping constants
    # q1=1.540793 and q2=0.862273.  Scaling q2 by the transition width keeps
    # the same transition shape for custom cutoffs.
    q2 = 0.862273 * (2.5 / (c - b))
    q1 = b / np.tanh(q2 * (c - b))
    return float(q1), float(q2)


def _wrapping_psi(z: np.ndarray, b: float, c: float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    a = np.abs(z)
    q1, q2 = _wrapping_constants(b, c)
    out = np.zeros_like(z)
    central = a <= b
    transition = (a > b) & (a < c)
    out[central] = z[central]
    out[transition] = (
        np.sign(z[transition])
        * q1
        * np.tanh(q2 * (c - a[transition]))
    )
    return out


def _wrapping_weight(z: np.ndarray, b: float, c: float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    psi = _wrapping_psi(z, b, c)
    out = np.ones_like(z)
    nonzero = np.abs(z) > np.sqrt(_EPS)
    out[nonzero] = psi[nonzero] / z[nonzero]
    return np.clip(out, 0.0, 1.0)


def _wrapping_rho(z: np.ndarray, b: float, c: float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    a = np.abs(z)
    q1, q2 = _wrapping_constants(b, c)
    out = np.empty_like(a)
    central = a <= b
    transition = (a > b) & (a < c)
    far = a >= c
    out[central] = 0.5 * a[central] ** 2
    base = 0.5 * b * b
    logcosh_start = np.log(np.cosh(q2 * (c - b)))
    out[transition] = base + (q1 / q2) * (
        logcosh_start - np.log(np.cosh(q2 * (c - a[transition])))
    )
    out[far] = base + (q1 / q2) * logcosh_start
    return out


def _deterministic_component_signs(components: np.ndarray) -> np.ndarray:
    components = np.asarray(components, dtype=np.float64).copy()
    if components.size == 0:
        return components
    rows = np.arange(components.shape[0])
    columns = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[rows, columns])
    signs[signs == 0.0] = 1.0
    components *= signs[:, None]
    return components


def _weighted_scores(
    X: np.ndarray,
    observed: np.ndarray,
    weights: np.ndarray,
    center: np.ndarray,
    loadings: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """Solve all weighted score normal equations in one batched operation."""
    n = X.shape[0]
    q = loadings.shape[1]
    safe = np.where(observed, X, center)
    centered = safe - center
    active = np.any(weights > 0.0, axis=1)
    scores = np.zeros((n, q), dtype=np.float64)
    if not np.any(active):
        return scores

    active_weights = weights[active]
    grams = np.einsum(
        "pq,np,pr->nqr",
        loadings,
        active_weights,
        loadings,
        optimize=True,
    )
    grams += ridge * np.eye(q)[None, :, :]
    rhs = np.einsum(
        "pq,np,np->nq",
        loadings,
        active_weights,
        centered[active],
        optimize=True,
    )
    scores[active] = np.linalg.solve(grams, rhs[..., None])[..., 0]
    return scores


def _weighted_center(
    X: np.ndarray,
    observed: np.ndarray,
    weights: np.ndarray,
    scores: np.ndarray,
    loadings: np.ndarray,
    previous: np.ndarray,
) -> np.ndarray:
    fitted_without_center = scores @ loadings.T
    safe = np.where(observed, X, 0.0)
    numerator = np.sum(weights * (safe - fitted_without_center), axis=0)
    denominator = np.sum(weights, axis=0)
    return np.divide(
        numerator,
        denominator,
        out=previous.copy(),
        where=denominator > np.sqrt(_EPS),
    )


def _weighted_loadings(
    X: np.ndarray,
    observed: np.ndarray,
    weights: np.ndarray,
    center: np.ndarray,
    scores: np.ndarray,
    previous: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """Solve all feature-loading normal equations in one batched operation."""
    p = X.shape[1]
    q = scores.shape[1]
    safe = np.where(observed, X, center)
    centered = safe - center
    active = np.any(weights > 0.0, axis=0)
    loadings = previous.copy()
    if np.any(active):
        active_weights = weights[:, active]
        grams = np.einsum(
            "nq,np,nr->pqr",
            scores,
            active_weights,
            scores,
            optimize=True,
        )
        grams += ridge * np.eye(q)[None, :, :]
        rhs = np.einsum(
            "nq,np,np->pq",
            scores,
            active_weights,
            centered[:, active],
            optimize=True,
        )
        loadings[active] = np.linalg.solve(grams, rhs[..., None])[..., 0]
    qmat, rmat = np.linalg.qr(loadings, mode="reduced")
    scores[:] = scores @ rmat.T
    return qmat


def _case_deviation(
    standardized_residuals: np.ndarray,
    observed: np.ndarray,
    residual_scales: np.ndarray,
    b: float,
    c: float,
) -> np.ndarray:
    losses = 2.0 * _wrapping_rho(standardized_residuals, b, c)
    losses = losses * (residual_scales[None, :] ** 2)
    losses = np.where(observed, losses, 0.0)
    counts = observed.sum(axis=1)
    return np.sqrt(
        np.divide(
            losses.sum(axis=1),
            counts,
            out=np.zeros(standardized_residuals.shape[0], dtype=float),
            where=counts > 0,
        )
    )


@dataclass
class CellwiseRobustPCA(EstimatorMixin):
    """PCA with cellwise and casewise iteratively reweighted least squares.

    Parameters
    ----------
    n_components : int, default=2
        Dimension of the fitted principal subspace.
    max_iter : int, default=100
        Maximum number of outer IRLS iterations.
    tol : float, default=1e-5
        Relative fitted-value change used as the convergence criterion.
    cell_b, cell_c : float, default=1.5, 4.0
        Inner and outer cutoffs of the redescending cellwise wrapping weight.
    case_b, case_c : float, default=1.5, 4.0
        Inner and outer cutoffs of the rowwise wrapping weight.
    ridge : float, default=1e-8
        Positive ridge used in weighted least-squares solves.
    weight_threshold : float, default=0.5
        Weights below this value are reported as outlying cells or rows.
    store_scores : bool, default=True
        Store training scores and fitted diagnostics.

    Notes
    -----
    This estimator implements the cellPCA strategy of combining cellwise and
    casewise redescending weights in a single weighted low-rank fit.  The
    reference algorithm initializes with MacroPCA and estimates fixed M-scales.
    This implementation instead uses robust marginal clipping followed by SVD
    and fixed MAD-type residual scales.  It is therefore an interoperable Python
    implementation of the weighting model, not a claim of numerical parity with
    the reference software.
    """

    n_components: int = 2
    max_iter: int = 100
    tol: float = 1e-5
    cell_b: float = 1.5
    cell_c: float = 4.0
    case_b: float = 1.5
    case_c: float = 4.0
    ridge: float = 1e-8
    weight_threshold: float = 0.5
    store_scores: bool = True

    def _validate_parameters(self, n_samples: int, n_features: int) -> None:
        if isinstance(self.n_components, (bool, np.bool_)) or not isinstance(
            self.n_components, (int, np.integer)
        ):
            raise TypeError("n_components must be an integer")
        if not 1 <= int(self.n_components) < min(n_samples, n_features):
            raise ValueError(
                "n_components must be between 1 and min(n_samples, n_features) - 1"
            )
        if int(self.max_iter) < 1:
            raise ValueError("max_iter must be at least 1")
        if not np.isfinite(self.tol) or float(self.tol) <= 0.0:
            raise ValueError("tol must be positive and finite")
        for prefix, b, c in (
            ("cell", self.cell_b, self.cell_c),
            ("case", self.case_b, self.case_c),
        ):
            if not np.isfinite(b) or not np.isfinite(c) or b <= 0.0 or c <= b:
                raise ValueError(f"{prefix}_b and {prefix}_c must satisfy 0 < b < c")
        if not np.isfinite(self.ridge) or float(self.ridge) <= 0.0:
            raise ValueError("ridge must be positive and finite")
        if not np.isfinite(self.weight_threshold) or not (
            0.0 < float(self.weight_threshold) < 1.0
        ):
            raise ValueError("weight_threshold must be in (0, 1)")
        if not isinstance(self.store_scores, (bool, np.bool_)):
            raise TypeError("store_scores must be a boolean")

    def _initialize(self, X: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        center, marginal_scale = _robust_center_scale(X)
        standardized = (X - center) / marginal_scale
        clipped = np.clip(np.where(observed, standardized, 0.0), -2.5, 2.5)
        initial = center + clipped * marginal_scale
        initial = np.where(observed, initial, center)
        centered = initial - center
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        loadings = vt[: int(self.n_components)].T
        scores = centered @ loadings
        return center, loadings, scores

    def fit(self, X: Any, y: Any | None = None) -> "CellwiseRobustPCA":
        """Fit a cellwise- and casewise-robust principal subspace."""
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

        center, loadings, scores = self._initialize(X, observed)
        fitted = center + scores @ loadings.T
        residuals = np.where(observed, X - fitted, np.nan)
        residual_center = np.nanmedian(residuals, axis=0)
        residual_scales = 1.482602218505602 * np.nanmedian(
            np.abs(residuals - residual_center), axis=0
        )
        marginal_scale = _robust_center_scale(X)[1]
        scale_floor = np.maximum(np.sqrt(_EPS) * np.maximum(marginal_scale, 1.0), 1e-12)
        residual_scales = np.where(
            np.isfinite(residual_scales) & (residual_scales > scale_floor),
            residual_scales,
            marginal_scale,
        )
        residual_scales = np.maximum(residual_scales, scale_floor)

        standardized = np.where(observed, (X - fitted) / residual_scales, 0.0)
        initial_case = _case_deviation(
            standardized,
            observed,
            residual_scales,
            float(self.cell_b),
            float(self.cell_c),
        )
        df = max(p - int(self.n_components), 1)
        median_factor = np.sqrt(chi2.ppf(0.5, df) / df)
        case_scale = float(np.median(initial_case) / max(median_factor, np.sqrt(_EPS)))
        if not np.isfinite(case_scale) or case_scale <= np.sqrt(_EPS):
            positive = initial_case[initial_case > np.sqrt(_EPS)]
            case_scale = float(np.median(positive)) if positive.size else 1.0

        objective_history: list[float] = []
        converged = False
        previous_fitted = fitted.copy()
        for iteration in range(1, int(self.max_iter) + 1):
            residuals_safe = np.where(observed, X - fitted, 0.0)
            standardized = residuals_safe / residual_scales
            cell_weights = _wrapping_weight(
                standardized, float(self.cell_b), float(self.cell_c)
            )
            cell_weights = np.where(observed, cell_weights, 0.0)
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
            center = _weighted_center(
                X, observed, weights, scores, loadings, center
            )
            scores = _weighted_scores(
                X, observed, weights, center, loadings, float(self.ridge)
            )
            loadings = _weighted_loadings(
                X,
                observed,
                weights,
                center,
                scores,
                loadings,
                float(self.ridge),
            )
            scores = _weighted_scores(
                X, observed, weights, center, loadings, float(self.ridge)
            )

            fitted = center + scores @ loadings.T
            residuals_safe = np.where(observed, X - fitted, 0.0)
            standardized = residuals_safe / residual_scales
            case_deviation = _case_deviation(
                standardized,
                observed,
                residual_scales,
                float(self.cell_b),
                float(self.cell_c),
            )
            objective = float(
                np.sum(
                    _wrapping_rho(
                        case_deviation / case_scale,
                        float(self.case_b),
                        float(self.case_c),
                    )
                )
            )
            objective_history.append(objective)

            denominator = max(float(np.linalg.norm(previous_fitted)), 1.0)
            relative_change = float(np.linalg.norm(fitted - previous_fitted) / denominator)
            if relative_change <= float(self.tol):
                converged = True
                break
            previous_fitted = fitted.copy()

        # Final diagnostics and an ordered basis inside the fitted subspace.
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

        weight_sum = max(float(np.sum(case_weights)), np.sqrt(_EPS))
        score_mean = np.sum(case_weights[:, None] * scores, axis=0) / weight_sum
        center = center + score_mean @ loadings.T
        scores = scores - score_mean
        score_covariance = (scores * case_weights[:, None]).T @ scores / weight_sum
        values, vectors = np.linalg.eigh(0.5 * (score_covariance + score_covariance.T))
        order = np.argsort(values)[::-1]
        values = np.maximum(values[order], 0.0)
        rotation = vectors[:, order]
        scores = scores @ rotation
        loadings = loadings @ rotation
        components = _deterministic_component_signs(loadings.T)
        signs = np.sum(components * loadings.T, axis=1)
        signs = np.where(signs < 0.0, -1.0, 1.0)
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

        total_variance = 0.0
        safe = np.where(observed, X, center)
        for j in range(p):
            w = cell_weights[:, j] * case_weights
            denom = float(np.sum(w))
            if denom > np.sqrt(_EPS):
                total_variance += float(np.sum(w * (safe[:, j] - center[j]) ** 2) / denom)
        total_variance = max(total_variance, float(np.sum(values)), np.sqrt(_EPS))

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
        self.loadings_ = components.T
        self.n_components_ = int(self.n_components)
        self.n_samples_in_ = n
        self.n_features_in_ = p
        self.residual_scales_ = residual_scales
        self.case_scale_ = float(case_scale)
        self.explained_variance_ = values
        self.eigenvalues_ = values.copy()
        self.explained_variance_ratio_ = values / total_variance
        self.noise_variance_ = float(
            np.nansum(residuals * residuals) / max(int(observed.sum()) - n * self.n_components_, 1)
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
        self.n_iter_ = iteration
        self.converged_ = converged
        self.cell_cutoff_ = float(self.cell_c)
        self.case_cutoff_ = float(self.case_c)

        if self.store_scores:
            self.scores_ = scores
        elif hasattr(self, "scores_"):
            delattr(self, "scores_")
        return self

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "components_"):
            raise AttributeError("CellPCA is not fitted yet")

    def _check_new_X(self, X: Any) -> np.ndarray:
        self._check_is_fitted()
        X = _as_matrix(X, min_rows=1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )
        if np.any(np.isfinite(X).sum(axis=1) == 0):
            raise ValueError("every row must contain at least one finite value")
        return X

    def _predict_diagnostics(self, X: Any) -> dict[str, np.ndarray]:
        X = self._check_new_X(X)
        observed = np.isfinite(X)
        safe = np.where(observed, X, self.center_)
        weights = observed.astype(float)
        scores = _weighted_scores(
            X, observed, weights, self.center_, self.loadings_, float(self.ridge)
        )
        for _ in range(30):
            fitted = self.center_ + scores @ self.components_
            standardized = np.where(
                observed,
                (safe - fitted) / self.residual_scales_,
                0.0,
            )
            new_weights = np.where(
                observed,
                _wrapping_weight(
                    standardized, float(self.cell_b), float(self.cell_c)
                ),
                0.0,
            )
            new_scores = _weighted_scores(
                X,
                observed,
                new_weights,
                self.center_,
                self.loadings_,
                float(self.ridge),
            )
            if np.linalg.norm(new_scores - scores) <= 1e-8 * max(
                np.linalg.norm(scores), 1.0
            ):
                scores = new_scores
                weights = new_weights
                break
            scores = new_scores
            weights = new_weights

        fitted = self.center_ + scores @ self.components_
        residuals = np.where(observed, X - fitted, np.nan)
        standardized = np.where(observed, residuals / self.residual_scales_, np.nan)
        cell_weights = np.where(
            observed,
            _wrapping_weight(
                np.nan_to_num(standardized, nan=0.0),
                float(self.cell_b),
                float(self.cell_c),
            ),
            0.0,
        )
        case_deviation = _case_deviation(
            np.nan_to_num(standardized, nan=0.0),
            observed,
            self.residual_scales_,
            float(self.cell_b),
            float(self.cell_c),
        )
        case_weights = _wrapping_weight(
            case_deviation / self.case_scale_,
            float(self.case_b),
            float(self.case_c),
        )
        cell_outliers = observed & (cell_weights < float(self.weight_threshold))
        case_outliers = case_weights < float(self.weight_threshold)
        corrected = np.array(X, copy=True)
        replace = (~observed) | cell_outliers
        corrected[replace] = fitted[replace]
        imputed = np.array(X, copy=True)
        imputed[~observed] = fitted[~observed]
        return {
            "scores": scores,
            "fitted_values": fitted,
            "residuals": residuals,
            "standardized_residuals": standardized,
            "cell_weights": cell_weights,
            "case_weights": case_weights,
            "cell_outlier_mask": cell_outliers,
            "case_outlier_mask": case_outliers,
            "missing_mask": ~observed,
            "case_deviations": case_deviation,
            "max_cell_residuals": np.nanmax(np.abs(standardized), axis=1),
            "corrected_data": corrected,
            "imputed_data": imputed,
        }

    def transform(self, X: Any) -> np.ndarray:
        """Project rows onto the fitted subspace using robust cell weights."""
        return self._predict_diagnostics(X)["scores"]

    def fit_transform(self, X: Any, y: Any | None = None) -> np.ndarray:
        """Fit the model and return training scores."""
        self.fit(X, y)
        return self.scores_.copy() if hasattr(self, "scores_") else self.transform(X)

    def inverse_transform(self, scores: Any) -> np.ndarray:
        """Map component scores back to the original feature space."""
        self._check_is_fitted()
        scores = np.asarray(scores, dtype=np.float64)
        if scores.ndim != 2:
            raise ValueError("scores must be a 2D array")
        if scores.shape[1] != self.n_components_:
            raise ValueError(
                f"scores has {scores.shape[1]} components, expected {self.n_components_}"
            )
        if not np.isfinite(scores).all():
            raise ValueError("scores must contain only finite values")
        return self.center_ + scores @ self.components_

    def reconstruct(self, X: Any) -> np.ndarray:
        """Return robust low-rank predictions for new rows."""
        return self._predict_diagnostics(X)["fitted_values"]

    def impute(self, X: Any) -> np.ndarray:
        """Replace missing cells by fitted values and retain observed cells."""
        return self._predict_diagnostics(X)["imputed_data"]

    def correct(self, X: Any) -> np.ndarray:
        """Replace missing and flagged cells by fitted values."""
        return self._predict_diagnostics(X)["corrected_data"]

    def cellwise_diagnostics(self, X: Any) -> dict[str, np.ndarray]:
        """Return scores, residual weights, flags, predictions, and corrections."""
        return self._predict_diagnostics(X)

    def outlier_map(self, X: Any | None = None) -> np.ndarray:
        """Return case deviation and maximum absolute cell residual.

        Column 0 contains the casewise total deviation from the fitted subspace.
        Column 1 contains the largest absolute standardized cell residual in the
        row.  Keeping these axes separate distinguishes an unusual complete row
        from a row driven by one or a few bad measurements.
        """
        self._check_is_fitted()
        if X is None:
            return np.column_stack(
                [self.case_deviations_, self.max_cell_residuals_]
            )
        diagnostics = self._predict_diagnostics(X)
        return np.column_stack(
            [diagnostics["case_deviations"], diagnostics["max_cell_residuals"]]
        )


CellPCA = CellwiseRobustPCA
CasewiseCellwisePCA = CellwiseRobustPCA


__all__ = ["CellwiseRobustPCA", "CellPCA", "CasewiseCellwisePCA"]
