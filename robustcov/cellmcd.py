# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Cellwise minimum covariance determinant estimation.

This module implements the observed-likelihood CellMCD concentration step of
Raymaekers and Rousseeuw (2024).  The reference implementation initializes the
algorithm with DDCW.  ``robustcov`` instead uses a deterministic robust
marginal and clipped-correlation start, so fitted solutions need not match the
reference software exactly even when the final objective and updates agree.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import chi2

from ._utils import check_array
from .covariance import BaseRobustCovariance, ConvergenceWarning


_LOG_2PI = float(np.log(2.0 * np.pi))
_EPS = np.finfo(np.float64).eps


def _mad_scale(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(X, axis=0)
    absolute = np.abs(X - center)
    scale = 1.482602218505602 * np.nanmedian(absolute, axis=0)
    fallback = np.nanstd(X, axis=0, ddof=1)
    valid = scale[np.isfinite(scale) & (scale > 0.0)]
    reference = float(np.median(valid)) if valid.size else 1.0
    floor = max(np.sqrt(_EPS) * max(reference, 1.0), np.finfo(float).tiny)
    scale = np.where(np.isfinite(scale) & (scale > floor), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > floor), scale, np.nan)
    if not np.isfinite(center).all():
        raise ValueError("every feature must contain at least one finite value")
    if not np.isfinite(scale).all():
        bad = np.flatnonzero(~np.isfinite(scale))
        raise ValueError(
            "every feature must have nonzero robust scale; failed columns: "
            + ", ".join(map(str, bad.tolist()))
        )
    return center, scale


def _truncate_covariance(covariance: np.ndarray, minimum_eigenvalue: float) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=np.float64)
    covariance = 0.5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, minimum_eigenvalue)
    result = (vectors * values) @ vectors.T
    return 0.5 * (result + result.T)


def _initial_covariance(Z: np.ndarray, finite: np.ndarray, minimum_eigenvalue: float) -> np.ndarray:
    filled = np.where(finite, Z, 0.0)
    clipped = np.clip(filled, -2.5, 2.5)
    covariance = clipped.T @ clipped / max(Z.shape[0], 1)
    diagonal = np.diag(covariance).copy()
    diagonal = np.where(np.isfinite(diagonal) & (diagonal > minimum_eigenvalue), diagonal, 1.0)
    np.fill_diagonal(covariance, diagonal)
    return _truncate_covariance(covariance, minimum_eigenvalue)


def _conditional_parameters(
    location: np.ndarray,
    covariance: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray,
    observed_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Conditional mean and covariance for ``target`` given ``observed``."""
    if target.size == 0:
        return np.empty(0, dtype=float), np.empty((0, 0), dtype=float)
    if observed.size == 0:
        return location[target].copy(), covariance[np.ix_(target, target)].copy()
    cov_oo = covariance[np.ix_(observed, observed)]
    cov_to = covariance[np.ix_(target, observed)]
    solve = np.linalg.solve(cov_oo, observed_values - location[observed])
    mean = location[target] + cov_to @ solve
    conditional = covariance[np.ix_(target, target)] - cov_to @ np.linalg.solve(
        cov_oo, covariance[np.ix_(observed, target)]
    )
    conditional = 0.5 * (conditional + conditional.T)
    return mean, conditional


def _partial_distance(
    row: np.ndarray,
    mask: np.ndarray,
    location: np.ndarray,
    covariance: np.ndarray,
) -> float:
    observed = np.flatnonzero(mask)
    if observed.size == 0:
        return 0.0
    centered = row[observed] - location[observed]
    subcov = covariance[np.ix_(observed, observed)]
    return float(centered @ np.linalg.solve(subcov, centered))


def _observed_objective(
    X: np.ndarray,
    W: np.ndarray,
    location: np.ndarray,
    covariance: np.ndarray,
    penalties: np.ndarray,
    finite_mask: np.ndarray,
) -> float:
    objective = 0.0
    for pattern in np.unique(W, axis=0):
        rows = np.flatnonzero(np.all(W == pattern, axis=1))
        observed = np.flatnonzero(pattern)
        if observed.size:
            subcov = covariance[np.ix_(observed, observed)]
            sign, logdet = np.linalg.slogdet(subcov)
            if sign <= 0:
                return float("inf")
            centered = X[np.ix_(rows, observed)] - location[observed]
            solved = np.linalg.solve(subcov, centered.T).T
            objective += float(np.sum(centered * solved))
            objective += rows.size * (float(logdet) + observed.size * _LOG_2PI)
    flagged = (~W) & finite_mask
    objective += float(np.sum(flagged * penalties[None, :]))
    return objective


def _em_update(
    X: np.ndarray,
    W: np.ndarray,
    location: np.ndarray,
    covariance: np.ndarray,
    minimum_eigenvalue: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, p = X.shape
    conditional_means = np.empty((n, p), dtype=np.float64)
    covariance_bias = np.zeros((p, p), dtype=np.float64)

    for i in range(n):
        observed = np.flatnonzero(W[i])
        missing = np.flatnonzero(~W[i])
        conditional_means[i, observed] = X[i, observed]
        mean_missing, conditional_cov = _conditional_parameters(
            location, covariance, missing, observed, X[i, observed]
        )
        conditional_means[i, missing] = mean_missing
        if missing.size:
            covariance_bias[np.ix_(missing, missing)] += conditional_cov

    new_location = conditional_means.mean(axis=0)
    centered = conditional_means - new_location
    new_covariance = (centered.T @ centered + covariance_bias) / n
    new_covariance = _truncate_covariance(new_covariance, minimum_eigenvalue)
    return new_location, new_covariance, conditional_means


class CellwiseMinimumCovarianceDeterminant(BaseRobustCovariance):
    """Cellwise-robust location and covariance estimator.

    Parameters
    ----------
    alpha : float, default=0.75
        Minimum fraction of cells retained in every feature.  The effective
        count is at least ``floor(n / 2) + 1``.
    quantile : float, default=0.99
        Chi-square probability used to calibrate the cellwise penalty and
        standardized-residual cutoff.
    max_iter : int, default=100
        Maximum number of concentration steps.
    tol : float, default=1e-4
        Maximum absolute covariance change used as the stopping criterion on
        robustly standardized data.
    min_eigenvalue : float, default=1e-4
        Eigenvalue floor imposed on the standardized covariance estimate.
    min_samples_per_feature : float or None, default=5.0
        Require ``n / p`` to meet this value.  Set to ``None`` to disable the
        check; CellMCD is not intended as a high-dimensional estimator.

    Notes
    -----
    The estimator minimizes an observed Gaussian likelihood plus a penalty for
    flagged cells while retaining at least ``h`` cells in every column.  Its
    concentration step alternates a columnwise update of the binary cell mask
    with one Gaussian EM update of location and covariance.

    The reference CellMCD implementation uses a DDCW start.  This implementation
    uses a deterministic median/MAD and clipped-correlation start.  Consequently
    it should be viewed as an implementation of the CellMCD objective and C-step,
    not a claim of numerical identity with the reference software.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.75,
        quantile: float = 0.99,
        max_iter: int = 100,
        tol: float = 1e-4,
        min_eigenvalue: float = 1e-4,
        min_samples_per_feature: float | None = 5.0,
    ):
        super().__init__(
            assume_centered=False,
            store_precision=True,
            scale_correction="none",
            tail_diagnostics=False,
            missing_values="native",
        )
        if not (0.5 <= float(alpha) <= 1.0):
            raise ValueError("alpha must be in [0.5, 1]")
        if not (0.5 < float(quantile) < 1.0):
            raise ValueError("quantile must be in (0.5, 1)")
        if int(max_iter) < 1:
            raise ValueError("max_iter must be at least 1")
        if float(tol) <= 0.0:
            raise ValueError("tol must be positive")
        if float(min_eigenvalue) < 1e-8:
            raise ValueError("min_eigenvalue must be at least 1e-8")
        if min_samples_per_feature is not None and float(min_samples_per_feature) <= 0.0:
            raise ValueError("min_samples_per_feature must be positive or None")
        self.alpha = float(alpha)
        self.quantile = float(quantile)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.min_eigenvalue = float(min_eigenvalue)
        self.min_samples_per_feature = (
            None if min_samples_per_feature is None else float(min_samples_per_feature)
        )

    def fit(self, X, y=None):
        X = check_array(X, allow_nan=True)
        n, p = X.shape
        if self.min_samples_per_feature is not None and n / p < self.min_samples_per_feature:
            raise ValueError(
                "CellMCD requires more observations per feature; "
                f"n / p = {n / p:.3g} < {self.min_samples_per_feature:.3g}. "
                "Set min_samples_per_feature=None to override this guard."
            )

        self.n_samples_in_, self.n_features_in_ = n, p
        self.h_ = max(int(np.floor(n / 2)) + 1, int(np.ceil(self.alpha * n)))
        finite = np.isfinite(X)
        finite_per_column = finite.sum(axis=0)
        if np.any(finite_per_column < self.h_):
            bad = np.flatnonzero(finite_per_column < self.h_)
            raise ValueError(
                "each feature must contain at least h finite cells; failed columns: "
                + ", ".join(map(str, bad.tolist()))
            )

        marginal_location, marginal_scale = _mad_scale(X)
        Z = (X - marginal_location) / marginal_scale
        cutoff = float(np.sqrt(chi2.ppf(self.quantile, 1)))
        W = finite & (np.abs(Z) <= cutoff)
        for j in range(p):
            if np.count_nonzero(W[:, j]) < self.h_:
                order = np.argsort(
                    np.where(finite[:, j], np.abs(Z[:, j]), np.inf), kind="stable"
                )
                W[:, j] = False
                W[order[: self.h_], j] = True

        location = np.zeros(p, dtype=np.float64)
        covariance = _initial_covariance(Z, finite, self.min_eigenvalue)
        precision = np.linalg.inv(covariance)
        conditional_variance = 1.0 / np.diag(precision)
        penalties = (
            chi2.ppf(self.quantile, 1)
            + _LOG_2PI
            + np.log(np.maximum(conditional_variance, self.min_eigenvalue))
        )

        objective_history = [
            _observed_objective(Z, W, location, covariance, penalties, finite)
        ]
        converged = False
        conditional_means = np.where(finite, Z, 0.0)

        for iteration in range(1, self.max_iter + 1):
            proposed_W = self._update_cell_support(
                Z, W.copy(), location, covariance, penalties, finite
            )
            proposed_location, proposed_covariance, proposed_means = _em_update(
                Z,
                proposed_W,
                location,
                covariance,
                self.min_eigenvalue,
            )
            objective = _observed_objective(
                Z,
                proposed_W,
                proposed_location,
                proposed_covariance,
                penalties,
                finite,
            )
            if objective > objective_history[-1] + 1e-8 * max(1.0, abs(objective_history[-1])):
                break
            change = float(np.max(np.abs(proposed_covariance - covariance)))
            W = proposed_W
            location = proposed_location
            covariance = proposed_covariance
            conditional_means = proposed_means
            objective_history.append(objective)
            if change <= self.tol:
                converged = True
                break

        self.n_iter_ = len(objective_history) - 1
        self.converged_ = converged
        if not converged and self.n_iter_ >= self.max_iter:
            warnings.warn("CellMCD did not converge", ConvergenceWarning)

        self.standardization_location_ = marginal_location
        self.standardization_scale_ = marginal_scale
        self.standardized_location_ = location
        self.standardized_covariance_ = covariance
        self.location_ = marginal_location + marginal_scale * location
        self.covariance_ = covariance * np.outer(marginal_scale, marginal_scale)
        self.precision_ = np.linalg.inv(self.covariance_)
        self.cell_support_ = W.astype(bool)
        self.cell_weights_ = self.cell_support_.astype(np.int8)
        self.missing_mask_ = ~finite
        self.cell_outlier_mask_ = (~self.cell_support_) & finite
        self.penalties_ = penalties.copy()
        self.objective_history_ = np.asarray(objective_history, dtype=float)
        self.objective_value_ = float(objective_history[-1])
        self.cell_cutoff_ = cutoff

        predictions, conditional_std, residuals = self._diagnostics_with_support(
            X, self.cell_support_
        )
        self.predicted_values_ = predictions
        self.conditional_std_ = conditional_std
        self.standardized_residuals_ = residuals
        self.imputed_data_ = np.asarray(X, dtype=float).copy()
        replace = ~self.cell_support_
        self.imputed_data_[replace] = predictions[replace]
        self.corrected_data_ = self.imputed_data_.copy()
        self.row_outlier_fraction_ = self.cell_outlier_mask_.sum(axis=1) / np.maximum(
            finite.sum(axis=1), 1
        )
        self.column_outlier_fraction_ = self.cell_outlier_mask_.sum(axis=0) / np.maximum(
            finite.sum(axis=0), 1
        )
        self.distances_ = self._partial_distances(X, self.cell_support_)
        self.expected_complete_data_ = (
            marginal_location + conditional_means * marginal_scale
        )
        return self

    def _update_cell_support(
        self,
        Z: np.ndarray,
        W: np.ndarray,
        location: np.ndarray,
        covariance: np.ndarray,
        penalties: np.ndarray,
        finite: np.ndarray,
    ) -> np.ndarray:
        column_order = np.argsort(W.sum(axis=0), kind="stable")
        for j in column_order:
            delta = np.full(Z.shape[0], np.inf, dtype=np.float64)
            other = np.arange(Z.shape[1]) != j
            patterns = np.unique(W[:, other], axis=0)
            for pattern in patterns:
                rows = np.flatnonzero(np.all(W[:, other] == pattern, axis=1))
                rows = rows[finite[rows, j]]
                if rows.size == 0:
                    continue
                observed = np.flatnonzero(other)[pattern]
                target = np.array([j], dtype=int)
                if observed.size:
                    cov_oo = covariance[np.ix_(observed, observed)]
                    cov_jo = covariance[np.ix_(target, observed)]
                    beta = np.linalg.solve(cov_oo, covariance[np.ix_(observed, target)])
                    conditional_variance = float(
                        (covariance[j, j] - cov_jo @ beta).item()
                    )
                    centered = Z[np.ix_(rows, observed)] - location[observed]
                    prediction = location[j] + centered @ np.linalg.solve(
                        cov_oo, covariance[np.ix_(observed, target)]
                    ).reshape(-1)
                else:
                    conditional_variance = float(covariance[j, j])
                    prediction = np.full(rows.size, location[j], dtype=float)
                conditional_variance = max(conditional_variance, self.min_eigenvalue)
                delta[rows] = (
                    (Z[rows, j] - prediction) ** 2 / conditional_variance
                    + np.log(conditional_variance)
                    + _LOG_2PI
                )

            good = np.flatnonzero(delta <= penalties[j])
            if good.size < self.h_:
                good = np.argsort(delta, kind="stable")[: self.h_]
            W[:, j] = False
            W[good, j] = True
            W[~finite[:, j], j] = False
        return W

    def _diagnostics_with_support(
        self, X: np.ndarray, support: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        n, p = X.shape
        predictions = np.empty((n, p), dtype=float)
        conditional_std = np.empty((n, p), dtype=float)
        residuals = np.full((n, p), np.nan, dtype=float)
        for i in range(n):
            for j in range(p):
                observed = np.flatnonzero(support[i] & (np.arange(p) != j))
                mean, conditional = _conditional_parameters(
                    self.location_,
                    self.covariance_,
                    np.array([j], dtype=int),
                    observed,
                    X[i, observed],
                )
                variance = max(float(conditional[0, 0]), np.finfo(float).tiny)
                predictions[i, j] = float(mean[0])
                conditional_std[i, j] = np.sqrt(variance)
                if np.isfinite(X[i, j]):
                    residuals[i, j] = (X[i, j] - mean[0]) / np.sqrt(variance)
        return predictions, conditional_std, residuals

    def cellwise_diagnostics(self, X, *, max_passes: int = 2):
        """Return predictions, residuals, and an outlier mask for new rows.

        Cell flags are refined by removing cells whose conditional standardized
        residual exceeds the fitted chi-square cutoff.  Missing cells are never
        labeled as outliers, but they are included in the returned corrected data.
        """
        if not hasattr(self, "covariance_"):
            raise RuntimeError("Estimator is not fitted")
        X = check_array(X, allow_nan=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the fitted data")
        if int(max_passes) < 1:
            raise ValueError("max_passes must be at least 1")
        finite = np.isfinite(X)
        support = finite.copy()
        predictions = np.empty_like(X)
        conditional_std = np.empty_like(X)
        residuals = np.full_like(X, np.nan)
        for _ in range(int(max_passes)):
            predictions, conditional_std, residuals = self._diagnostics_with_support(X, support)
            proposed = finite & (np.abs(residuals) <= self.cell_cutoff_)
            if np.array_equal(proposed, support):
                break
            support = proposed
        outliers = finite & ~support
        corrected = X.copy()
        corrected[~support] = predictions[~support]
        return {
            "predictions": predictions,
            "conditional_std": conditional_std,
            "standardized_residuals": residuals,
            "cell_support": support,
            "cell_outlier_mask": outliers,
            "missing_mask": ~finite,
            "corrected_data": corrected,
        }

    def predict_cells(self, X, *, max_passes: int = 2) -> np.ndarray:
        """Return a boolean mask marking conditionally outlying cells."""
        return self.cellwise_diagnostics(X, max_passes=max_passes)["cell_outlier_mask"]

    def cell_scores(self, X, *, max_passes: int = 2) -> np.ndarray:
        """Return absolute conditional standardized residuals."""
        residuals = self.cellwise_diagnostics(X, max_passes=max_passes)[
            "standardized_residuals"
        ]
        return np.abs(residuals)

    def transform(self, X, *, replace_outliers: bool = True, max_passes: int = 2) -> np.ndarray:
        """Impute missing cells and optionally replace flagged cells by predictions."""
        diagnostics = self.cellwise_diagnostics(X, max_passes=max_passes)
        if replace_outliers:
            return diagnostics["corrected_data"]
        X = np.asarray(X, dtype=float).copy()
        missing = diagnostics["missing_mask"]
        X[missing] = diagnostics["predictions"][missing]
        return X

    def _partial_distances(self, X: np.ndarray, support: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                _partial_distance(row, mask, self.location_, self.covariance_)
                for row, mask in zip(np.asarray(X, dtype=float), support)
            ],
            dtype=float,
        )

    def mahalanobis(self, X):
        diagnostics = self.cellwise_diagnostics(X)
        return self._partial_distances(
            np.asarray(X, dtype=float), diagnostics["cell_support"]
        )

    def predict(self, X, alpha: float = 0.975):
        diagnostics = self.cellwise_diagnostics(X)
        distances = self._partial_distances(
            np.asarray(X, dtype=float), diagnostics["cell_support"]
        )
        dimensions = diagnostics["cell_support"].sum(axis=1)
        cutoff = chi2.ppf(alpha, np.maximum(dimensions, 1))
        return np.where(distances <= cutoff, 1, -1)


CellMCD = CellwiseMinimumCovarianceDeterminant
CellwiseMCD = CellwiseMinimumCovarianceDeterminant


__all__ = [
    "CellwiseMinimumCovarianceDeterminant",
    "CellMCD",
    "CellwiseMCD",
]
