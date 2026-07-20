# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Experimental spectral filtering for adversarial row contamination.

This module provides a practical robustcov composite inspired by the iterative
filtering literature in algorithmic high-dimensional robust statistics.  It is
not an implementation of the optimal Gaussian covariance estimators from that
literature and does not inherit their finite-sample guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import chi2

from .._estimator import EstimatorMixin
from .._utils import check_array, mahalanobis_squared, median_impute


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _geometric_median(
    X: np.ndarray,
    *,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """Compute a deterministic Weiszfeld geometric median."""

    estimate = np.median(X, axis=0)
    tiny = np.finfo(float).eps
    for _ in range(max_iter):
        distances = np.linalg.norm(X - estimate, axis=1)
        coincident = distances <= tiny
        if np.any(coincident):
            return X[np.flatnonzero(coincident)[0]].copy()
        weights = 1.0 / np.maximum(distances, tiny)
        updated = np.sum(weights[:, None] * X, axis=0) / np.sum(weights)
        if np.linalg.norm(updated - estimate) <= tol * max(
            1.0, np.linalg.norm(estimate)
        ):
            return updated
        estimate = updated
    return estimate


def _regularized_winsorized_covariance(
    X: np.ndarray,
    location: np.ndarray,
    *,
    shrinkage: float,
    winsorize: float,
    ridge: float,
) -> np.ndarray:
    centered = X - location
    coordinate_median = np.median(centered, axis=0)
    mad = 1.4826 * np.median(
        np.abs(centered - coordinate_median),
        axis=0,
    )
    standard_deviation = np.std(centered, axis=0, ddof=0)
    scales = np.where(
        mad > np.sqrt(np.finfo(float).eps),
        mad,
        np.where(standard_deviation > np.sqrt(np.finfo(float).eps), standard_deviation, 1.0),
    )
    clipped = np.clip(centered / scales, -winsorize, winsorize) * scales
    covariance = _symmetrize(clipped.T @ clipped / float(X.shape[0]))
    diagonal = np.diag(np.diag(covariance))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal

    values, vectors = np.linalg.eigh(covariance)
    average_variance = max(float(np.trace(covariance)) / covariance.shape[0], 1.0)
    floor = ridge * average_variance
    values = np.maximum(values, floor)
    return _symmetrize((vectors * values) @ vectors.T)


def _inverse_square_root(covariance: np.ndarray, ridge: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(_symmetrize(covariance))
    scale = max(float(values[-1]), 1.0)
    values = np.maximum(values, ridge * scale)
    return _symmetrize((vectors * (1.0 / np.sqrt(values))) @ vectors.T)


def _quadratic_operator(
    whitened: np.ndarray,
    direction: np.ndarray,
    second_moment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scores = (
        np.einsum("ij,jk,ik->i", whitened, direction, whitened)
        - float(np.sum(direction * second_moment))
    )
    transformed = (
        np.einsum("i,ij,ik->jk", scores, whitened, whitened)
        / float(whitened.shape[0])
        - float(np.mean(scores)) * second_moment
    )
    return _symmetrize(transformed), scores


def _dominant_quadratic_direction(
    whitened: np.ndarray,
    second_moment: np.ndarray,
    *,
    random_state: int,
    n_starts: int,
    power_iterations: int,
    tol: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    p = whitened.shape[1]
    rng = np.random.default_rng(random_state)
    starts = [np.eye(p) / np.sqrt(float(p))]
    for _ in range(n_starts - 1):
        candidate = rng.normal(size=(p, p))
        candidate = _symmetrize(candidate)
        candidate /= max(np.linalg.norm(candidate, ord="fro"), np.finfo(float).tiny)
        starts.append(candidate)

    best_value = -np.inf
    best_direction = starts[0]
    best_scores = np.zeros(whitened.shape[0])
    for start in starts:
        direction = start
        for _ in range(power_iterations):
            transformed, _ = _quadratic_operator(
                whitened,
                direction,
                second_moment,
            )
            norm = np.linalg.norm(transformed, ord="fro")
            if norm <= np.finfo(float).tiny:
                break
            updated = transformed / norm
            if float(np.sum(updated * direction)) < 0.0:
                updated = -updated
            if np.linalg.norm(updated - direction, ord="fro") <= tol:
                direction = updated
                break
            direction = updated
        transformed, scores = _quadratic_operator(
            whitened,
            direction,
            second_moment,
        )
        value = float(np.sum(direction * transformed))
        if value > best_value:
            best_value = value
            best_direction = direction.copy()
            best_scores = scores.copy()
    return best_value, best_direction, best_scores


def _robust_absolute_z(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    scale = 1.4826 * float(np.median(np.abs(values - median)))
    scale = max(scale, np.sqrt(np.finfo(float).eps))
    return np.abs(values - median) / scale


def _robust_upper_z(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    scale = 1.4826 * float(np.median(np.abs(values - median)))
    scale = max(scale, np.sqrt(np.finfo(float).eps))
    return np.maximum(values - median, 0.0) / scale


@dataclass(frozen=True)
class SpectralFilterStep:
    """Diagnostics for one filtering iteration."""

    iteration: int
    support_size: int
    n_removed: int
    operator_eigenvalue: float
    operator_threshold: float
    max_filter_score: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "iteration": int(self.iteration),
            "support_size": int(self.support_size),
            "n_removed": int(self.n_removed),
            "operator_eigenvalue": float(self.operator_eigenvalue),
            "operator_threshold": float(self.operator_threshold),
            "max_filter_score": float(self.max_filter_score),
        }


@dataclass
class SpectralFilteringCovariance(EstimatorMixin):
    """Estimate covariance by iteratively filtering adversarially corrupted rows.

    The estimator repeatedly fits a regularized provisional covariance, whitens
    the retained rows, and finds a high-variance direction in the lifted
    quadratic features ``zz.T - I``.  Rows with extreme directional quadratic
    scores or radial scores are removed until the lifted covariance operator is
    compatible with a near-Gaussian reference or the declared contamination
    budget is exhausted.

    This is an experimental robustcov composite inspired by filtering-based
    high-dimensional robust covariance estimation.  It is not the exact
    algorithm of Diakonikolas et al. or Cheng et al. and carries none of their
    optimal-error guarantees.

    Parameters
    ----------
    contamination : float, default=0.1
        Upper bound on the fraction of adversarial rows.  It defines a hard
        removal budget and must lie in ``[0, 0.4)``.
    max_iter : int, default=25
        Maximum number of filtering iterations.
    filter_strength : float, default=8.0
        Multiplier in the finite-sample lifted-operator tolerance.  Larger
        values make filtering more conservative.
    score_threshold : float, default=6.0
        Robust standardized score required before a row can be removed.
    removal_fraction : float, default=0.25
        Fraction of the remaining removal budget used in one iteration.
    shrinkage : float, default=0.05
        Diagonal shrinkage of provisional and final covariance estimates.
    winsorize : float, default=5.0
        Coordinatewise robust-standardization clipping used only to stabilize
        provisional whitening and the final support covariance.
    ridge : float, default=1e-8
        Relative positive-eigenvalue floor.
    n_starts : int, default=2
        Number of deterministic/random power-iteration starts.
    power_iterations : int, default=20
        Matrix-free power iterations per start in quadratic-feature space.
    tol : float, default=1e-5
        Power-iteration convergence tolerance.
    center_max_iter : int, default=100
        Maximum Weiszfeld iterations for the geometric median.
    center_tol : float, default=1e-7
        Geometric-median convergence tolerance.
    scale_correction : {"radial_median", "none"}, default="radial_median"
        Optional final radial-median consistency correction.
    missing_values : {"raise", "median"}, default="raise"
        Missing-value handling.
    random_state : int, default=0
        Seed for extra power-iteration starts.
    store_history : bool, default=True
        Store :class:`SpectralFilterStep` diagnostics in ``filter_history_``.

    Notes
    -----
    The clean reference is expected to be approximately Gaussian after an
    affine transform.  Heavy-tailed clean data can look anomalous in quadratic
    feature space; use Tyler, Student-t, Cauchy, or spatial-sign estimators for
    that setting.  Cellwise corruption should be handled by CellMCD or CellRCov.
    The matrix-free lifted operator costs roughly ``O(n p^2)`` per power step,
    so this experimental estimator is not intended for extremely large feature
    counts.
    """

    contamination: float = 0.1
    max_iter: int = 25
    filter_strength: float = 8.0
    score_threshold: float = 6.0
    removal_fraction: float = 0.25
    shrinkage: float = 0.05
    winsorize: float = 5.0
    ridge: float = 1e-8
    n_starts: int = 2
    power_iterations: int = 20
    tol: float = 1e-5
    center_max_iter: int = 100
    center_tol: float = 1e-7
    scale_correction: str = "radial_median"
    missing_values: str = "raise"
    random_state: int = 0
    store_history: bool = True

    def _validate_parameters(self) -> None:
        if isinstance(self.contamination, (bool, np.bool_)):
            raise TypeError("contamination must be a real number")
        contamination = float(self.contamination)
        if not np.isfinite(contamination) or not 0.0 <= contamination < 0.4:
            raise ValueError("contamination must be in [0, 0.4)")
        for name in (
            "max_iter",
            "n_starts",
            "power_iterations",
            "center_max_iter",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        for name in (
            "filter_strength",
            "score_threshold",
            "winsorize",
            "ridge",
            "tol",
            "center_tol",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
                raise TypeError(f"{name} must be a real number")
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{name} must be a positive finite number")
        for name in ("removal_fraction", "shrinkage"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
                raise TypeError(f"{name} must be a real number")
            value = float(value)
            if not np.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1)")
        if self.scale_correction not in {"radial_median", "none"}:
            raise ValueError("scale_correction must be 'radial_median' or 'none'")
        if self.missing_values not in {"raise", "median"}:
            raise ValueError("missing_values must be 'raise' or 'median'")
        if isinstance(self.random_state, (bool, np.bool_)) or not isinstance(
            self.random_state, (int, np.integer)
        ):
            raise TypeError("random_state must be an integer")
        if not isinstance(self.store_history, (bool, np.bool_)):
            raise TypeError("store_history must be a boolean")

    def _provisional_fit(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        location = _geometric_median(
            X,
            max_iter=int(self.center_max_iter),
            tol=float(self.center_tol),
        )
        covariance = _regularized_winsorized_covariance(
            X,
            location,
            shrinkage=float(self.shrinkage),
            winsorize=float(self.winsorize),
            ridge=float(self.ridge),
        )
        return location, covariance

    def fit(self, X: Any, y: Any = None) -> "SpectralFilteringCovariance":
        """Fit the filtering estimator."""

        del y
        self._validate_parameters()
        X = check_array(X, allow_nan=self.missing_values == "median")
        if self.missing_values == "median":
            X, self.impute_values_ = median_impute(X)
        self.n_samples_in_, self.n_features_in_ = X.shape
        if self.n_samples_in_ < 8:
            raise ValueError("SpectralFilteringCovariance requires at least 8 rows")

        support = np.ones(self.n_samples_in_, dtype=bool)
        budget = int(np.floor(float(self.contamination) * self.n_samples_in_))
        history: list[SpectralFilterStep] = []
        filter_scores = np.zeros(self.n_samples_in_, dtype=float)
        stopping_reason = "max_iter"
        converged = False
        effective_dimension = self.n_features_in_ * (self.n_features_in_ + 1) / 2.0

        for iteration in range(int(self.max_iter)):
            active_indices = np.flatnonzero(support)
            active = X[active_indices]
            location, covariance = self._provisional_fit(active)
            inverse_root = _inverse_square_root(covariance, float(self.ridge))
            whitened = (active - location) @ inverse_root.T
            second_moment = _symmetrize(
                whitened.T @ whitened / float(whitened.shape[0])
            )
            operator_value, _, quadratic_scores = _dominant_quadratic_direction(
                whitened,
                second_moment,
                random_state=int(self.random_state) + iteration,
                n_starts=int(self.n_starts),
                power_iterations=int(self.power_iterations),
                tol=float(self.tol),
            )
            operator_threshold = 2.0 + float(self.filter_strength) * (
                np.sqrt(effective_dimension / whitened.shape[0])
                + effective_dimension / whitened.shape[0]
            )
            radial_scores = np.sum(whitened * whitened, axis=1)
            combined_scores = np.maximum(
                _robust_absolute_z(quadratic_scores),
                _robust_upper_z(radial_scores),
            )
            filter_scores[active_indices] = np.maximum(
                filter_scores[active_indices],
                combined_scores,
            )

            remaining_budget = budget - int(np.sum(~support))
            candidates = np.flatnonzero(
                combined_scores > float(self.score_threshold)
            )
            n_removed = 0
            if operator_value <= operator_threshold:
                stopping_reason = "operator_within_tolerance"
                converged = True
            elif remaining_budget <= 0:
                stopping_reason = "removal_budget_exhausted"
            elif candidates.size == 0:
                stopping_reason = "no_extreme_filter_scores"
            else:
                n_removed = min(
                    candidates.size,
                    remaining_budget,
                    max(
                        1,
                        int(
                            np.ceil(
                                float(self.removal_fraction)
                                * float(remaining_budget)
                            )
                        ),
                    ),
                )
                order = candidates[
                    np.argsort(combined_scores[candidates])[::-1][:n_removed]
                ]
                support[active_indices[order]] = False

            if self.store_history:
                history.append(
                    SpectralFilterStep(
                        iteration=iteration,
                        support_size=int(active.shape[0]),
                        n_removed=int(n_removed),
                        operator_eigenvalue=float(operator_value),
                        operator_threshold=float(operator_threshold),
                        max_filter_score=float(np.max(combined_scores)),
                    )
                )
            if n_removed == 0:
                break

        retained = X[support]
        if retained.shape[0] < 2:
            raise RuntimeError("filtering retained fewer than two observations")
        self.location_, covariance = self._provisional_fit(retained)
        self.covariance_ = covariance
        self.precision_ = np.linalg.pinv(self.covariance_)
        retained_distances = mahalanobis_squared(
            retained,
            self.location_,
            self.precision_,
        )
        if self.scale_correction == "radial_median":
            denominator = max(
                float(chi2.ppf(0.5, self.n_features_in_)),
                np.finfo(float).tiny,
            )
            self.scale_ = float(np.median(retained_distances) / denominator)
            self.covariance_ = self.scale_ * self.covariance_
            self.precision_ = np.linalg.pinv(self.covariance_)
        else:
            self.scale_ = 1.0

        self.shape_ = (
            self.n_features_in_
            * self.covariance_
            / max(float(np.trace(self.covariance_)), np.finfo(float).tiny)
        )
        self.support_ = support
        self.filter_scores_ = filter_scores
        self.distances_ = self.mahalanobis(X)
        self.n_removed_ = int(np.sum(~support))
        self.effective_contamination_ = self.n_removed_ / float(self.n_samples_in_)
        self.n_iter_ = len(history) if self.store_history else iteration + 1
        self.filter_history_ = history
        self.converged_ = bool(converged)
        self.stopping_reason_ = stopping_reason
        return self

    def mahalanobis(self, X: Any) -> np.ndarray:
        """Return squared Mahalanobis distances from the fitted covariance."""

        if not hasattr(self, "precision_"):
            raise AttributeError("SpectralFilteringCovariance is not fitted")
        X = check_array(X, allow_nan=self.missing_values == "median")
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the fitted data")
        if self.missing_values == "median" and np.isnan(X).any():
            X, _ = median_impute(X, self.impute_values_)
        return mahalanobis_squared(X, self.location_, self.precision_)

    def score_samples(self, X: Any) -> np.ndarray:
        """Return larger-is-more-normal covariance log-score surrogates."""

        return -0.5 * self.mahalanobis(X)

    def predict(self, X: Any, alpha: float = 0.975) -> np.ndarray:
        """Label observations by a chi-square distance cutoff."""

        if not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        cutoff = float(chi2.ppf(float(alpha), self.n_features_in_))
        return np.where(self.mahalanobis(X) <= cutoff, 1, -1)

    def fit_predict(self, X: Any, y: Any = None) -> np.ndarray:
        """Fit the estimator and return inlier/outlier labels."""

        return self.fit(X, y).predict(X)

    def history_records(self) -> list[dict[str, float | int]]:
        """Return serializable filtering diagnostics."""

        if not hasattr(self, "filter_history_"):
            raise AttributeError("SpectralFilteringCovariance is not fitted")
        return [step.as_dict() for step in self.filter_history_]
