# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Deterministic multivariate S- and MM-estimators.

The estimators follow the S-scale and MM-refinement equations of Hubert,
Rousseeuw, Vanpaemel, and Verdonck.  The deterministic start family is inspired
by DetS, but is package-specific: it combines several robust correlation and
projection starts instead of reproducing the six exact DetMCD starts from the
reference implementation.
"""

from __future__ import annotations

from functools import lru_cache
import warnings

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import chi, chi2, norm, rankdata

from ._utils import check_array, mahalanobis_squared, median_impute, radial_kurtosis
from .covariance import BaseRobustCovariance, ConvergenceWarning
from .mrcd import _mad_scale, _qn_scale, _robust_standardize


_EPS = np.finfo(np.float64).eps


def _symmetrize_spd(matrix: np.ndarray, ridge: float) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    scale = max(
        float(np.median(np.abs(values))),
        abs(float(np.trace(matrix) / matrix.shape[0])),
    )
    if not np.isfinite(scale) or scale <= np.finfo(float).tiny:
        scale = 1.0
    floor = max(
        float(ridge) * scale,
        np.sqrt(_EPS) * scale,
        np.finfo(float).tiny,
    )
    values = np.maximum(values, floor)
    return vectors @ np.diag(values) @ vectors.T


def _normalize_shape(scatter: np.ndarray, ridge: float) -> np.ndarray:
    scatter = _symmetrize_spd(scatter, ridge)
    sign, logdet = np.linalg.slogdet(scatter)
    if sign <= 0 or not np.isfinite(logdet):
        raise np.linalg.LinAlgError("scatter matrix is not positive definite")
    return scatter / np.exp(logdet / scatter.shape[0])


def _bisquare_rho(u: np.ndarray, c: float) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    absolute = np.abs(u)
    out = np.full_like(absolute, c * c / 6.0)
    inside = absolute <= c
    z = absolute[inside]
    out[inside] = 0.5 * z**2 - 0.5 * z**4 / c**2 + z**6 / (6.0 * c**4)
    return out


def _bisquare_weight(u: np.ndarray, c: float) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    absolute = np.abs(u)
    out = np.zeros_like(absolute)
    inside = absolute < c
    t = 1.0 - (absolute[inside] / c) ** 2
    out[inside] = t * t
    return out


def _s_expectation(c: float, p: int) -> float:
    c2 = c * c
    return float(
        (p / 2.0) * chi2.cdf(c2, p + 2)
        - (p * (p + 2) / (2.0 * c2)) * chi2.cdf(c2, p + 4)
        + (p * (p + 2) * (p + 4) / (6.0 * c2 * c2)) * chi2.cdf(c2, p + 6)
        + (c2 / 6.0) * (1.0 - chi2.cdf(c2, p))
    )


@lru_cache(maxsize=128)
def _s_tuning_constant(p: int, breakdown: float) -> tuple[float, float]:
    if not (0.05 <= breakdown <= 0.5):
        raise ValueError("breakdown must be in [0.05, 0.5]")

    def equation(c: float) -> float:
        return _s_expectation(c, p) / (c * c / 6.0) - breakdown

    lower = max(0.05, 0.05 * np.sqrt(p))
    upper = max(20.0, 6.0 * np.sqrt(p))
    while equation(upper) > 0:
        upper *= 2.0
    c = float(brentq(equation, lower, upper, xtol=1e-12, rtol=1e-12))
    return c, _s_expectation(c, p)


def _location_efficiency(c: float, p: int) -> float:
    def density(r: float) -> float:
        return float(chi.pdf(r, p))

    def weight(r: float) -> float:
        if r >= c:
            return 0.0
        t = 1.0 - (r / c) ** 2
        return t * t

    def derivative(r: float) -> float:
        if r >= c:
            return 0.0
        return -4.0 * r / (c * c) * (1.0 - (r / c) ** 2)

    expected_weight = quad(lambda r: weight(r) * density(r), 0.0, c, epsabs=1e-10)[0]
    expected_derivative = quad(lambda r: r * derivative(r) * density(r), 0.0, c, epsabs=1e-10)[0]
    second_moment = quad(lambda r: weight(r) ** 2 * r * r * density(r), 0.0, c, epsabs=1e-10)[0] / p
    derivative_term = expected_weight + expected_derivative / p
    return float(derivative_term * derivative_term / second_moment)


@lru_cache(maxsize=128)
def _mm_tuning_constant(p: int, efficiency: float) -> float:
    if not (0.5 <= efficiency < 1.0):
        raise ValueError("efficiency must be in [0.5, 1)")

    lower = max(0.5, 0.4 * np.sqrt(p))
    upper = max(12.0, 4.0 * np.sqrt(p))
    while _location_efficiency(upper, p) < efficiency:
        upper *= 1.5
    return float(brentq(lambda c: _location_efficiency(c, p) - efficiency, lower, upper, xtol=1e-10))


def _radial_distances(X: np.ndarray, location: np.ndarray, shape: np.ndarray) -> np.ndarray:
    precision = np.linalg.pinv(shape)
    d2 = mahalanobis_squared(X, location, precision)
    return np.sqrt(np.maximum(d2, 0.0))


def _solve_scale(
    radii: np.ndarray,
    *,
    initial: float,
    c: float,
    b: float,
    max_iter: int,
    tol: float,
) -> tuple[float, int, bool]:
    scale = max(float(initial), np.sqrt(_EPS))
    for iteration in range(1, max_iter + 1):
        ratio = float(np.mean(_bisquare_rho(radii / scale, c)) / b)
        new_scale = scale * np.sqrt(max(ratio, np.finfo(float).tiny))
        if abs(new_scale - scale) <= tol * max(scale, 1.0):
            return float(new_scale), iteration, True
        scale = float(new_scale)
    return scale, max_iter, False


def _weighted_location_shape(
    X: np.ndarray,
    weights: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    total = float(np.sum(weights))
    if (not np.isfinite(total) or total <= np.sqrt(_EPS)
            or np.count_nonzero(weights > 0) <= X.shape[1] + 1):
        raise np.linalg.LinAlgError("too few observations receive positive weight")
    location = np.sum(weights[:, None] * X, axis=0) / total
    centered = X - location
    scatter = (centered.T * weights) @ centered / total
    return location, _normalize_shape(scatter, ridge)


def _qn_columns(X: np.ndarray, *, finite_correction: bool = True) -> np.ndarray:
    values = np.array(
        [_qn_scale(X[:, j], finite_correction=finite_correction) for j in range(X.shape[1])],
        dtype=np.float64,
    )
    fallback = np.std(X, axis=0, ddof=1)
    values = np.where(np.isfinite(values) & (values > np.sqrt(_EPS)), values, fallback)
    return np.where(np.isfinite(values) & (values > np.sqrt(_EPS)), values, 1.0)


def _rank_correlation(Z: np.ndarray, *, normal_scores: bool) -> np.ndarray:
    n, p = Z.shape
    transformed = np.empty_like(Z)
    for j in range(p):
        ranks = rankdata(Z[:, j], method="average")
        if normal_scores:
            transformed[:, j] = norm.ppf((ranks - 0.5) / n)
        else:
            transformed[:, j] = ranks
    # Constant columns have undefined rank correlation.  Exclude them from
    # corrcoef and treat them as uncorrelated in this deterministic start; the
    # later SPD floor supplies their regularization without emitting warnings.
    spread = np.ptp(transformed, axis=0)
    active = np.isfinite(spread) & (spread > np.sqrt(_EPS))
    correlation = np.eye(p, dtype=np.float64)
    if np.count_nonzero(active) >= 2:
        active_correlation = np.asarray(
            np.corrcoef(transformed[:, active], rowvar=False), dtype=np.float64
        )
        correlation[np.ix_(active, active)] = active_correlation
    correlation = np.where(np.isfinite(correlation), correlation, 0.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def _gk_covariance(Z: np.ndarray) -> np.ndarray:
    p = Z.shape[1]
    scales = _qn_columns(Z)
    covariance = np.diag(scales**2)
    for j in range(p):
        for k in range(j + 1, p):
            plus = _qn_scale(Z[:, j] + Z[:, k])
            minus = _qn_scale(Z[:, j] - Z[:, k])
            value = 0.25 * (plus * plus - minus * minus)
            covariance[j, k] = covariance[k, j] = value
    return covariance


def _projection_scatter(Z: np.ndarray, basis: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    projected = Z @ basis
    scales = _qn_columns(projected)
    scatter = basis @ np.diag(scales**2) @ basis.T
    scatter = _symmetrize_spd(scatter, ridge)
    values, vectors = np.linalg.eigh(scatter)
    sqrt_scatter = vectors @ np.diag(np.sqrt(values)) @ vectors.T
    inverse_sqrt = vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T
    sphered = Z @ inverse_sqrt
    center = np.median(sphered, axis=0) @ sqrt_scatter
    return center, scatter


def _deterministic_initial_models(Z: np.ndarray, *, ridge: float) -> list[tuple[np.ndarray, np.ndarray]]:
    n, p = Z.shape
    candidates: list[np.ndarray] = []
    candidates.append(np.eye(p))

    spearman = _rank_correlation(Z, normal_scores=False)
    candidates.append(_symmetrize_spd(spearman, ridge))

    normal_scores = _rank_correlation(Z, normal_scores=True)
    candidates.append(_symmetrize_spd(normal_scores, ridge))

    candidates.append(_symmetrize_spd(_gk_covariance(Z), ridge))

    clipped = np.clip(Z, np.quantile(Z, 0.05, axis=0), np.quantile(Z, 0.95, axis=0))
    candidates.append(_symmetrize_spd(np.cov(clipped, rowvar=False), ridge))

    centered = Z - np.median(Z, axis=0)
    norms = np.linalg.norm(centered, axis=1)
    signs = centered / np.maximum(norms[:, None], np.sqrt(_EPS))
    sign_scatter = signs.T @ signs / n
    candidates.append(_symmetrize_spd(sign_scatter, ridge))

    models: list[tuple[np.ndarray, np.ndarray]] = []
    for candidate in candidates:
        _, basis = np.linalg.eigh(candidate)
        center, scatter = _projection_scatter(Z, basis, ridge)
        models.append((center, scatter))
    return models


def _initial_triplets(
    Z: np.ndarray,
    *,
    ridge: float,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    n, p = Z.shape
    h = int(np.ceil(n / 2.0))
    triplets = []
    for center, scatter in _deterministic_initial_models(Z, ridge=ridge):
        distances = mahalanobis_squared(Z, center, np.linalg.pinv(scatter))
        support = np.argsort(distances, kind="mergesort")[:h]
        subset = Z[support]
        location = subset.mean(axis=0)
        subset_scatter = _symmetrize_spd(np.cov(subset, rowvar=False), ridge)
        shape = _normalize_shape(subset_scatter, ridge)
        radii = _radial_distances(Z, location, shape)
        scale = float(np.median(radii))
        triplets.append((location, shape, max(scale, np.sqrt(_EPS))))
    return triplets


def _i_step(
    X: np.ndarray,
    location: np.ndarray,
    shape: np.ndarray,
    scale: float,
    *,
    c: float,
    b: float,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    radii = _radial_distances(X, location, shape)
    scale = scale * np.sqrt(max(float(np.mean(_bisquare_rho(radii / scale, c)) / b), np.finfo(float).tiny))
    radii = _radial_distances(X, location, shape)
    weights = _bisquare_weight(radii / scale, c)
    location, shape = _weighted_location_shape(X, weights, ridge=ridge)
    return location, shape, float(scale), weights


def _fully_refine_s(
    X: np.ndarray,
    triplet: tuple[np.ndarray, np.ndarray, float],
    *,
    c: float,
    b: float,
    ridge: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, int, bool, list[float]]:
    location, shape, scale = triplet
    history: list[float] = []
    converged = False
    for iteration in range(1, max_iter + 1):
        old_location = location.copy()
        old_shape = shape.copy()
        old_scale = scale
        location, shape, scale, weights = _i_step(
            X, location, shape, scale, c=c, b=b, ridge=ridge
        )
        radii = _radial_distances(X, location, shape)
        scale, _, _ = _solve_scale(
            radii, initial=scale, c=c, b=b, max_iter=30, tol=tol * 0.1
        )
        history.append(scale)
        change = max(
            np.linalg.norm(location - old_location) / max(np.linalg.norm(old_location), 1.0),
            np.linalg.norm(shape - old_shape, ord="fro") / max(np.linalg.norm(old_shape, ord="fro"), 1.0),
            abs(scale - old_scale) / max(old_scale, 1.0),
        )
        if change <= tol:
            converged = True
            break
    radii = _radial_distances(X, location, shape)
    weights = _bisquare_weight(radii / scale, c)
    return location, shape, scale, weights, iteration, converged, history


class DeterministicSEstimator(BaseRobustCovariance):
    """Deterministic multivariate S-estimator of location and scatter.

    Parameters
    ----------
    breakdown : float, default=0.5
        Target breakdown value for the Tukey-bisquare S-scale.  Values from
        0.05 to 0.5 are accepted.
    initial_steps : int, default=2
        Number of short I-steps applied to each deterministic start before the
        best candidates are selected.
    n_best : int, default=2
        Number of starts polished to convergence.
    """

    def __init__(
        self,
        *,
        breakdown: float = 0.5,
        initial_steps: int = 2,
        n_best: int = 2,
        max_iter: int = 100,
        tol: float = 1e-6,
        ridge: float = 1e-8,
        standardization: str = "qn",
        missing_values: str = "raise",
        tail_diagnostics: bool = True,
    ):
        super().__init__(
            assume_centered=False,
            store_precision=True,
            scale_correction="none",
            tail_diagnostics=tail_diagnostics,
            missing_values=missing_values,
        )
        if not (0.05 <= float(breakdown) <= 0.5):
            raise ValueError("breakdown must be in [0.05, 0.5]")
        if int(initial_steps) < 0:
            raise ValueError("initial_steps must be non-negative")
        if int(n_best) < 1:
            raise ValueError("n_best must be positive")
        if int(max_iter) < 1:
            raise ValueError("max_iter must be positive")
        if float(tol) <= 0:
            raise ValueError("tol must be positive")
        if float(ridge) < 0:
            raise ValueError("ridge must be non-negative")
        if standardization not in {"qn", "mad"}:
            raise ValueError("standardization must be 'qn' or 'mad'")
        if missing_values not in {"raise", "median"}:
            raise ValueError("missing_values must be 'raise' or 'median'")
        self.breakdown = float(breakdown)
        self.initial_steps = int(initial_steps)
        self.n_best = int(n_best)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.ridge = float(ridge)
        self.standardization = standardization
        self.missing_values = missing_values

    def fit(self, X, y=None):
        X = check_array(X, allow_nan=self.missing_values == "median")
        if self.missing_values == "median":
            X, self.impute_values_ = median_impute(X)
        self.n_samples_in_, self.n_features_in_ = X.shape
        if int(np.ceil(self.n_samples_in_ / 2.0)) <= self.n_features_in_:
            raise ValueError(
                "DetS requires ceil(n_samples / 2) > n_features; use MRCD or a regularized estimator when p is large"
            )

        Z, center, scale = _robust_standardize(
            X, method=self.standardization, finite_correction=True
        )
        self.standardization_center_ = center
        self.standardization_scale_ = scale
        c, b = _s_tuning_constant(self.n_features_in_, self.breakdown)
        self.tuning_constant_ = c
        self.rho_expectation_ = b

        candidates = _initial_triplets(Z, ridge=self.ridge)
        short = []
        for candidate in candidates:
            location, shape, candidate_scale = candidate
            weights = np.ones(self.n_samples_in_)
            for _ in range(self.initial_steps):
                location, shape, candidate_scale, weights = _i_step(
                    Z, location, shape, candidate_scale,
                    c=c, b=b, ridge=self.ridge,
                )
            radii = _radial_distances(Z, location, shape)
            candidate_scale, _, _ = _solve_scale(
                radii,
                initial=candidate_scale,
                c=c,
                b=b,
                max_iter=100,
                tol=self.tol * 0.1,
            )
            short.append((candidate_scale, location, shape, weights))

        short.sort(key=lambda item: item[0])
        polished = []
        for _, location, shape, _ in short[: min(self.n_best, len(short))]:
            radii = _radial_distances(Z, location, shape)
            initial_scale = float(np.median(radii))
            initial_scale, _, _ = _solve_scale(
                radii, initial=initial_scale, c=c, b=b, max_iter=100, tol=self.tol * 0.1
            )
            polished.append(
                _fully_refine_s(
                    Z,
                    (location, shape, initial_scale),
                    c=c,
                    b=b,
                    ridge=self.ridge,
                    max_iter=self.max_iter,
                    tol=self.tol,
                )
            )

        best = min(polished, key=lambda item: item[2])
        z_location, z_shape, z_scale, weights, n_iter, converged, history = best
        z_covariance = z_scale * z_scale * z_shape

        D = np.diag(scale)
        self.location_ = center + scale * z_location
        self.covariance_ = D @ z_covariance @ D
        self.covariance_ = _symmetrize_spd(self.covariance_, self.ridge)
        self.precision_ = np.linalg.pinv(self.covariance_)
        self.shape_ = self.covariance_ / np.exp(np.linalg.slogdet(self.covariance_)[1] / self.n_features_in_)
        self.scale_ = float(np.exp(np.linalg.slogdet(self.covariance_)[1] / (2.0 * self.n_features_in_)))
        self.distances_ = mahalanobis_squared(X, self.location_, self.precision_)
        self.weights_ = np.asarray(weights, dtype=np.float64)
        self.support_ = self.weights_ > 0
        self.n_iter_ = int(n_iter)
        self.converged_ = bool(converged)
        self.objective_history_ = np.asarray(history, dtype=np.float64)
        self.objective_value_ = self.scale_
        self.n_initial_models_ = len(candidates)
        if not self.converged_:
            warnings.warn("DetS did not converge", ConvergenceWarning)
        if self.tail_diagnostics:
            self.radial_kurtosis_ = radial_kurtosis(self.distances_, self.n_features_in_)
            self.tail_index_ = self.radial_kurtosis_
        return self


class DeterministicMMEstimator(BaseRobustCovariance):
    """Deterministic multivariate MM-estimator.

    A high-breakdown :class:`DeterministicSEstimator` supplies the fixed robust
    scale.  A second Tukey-bisquare IRLS fit then estimates a more efficient
    location and shape while retaining the starting estimator's breakdown
    protection under the usual MM conditions.
    """

    def __init__(
        self,
        *,
        breakdown: float = 0.5,
        efficiency: float = 0.95,
        initial_steps: int = 2,
        n_best: int = 2,
        max_iter: int = 100,
        tol: float = 1e-6,
        ridge: float = 1e-8,
        standardization: str = "qn",
        missing_values: str = "raise",
        tail_diagnostics: bool = True,
    ):
        super().__init__(
            assume_centered=False,
            store_precision=True,
            scale_correction="none",
            tail_diagnostics=tail_diagnostics,
            missing_values=missing_values,
        )
        if not (0.5 <= float(efficiency) < 1.0):
            raise ValueError("efficiency must be in [0.5, 1)")
        self.breakdown = float(breakdown)
        self.efficiency = float(efficiency)
        self.initial_steps = int(initial_steps)
        self.n_best = int(n_best)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.ridge = float(ridge)
        self.standardization = standardization
        self.missing_values = missing_values
        self.tail_diagnostics = tail_diagnostics

    def fit(self, X, y=None):
        X = check_array(X, allow_nan=self.missing_values == "median")
        if self.missing_values == "median":
            X, self.impute_values_ = median_impute(X)
        self.n_samples_in_, self.n_features_in_ = X.shape

        initial = DeterministicSEstimator(
            breakdown=self.breakdown,
            initial_steps=self.initial_steps,
            n_best=self.n_best,
            max_iter=self.max_iter,
            tol=self.tol,
            ridge=self.ridge,
            standardization=self.standardization,
            missing_values="raise",
            tail_diagnostics=False,
        ).fit(X)
        self.initial_estimator_ = initial
        self.initial_location_ = initial.location_.copy()
        self.initial_covariance_ = initial.covariance_.copy()
        self.s_scale_ = float(initial.scale_)
        self.s_tuning_constant_ = float(initial.tuning_constant_)
        c = _mm_tuning_constant(self.n_features_in_, self.efficiency)
        self.tuning_constant_ = c
        self.nominal_location_efficiency_ = _location_efficiency(c, self.n_features_in_)

        location = initial.location_.copy()
        shape = _normalize_shape(initial.covariance_, self.ridge)
        scale = self.s_scale_
        history: list[float] = []
        converged = False
        for iteration in range(1, self.max_iter + 1):
            old_location = location.copy()
            old_shape = shape.copy()
            radii = _radial_distances(X, location, shape)
            weights = _bisquare_weight(radii / scale, c)
            location, shape = _weighted_location_shape(X, weights, ridge=self.ridge)
            objective = float(np.mean(_bisquare_rho(_radial_distances(X, location, shape) / scale, c) / (c * c / 6.0)))
            history.append(objective)
            change = max(
                np.linalg.norm(location - old_location) / max(np.linalg.norm(old_location), 1.0),
                np.linalg.norm(shape - old_shape, ord="fro") / max(np.linalg.norm(old_shape, ord="fro"), 1.0),
            )
            if change <= self.tol:
                converged = True
                break

        self.location_ = location
        self.shape_ = shape
        self.scale_ = scale
        self.covariance_ = _symmetrize_spd(scale * scale * shape, self.ridge)
        self.precision_ = np.linalg.pinv(self.covariance_)
        self.distances_ = mahalanobis_squared(X, self.location_, self.precision_)
        radii = np.sqrt(np.maximum(self.distances_, 0.0))
        self.weights_ = _bisquare_weight(radii, c)
        self.support_ = self.weights_ > 0
        self.n_iter_ = int(iteration)
        self.converged_ = bool(converged)
        self.objective_history_ = np.asarray(history, dtype=np.float64)
        self.objective_value_ = history[-1] if history else float("nan")
        if not self.converged_:
            warnings.warn("DetMM did not converge", ConvergenceWarning)
        if self.tail_diagnostics:
            self.radial_kurtosis_ = radial_kurtosis(self.distances_, self.n_features_in_)
            self.tail_index_ = self.radial_kurtosis_
        return self


DetS = DeterministicSEstimator
DetMM = DeterministicMMEstimator


__all__ = [
    "DeterministicSEstimator",
    "DeterministicMMEstimator",
    "DetS",
    "DetMM",
]
