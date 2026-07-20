# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Minimum Regularized Covariance Determinant estimator.

The implementation follows the MRCD estimator of Boudt, Rousseeuw,
Vanduffel, and Verdonck.  It uses robust marginal standardization, a fixed
positive-definite target, automatic condition-number calibration, and
regularized concentration steps.  The global subset problem is approximate,
as it is for practical MCD implementations; this implementation uses a mix of
deterministic central starts and randomized starts rather than the six DetMCD
starts used by the reference R implementation.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import chi2, rankdata

from ._utils import check_array, mahalanobis_squared, median_impute, radial_kurtosis
from .covariance import BaseRobustCovariance, ConvergenceWarning


_EPS = np.finfo(np.float64).eps


def _qn_finite_sample_factor(n: int) -> float:
    """Classical finite-sample correction used with the Qn scale estimator."""
    small = {
        2: 0.399,
        3: 0.994,
        4: 0.512,
        5: 0.844,
        6: 0.611,
        7: 0.857,
        8: 0.669,
        9: 0.872,
    }
    if n in small:
        return small[n]
    if n % 2:
        return n / (n + 1.4)
    return n / (n + 3.8)


def _qn_scale(x: np.ndarray, *, finite_correction: bool = True) -> float:
    """Compute the exact Qn scale using pairwise absolute differences.

    This direct implementation is O(n^2) in memory and time.  MRCD is usually
    used in small- or moderate-sample, high-dimensional settings, where that
    tradeoff is acceptable and avoids an additional compiled dependency.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 2:
        return 0.0
    distances = pdist(x.reshape(-1, 1), metric="cityblock")
    h = n // 2 + 1
    k = h * (h - 1) // 2  # one-based order statistic
    value = float(np.partition(distances, k - 1)[k - 1])
    factor = 2.2219
    if finite_correction:
        factor *= _qn_finite_sample_factor(n)
    return factor * value


def _mad_scale(x: np.ndarray) -> float:
    median = np.median(x)
    return float(1.482602218505602 * np.median(np.abs(x - median)))


def _robust_standardize(
    X: np.ndarray,
    *,
    method: str,
    finite_correction: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.median(X, axis=0)
    if method == "qn":
        scale = np.array(
            [_qn_scale(X[:, j], finite_correction=finite_correction) for j in range(X.shape[1])],
            dtype=np.float64,
        )
    elif method == "mad":
        scale = np.array([_mad_scale(X[:, j]) for j in range(X.shape[1])], dtype=np.float64)
    else:
        raise ValueError("standardization must be 'qn' or 'mad'")

    fallback = np.std(X, axis=0, ddof=1)
    positive = scale[np.isfinite(scale) & (scale > 0)]
    fallback_positive = fallback[np.isfinite(fallback) & (fallback > 0)]
    if positive.size:
        reference = float(np.median(positive))
    elif fallback_positive.size:
        reference = float(np.median(fallback_positive))
    else:
        reference = 1.0
    # Use a relative floor.  An absolute unit-scale floor destroys scale
    # equivariance for legitimately small-valued data (for example, measurements
    # expressed in meters instead of nanometers).  Truly constant columns still
    # fall back to 1.0 so they standardize to zeros and can be regularized later.
    floor = max(np.sqrt(_EPS) * reference, np.finfo(float).tiny)
    scale = np.where(np.isfinite(scale) & (scale > floor), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > floor), scale, 1.0)
    U = (X - center) / scale
    return np.asarray(U, dtype=np.float64, order="C"), center, scale


def _equicorrelation_target(U: np.ndarray) -> tuple[np.ndarray, float]:
    p = U.shape[1]
    if p == 1:
        return np.ones((1, 1), dtype=np.float64), 0.0
    ranks = np.empty_like(U)
    for j in range(p):
        ranks[:, j] = rankdata(U[:, j], method="average")
    corr = np.corrcoef(ranks, rowvar=False)
    off_diagonal = corr[np.triu_indices(p, k=1)]
    off_diagonal = off_diagonal[np.isfinite(off_diagonal)]
    c = float(np.median(off_diagonal)) if off_diagonal.size else 0.0
    lower = -1.0 / (p - 1) + 1e-6
    c = float(np.clip(c, lower, 1.0 - 1e-6))
    target = c * np.ones((p, p), dtype=np.float64) + (1.0 - c) * np.eye(p)
    return target, c


def _validate_target(target, U: np.ndarray) -> tuple[np.ndarray, str, float | None]:
    p = U.shape[1]
    correlation = None
    if isinstance(target, str):
        name = target.lower()
        if name == "identity":
            matrix = np.eye(p, dtype=np.float64)
        elif name in {"equicorrelation", "equicorrelated"}:
            matrix, correlation = _equicorrelation_target(U)
            name = "equicorrelation"
        else:
            raise ValueError("target must be 'identity', 'equicorrelation', or an SPD array")
    else:
        matrix = np.asarray(target, dtype=np.float64)
        name = "custom"
        if matrix.shape != (p, p):
            raise ValueError(f"target must have shape {(p, p)}")
        if not np.isfinite(matrix).all():
            raise ValueError("target contains NaN or infinity")
        matrix = 0.5 * (matrix + matrix.T)
        average_variance = float(np.trace(matrix) / p)
        if not np.isfinite(average_variance) or average_variance <= 0:
            raise ValueError("target must have positive average diagonal")
        matrix = matrix / average_variance

    values = np.linalg.eigvalsh(matrix)
    if values[0] <= 0:
        raise ValueError("target must be positive definite")
    return matrix, name, correlation


def _consistency_factor(retained_fraction: float, p: int) -> float:
    if retained_fraction >= 1.0 - 1e-15:
        return 1.0
    quantile = chi2.ppf(retained_fraction, p)
    denominator = chi2.cdf(quantile, p + 2)
    if not np.isfinite(denominator) or denominator <= 0:
        return 1.0
    return float(retained_fraction / denominator)


def _target_whiten(U: np.ndarray, target: np.ndarray):
    values, vectors = np.linalg.eigh(target)
    inverse_sqrt = vectors @ np.diag(1.0 / np.sqrt(values))
    W = U @ inverse_sqrt
    return np.asarray(W, dtype=np.float64, order="C"), values, vectors


def _subset_singular_values(W: np.ndarray, support: np.ndarray):
    subset = W[support]
    center = subset.mean(axis=0)
    centered = subset - center
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    return center, centered, singular_values


def _rho_for_eigenvalues(eigenvalues: np.ndarray, max_condition_number: float) -> float:
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    maximum = float(np.max(eigenvalues)) if eigenvalues.size else 0.0
    minimum = float(np.min(eigenvalues)) if eigenvalues.size else 0.0
    minimum = max(minimum, 0.0)
    if maximum <= np.sqrt(_EPS):
        return 1e-8
    if minimum > 0 and maximum / minimum <= max_condition_number:
        return 0.0
    numerator = maximum - max_condition_number * minimum
    denominator = (max_condition_number - 1.0) + numerator
    if denominator <= 0:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0 - 1e-12))


def _required_rho(
    singular_values: np.ndarray,
    *,
    h: int,
    p: int,
    consistency_factor: float,
    max_condition_number: float,
) -> float:
    eigenvalues = consistency_factor * singular_values**2 / max(h - 1, 1)
    if eigenvalues.size < p:
        eigenvalues = np.concatenate([eigenvalues, np.zeros(p - eigenvalues.size)])
    return _rho_for_eigenvalues(eigenvalues, max_condition_number)


def _subset_model(
    W: np.ndarray,
    support: np.ndarray,
    *,
    rho: float,
    consistency_factor: float,
):
    h = support.size
    p = W.shape[1]
    center, centered, singular_values = _subset_singular_values(W, support)
    beta = (1.0 - rho) * consistency_factor
    eigenvalues = rho + beta * singular_values**2 / max(h - 1, 1)
    remaining = p - singular_values.size
    if np.any(eigenvalues <= 0) or (remaining > 0 and rho <= 0):
        return center, centered, singular_values, float("inf"), float("inf")
    logdet = float(np.log(eigenvalues).sum())
    if remaining > 0:
        logdet += remaining * float(np.log(rho))
    condition = float(np.max(eigenvalues) / min(np.min(eigenvalues), rho if remaining > 0 else np.inf))
    if remaining == 0:
        condition = float(np.max(eigenvalues) / np.min(eigenvalues))
    return center, centered, singular_values, logdet, condition


def _regularized_distances(
    W: np.ndarray,
    center: np.ndarray,
    centered_subset: np.ndarray,
    singular_values: np.ndarray,
    *,
    rho: float,
    consistency_factor: float,
) -> np.ndarray:
    h = centered_subset.shape[0]
    p = W.shape[1]
    Y = W - center
    _, _, vt = np.linalg.svd(centered_subset, full_matrices=False)
    beta = (1.0 - rho) * consistency_factor
    eigenvalues = rho + beta * singular_values**2 / max(h - 1, 1)
    coordinates = Y @ vt.T
    if rho > 0:
        projected_norm = np.einsum("ij,ij->i", coordinates, coordinates)
        total_norm = np.einsum("ij,ij->i", Y, Y)
        residual = np.maximum(total_norm - projected_norm, 0.0)
        distances = np.sum(coordinates**2 / eigenvalues, axis=1)
        if vt.shape[0] < p:
            distances += residual / rho
        return distances
    if vt.shape[0] < p or np.any(eigenvalues <= 0):
        return np.full(W.shape[0], np.inf)
    return np.sum(coordinates**2 / eigenvalues, axis=1)


def _central_supports(W: np.ndarray, h: int) -> list[np.ndarray]:
    median = np.median(W, axis=0)
    absolute = np.abs(W - median)
    score_sets = [
        np.einsum("ij,ij->i", absolute, absolute),
        np.sum(absolute, axis=1),
        np.max(absolute, axis=1),
    ]
    supports = []
    for scores in score_sets:
        supports.append(np.sort(np.argpartition(scores, h - 1)[:h]))
    return supports


def _random_projection_support(
    W: np.ndarray,
    h: int,
    rng: np.random.Generator,
) -> np.ndarray:
    p = W.shape[1]
    n_directions = min(max(3, int(np.ceil(np.log2(p + 1)))), 12)
    directions = rng.normal(size=(p, n_directions))
    norms = np.linalg.norm(directions, axis=0)
    directions /= np.where(norms > 0, norms, 1.0)
    projected = W @ directions
    center = np.median(projected, axis=0)
    scale = np.median(np.abs(projected - center), axis=0)
    scale = np.where(scale > np.sqrt(_EPS), scale, 1.0)
    outlyingness = np.max(np.abs(projected - center) / scale, axis=1)
    return np.sort(np.argpartition(outlyingness, h - 1)[:h])


def _initial_supports(
    W: np.ndarray,
    *,
    h: int,
    n_init: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    n, p = W.shape
    supports: list[np.ndarray] = []
    seen: set[bytes] = set()

    def add(support: np.ndarray):
        support = np.sort(np.asarray(support, dtype=np.int64))
        key = support.tobytes()
        if support.size == h and key not in seen:
            seen.add(key)
            supports.append(support)

    for support in _central_supports(W, h):
        add(support)

    attempts = 0
    max_attempts = max(20, 10 * n_init)
    while len(supports) < n_init and attempts < max_attempts:
        attempts += 1
        if attempts % 3:
            add(_random_projection_support(W, h, rng))
            continue

        elemental_size = min(h, p + 1)
        elemental = rng.choice(n, size=elemental_size, replace=False)
        subset = W[elemental]
        center = subset.mean(axis=0)
        covariance = np.cov(subset, rowvar=False, ddof=1)
        covariance = np.atleast_2d(covariance)
        trace = float(np.trace(covariance))
        ridge = max(trace / max(p, 1), 1.0) * 1e-6
        covariance = 0.5 * (covariance + covariance.T) + ridge * np.eye(p)
        precision = np.linalg.pinv(covariance, hermitian=True)
        centered = W - center
        scores = np.einsum("ij,jk,ik->i", centered, precision, centered)
        add(np.argpartition(scores, h - 1)[:h])

    if h == n:
        return [np.arange(n, dtype=np.int64)]
    fallback_attempts = 0
    while len(supports) < n_init and fallback_attempts < 20 * n_init:
        fallback_attempts += 1
        add(rng.choice(n, size=h, replace=False))
    return supports


def _c_steps(
    W: np.ndarray,
    support: np.ndarray,
    *,
    h: int,
    rho: float,
    consistency_factor: float,
    max_iter: int,
    tol: float,
):
    support = np.sort(np.asarray(support, dtype=np.int64))
    path: list[float] = []
    converged = False
    previous = np.inf
    for iteration in range(max_iter):
        center, centered, singular_values, logdet, condition = _subset_model(
            W,
            support,
            rho=rho,
            consistency_factor=consistency_factor,
        )
        path.append(logdet)
        distances = _regularized_distances(
            W,
            center,
            centered,
            singular_values,
            rho=rho,
            consistency_factor=consistency_factor,
        )
        new_support = np.sort(np.argpartition(distances, h - 1)[:h])
        if np.array_equal(new_support, support):
            converged = True
            break
        if np.isfinite(previous) and previous - logdet <= tol * max(1.0, abs(previous)):
            support = new_support
            converged = True
            break
        previous = logdet
        support = new_support

    center, centered, singular_values, logdet, condition = _subset_model(
        W,
        support,
        rho=rho,
        consistency_factor=consistency_factor,
    )
    if not path or path[-1] != logdet:
        path.append(logdet)
    return {
        "support": support,
        "center": center,
        "centered": centered,
        "singular_values": singular_values,
        "logdet": logdet,
        "condition": condition,
        "n_iter": min(max_iter, len(path)),
        "converged": converged,
        "path": np.asarray(path, dtype=np.float64),
    }


class MinimumRegularizedCovarianceDeterminant(BaseRobustCovariance):
    """Minimum Regularized Covariance Determinant (MRCD) estimator.

    MRCD searches for an ``h``-subset whose regularized covariance has a small
    determinant.  Regularization toward a positive-definite target keeps the
    estimate invertible when the feature dimension is close to or greater than
    the sample size.

    Parameters
    ----------
    support_fraction : float, default=0.75
        Fraction of observations retained in the raw subset. Must lie in
        ``[0.5, 1]``. Smaller values tolerate more rowwise contamination.
    contamination : float or None, default=None
        Alternative way to set ``support_fraction`` as ``1 - contamination``.
    target : {'identity', 'equicorrelation'} or array-like, default='identity'
        Positive-definite target in robustly standardized coordinates. Custom
        targets are symmetrized and normalized to have average diagonal one.
    regularization : {'auto'} or float, default='auto'
        Target weight ``rho``. Automatic calibration chooses the smallest
        regularization needed to bound the relative condition number.
    max_condition_number : float, default=50
        Maximum condition number of the target-whitened regularized covariance
        when ``regularization='auto'``.
    standardization : {'qn', 'mad'}, default='qn'
        Robust marginal scale estimator used before subset optimization.
    quality : {'fast', 'balanced', 'high'}, default='balanced'
        Multi-start search preset. Explicit search parameters override it.
    """

    _QUALITY_PRESETS = {
        "fast": {"n_init": 20, "n_best": 5, "initial_c_steps": 2, "max_iter": 50},
        "balanced": {"n_init": 60, "n_best": 10, "initial_c_steps": 2, "max_iter": 100},
        "high": {"n_init": 200, "n_best": 20, "initial_c_steps": 3, "max_iter": 150},
    }

    def __init__(
        self,
        support_fraction=None,
        contamination=None,
        *,
        target="identity",
        regularization="auto",
        max_condition_number=50.0,
        standardization="qn",
        finite_sample_correction=True,
        consistency_correction=True,
        quality="balanced",
        n_init=None,
        n_best=None,
        initial_c_steps=None,
        max_iter=None,
        tol=1e-7,
        random_state=0,
        tail_diagnostics=True,
        missing_values="raise",
    ):
        super().__init__(
            assume_centered=False,
            store_precision=True,
            scale_correction="none",
            tail_diagnostics=tail_diagnostics,
            missing_values=missing_values,
        )
        if quality not in self._QUALITY_PRESETS:
            raise ValueError("quality must be one of 'fast', 'balanced', or 'high'")
        if contamination is not None:
            contamination = float(contamination)
            if not (0.0 <= contamination < 0.5):
                raise ValueError("contamination must be in [0, 0.5)")
            if support_fraction is not None:
                raise ValueError("Specify either support_fraction or contamination, not both")
            support_fraction = 1.0 - contamination
        if support_fraction is None:
            support_fraction = 0.75
        support_fraction = float(support_fraction)
        if not (0.5 <= support_fraction <= 1.0):
            raise ValueError("support_fraction must be in [0.5, 1]")
        if isinstance(regularization, str):
            if regularization.lower() != "auto":
                raise ValueError("regularization must be 'auto' or a float in [0, 1]")
            regularization = "auto"
        else:
            regularization = float(regularization)
            if not (0.0 <= regularization <= 1.0):
                raise ValueError("regularization must be in [0, 1]")
        max_condition_number = float(max_condition_number)
        if not np.isfinite(max_condition_number) or max_condition_number <= 1.0:
            raise ValueError("max_condition_number must be greater than 1")
        if standardization not in {"qn", "mad"}:
            raise ValueError("standardization must be 'qn' or 'mad'")

        preset = self._QUALITY_PRESETS[quality]
        self.support_fraction = support_fraction
        self.contamination = contamination
        self.target = target
        self.regularization = regularization
        self.max_condition_number = max_condition_number
        self.standardization = standardization
        self.finite_sample_correction = bool(finite_sample_correction)
        self.consistency_correction = bool(consistency_correction)
        self.quality = quality
        self.n_init = preset["n_init"] if n_init is None else int(n_init)
        self.n_best = preset["n_best"] if n_best is None else int(n_best)
        self.initial_c_steps = preset["initial_c_steps"] if initial_c_steps is None else int(initial_c_steps)
        self.max_iter = preset["max_iter"] if max_iter is None else int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.tail_diagnostics = bool(tail_diagnostics)
        self.missing_values = missing_values
        if self.n_init < 1 or self.n_best < 1 or self.initial_c_steps < 0 or self.max_iter < 1:
            raise ValueError("search iteration counts must be positive")

    def fit(self, X, y=None):
        X = check_array(X, allow_nan=self.missing_values == "median")
        if self.missing_values == "median":
            X, self.impute_values_ = median_impute(X)
        elif self.missing_values != "raise":
            raise ValueError("missing_values must be 'raise' or 'median'")

        self.n_samples_in_, self.n_features_in_ = X.shape
        self.h_ = max(2, min(self.n_samples_in_, int(np.ceil(self.support_fraction * self.n_samples_in_))))
        self.effective_support_fraction_ = self.h_ / self.n_samples_in_
        self.support_fraction_ = self.effective_support_fraction_
        self.consistency_factor_ = (
            _consistency_factor(self.effective_support_fraction_, self.n_features_in_)
            if self.consistency_correction
            else 1.0
        )

        U, marginal_location, marginal_scale = _robust_standardize(
            X,
            method=self.standardization,
            finite_correction=self.finite_sample_correction,
        )
        target, target_name, target_correlation = _validate_target(self.target, U)
        W, target_eigenvalues, target_eigenvectors = _target_whiten(U, target)

        rng = np.random.default_rng(self.random_state)
        supports = _initial_supports(W, h=self.h_, n_init=self.n_init, rng=rng)
        required = []
        for support in supports:
            _, _, singular_values = _subset_singular_values(W, support)
            required.append(
                _required_rho(
                    singular_values,
                    h=self.h_,
                    p=self.n_features_in_,
                    consistency_factor=self.consistency_factor_,
                    max_condition_number=self.max_condition_number,
                )
            )
        required_array = np.asarray(required, dtype=np.float64)
        if self.regularization == "auto":
            maximum = float(np.max(required_array))
            if maximum <= 0.1:
                rho = maximum
            else:
                rho = max(0.1, float(np.median(required_array)))
        else:
            rho = float(self.regularization)

        if rho == 0.0 and self.n_features_in_ >= self.h_:
            raise ValueError(
                "regularization=0 produces a singular subset covariance when p >= h; "
                "use regularization='auto' or a positive value"
            )

        eligible = [
            support for support, needed in zip(supports, required_array)
            if needed <= rho + 1e-12
        ]
        if not eligible:
            eligible = [supports[int(np.argmin(required_array))]]
            if self.regularization == "auto":
                rho = float(np.min(required_array))

        short_results = [
            _c_steps(
                W,
                support,
                h=self.h_,
                rho=rho,
                consistency_factor=self.consistency_factor_,
                max_iter=max(1, self.initial_c_steps),
                tol=self.tol,
            )
            for support in eligible
        ]
        short_results.sort(key=lambda item: item["logdet"])
        finalists = short_results[: min(self.n_best, len(short_results))]
        polished = [
            _c_steps(
                W,
                item["support"],
                h=self.h_,
                rho=rho,
                consistency_factor=self.consistency_factor_,
                max_iter=self.max_iter,
                tol=self.tol,
            )
            for item in finalists
        ]
        best = min(polished, key=lambda item: item["logdet"])

        support_indices = best["support"]
        support = np.zeros(self.n_samples_in_, dtype=bool)
        support[support_indices] = True
        subset_U = U[support]
        mean_U = subset_U.mean(axis=0)
        covariance_U = np.cov(subset_U, rowvar=False, ddof=1)
        covariance_U = np.atleast_2d(covariance_U)
        covariance_U = self.consistency_factor_ * covariance_U
        regularized_U = rho * target + (1.0 - rho) * covariance_U
        regularized_U = 0.5 * (regularized_U + regularized_U.T)

        D = np.diag(marginal_scale)
        location = marginal_location + marginal_scale * mean_U
        covariance = D @ regularized_U @ D
        covariance = 0.5 * (covariance + covariance.T)
        precision = np.linalg.inv(covariance)
        distances = mahalanobis_squared(X, location, precision)

        target_covariance = D @ target @ D
        relative_covariance = np.atleast_2d(
            np.cov(W[support], rowvar=False, ddof=1)
        )
        relative_values = np.linalg.eigvalsh(
            rho * np.eye(self.n_features_in_)
            + (1.0 - rho)
            * self.consistency_factor_
            * relative_covariance
        )
        relative_values = np.maximum(
            np.asarray(relative_values, dtype=np.float64), np.finfo(float).tiny
        )

        self.location_ = location
        self.covariance_ = covariance
        self.shape_ = covariance
        self.precision_ = precision
        self.distances_ = distances
        self.support_ = support
        self.raw_location_ = location.copy()
        self.raw_covariance_ = covariance.copy()
        self.raw_distances_ = distances.copy()
        self.raw_support_ = support.copy()
        self.target_ = target
        self.target_name_ = target_name
        self.target_correlation_ = target_correlation
        self.target_covariance_ = target_covariance
        self.target_eigenvalues_ = target_eigenvalues
        self.target_eigenvectors_ = target_eigenvectors
        self.marginal_location_ = marginal_location
        self.marginal_scale_ = marginal_scale
        self.regularization_ = rho
        self.initial_regularizations_ = required_array
        self.log_objective_ = float(best["logdet"] / self.n_features_in_)
        self.objective_value_ = float(np.exp(self.log_objective_))
        self.objective_path_ = best["path"] / self.n_features_in_
        self.n_iter_ = int(best["n_iter"])
        self.converged_ = bool(best["converged"])
        self.standardized_condition_number_ = float(relative_values[-1] / relative_values[0])
        self.condition_number_ = float(np.linalg.cond(covariance))
        self.det_ = float(np.linalg.det(covariance))
        self.raw_det_ = self.det_
        self.raw_scale_ = 1.0
        self.scale_ = 1.0
        if not self.converged_:
            warnings.warn("MRCD concentration steps did not converge", ConvergenceWarning)
        if self.tail_diagnostics:
            self.radial_kurtosis_ = radial_kurtosis(self.distances_, self.n_features_in_)
            self.tail_index_ = self.radial_kurtosis_
        return self


MRCD = MinimumRegularizedCovarianceDeterminant
MinRegularizedCovDet = MinimumRegularizedCovarianceDeterminant


__all__ = [
    "MinimumRegularizedCovarianceDeterminant",
    "MRCD",
    "MinRegularizedCovDet",
]
