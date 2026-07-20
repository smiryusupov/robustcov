# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Kernel Minimum Regularized Covariance Determinant estimator.

This module implements the KMRCD subset objective and kernel C-steps of
Schreurs et al. (2021).  The search is approximate: it combines deterministic
kernel-central starts with randomized h-subsets instead of reproducing the four
refined initial estimators from the reference MATLAB implementation.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import norm

from ._estimator import EstimatorMixin
from ._utils import check_array, median_impute
from .covariance import ConvergenceWarning
from .mrcd import _robust_standardize

_EPS = np.finfo(np.float64).eps


def _resolve_h(n: int, support_fraction: float | None, contamination: float | None) -> int:
    if contamination is not None:
        if support_fraction is not None:
            raise ValueError("Specify either support_fraction or contamination, not both")
        contamination = float(contamination)
        if not 0.0 <= contamination < 0.5:
            raise ValueError("contamination must be in [0, 0.5)")
        support_fraction = 1.0 - contamination
    if support_fraction is None:
        support_fraction = 0.75
    support_fraction = float(support_fraction)
    if not 0.5 <= support_fraction <= 1.0:
        raise ValueError("support_fraction must be in [0.5, 1]")
    return max(2, min(n, int(np.ceil(support_fraction * n))))


def _rbf_gamma_from_median(X: np.ndarray) -> tuple[float, float]:
    distances = pdist(X, metric="sqeuclidean")
    positive = distances[np.isfinite(distances) & (distances > 0)]
    sigma2 = float(np.median(positive)) if positive.size else 1.0
    sigma2 = max(sigma2, np.finfo(np.float64).tiny)
    return 1.0 / (2.0 * sigma2), sigma2


def _kernel_matrix(
    X: np.ndarray,
    Y: np.ndarray | None,
    *,
    kernel: str | Callable,
    gamma: float | None,
    degree: int,
    coef0: float,
) -> np.ndarray:
    if callable(kernel):
        value = kernel(X, X if Y is None else Y)
        return np.asarray(value, dtype=np.float64)
    name = str(kernel).lower()
    Yv = X if Y is None else Y
    if name == "linear":
        return np.asarray(X @ Yv.T, dtype=np.float64)
    if name in {"rbf", "gaussian"}:
        distances = cdist(X, Yv, metric="sqeuclidean")
        return np.exp(-float(gamma) * distances)
    if name in {"poly", "polynomial"}:
        return np.power(float(gamma) * (X @ Yv.T) + float(coef0), int(degree))
    raise ValueError("kernel must be 'linear', 'rbf', 'polynomial', 'precomputed', or callable")


def _validate_psd_kernel(K: np.ndarray) -> tuple[np.ndarray, float]:
    K = np.asarray(K, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("precomputed kernel must be a square matrix")
    if not np.isfinite(K).all():
        raise ValueError("kernel matrix contains NaN or infinity")
    K = 0.5 * (K + K.T)
    values, vectors = np.linalg.eigh(K)
    scale = max(float(np.max(np.abs(values))), np.finfo(np.float64).tiny)
    if values[0] < -1e-8 * scale:
        raise ValueError("kernel matrix is not positive semidefinite")
    clipped = np.maximum(values, 0.0)
    correction = float(np.max(np.abs(clipped - values)))
    if correction:
        K = (vectors * clipped) @ vectors.T
        K = 0.5 * (K + K.T)
    return K, correction


def _center_subset_kernel(K_hh: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    row_mean = np.mean(K_hh, axis=1)
    grand_mean = float(np.mean(row_mean))
    centered = K_hh - row_mean[:, None] - row_mean[None, :] + grand_mean
    centered = 0.5 * (centered + centered.T)
    return centered, row_mean, grand_mean


def _required_rho_kernel(eigenvalues: np.ndarray, h: int, max_condition_number: float) -> float:
    maximum = max(float(np.max(eigenvalues)), 0.0)
    minimum = max(float(np.min(eigenvalues)), 0.0)
    a = float(h - 1)
    if maximum <= np.sqrt(_EPS):
        return 1e-8
    current = np.inf if minimum <= 0 else maximum / minimum
    if current <= max_condition_number:
        return 1e-8
    numerator = maximum - max_condition_number * minimum
    denominator = (max_condition_number - 1.0) * a + numerator
    if denominator <= 0:
        return 1e-8
    return float(np.clip(numerator / denominator, 1e-8, 1.0 - 1e-10))


def _subset_fit(K: np.ndarray, support: np.ndarray, rho: float):
    K_hh = K[np.ix_(support, support)]
    centered, row_mean, grand_mean = _center_subset_kernel(K_hh)
    h = support.size
    reg = (1.0 - rho) * centered + (h - 1.0) * rho * np.eye(h)
    sign, logdet = np.linalg.slogdet(reg)
    if sign <= 0 or not np.isfinite(logdet):
        return None
    values = np.linalg.eigvalsh(reg)
    condition = float(values[-1] / max(values[0], np.finfo(float).tiny))
    return {
        "kernel_hh": K_hh,
        "centered_kernel": centered,
        "row_mean": row_mean,
        "grand_mean": grand_mean,
        "regularized_kernel": reg,
        "logdet": float(logdet),
        "condition": condition,
    }


def _distances_from_subset(
    K_cross_hn: np.ndarray,
    kernel_diag: np.ndarray,
    model: dict,
    rho: float,
) -> np.ndarray:
    # K_cross_hn contains k(x_i, x) for subset rows i and evaluation points x.
    mean_hx = np.mean(K_cross_hn, axis=0)
    kvec = (
        K_cross_hn
        - model["row_mean"][:, None]
        - mean_hx[None, :]
        + model["grand_mean"]
    )
    diag_centered = kernel_diag - 2.0 * mean_hx + model["grand_mean"]
    solved = np.linalg.solve(model["regularized_kernel"], kvec)
    quadratic = np.sum(kvec * solved, axis=0)
    distances = (diag_centered - (1.0 - rho) * quadratic) / rho
    return np.maximum(np.asarray(distances, dtype=np.float64), 0.0)


def _kernel_spatial_median_scores(K: np.ndarray, max_iter: int = 200, tol: float = 1e-9):
    n = K.shape[0]
    weights = np.full(n, 1.0 / n)
    diag = np.diag(K)
    for _ in range(max_iter):
        quadratic = float(weights @ K @ weights)
        d2 = np.maximum(diag - 2.0 * (K @ weights) + quadratic, 0.0)
        inverse = 1.0 / np.maximum(np.sqrt(d2), 1e-10)
        updated = inverse / inverse.sum()
        if np.linalg.norm(updated - weights, ord=1) <= tol:
            weights = updated
            break
        weights = updated
    quadratic = float(weights @ K @ weights)
    return np.maximum(diag - 2.0 * (K @ weights) + quadratic, 0.0)


def _initial_supports(K: np.ndarray, h: int, n_init: int, rng: np.random.Generator):
    n = K.shape[0]
    supports: list[np.ndarray] = []
    seen: set[bytes] = set()

    def add(indices):
        indices = np.sort(np.asarray(indices, dtype=np.int64))
        if indices.size != h:
            return
        key = indices.tobytes()
        if key not in seen:
            seen.add(key)
            supports.append(indices)

    diag = np.diag(K)
    global_mean_dist = np.maximum(diag - 2.0 * K.mean(axis=1) + K.mean(), 0.0)
    add(np.argpartition(global_mean_dist, h - 1)[:h])
    add(np.argpartition(_kernel_spatial_median_scores(K), h - 1)[:h])

    pairwise_d2 = np.maximum(diag[:, None] + diag[None, :] - 2.0 * K, 0.0)
    medoid_scores = np.median(pairwise_d2, axis=1)
    add(np.argpartition(medoid_scores, h - 1)[:h])

    attempts = 0
    while len(supports) < n_init and attempts < max(100, 30 * n_init):
        attempts += 1
        if attempts % 2:
            anchors = rng.choice(n, size=min(4, n), replace=False)
            score = np.min(pairwise_d2[:, anchors], axis=1)
            score += 0.05 * rng.random(n) * max(float(np.median(score)), 1.0)
            add(np.argpartition(score, h - 1)[:h])
        else:
            add(rng.choice(n, size=h, replace=False))
    return supports


def _c_steps(K: np.ndarray, support: np.ndarray, h: int, rho: float, max_iter: int, tol: float):
    support = np.sort(np.asarray(support, dtype=np.int64))
    path: list[float] = []
    converged = False
    final_model = None
    diag = np.diag(K)
    for iteration in range(max_iter):
        model = _subset_fit(K, support, rho)
        if model is None:
            break
        path.append(model["logdet"])
        cross = K[np.ix_(support, np.arange(K.shape[0]))]
        distances = _distances_from_subset(cross, diag, model, rho)
        new_support = np.sort(np.argpartition(distances, h - 1)[:h])
        if np.array_equal(new_support, support):
            converged = True
            final_model = model
            break
        new_model = _subset_fit(K, new_support, rho)
        if new_model is None or new_model["logdet"] > model["logdet"] + max(tol, 1e-9):
            final_model = model
            converged = True
            break
        if abs(model["logdet"] - new_model["logdet"]) <= tol:
            support = new_support
            final_model = new_model
            path.append(new_model["logdet"])
            converged = True
            break
        support = new_support
        final_model = new_model
    if final_model is None:
        final_model = _subset_fit(K, support, rho)
    return {
        "support": support,
        "model": final_model,
        "path": np.asarray(path, dtype=np.float64),
        "logdet": float(final_model["logdet"]) if final_model is not None else np.inf,
        "n_iter": iteration + 1 if max_iter else 0,
        "converged": converged,
    }


def _univariate_mcd_location_scale(values: np.ndarray, h: int) -> tuple[float, float]:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    h = min(max(2, h), ordered.size)
    csum = np.concatenate([[0.0], np.cumsum(ordered)])
    csum2 = np.concatenate([[0.0], np.cumsum(ordered**2)])
    sums = csum[h:] - csum[:-h]
    sums2 = csum2[h:] - csum2[:-h]
    variances = np.maximum((sums2 - sums**2 / h) / max(h - 1, 1), 0.0)
    start = int(np.argmin(variances))
    subset = ordered[start : start + h]
    location = float(np.mean(subset))
    scale = float(np.std(subset, ddof=1))
    if not np.isfinite(scale) or scale <= np.sqrt(_EPS):
        scale = float(1.482602218505602 * np.median(np.abs(values - np.median(values))))
    return location, max(scale, np.sqrt(_EPS))


class KernelMinimumRegularizedCovarianceDeterminant(EstimatorMixin):
    """Robust kernel-space subset estimator for nonlinear outlier detection.

    Parameters
    ----------
    kernel : {'rbf', 'linear', 'polynomial', 'precomputed'} or callable
        Positive-semidefinite kernel.  A callable receives ``(X, Y)``.
    gamma : {'median', 'scale'} or float, default='median'
        Kernel coefficient.  For the RBF kernel, ``'median'`` implements the
        KMRCD paper's median squared-distance bandwidth heuristic.
    support_fraction, contamination : float, optional
        Size of the retained h-subset.  Specify at most one.
    regularization : {'auto'} or float, default='auto'
        Target weight rho in feature space.  It must be strictly positive.
    """

    def __init__(
        self,
        *,
        kernel: str | Callable = "rbf",
        gamma: str | float = "median",
        degree: int = 3,
        coef0: float = 1.0,
        support_fraction: float | None = None,
        contamination: float | None = None,
        regularization: str | float = "auto",
        max_condition_number: float = 50.0,
        standardization: str = "qn",
        finite_sample_correction: bool = True,
        n_init: int = 20,
        n_best: int = 5,
        initial_c_steps: int = 2,
        max_iter: int = 50,
        tol: float = 1e-8,
        cutoff_quantile: float = 0.995,
        random_state: int | None = 0,
        missing_values: str = "raise",
    ):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.support_fraction = support_fraction
        self.contamination = contamination
        self.regularization = regularization
        self.max_condition_number = max_condition_number
        self.standardization = standardization
        self.finite_sample_correction = finite_sample_correction
        self.n_init = n_init
        self.n_best = n_best
        self.initial_c_steps = initial_c_steps
        self.max_iter = max_iter
        self.tol = tol
        self.cutoff_quantile = cutoff_quantile
        self.random_state = random_state
        self.missing_values = missing_values

    def _validate_parameters(self):
        if isinstance(self.regularization, str):
            if self.regularization != "auto":
                raise ValueError("regularization must be 'auto' or a float in (0, 1)")
        elif not 0.0 < float(self.regularization) < 1.0:
            raise ValueError("regularization must be strictly between 0 and 1")
        if float(self.max_condition_number) <= 1.0:
            raise ValueError("max_condition_number must be greater than 1")
        if int(self.n_init) < 1 or int(self.n_best) < 1:
            raise ValueError("n_init and n_best must be positive")
        if int(self.initial_c_steps) < 1 or int(self.max_iter) < 1:
            raise ValueError("initial_c_steps and max_iter must be positive")
        if not 0.5 < float(self.cutoff_quantile) < 1.0:
            raise ValueError("cutoff_quantile must be in (0.5, 1)")
        if int(self.degree) < 1:
            raise ValueError("degree must be positive")

    def fit(self, X, y=None):
        self._validate_parameters()
        precomputed = isinstance(self.kernel, str) and self.kernel.lower() == "precomputed"
        if precomputed:
            K, correction = _validate_psd_kernel(np.asarray(X, dtype=np.float64))
            self.n_samples_in_ = K.shape[0]
            self.n_features_in_ = None
            self.X_fit_standardized_ = None
            self.marginal_location_ = None
            self.marginal_scale_ = None
            self.gamma_ = None
            self.bandwidth_squared_ = None
        else:
            X = check_array(X, allow_nan=self.missing_values == "median")
            if self.missing_values == "median":
                X, self.impute_values_ = median_impute(X)
            elif self.missing_values != "raise":
                raise ValueError("missing_values must be 'raise' or 'median'")
            self.n_samples_in_, self.n_features_in_ = X.shape
            if self.standardization == "none":
                U = np.asarray(X, dtype=np.float64, order="C")
                center = np.zeros(self.n_features_in_)
                scale = np.ones(self.n_features_in_)
            else:
                U, center, scale = _robust_standardize(
                    X,
                    method=self.standardization,
                    finite_correction=self.finite_sample_correction,
                )
            self.X_fit_standardized_ = U
            self.marginal_location_ = center
            self.marginal_scale_ = scale
            if isinstance(self.gamma, str):
                if self.gamma == "median":
                    gamma, sigma2 = _rbf_gamma_from_median(U)
                elif self.gamma == "scale":
                    variance = float(np.var(U))
                    gamma = 1.0 / max(
                        self.n_features_in_ * variance, np.finfo(np.float64).tiny
                    )
                    sigma2 = 1.0 / (2.0 * gamma)
                else:
                    raise ValueError("gamma must be 'median', 'scale', or a positive float")
            else:
                gamma = float(self.gamma)
                if gamma <= 0:
                    raise ValueError("gamma must be positive")
                sigma2 = 1.0 / (2.0 * gamma)
            self.gamma_ = gamma
            self.bandwidth_squared_ = sigma2
            K = _kernel_matrix(
                U,
                None,
                kernel=self.kernel,
                gamma=gamma,
                degree=self.degree,
                coef0=self.coef0,
            )
            K, correction = _validate_psd_kernel(K)

        self.kernel_matrix_ = K
        self.kernel_psd_correction_ = correction
        self.h_ = _resolve_h(self.n_samples_in_, self.support_fraction, self.contamination)
        self.support_fraction_ = self.h_ / self.n_samples_in_
        rng = np.random.default_rng(self.random_state)
        supports = _initial_supports(K, self.h_, int(self.n_init), rng)

        required = []
        for support in supports:
            centered, _, _ = _center_subset_kernel(K[np.ix_(support, support)])
            required.append(
                _required_rho_kernel(
                    np.linalg.eigvalsh(centered),
                    self.h_,
                    float(self.max_condition_number),
                )
            )
        required = np.asarray(required, dtype=np.float64)
        if self.regularization == "auto":
            maximum = float(np.max(required))
            rho = maximum if maximum <= 0.1 else max(0.1, float(np.median(required)))
        else:
            rho = float(self.regularization)
        eligible = [s for s, needed in zip(supports, required) if needed <= rho + 1e-12]
        if not eligible:
            best_idx = int(np.argmin(required))
            eligible = [supports[best_idx]]
            if self.regularization == "auto":
                rho = float(required[best_idx])

        screened = [
            _c_steps(K, s, self.h_, rho, int(self.initial_c_steps), float(self.tol))
            for s in eligible
        ]
        screened.sort(key=lambda item: item["logdet"])
        finalists = screened[: min(int(self.n_best), len(screened))]
        polished = [
            _c_steps(K, item["support"], self.h_, rho, int(self.max_iter), float(self.tol))
            for item in finalists
        ]
        best = min(polished, key=lambda item: item["logdet"])
        if best["model"] is None:
            raise RuntimeError("KMRCD failed to construct a positive-definite subset model")

        support_mask = np.zeros(self.n_samples_in_, dtype=bool)
        support_mask[best["support"]] = True
        self.support_indices_ = best["support"]
        self.support_ = support_mask
        self.raw_support_ = support_mask.copy()
        self.regularization_ = float(rho)
        self.initial_regularizations_ = required
        self.regularized_kernel_ = best["model"]["regularized_kernel"]
        self.centered_support_kernel_ = best["model"]["centered_kernel"]
        self.support_kernel_row_mean_ = best["model"]["row_mean"]
        self.support_kernel_grand_mean_ = best["model"]["grand_mean"]
        self.standardized_condition_number_ = best["model"]["condition"]
        self.log_objective_ = float(best["logdet"] / self.h_)
        self.objective_value_ = float(np.exp(self.log_objective_))
        self.objective_path_ = best["path"] / self.h_
        self.n_iter_ = int(best["n_iter"])
        self.converged_ = bool(best["converged"])
        if not self.converged_:
            warnings.warn("KMRCD concentration steps did not converge", ConvergenceWarning)

        self.distances_ = self._mahalanobis_from_kernel(K, np.diag(K))
        self.raw_distances_ = self.distances_.copy()
        robust_distance = np.sqrt(self.distances_)
        log_distance = np.log(0.1 + robust_distance)
        cutoff_location, cutoff_scale = _univariate_mcd_location_scale(log_distance, self.h_)
        cutoff_log = cutoff_location + norm.ppf(self.cutoff_quantile) * cutoff_scale
        self.cutoff_location_ = cutoff_location
        self.cutoff_scale_ = cutoff_scale
        self.robust_distance_threshold_ = float(np.exp(cutoff_log) - 0.1)
        self.distance_threshold_ = self.robust_distance_threshold_**2
        self.outlier_mask_ = self.distances_ > self.distance_threshold_
        return self

    def _standardize_new(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X, allow_nan=self.missing_values == "median")
        if self.missing_values == "median":
            X = np.where(np.isnan(X), self.impute_values_, X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have {self.n_features_in_} features")
        return (X - self.marginal_location_) / self.marginal_scale_

    def _mahalanobis_from_kernel(self, K_new_train: np.ndarray, kernel_diag: np.ndarray) -> np.ndarray:
        K_new_train = np.asarray(K_new_train, dtype=np.float64)
        if K_new_train.ndim != 2 or K_new_train.shape[1] != self.n_samples_in_:
            raise ValueError("cross-kernel matrix must have shape (n_new, n_training)")
        kernel_diag = np.asarray(kernel_diag, dtype=np.float64)
        if kernel_diag.shape != (K_new_train.shape[0],):
            raise ValueError("kernel_diag must contain one self-kernel value per row")
        cross_hn = K_new_train[:, self.support_indices_].T
        model = {
            "row_mean": self.support_kernel_row_mean_,
            "grand_mean": self.support_kernel_grand_mean_,
            "regularized_kernel": self.regularized_kernel_,
        }
        return _distances_from_subset(cross_hn, kernel_diag, model, self.regularization_)

    def mahalanobis(self, X, *, kernel_diag=None):
        if not hasattr(self, "support_indices_"):
            raise RuntimeError("Estimator is not fitted")
        precomputed = isinstance(self.kernel, str) and self.kernel.lower() == "precomputed"
        if precomputed:
            if kernel_diag is None:
                raise ValueError("kernel_diag is required when kernel='precomputed'")
            return self._mahalanobis_from_kernel(np.asarray(X, dtype=np.float64), kernel_diag)
        U = self._standardize_new(X)
        cross = _kernel_matrix(
            U,
            self.X_fit_standardized_,
            kernel=self.kernel,
            gamma=self.gamma_,
            degree=self.degree,
            coef0=self.coef0,
        )
        if callable(self.kernel):
            diag = np.diag(_kernel_matrix(U, None, kernel=self.kernel, gamma=self.gamma_, degree=self.degree, coef0=self.coef0))
        elif str(self.kernel).lower() in {"rbf", "gaussian"}:
            diag = np.ones(U.shape[0])
        elif str(self.kernel).lower() == "linear":
            diag = np.einsum("ij,ij->i", U, U)
        else:
            diag = np.power(self.gamma_ * np.einsum("ij,ij->i", U, U) + self.coef0, self.degree)
        return self._mahalanobis_from_kernel(cross, diag)

    def score_samples(self, X, *, kernel_diag=None):
        return -0.5 * self.mahalanobis(X, kernel_diag=kernel_diag)

    def decision_function(self, X, *, kernel_diag=None):
        return self.distance_threshold_ - self.mahalanobis(X, kernel_diag=kernel_diag)

    def predict(self, X, *, kernel_diag=None):
        return np.where(self.mahalanobis(X, kernel_diag=kernel_diag) <= self.distance_threshold_, 1, -1)

    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X, kernel_diag=np.diag(X) if isinstance(self.kernel, str) and self.kernel.lower() == "precomputed" else None)


KernelMRCD = KernelMinimumRegularizedCovarianceDeterminant
KMRCD = KernelMinimumRegularizedCovarianceDeterminant

__all__ = [
    "KernelMinimumRegularizedCovarianceDeterminant",
    "KernelMRCD",
    "KMRCD",
]
