# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Classical and robust second-order blind identification (SOBI)."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from ._estimator import EstimatorMixin
from ._utils import check_array
from .joint_diagonalization import canonicalize_unmixing, joint_diagonalize_symmetric
from .m_estimators import StudentTScatter


def _resolve_lags(lags, n_samples: int) -> np.ndarray:
    if lags is None:
        upper = max(1, min(12, n_samples // 10))
        result = np.arange(1, upper + 1, dtype=int)
    elif isinstance(lags, (int, np.integer)) and not isinstance(lags, (bool, np.bool_)):
        if int(lags) < 1:
            raise ValueError("integer lags must be positive")
        result = np.arange(1, int(lags) + 1, dtype=int)
    else:
        result = np.asarray(list(lags), dtype=int)
        if result.ndim != 1 or result.size == 0:
            raise ValueError("lags must be a non-empty one-dimensional collection")
    result = np.unique(result)
    if np.any(result < 1) or np.any(result >= n_samples):
        raise ValueError("every lag must be between 1 and n_samples - 1")
    return result


def _weighted_lag_scatter(
    whitened: np.ndarray,
    lag: int,
    *,
    weighting: str,
    tuning: float,
) -> np.ndarray:
    current = whitened[lag:]
    previous = whitened[:-lag]
    if weighting == "none":
        cross = current.T @ previous / current.shape[0]
        return 0.5 * (cross + cross.T)

    radial = np.sqrt(
        np.sum(current * current, axis=1) + np.sum(previous * previous, axis=1)
    )
    positive = radial[radial > np.finfo(np.float64).tiny]
    if positive.size == 0:
        raise ValueError("lagged scatter is undefined for coincident observations")
    scale = float(np.median(positive))
    standardized = radial / max(scale, np.finfo(np.float64).tiny)
    if weighting == "huber":
        weights = np.minimum(1.0, tuning / np.maximum(standardized, 1e-300))
    elif weighting == "tukey":
        ratio = standardized / tuning
        weights = np.where(ratio < 1.0, (1.0 - ratio * ratio) ** 2, 0.0)
    else:
        raise ValueError("lag_weighting must be 'none', 'huber', or 'tukey'")
    weight_sum = float(np.sum(weights))
    if weight_sum <= np.finfo(np.float64).tiny:
        raise ValueError("all robust lag weights are zero; increase lag_tuning")
    current_location = np.sum(weights[:, None] * current, axis=0) / weight_sum
    previous_location = np.sum(weights[:, None] * previous, axis=0) / weight_sum
    current_centered = current - current_location
    previous_centered = previous - previous_location
    cross = current_centered.T @ (weights[:, None] * previous_centered) / weight_sum
    return 0.5 * (cross + cross.T)


class SOBI(EstimatorMixin):
    """Second-order blind identification for temporally correlated sources.

    SOBI whitens a multivariate time series and jointly diagonalizes several
    lagged covariance matrices.  Sources must have distinguishable temporal
    autocorrelation signatures; unlike ordinary ICA, Gaussian sources can be
    separated when those signatures differ.

    Parameters
    ----------
    n_components : int or None, default=None
        Number of sources to retain.
    lags : int, iterable of int, or None, default=None
        Positive time lags.  An integer ``m`` means lags ``1..m``.  ``None``
        uses up to twelve short lags.
    whitening_estimator : object or None, default=None
        Optional scatter estimator used for centering and whitening.  ``None``
        uses the empirical mean and covariance.
    center : {'mean', 'median'}, default='mean'
        Center used when no whitening estimator is supplied.
    lag_weighting : {'none', 'huber', 'tukey'}, default='none'
        Robust weighting applied to each lagged scatter.
    lag_tuning : float, default=2.5
        Huber or Tukey cutoff in units of the median pair radius.
    max_sweeps : int, default=100
        Maximum Jacobi joint-diagonalization sweeps.
    tol : float, default=1e-10
        Joint-diagonalization convergence tolerance.
    eigenvalue_floor : float, default=1e-12
        Relative whitening eigenvalue floor.
    backend : {'auto', 'python', 'cpp'}, default='auto'
        Joint-diagonalization backend.
    """

    def __init__(
        self,
        n_components=None,
        lags=None,
        whitening_estimator=None,
        center="mean",
        lag_weighting="none",
        lag_tuning=2.5,
        max_sweeps=100,
        tol=1e-10,
        eigenvalue_floor=1e-12,
        backend="auto",
    ):
        self.n_components = n_components
        self.lags = lags
        self.whitening_estimator = whitening_estimator
        self.center = center
        self.lag_weighting = lag_weighting
        self.lag_tuning = lag_tuning
        self.max_sweeps = max_sweeps
        self.tol = tol
        self.eigenvalue_floor = eigenvalue_floor
        self.backend = backend

    def _validate_parameters(self, n_features: int, n_samples: int) -> int:
        if self.n_components is None:
            n_components = n_features
        elif isinstance(self.n_components, (int, np.integer)) and not isinstance(
            self.n_components, (bool, np.bool_)
        ):
            n_components = int(self.n_components)
        else:
            raise TypeError("n_components must be an integer or None")
        if not 1 <= n_components <= min(n_features, n_samples - 1):
            raise ValueError("n_components must be between 1 and min(n_features, n_samples - 1)")
        if str(self.center).lower() not in {"mean", "median"}:
            raise ValueError("center must be 'mean' or 'median'")
        if str(self.lag_weighting).lower() not in {"none", "huber", "tukey"}:
            raise ValueError("lag_weighting must be 'none', 'huber', or 'tukey'")
        if not np.isscalar(self.lag_tuning) or not np.isfinite(self.lag_tuning) or self.lag_tuning <= 0:
            raise ValueError("lag_tuning must be a positive finite number")
        if not isinstance(self.max_sweeps, (int, np.integer)) or self.max_sweeps < 1:
            raise ValueError("max_sweeps must be a positive integer")
        if not np.isscalar(self.tol) or not np.isfinite(self.tol) or self.tol <= 0:
            raise ValueError("tol must be a positive finite number")
        if not np.isscalar(self.eigenvalue_floor) or self.eigenvalue_floor <= 0:
            raise ValueError("eigenvalue_floor must be positive")
        if self.backend not in {"auto", "python", "cpp"}:
            raise ValueError("backend must be 'auto', 'python', or 'cpp'")
        return n_components

    def fit(self, X, y=None):
        del y
        X = check_array(X)
        n_samples, n_features = X.shape
        n_components = self._validate_parameters(n_features, n_samples)
        lags = _resolve_lags(self.lags, n_samples)

        if self.whitening_estimator is None:
            if str(self.center).lower() == "median":
                location = np.median(X, axis=0)
            else:
                location = np.mean(X, axis=0)
            centered = X - location
            covariance = centered.T @ centered / n_samples
            whitening_estimator = None
        else:
            whitening_estimator = deepcopy(self.whitening_estimator)
            whitening_estimator.fit(X)
            if not hasattr(whitening_estimator, "covariance_"):
                raise AttributeError("whitening_estimator must expose covariance_ after fit")
            covariance = np.asarray(whitening_estimator.covariance_, dtype=np.float64)
            location = np.asarray(
                getattr(whitening_estimator, "location_", np.mean(X, axis=0)),
                dtype=np.float64,
            )
            centered = X - location

        covariance = 0.5 * (covariance + covariance.T)
        if covariance.shape != (n_features, n_features) or not np.all(np.isfinite(covariance)):
            raise ValueError("whitening covariance has incompatible shape or non-finite values")
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        scale = max(float(eigenvalues[0]), np.finfo(np.float64).tiny)
        floor = max(float(self.eigenvalue_floor) * scale, np.finfo(np.float64).tiny)
        if eigenvalues[n_components - 1] <= floor:
            raise ValueError("whitening covariance is rank deficient for n_components")
        retained_values = np.maximum(eigenvalues[:n_components], floor)
        retained_vectors = eigenvectors[:, :n_components]
        whitening = retained_vectors.T / np.sqrt(retained_values)[:, None]
        dewhitening = retained_vectors * np.sqrt(retained_values)[None, :]
        whitened = centered @ whitening.T

        lag_weighting = str(self.lag_weighting).lower()
        autocovariances = np.asarray(
            [
                _weighted_lag_scatter(
                    whitened,
                    int(lag),
                    weighting=lag_weighting,
                    tuning=float(self.lag_tuning),
                )
                for lag in lags
            ]
        )
        rotation, diagonalized, info = joint_diagonalize_symmetric(
            autocovariances,
            max_sweeps=int(self.max_sweeps),
            tol=float(self.tol),
            backend=self.backend,
        )
        signatures = np.sum(
            np.diagonal(diagonalized, axis1=1, axis2=2) ** 2,
            axis=0,
        )
        component_order = np.argsort(signatures)[::-1]
        rotation = rotation[:, component_order]
        diagonalized = diagonalized[:, component_order][:, :, component_order]
        signatures = signatures[component_order]

        unmixing = rotation.T @ whitening
        mixing = np.linalg.pinv(unmixing)
        unmixing, mixing, permutation, signs = canonicalize_unmixing(
            unmixing,
            mixing=mixing,
            order=np.arange(n_components),
        )
        sources = centered @ unmixing.T

        self.whitening_estimator_ = whitening_estimator
        self.location_ = location
        self.mean_ = location
        self.covariance_ = covariance
        self.whitening_ = whitening
        self.dewhitening_ = dewhitening
        self.rotation_ = rotation
        self.unmixing_ = unmixing
        self.components_ = unmixing
        self.mixing_ = mixing
        self.sources_ = sources
        self.lags_ = lags
        self.autocovariances_ = autocovariances
        self.diagonal_autocovariances_ = diagonalized
        self.temporal_signatures_ = signatures
        self.permutation_ = permutation
        self.signs_ = signs
        self.n_components_ = n_components
        self.n_features_in_ = n_features
        self.n_samples_in_ = n_samples
        self.eigenvalue_floor_ = floor
        self.converged_ = bool(info["converged"])
        self.n_sweeps_ = int(info["n_sweeps"])
        self.initial_off_diagonal_energy_ = float(info["initial_off_diagonal_energy"])
        self.off_diagonal_energy_ = float(info["off_diagonal_energy"])
        return self

    def _check_fitted(self):
        if not hasattr(self, "unmixing_"):
            raise AttributeError(f"{type(self).__name__} is not fitted yet")

    def transform(self, X):
        self._check_fitted()
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has an incompatible number of features")
        return (X - self.location_) @ self.unmixing_.T

    def fit_transform(self, X, y=None):
        return self.fit(X, y).sources_

    def inverse_transform(self, sources):
        self._check_fitted()
        sources = np.asarray(sources, dtype=np.float64)
        if sources.ndim != 2 or sources.shape[1] != self.n_components_:
            raise ValueError("sources has incompatible shape")
        if not np.all(np.isfinite(sources)):
            raise ValueError("sources must contain only finite values")
        return sources @ self.mixing_.T + self.location_


class RobustSOBI(SOBI):
    """SOBI with robust whitening and robust lagged scatter estimates.

    The defaults combine an affine-equivariant Student-t scatter estimator with
    Huber-weighted lagged cross-scatter matrices.  This targets isolated temporal
    impulses and heavy-tailed observations while preserving the classical SOBI
    model and API.
    """

    def __init__(
        self,
        n_components=None,
        lags=None,
        whitening_estimator=None,
        center="median",
        lag_weighting="huber",
        lag_tuning=2.5,
        max_sweeps=100,
        tol=1e-10,
        eigenvalue_floor=1e-12,
        backend="auto",
    ):
        super().__init__(
            n_components=n_components,
            lags=lags,
            whitening_estimator=whitening_estimator,
            center=center,
            lag_weighting=lag_weighting,
            lag_tuning=lag_tuning,
            max_sweeps=max_sweeps,
            tol=tol,
            eigenvalue_floor=eigenvalue_floor,
            backend=backend,
        )

    def fit(self, X, y=None):
        if self.whitening_estimator is None:
            original = self.whitening_estimator
            self.whitening_estimator = StudentTScatter(
                df=3.0,
                alpha=0.0,
                max_iter=300,
                tol=1e-7,
                warn_on_nonconvergence=False,
            )
            try:
                return super().fit(X, y)
            finally:
                self.whitening_estimator = original
        return super().fit(X, y)
