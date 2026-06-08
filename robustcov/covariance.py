# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from . import _robustcov_cpp as _cpp
from ._utils import check_array, mahalanobis_squared, median_impute, radial_kurtosis, robust_radial_scale
from .parallel import thread_limit


class ConvergenceWarning(UserWarning):
    pass


@dataclass
class BaseRobustCovariance:
    assume_centered: bool = False
    store_precision: bool = True
    scale_correction: str = "none"
    tail_diagnostics: bool = True
    missing_values: str = "raise"

    def mahalanobis(self, X):
        if not hasattr(self, "precision_"):
            raise RuntimeError("Estimator is not fitted")
        return mahalanobis_squared(
            X,
            self.location_,
            self.precision_,
            allow_nan=getattr(self, "missing_values", "raise") == "median",
            impute_values=getattr(self, "impute_values_", None),
        )

    def score_samples(self, X):
        return -0.5 * self.mahalanobis(X)

    def predict(self, X, alpha: float = 0.975):
        d2 = self.mahalanobis(X)
        cutoff = chi2.ppf(alpha, self.n_features_in_)
        return np.where(d2 <= cutoff, 1, -1)

    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)


class FastMCD(BaseRobustCovariance):
    """Fast minimum covariance determinant estimator backed by C++ kernels.

    The implementation uses random elemental starts, short initial C-steps, keeps the
    best determinant candidates, then fully polishes those candidates. ``quality``
    controls the speed/accuracy tradeoff while explicit ``n_init``/``n_best`` values
    remain available for benchmarking.
    """

    _QUALITY_PRESETS = {
        "fast": {"n_init": 100, "n_best": 5, "initial_c_steps": 2, "max_iter": 50},
        "balanced": {"n_init": 500, "n_best": 10, "initial_c_steps": 2, "max_iter": 100},
        "high": {"n_init": 2000, "n_best": 20, "initial_c_steps": 3, "max_iter": 150},
    }

    def __init__(
        self,
        support_fraction=None,
        contamination=None,
        n_init=None,
        max_iter=None,
        tol=1e-6,
        reweight=True,
        random_state=0,
        scale_correction="none",
        tail_diagnostics=True,
        missing_values="raise",
        quality="fast",
        n_best=None,
        initial_c_steps=None,
        reweight_alpha=0.975,
        n_jobs=None,
    ):
        super().__init__(assume_centered=False, scale_correction=scale_correction, tail_diagnostics=tail_diagnostics, missing_values=missing_values)
        if quality not in self._QUALITY_PRESETS:
            raise ValueError("quality must be one of 'fast', 'balanced', or 'high'")
        preset = self._QUALITY_PRESETS[quality]
        if contamination is not None:
            contamination = float(contamination)
            if not (0.0 <= contamination < 0.5):
                raise ValueError("contamination must be in [0, 0.5)")
            if support_fraction is not None:
                raise ValueError("Specify either support_fraction or contamination, not both")
        if support_fraction is not None:
            support_fraction = float(support_fraction)
            if not (0.0 < support_fraction <= 1.0):
                raise ValueError("support_fraction must be in (0, 1]")

        self.support_fraction = support_fraction
        self.contamination = contamination
        self.quality = quality
        self.n_init = preset["n_init"] if n_init is None else n_init
        self.max_iter = preset["max_iter"] if max_iter is None else max_iter
        self.n_best = preset["n_best"] if n_best is None else n_best
        self.initial_c_steps = preset["initial_c_steps"] if initial_c_steps is None else initial_c_steps
        self.tol = tol
        self.reweight = reweight
        self.random_state = random_state
        self.missing_values = missing_values
        self.reweight_alpha = reweight_alpha
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        X = check_array(X, allow_nan=self.missing_values == "median")
        if self.missing_values == "median":
            X, self.impute_values_ = median_impute(X)
        elif self.missing_values != "raise":
            raise ValueError("missing_values must be 'raise' or 'median'")
        self.n_samples_in_, self.n_features_in_ = X.shape
        if self.contamination is not None:
            sf = 1.0 - float(self.contamination)
        else:
            sf = -1.0 if self.support_fraction is None else float(self.support_fraction)
        with thread_limit(self.n_jobs):
            result = _cpp.fit_fast_mcd(
                X,
                support_fraction=sf,
                n_init=int(self.n_init),
                max_iter=int(self.max_iter),
                tol=float(self.tol),
                reweight=bool(self.reweight),
                random_state=int(self.random_state),
                n_best=int(self.n_best),
                initial_c_steps=int(self.initial_c_steps),
                reweight_alpha=float(self.reweight_alpha),
            )
        self._load_result(result)
        return self

    def _load_result(self, result):
        self.location_ = result["location"]
        self.shape_ = result["shape"]
        self.covariance_ = result["covariance"]
        self.precision_ = result["precision"]
        self.distances_ = result["distances"]
        self.support_ = result["support"]
        self.raw_location_ = result["raw_location"]
        self.raw_covariance_ = result["raw_covariance"]
        self.raw_distances_ = result["raw_distances"]
        self.raw_support_ = result["raw_support"]
        self.raw_scale_ = float(result.get("raw_scale", 1.0))
        self.h_ = int(result["h"])
        self.effective_support_fraction_ = self.h_ / float(getattr(self, "n_samples_in_", len(self.support_)))
        self.det_ = float(np.linalg.det(self.covariance_))
        self.raw_det_ = float(np.linalg.det(self.raw_covariance_))
        self.objective_value_ = float(result["objective_value"])
        self.n_iter_ = int(result["n_iter"])
        self.converged_ = bool(result["converged"])
        self.scale_ = robust_radial_scale(self.distances_, self.n_features_in_, self.scale_correction)
        if self.scale_correction not in {"none", "identity"}:
            self.covariance_ = self.scale_ * self.shape_
            self.precision_ = np.linalg.pinv(self.covariance_)
            self.distances_ = mahalanobis_squared_from_fitted(self)
        if self.tail_diagnostics:
            self.radial_kurtosis_ = radial_kurtosis(self.distances_, self.n_features_in_)
            self.tail_index_ = self.radial_kurtosis_


MinCovDet = FastMCD


class TylerShape(BaseRobustCovariance):
    """Tyler fixed-point shape estimator.

    The returned scatter is trace-normalized by default, i.e. trace(``shape_``) = p.
    Use scale_correction='radial_median', 'mean', 'trimmed', or 'winsorized' to
    convert the shape estimate into a covariance-like matrix.
    """

    def __init__(
        self,
        max_iter=500,
        tol=1e-7,
        assume_centered=False,
        scale_correction="none",
        tail_diagnostics=True,
        missing_values="raise",
        n_jobs=None,
    ):
        super().__init__(assume_centered=assume_centered, scale_correction=scale_correction, tail_diagnostics=tail_diagnostics, missing_values=missing_values)
        self.max_iter = max_iter
        self.tol = tol
        self.regularization = 0.0
        self.missing_values = missing_values
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        X = check_array(X, allow_nan=self.missing_values == "median")
        if self.missing_values == "median":
            X, self.impute_values_ = median_impute(X)
        elif self.missing_values != "raise":
            raise ValueError("missing_values must be 'raise' or 'median'")
        self.n_features_in_ = X.shape[1]
        with thread_limit(self.n_jobs):
            result = _cpp.fit_tyler(
                X,
                max_iter=int(self.max_iter),
                tol=float(self.tol),
                regularization=float(self.regularization),
                assume_centered=bool(self.assume_centered),
            )
        self._load_result(X, result)
        return self

    def _load_result(self, X, result):
        self.location_ = result["location"]
        self.shape_ = result["shape"]
        self.scale_ = robust_radial_scale(result["distances"], self.n_features_in_, self.scale_correction)
        self.covariance_ = self.scale_ * self.shape_
        self.precision_ = np.linalg.pinv(self.covariance_)
        self.distances_ = mahalanobis_squared(X, self.location_, self.precision_)
        self.n_iter_ = int(result["n_iter"])
        self.converged_ = bool(result["converged"])
        if not self.converged_:
            warnings.warn("TylerShape did not converge", ConvergenceWarning)
        if self.tail_diagnostics:
            self.radial_kurtosis_ = radial_kurtosis(self.distances_, self.n_features_in_)
            self.tail_index_ = self.radial_kurtosis_


class RegularizedTyler(TylerShape):
    """Regularized Tyler shape estimator.

    regularization=alpha shrinks the Tyler update toward identity.
    This works in more difficult or high-dimensional regimes than unregularized Tyler.
    """

    def __init__(
        self,
        alpha=0.1,
        max_iter=500,
        tol=1e-7,
        assume_centered=False,
        scale_correction="none",
        tail_diagnostics=True,
        missing_values="raise",
        n_jobs=None,
    ):
        super().__init__(max_iter=max_iter, tol=tol, assume_centered=assume_centered,
                         scale_correction=scale_correction, tail_diagnostics=tail_diagnostics,
                         missing_values=missing_values, n_jobs=n_jobs)
        if not (0 <= alpha < 1):
            raise ValueError("alpha must be in [0, 1)")
        self.alpha = alpha
        self.regularization = float(alpha)


def mahalanobis_squared_from_fitted(est):
    centered = np.asarray(est.distances_)  # placeholder to keep helper private/simple
    # Recompute only when caller has original fitting distances unavailable. For this MVP,
    # the C++ result already contained distances before optional scale correction; scaling
    # by scale divides squared distances.
    if getattr(est, "scale_", 1.0) > 0:
        return centered / est.scale_
    return centered
