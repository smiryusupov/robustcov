# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import sqrtm

from .covariance import BaseRobustCovariance, RegularizedTyler
from ._utils import check_array, median_impute, mahalanobis_squared, radial_kurtosis, robust_radial_scale


class MEstimatorWarning(UserWarning):
    pass


def _trace_normalize(S: np.ndarray, p: int) -> np.ndarray:
    S = 0.5 * (S + S.T)
    tr = float(np.trace(S))
    if not np.isfinite(tr) or tr <= 0:
        S = np.eye(p)
        tr = float(p)
    return S * (p / tr)


def _safe_pinv(S: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(0.5 * (S + S.T), hermitian=True)


def _empirical_initial_shape(X: np.ndarray, location: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    p = X.shape[1]
    Xc = X - location
    S = (Xc.T @ Xc) / max(1, X.shape[0])
    S = 0.5 * (S + S.T) + ridge * np.eye(p)
    return _trace_normalize(S, p)


def _target_matrix(target: str, X: np.ndarray, location: np.ndarray) -> np.ndarray:
    target = target.lower()
    p = X.shape[1]
    if target == "identity":
        return np.eye(p)
    if target == "diagonal":
        Xc = X - location
        diag = np.var(Xc, axis=0)
        diag = np.where(np.isfinite(diag) & (diag > 1e-12), diag, 1.0)
        return _trace_normalize(np.diag(diag), p)
    raise ValueError("target must be 'identity' or 'diagonal'")


def _sqrt_shrink(S: np.ndarray, T: np.ndarray, alpha: float, p: int) -> np.ndarray:
    """Geometry-inspired covariance shrinkage in square-root space.

    This is intended as an experimental Hellinger-style prototype. It is not presented
    as a final exact optimizer for a specific published Hellinger objective.
    """
    try:
        A = np.real_if_close(sqrtm(0.5 * (S + S.T))).astype(float)
        B = np.real_if_close(sqrtm(0.5 * (T + T.T))).astype(float)
        C = (1.0 - alpha) * A + alpha * B
        out = C @ C.T
    except Exception:
        out = (1.0 - alpha) * S + alpha * T
    out = 0.5 * (out + out.T) + 1e-10 * np.eye(p)
    return _trace_normalize(out, p)


class IterativeMScatter(BaseRobustCovariance):
    """Generic iteratively reweighted robust scatter estimator.

    This class is public but lower-level. Prefer `StudentTScatter` or
    `RegularizedCauchy` for user-facing code.
    """

    def __init__(
        self,
        weight="student_t",
        df=3.0,
        alpha=0.0,
        target="identity",
        shrinkage="linear",
        max_iter=300,
        tol=1e-6,
        assume_centered=False,
        update_location=True,
        scale_correction="radial_median",
        tail_diagnostics=True,
        missing_values="raise",
        ridge=1e-8,
        damping=1.0,
        warn_on_nonconvergence=True,
    ):
        super().__init__(assume_centered=assume_centered, scale_correction=scale_correction, tail_diagnostics=tail_diagnostics, missing_values=missing_values)
        self.weight = weight
        self.df = df
        self.alpha = float(alpha)
        self.target = target
        self.shrinkage = shrinkage
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.update_location = bool(update_location)
        self.ridge = float(ridge)
        self.damping = float(damping)
        self.warn_on_nonconvergence = bool(warn_on_nonconvergence)
        if not (0.0 < self.damping <= 1.0):
            raise ValueError("damping must be in (0, 1]")
        if not (0 <= self.alpha < 1):
            raise ValueError("alpha must be in [0, 1)")
        if self.df is not None and self.df <= 0:
            raise ValueError("df must be positive")

    def _weights(self, d2: np.ndarray, p: int) -> np.ndarray:
        d2 = np.maximum(np.asarray(d2, dtype=float), 1e-12)
        weight = self.weight.lower()
        if weight in {"student_t", "t"}:
            return (float(self.df) + p) / (float(self.df) + d2)
        if weight == "cauchy":
            return (1.0 + p) / (1.0 + d2)
        if weight == "tyler":
            return p / d2
        raise ValueError("weight must be 'student_t', 'cauchy', or 'tyler'")

    def fit(self, X, y=None):
        X = check_array(X, allow_nan=self.missing_values == "median")
        if self.missing_values == "median":
            X, self.impute_values_ = median_impute(X)
        elif self.missing_values != "raise":
            raise ValueError("missing_values must be 'raise' or 'median'")
        n, p = X.shape
        self.n_samples_in_, self.n_features_in_ = n, p

        if self.assume_centered:
            location = np.zeros(p)
        else:
            location = np.median(X, axis=0)
        S = _empirical_initial_shape(X, location, ridge=max(self.ridge, 1e-8))
        target = _target_matrix(self.target, X, location)

        converged = False
        last_diff = np.inf
        for it in range(1, self.max_iter + 1):
            precision = _safe_pinv(S + self.ridge * np.eye(p))
            Xc = X - location
            d2 = np.einsum("ij,jk,ik->i", Xc, precision, Xc)
            w = self._weights(d2, p)
            w = np.where(np.isfinite(w), w, 0.0)
            sw = float(np.sum(w))
            if sw <= 1e-12:
                warnings.warn("All M-estimator weights vanished; keeping previous iterate", MEstimatorWarning)
                break
            if self.update_location and not self.assume_centered:
                new_location = (w[:, None] * X).sum(axis=0) / sw
            else:
                new_location = location
            Xc_new = X - new_location
            scatter = (Xc_new.T * w) @ Xc_new / max(1, n)
            scatter = 0.5 * (scatter + scatter.T) + self.ridge * np.eye(p)
            scatter = _trace_normalize(scatter, p)
            if self.alpha > 0:
                if self.shrinkage in {"linear", "kl", "wiesel"}:
                    S_candidate = _trace_normalize((1.0 - self.alpha) * scatter + self.alpha * target, p)
                elif self.shrinkage in {"sqrt", "hellinger"}:
                    S_candidate = _sqrt_shrink(scatter, target, self.alpha, p)
                else:
                    raise ValueError("shrinkage must be 'linear', 'kl', 'wiesel', 'sqrt', or 'hellinger'")
            else:
                S_candidate = scatter
            # Damping stabilizes high-dimensional/small-sample iterations. It is a
            # convex blend in scatter and location space followed by trace normalization.
            if self.damping < 1.0:
                S_new = _trace_normalize((1.0 - self.damping) * S + self.damping * S_candidate, p)
                new_location = (1.0 - self.damping) * location + self.damping * new_location
            else:
                S_new = S_candidate
            diff = np.linalg.norm(S_new - S, ord="fro") / max(1e-12, np.linalg.norm(S, ord="fro"))
            loc_diff = np.linalg.norm(new_location - location) / max(1e-12, 1.0 + np.linalg.norm(location))
            S, location = S_new, new_location
            last_diff = max(float(diff), float(loc_diff))
            if last_diff < self.tol:
                converged = True
                break

        self.location_ = location
        self.shape_ = _trace_normalize(S, p)
        raw_precision = _safe_pinv(self.shape_ + self.ridge * np.eye(p))
        raw_distances = mahalanobis_squared(X, self.location_, raw_precision)
        self.scale_ = robust_radial_scale(raw_distances, p, self.scale_correction)
        self.covariance_ = self.scale_ * self.shape_
        self.precision_ = _safe_pinv(self.covariance_ + self.ridge * np.eye(p))
        self.distances_ = mahalanobis_squared(X, self.location_, self.precision_)
        self.weights_ = self._weights(np.maximum(raw_distances, 1e-12), p)
        self.n_iter_ = int(it)
        self.converged_ = bool(converged)
        self.objective_value_ = float(last_diff)
        if not self.converged_ and self.warn_on_nonconvergence:
            warnings.warn(f"{type(self).__name__} did not converge", MEstimatorWarning)
        if self.tail_diagnostics:
            self.radial_kurtosis_ = radial_kurtosis(self.distances_, p)
            self.tail_index_ = self.radial_kurtosis_
        return self


class StudentTScatter(IterativeMScatter):
    """Student-t inspired robust scatter estimator with fixed degrees of freedom.

    This is exposed as a robust covariance/scatter estimator, not as a full
    probability distribution fit. `df=1` corresponds to the Cauchy weight shape.
    """

    def __init__(self, df=3.0, alpha=0.0, target="identity", max_iter=300, tol=1e-6,
                 assume_centered=False, scale_correction="radial_median", tail_diagnostics=True,
                 missing_values="raise", update_location=True, damping=1.0, warn_on_nonconvergence=True):
        super().__init__(weight="student_t", df=df, alpha=alpha, target=target,
                         shrinkage="linear", max_iter=max_iter, tol=tol,
                         assume_centered=assume_centered, update_location=update_location,
                         scale_correction=scale_correction, tail_diagnostics=tail_diagnostics,
                         missing_values=missing_values, damping=damping,
                         warn_on_nonconvergence=warn_on_nonconvergence)


class RegularizedCauchy(StudentTScatter):
    """Regularized Cauchy scatter estimator for very heavy-tailed small samples."""

    def __init__(self, alpha=0.1, target="identity", max_iter=300, tol=1e-6,
                 assume_centered=False, scale_correction="radial_median", tail_diagnostics=True,
                 missing_values="raise", update_location=True, damping=0.7, warn_on_nonconvergence=True):
        super().__init__(df=1.0, alpha=alpha, target=target, max_iter=max_iter, tol=tol,
                         assume_centered=assume_centered, scale_correction=scale_correction,
                         tail_diagnostics=tail_diagnostics, missing_values=missing_values,
                         update_location=update_location, damping=damping,
                         warn_on_nonconvergence=warn_on_nonconvergence)


class KLRegularizedTyler(RegularizedTyler):
    """KL/Wiesel-style regularized Tyler shape estimator alias.

    In the current prototype this is intentionally a named alias around
    `RegularizedTyler`. It is kept as a separate class so benchmarks can track the
    intended estimator family while the exact KL objective/update is finalized.
    """

    def __init__(self, alpha=0.1, max_iter=500, tol=1e-7, assume_centered=False,
                 scale_correction="none", tail_diagnostics=True, missing_values="raise"):
        super().__init__(alpha=alpha, max_iter=max_iter, tol=tol, assume_centered=assume_centered,
                         scale_correction=scale_correction, tail_diagnostics=tail_diagnostics,
                         missing_values=missing_values)
        self.penalty = "kl"


class WieselTyler(KLRegularizedTyler):
    """Wiesel-style regularized Tyler estimator alias.

    The implementation currently uses the same fixed-point shrinkage engine as
    `KLRegularizedTyler`; the separate name makes experiments and benchmarks clearer.
    """

    def __init__(self, alpha=0.1, max_iter=500, tol=1e-7, assume_centered=False,
                 scale_correction="none", tail_diagnostics=True, missing_values="raise"):
        super().__init__(alpha=alpha, max_iter=max_iter, tol=tol, assume_centered=assume_centered,
                         scale_correction=scale_correction, tail_diagnostics=tail_diagnostics,
                         missing_values=missing_values)
        self.penalty = "wiesel"


class HellingerRegularizedTyler(IterativeMScatter):
    """Experimental Hellinger-style regularized Tyler estimator.

    This uses Tyler radial weights with shrinkage in matrix square-root space. It is
    useful for exploratory comparisons, but should be treated as experimental until
    the exact Hellinger objective/update is finalized.
    """

    def __init__(self, alpha=0.1, target="identity", max_iter=300, tol=1e-6,
                 assume_centered=False, scale_correction="none", tail_diagnostics=True,
                 missing_values="raise", damping=0.7, warn_on_nonconvergence=True):
        super().__init__(weight="tyler", df=None, alpha=alpha, target=target,
                         shrinkage="hellinger", max_iter=max_iter, tol=tol,
                         assume_centered=assume_centered, update_location=not assume_centered,
                         scale_correction=scale_correction, tail_diagnostics=tail_diagnostics,
                         missing_values=missing_values, damping=damping,
                         warn_on_nonconvergence=warn_on_nonconvergence)
