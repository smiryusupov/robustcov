# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""SPD-geometry utilities for robust covariance and scatter estimators.

This module provides small, dependency-light utilities for working with
symmetric positive definite (SPD) covariance and scatter matrices.  The functions
are useful for Tyler-style shape estimators, regularized scatter paths,
covariance comparison, and robust Mahalanobis geometry.

The module intentionally provides utilities and diagnostics only.  It does not
claim convergence or optimality guarantees for every estimator using these
geometric quantities.
"""

from __future__ import annotations

import numpy as np


_EPS = np.finfo(float).eps


def _as_square_matrix(A: np.ndarray, *, name: str = "A") -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"{name} must be a square 2D array")
    if not np.all(np.isfinite(A)):
        raise ValueError(f"{name} must contain only finite values")
    return 0.5 * (A + A.T)


def _eigh_spd(A: np.ndarray, *, name: str = "A", tol: float | None = None):
    A = _as_square_matrix(A, name=name)
    vals, vecs = np.linalg.eigh(A)
    if tol is None:
        tol = 100.0 * _EPS * max(1.0, float(np.max(np.abs(vals))))
    if np.min(vals) <= tol:
        raise ValueError(f"{name} must be symmetric positive definite")
    return vals, vecs


def trace_normalize(S: np.ndarray, p: int | None = None) -> np.ndarray:
    """Normalize an SPD matrix to have trace ``p``.

    Parameters
    ----------
    S : array-like of shape (p, p)
        Symmetric positive definite matrix.
    p : int, optional
        Target trace.  Defaults to the matrix dimension.

    Returns
    -------
    S_norm : ndarray of shape (p, p)
        Trace-normalized matrix.
    """
    S = _as_square_matrix(S, name="S")
    dim = S.shape[0]
    target = dim if p is None else int(p)
    tr = float(np.trace(S))
    if tr <= 0.0 or not np.isfinite(tr):
        raise ValueError("S must have positive finite trace")
    return S * (target / tr)


def det_normalize(S: np.ndarray) -> np.ndarray:
    """Normalize an SPD matrix to have determinant one."""
    S = _as_square_matrix(S, name="S")
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0 or not np.isfinite(logdet):
        raise ValueError("S must be symmetric positive definite")
    dim = S.shape[0]
    return S / np.exp(logdet / dim)


def spd_power(A: np.ndarray, power: float) -> np.ndarray:
    """Matrix power for a symmetric positive definite matrix."""
    vals, vecs = _eigh_spd(A, name="A")
    out = (vecs * (vals**power)) @ vecs.T
    return 0.5 * (out + out.T)


def spd_log(A: np.ndarray) -> np.ndarray:
    """Principal matrix logarithm of an SPD matrix."""
    vals, vecs = _eigh_spd(A, name="A")
    out = (vecs * np.log(vals)) @ vecs.T
    return 0.5 * (out + out.T)


def spd_exp(A: np.ndarray) -> np.ndarray:
    """Matrix exponential of a symmetric matrix.

    The input only needs to be symmetric; the result is SPD.
    """
    A = _as_square_matrix(A, name="A")
    vals, vecs = np.linalg.eigh(A)
    out = (vecs * np.exp(vals)) @ vecs.T
    return 0.5 * (out + out.T)


def affine_invariant_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Affine-invariant Riemannian distance between two SPD matrices.

    Computes

    ``|| log(A^{-1/2} B A^{-1/2}) ||_F``.
    """
    A = _as_square_matrix(A, name="A")
    B = _as_square_matrix(B, name="B")
    A_inv_sqrt = spd_power(A, -0.5)
    C = A_inv_sqrt @ B @ A_inv_sqrt
    vals, _ = _eigh_spd(C, name="A^{-1/2} B A^{-1/2}")
    return float(np.linalg.norm(np.log(vals)))


def logeuclidean_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Log-Euclidean distance between two SPD matrices."""
    LA = spd_log(A)
    LB = spd_log(B)
    return float(np.linalg.norm(LA - LB, ord="fro"))


def spd_geodesic(A: np.ndarray, B: np.ndarray, t: float) -> np.ndarray:
    """Affine-invariant geodesic between two SPD matrices.

    Parameters
    ----------
    A, B : array-like of shape (p, p)
        SPD endpoint matrices.
    t : float
        Geodesic parameter.  ``t=0`` returns ``A`` and ``t=1`` returns ``B``.
    """
    A = _as_square_matrix(A, name="A")
    B = _as_square_matrix(B, name="B")
    t = float(t)

    A_sqrt = spd_power(A, 0.5)
    A_inv_sqrt = spd_power(A, -0.5)
    C = A_inv_sqrt @ B @ A_inv_sqrt
    out = A_sqrt @ spd_power(C, t) @ A_sqrt
    return 0.5 * (out + out.T)


def _center_data(X: np.ndarray, location: np.ndarray | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")

    if location is None:
        return X

    loc = np.asarray(location, dtype=float)
    if loc.shape != (X.shape[1],):
        raise ValueError("location must have shape (n_features,)")
    return X - loc


def _normalize_shape(S: np.ndarray, normalize: str | None) -> np.ndarray:
    if normalize is None:
        return _as_square_matrix(S, name="S")
    if normalize == "trace":
        return trace_normalize(S)
    if normalize == "det":
        return det_normalize(S)
    raise ValueError("normalize must be one of {'trace', 'det', None}")


def tyler_objective(
    S: np.ndarray,
    X: np.ndarray,
    *,
    location: np.ndarray | None = None,
    normalize: str | None = "trace",
) -> float:
    """Scale-invariant Tyler shape objective.

    For centered observations ``z_i`` and dimension ``p``, this computes

    ``log det(S) + (p / n) sum_i log(z_i.T inv(S) z_i)``.

    The objective is scale-invariant in ``S``; optional normalization is applied
    for numerical consistency and to match common shape conventions.
    """
    Z = _center_data(X, location=location)
    n_samples, n_features = Z.shape

    if n_samples == 0:
        raise ValueError("X must contain at least one row")

    S = _normalize_shape(S, normalize)
    if S.shape != (n_features, n_features):
        raise ValueError("S shape must match X feature dimension")

    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:
        raise ValueError("S must be symmetric positive definite")

    precision = np.linalg.inv(S)
    q = np.einsum("ij,jk,ik->i", Z, precision, Z)
    if np.any(q <= 0.0) or not np.all(np.isfinite(q)):
        raise ValueError("all centered observations must have positive quadratic form")

    return float(logdet + (n_features / n_samples) * np.sum(np.log(q)))


def tyler_fixed_point_residual(
    S: np.ndarray,
    X: np.ndarray,
    *,
    location: np.ndarray | None = None,
    normalize: str | None = "trace",
    relative: bool = True,
) -> float:
    """Fixed-point residual for the Tyler shape equation.

    The Tyler update is

    ``T(S) = (p / n) sum_i z_i z_i.T / (z_i.T inv(S) z_i)``.

    This function returns the Frobenius norm of ``T(S) - S`` after applying the
    requested shape normalization.  If ``relative=True``, the value is divided by
    ``||S||_F``.
    """
    Z = _center_data(X, location=location)
    n_samples, n_features = Z.shape

    if n_samples == 0:
        raise ValueError("X must contain at least one row")

    S = _normalize_shape(S, normalize)
    if S.shape != (n_features, n_features):
        raise ValueError("S shape must match X feature dimension")

    precision = np.linalg.inv(S)
    q = np.einsum("ij,jk,ik->i", Z, precision, Z)
    if np.any(q <= 0.0) or not np.all(np.isfinite(q)):
        raise ValueError("all centered observations must have positive quadratic form")

    update = (n_features / n_samples) * ((Z / q[:, None]).T @ Z)
    update = _normalize_shape(update, normalize)

    residual = float(np.linalg.norm(update - S, ord="fro"))
    if relative:
        denom = float(np.linalg.norm(S, ord="fro"))
        if denom > 0.0:
            residual /= denom
    return residual
