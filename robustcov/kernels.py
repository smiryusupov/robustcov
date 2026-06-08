# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import numpy as np
from scipy.special import kv, gamma

from .metrics import pairwise_mahalanobis_squared


def _check_length_scale(length_scale: float) -> float:
    length_scale = float(length_scale)
    if not np.isfinite(length_scale) or length_scale <= 0:
        raise ValueError("length_scale must be a positive finite scalar")
    return length_scale


def robust_rbf_kernel(
    X,
    Y=None,
    *,
    precision,
    length_scale: float = 1.0,
    center=None,
    ridge: float = 0.0,
):
    """RBF kernel using a full robust Mahalanobis input metric.

    The kernel is

    ``k(x, y) = exp(-0.5 * (x-y)^T P (x-y) / length_scale**2)``

    where ``P`` is usually a fitted robust precision matrix.
    """
    length_scale = _check_length_scale(length_scale)
    D2 = pairwise_mahalanobis_squared(X, Y, precision=precision, center=center, ridge=ridge)
    return np.exp(-0.5 * D2 / (length_scale * length_scale))


def robust_matern_kernel(
    X,
    Y=None,
    *,
    precision,
    length_scale: float = 1.0,
    nu: float = 1.5,
    center=None,
    ridge: float = 0.0,
):
    """Matérn kernel using a full robust Mahalanobis input metric.

    Common closed forms are used for ``nu`` equal to 0.5, 1.5, 2.5, or infinity.
    Other positive ``nu`` values use the standard Bessel-function expression.
    """
    length_scale = _check_length_scale(length_scale)
    nu = float(nu)
    if not np.isfinite(nu) and not math.isinf(nu):
        raise ValueError("nu must be positive or np.inf")
    if nu <= 0:
        raise ValueError("nu must be positive")
    D2 = pairwise_mahalanobis_squared(X, Y, precision=precision, center=center, ridge=ridge)
    D = np.sqrt(np.maximum(D2, 0.0)) / length_scale
    if math.isinf(nu):
        return np.exp(-0.5 * D * D)
    if nu == 0.5:
        return np.exp(-D)
    if nu == 1.5:
        z = math.sqrt(3.0) * D
        return (1.0 + z) * np.exp(-z)
    if nu == 2.5:
        z = math.sqrt(5.0) * D
        return (1.0 + z + z * z / 3.0) * np.exp(-z)

    z = np.sqrt(2.0 * nu) * D
    K = np.empty_like(z)
    zero = z == 0
    K[zero] = 1.0
    nz = ~zero
    if np.any(nz):
        znz = z[nz]
        K[nz] = (2.0 ** (1.0 - nu) / gamma(nu)) * (znz ** nu) * kv(nu, znz)
    return K


__all__ = ["robust_rbf_kernel", "robust_matern_kernel"]
