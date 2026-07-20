#!/usr/bin/env python
# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Shared numerical helpers for external DRO-PCA examples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Standardization:
    location: np.ndarray
    scale: np.ndarray
    keep: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float)[:, self.keep] - self.location) / self.scale


def robust_standardization(X: np.ndarray, *, floor: float = 1e-8) -> Standardization:
    X = np.asarray(X, dtype=float)
    location = np.median(X, axis=0)
    mad = np.median(np.abs(X - location), axis=0)
    scale = 1.4826 * mad
    fallback = np.std(X, axis=0, ddof=0)
    scale = np.where(scale > floor * max(float(np.median(scale[scale > 0])) if np.any(scale > 0) else 1.0, 1.0), scale, fallback)
    absolute_floor = floor * max(float(np.median(scale[scale > 0])) if np.any(scale > 0) else 1.0, np.finfo(float).tiny)
    keep = scale > absolute_floor
    if int(np.sum(keep)) < 2:
        raise ValueError("fewer than two non-constant features remain after standardization")
    return Standardization(location=location[keep], scale=scale[keep], keep=keep)


def empirical_pca(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    location = np.mean(X, axis=0)
    centered = X - location
    covariance = centered.T @ centered / max(X.shape[0], 1)
    values, vectors = np.linalg.eigh((covariance + covariance.T) * 0.5)
    basis = vectors[:, np.argsort(values)[::-1][:n_components]]
    for column in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0
    return location, basis


def reconstruction_errors(X: np.ndarray, location: np.ndarray, basis: np.ndarray) -> np.ndarray:
    centered = np.asarray(X, dtype=float) - location
    residual = centered - (centered @ basis) @ basis.T
    return np.einsum("ij,ij->i", residual, residual)


def upper_order_statistic(values: np.ndarray, false_alarm_rate: float) -> float:
    values = np.sort(np.asarray(values, dtype=float).reshape(-1))
    if values.size == 0:
        raise ValueError("calibration values must not be empty")
    if not 0.0 < false_alarm_rate < 1.0:
        raise ValueError("false_alarm_rate must lie strictly between zero and one")
    rank = int(np.ceil((values.size + 1) * (1.0 - false_alarm_rate)))
    return float(values[min(max(rank, 1), values.size) - 1])


def window_means(values: np.ndarray, window_size: int, step: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size < window_size:
        return np.array([float(np.mean(values))])
    return np.asarray(
        [float(np.mean(values[start : start + window_size])) for start in range(0, values.size - window_size + 1, step)],
        dtype=float,
    )


def diagonal_transport_from_domain_means(
    X: np.ndarray,
    domains: np.ndarray,
    *,
    ridge_fraction: float = 0.05,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    domains = np.asarray(domains)
    means = np.vstack([np.mean(X[domains == domain], axis=0) for domain in np.unique(domains)])
    if means.shape[0] < 2:
        raise ValueError("at least two calibration domains are required to estimate transport geometry")
    drift_variance = np.var(means, axis=0, ddof=0)
    base = float(np.mean(np.var(X, axis=0, ddof=0)))
    ridge = max(ridge_fraction * base, np.finfo(float).tiny)
    return np.diag(1.0 / (drift_variance + ridge))
