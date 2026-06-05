from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def check_array(X, *, allow_nan: bool = False) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64, order="C")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("X must have at least 2 rows and 1 column")
    if allow_nan:
        if np.isinf(X).any():
            raise ValueError("X contains infinity")
    else:
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinity")
    return X


def median_impute(X: np.ndarray, values: np.ndarray | None = None):
    """Column-median imputation for NaNs.

    Returns the imputed copy and the imputation values. This is intentionally
    simple and deterministic; it makes examples and benchmarks resilient to
    missingness without turning the package into a missing-data framework.
    """
    X = np.asarray(X, dtype=np.float64, order="C").copy()
    if values is None:
        values = np.nanmedian(X, axis=0)
        values = np.where(np.isfinite(values), values, 0.0)
    else:
        values = np.asarray(values, dtype=np.float64)
    mask = np.isnan(X)
    if mask.any():
        rows, cols = np.where(mask)
        X[rows, cols] = values[cols]
    return np.asarray(X, dtype=np.float64, order="C"), values


def robust_radial_scale(distances: np.ndarray, p: int, method: str = "median", trim_fraction: float = 0.1) -> float:
    """Estimate scalar scale from squared robust radial distances.

    Shape estimators such as Tyler return shape up to scale. This function estimates
    a scalar covariance scale under an elliptical model.
    """
    d = np.asarray(distances, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 1.0
    method = method.lower()
    if method in {"none", "identity"}:
        return 1.0
    if method in {"median", "radial_median"}:
        denom = chi2.ppf(0.5, p)
        return float(np.median(d) / max(denom, np.finfo(float).tiny))
    if method in {"mean", "radial_mean"}:
        return float(np.mean(d) / p)
    if method in {"trimmed", "trimmed_mean"}:
        qlo, qhi = np.quantile(d, [trim_fraction, 1.0 - trim_fraction])
        kept = d[(d >= qlo) & (d <= qhi)]
        return float(np.mean(kept) / p) if kept.size else 1.0
    if method in {"winsorized", "winsorized_mean"}:
        qlo, qhi = np.quantile(d, [trim_fraction, 1.0 - trim_fraction])
        return float(np.mean(np.clip(d, qlo, qhi)) / p)
    raise ValueError(f"Unknown scale correction method: {method!r}")


def radial_kurtosis(distances: np.ndarray, p: int, method: str = "winsorized", trim_fraction: float = 0.05) -> float:
    """Normalized multivariate radial kurtosis.

    For Gaussian data this is approximately 1. Values above 1 indicate heavier tails.
    """
    d = np.asarray(distances, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    z = d * d
    method = method.lower()
    if method == "classical":
        val = np.mean(z)
    elif method in {"trimmed", "trimmed_mean"}:
        qlo, qhi = np.quantile(z, [trim_fraction, 1.0 - trim_fraction])
        kept = z[(z >= qlo) & (z <= qhi)]
        val = np.mean(kept) if kept.size else np.nan
    elif method in {"winsorized", "winsorized_mean"}:
        qlo, qhi = np.quantile(z, [trim_fraction, 1.0 - trim_fraction])
        val = np.mean(np.clip(z, qlo, qhi))
    else:
        raise ValueError(f"Unknown radial kurtosis method: {method!r}")
    return float(val / (p * (p + 2)))


def mahalanobis_squared(X, location, precision, *, allow_nan: bool = False, impute_values=None):
    X = check_array(X, allow_nan=allow_nan)
    if allow_nan and np.isnan(X).any():
        X, _ = median_impute(X, impute_values)
    centered = X - np.asarray(location, dtype=np.float64)
    return np.einsum("ij,jk,ik->i", centered, precision, centered)
