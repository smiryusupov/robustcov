"""Cross-method benchmarks and a generated suitability report.

This benchmark is deliberately task-specific. It does not combine covariance,
PCA, matrix-valued estimation, and sparse precision recovery into one ranking.
Each scenario has known synthetic ground truth and reports metrics appropriate
to that task.

Quick run used by the documentation::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
      python benchmarks/compare_methods.py \
        --profile quick \
        --csv docs/_static/benchmarks/method_comparison_quick.csv \
        --rst docs/_generated/method_comparison_results.rst

A fuller local run::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
      python benchmarks/compare_methods.py \
        --profile full --repeats 3 --csv results/method_comparison.csv

The timing numbers are machine-dependent. Accuracy metrics are deterministic for
fixed NumPy and robustcov versions, apart from small floating-point differences.
"""
from __future__ import annotations

import argparse
import csv
import gc
import math
import sys
import time
import tracemalloc
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

import robustcov as rc


@dataclass(frozen=True)
class Profile:
    scatter_n: int
    scatter_p: int
    high_n: int
    high_p: int
    pca_n: int
    pca_p: int
    matrix_n: int
    graph_n: int
    graph_p: int
    subset_starts: int
    kernel_subset_starts: int
    cell_max_iter: int
    pca_max_iter: int
    graph_n_alphas: int
    graph_max_iter: int


_MEASURE_PYTHON_MEMORY = False


PROFILES = {
    "quick": Profile(
        scatter_n=180,
        scatter_p=8,
        high_n=64,
        high_p=80,
        pca_n=160,
        pca_p=18,
        matrix_n=72,
        graph_n=170,
        graph_p=10,
        subset_starts=24,
        kernel_subset_starts=20,
        cell_max_iter=35,
        pca_max_iter=60,
        graph_n_alphas=7,
        graph_max_iter=140,
    ),
    "full": Profile(
        scatter_n=360,
        scatter_p=12,
        high_n=100,
        high_p=140,
        pca_n=260,
        pca_p=24,
        matrix_n=120,
        graph_n=240,
        graph_p=12,
        subset_starts=60,
        kernel_subset_starts=30,
        cell_max_iter=60,
        pca_max_iter=100,
        graph_n_alphas=12,
        graph_max_iter=220,
    ),
}


CSV_FIELDS = [
    "family",
    "scenario",
    "method",
    "status",
    "n_samples",
    "n_features",
    "repeat",
    "seconds",
    "python_peak_mb",
    "covariance_error",
    "location_error",
    "row_outlier_auc",
    "cell_outlier_auc",
    "subspace_error",
    "missing_reconstruction_mae",
    "loading_support_precision",
    "loading_support_recall",
    "loading_support_f1",
    "loading_sparsity",
    "matrix_covariance_error",
    "precision_error",
    "edge_precision",
    "edge_recall",
    "edge_f1",
    "n_edges",
    "notes",
]


# ---------------------------------------------------------------------------
# Generic metrics and timing


def _finite_float(value: Any) -> float | str:
    try:
        value = float(value)
    except Exception:
        return ""
    return value if np.isfinite(value) else ""


def relative_frobenius(estimate: np.ndarray, truth: np.ndarray) -> float:
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return float(np.linalg.norm(estimate - truth, ord="fro") / np.linalg.norm(truth, ord="fro"))


def normalized_location_error(estimate: np.ndarray, truth: np.ndarray, covariance: np.ndarray) -> float:
    scale = math.sqrt(max(float(np.trace(covariance)), np.finfo(float).eps))
    return float(np.linalg.norm(np.asarray(estimate) - np.asarray(truth)) / scale)


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def projection_error(components: np.ndarray, truth_basis: np.ndarray) -> float:
    components = np.asarray(components, dtype=float)
    truth_basis = np.asarray(truth_basis, dtype=float)
    # Sparse loading vectors need not be mutually orthogonal. Compare the
    # column spaces through orthonormal bases instead of treating B B^T as a
    # projection matrix. Components and truth_basis are q x p.
    estimated_basis, _ = np.linalg.qr(components.T)
    truth_orthonormal, _ = np.linalg.qr(truth_basis.T)
    P_est = estimated_basis @ estimated_basis.T
    P_true = truth_orthonormal @ truth_orthonormal.T
    q = truth_basis.shape[0]
    return float(np.linalg.norm(P_est - P_true, ord="fro") / math.sqrt(2.0 * q))


def loading_support_metrics(components: np.ndarray, truth_basis: np.ndarray) -> tuple[float, float, float, float]:
    components = np.asarray(components, dtype=float)
    truth_basis = np.asarray(truth_basis, dtype=float)
    similarity = np.abs(truth_basis @ components.T)
    truth_order, estimate_order = linear_sum_assignment(-similarity)
    aligned = components[estimate_order]
    expected = np.abs(truth_basis[truth_order]) > 1e-12
    predicted = np.abs(aligned) > 1e-12
    tp = int(np.count_nonzero(expected & predicted))
    fp = int(np.count_nonzero(~expected & predicted))
    fn = int(np.count_nonzero(expected & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    sparsity = 1.0 - np.count_nonzero(components) / components.size
    return float(precision), float(recall), float(f1), float(sparsity)


def partial_correlation(precision: np.ndarray) -> np.ndarray:
    precision = np.asarray(precision, dtype=float)
    scale = np.sqrt(np.maximum(np.diag(precision), np.finfo(float).eps))
    partial = -precision / np.outer(scale, scale)
    np.fill_diagonal(partial, 1.0)
    return partial


def graph_metrics(adjacency: np.ndarray, truth_adjacency: np.ndarray) -> tuple[float, float, float, int]:
    adjacency = np.asarray(adjacency, dtype=bool)
    truth_adjacency = np.asarray(truth_adjacency, dtype=bool)
    upper = np.triu_indices_from(adjacency, k=1)
    pred = adjacency[upper]
    truth = truth_adjacency[upper]
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1), int(pred.sum())


def median_impute(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    medians = np.nanmedian(X, axis=0)
    if np.any(~np.isfinite(medians)):
        raise ValueError("every feature needs at least one finite value")
    return np.where(np.isnan(X), medians, X), medians


def mahalanobis_squared(X: np.ndarray, location: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    precision = np.linalg.pinv(np.asarray(covariance, dtype=float))
    centered = np.asarray(X, dtype=float) - np.asarray(location, dtype=float)
    return np.einsum("ij,jk,ik->i", centered, precision, centered)


def _measure(call: Callable[[], Any]) -> tuple[Any, float, float | str]:
    gc.collect()
    if _MEASURE_PYTHON_MEMORY:
        tracemalloc.start()
    started = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = call()
    finally:
        seconds = time.perf_counter() - started
        if _MEASURE_PYTHON_MEMORY:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mb: float | str = float(peak / (1024.0**2))
        else:
            peak_mb = ""
    return result, float(seconds), peak_mb


def _base_row(*, family: str, scenario: str, method: str, n: int, p: int, repeat: int) -> dict[str, Any]:
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        family=family,
        scenario=scenario,
        method=method,
        status="ok",
        n_samples=n,
        n_features=p,
        repeat=repeat,
    )
    return row


def _failure_row(*, family: str, scenario: str, method: str, n: int, p: int, repeat: int, exc: Exception) -> dict[str, Any]:
    row = _base_row(family=family, scenario=scenario, method=method, n=n, p=p, repeat=repeat)
    row["status"] = "failed"
    row["notes"] = f"{type(exc).__name__}: {exc}"
    return row


# ---------------------------------------------------------------------------
# Data generators


def toeplitz_covariance(p: int, rho: float = 0.55) -> np.ndarray:
    indices = np.arange(p)
    return rho ** np.abs(indices[:, None] - indices[None, :])


def sample_multivariate_t(
    rng: np.random.Generator,
    n: int,
    covariance: np.ndarray,
    df: float,
) -> np.ndarray:
    p = covariance.shape[0]
    gaussian = rng.multivariate_normal(np.zeros(p), covariance, size=n)
    scale = np.sqrt(rng.chisquare(df, size=n) / df)
    return gaussian / scale[:, None]


def add_rowwise_outliers(
    rng: np.random.Generator,
    X: np.ndarray,
    fraction: float,
    magnitude: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float).copy()
    n, p = X.shape
    count = max(1, int(round(fraction * n)))
    indices = rng.choice(n, size=count, replace=False)
    direction = rng.normal(size=p)
    direction /= np.linalg.norm(direction)
    X[indices] += magnitude * direction + rng.normal(scale=0.6, size=(count, p))
    labels = np.zeros(n, dtype=bool)
    labels[indices] = True
    return X, labels


def add_cellwise_errors(
    rng: np.random.Generator,
    X: np.ndarray,
    contamination: float,
    missing: float,
    magnitude: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float).copy()
    n, p = X.shape
    cell_labels = np.zeros((n, p), dtype=bool)
    n_bad = max(1, int(round(contamination * n * p)))
    flat = rng.choice(n * p, size=n_bad, replace=False)
    rows, cols = np.unravel_index(flat, (n, p))
    signs = rng.choice([-1.0, 1.0], size=n_bad)
    scales = np.std(X, axis=0, ddof=1)
    X[rows, cols] += signs * magnitude * np.maximum(scales[cols], 0.2)
    cell_labels[rows, cols] = True

    missing_mask = np.zeros((n, p), dtype=bool)
    n_missing = int(round(missing * n * p))
    if n_missing:
        available = np.flatnonzero(~cell_labels.ravel())
        missing_flat = rng.choice(available, size=min(n_missing, available.size), replace=False)
        mr, mc = np.unravel_index(missing_flat, (n, p))
        X[mr, mc] = np.nan
        missing_mask[mr, mc] = True
    return X, cell_labels, missing_mask


def make_low_rank(
    rng: np.random.Generator,
    n: int,
    p: int,
    q: int,
    noise: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    raw = rng.normal(size=(p, q))
    basis, _ = np.linalg.qr(raw)
    strengths = np.linspace(3.0, 1.4, q)
    scores = rng.normal(size=(n, q)) * strengths
    X = scores @ basis.T + rng.normal(scale=noise, size=(n, p))
    return X, basis.T


def make_sparse_precision(p: int) -> tuple[np.ndarray, np.ndarray]:
    precision = np.eye(p)
    for j in range(p - 1):
        value = 0.22 + 0.08 * (j % 3)
        precision[j, j + 1] = precision[j + 1, j] = -value
    for j in range(0, p - 3, 4):
        precision[j, j + 3] = precision[j + 3, j] = 0.18
    # Strict diagonal dominance keeps the matrix SPD.
    for j in range(p):
        precision[j, j] = 1.0 + np.sum(np.abs(precision[j]))
    covariance = np.linalg.inv(precision)
    return precision, covariance


def make_sparse_low_rank(
    rng: np.random.Generator,
    n: int,
    p: int,
    q: int,
    noise: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    block = max(4, min(7, p // (q + 2)))
    starts = np.linspace(1, p - block - 1, q, dtype=int)
    loadings = np.zeros((p, q), dtype=float)
    shape = np.linspace(1.0, 0.25, block)
    for k, start in enumerate(starts):
        loadings[start : start + block, k] = shape * (-1.0 if k % 2 else 1.0)
    loadings, _ = np.linalg.qr(loadings)
    scores = rng.normal(size=(n, q)) * np.linspace(3.0, 1.4, q)
    X = scores @ loadings.T + noise * rng.normal(size=(n, p))
    return X, loadings.T


def matrix_normal_sample(
    rng: np.random.Generator,
    n: int,
    row_covariance: np.ndarray,
    column_covariance: np.ndarray,
) -> np.ndarray:
    r = row_covariance.shape[0]
    c = column_covariance.shape[0]
    Lr = np.linalg.cholesky(row_covariance)
    Lc = np.linalg.cholesky(column_covariance)
    Z = rng.normal(size=(n, r, c))
    return np.einsum("ab,nbc,dc->nad", Lr, Z, Lc)


# ---------------------------------------------------------------------------
# Baseline estimators used only by the benchmark


class EmpiricalScatter:
    def __init__(self, *, impute: bool = False, ridge: float = 1e-8):
        self.impute = bool(impute)
        self.ridge = float(ridge)

    def fit(self, X: np.ndarray) -> "EmpiricalScatter":
        X = np.asarray(X, dtype=float)
        if self.impute:
            X, self.impute_values_ = median_impute(X)
        elif not np.all(np.isfinite(X)):
            raise ValueError("EmpiricalScatter requires finite input")
        self.location_ = np.mean(X, axis=0)
        covariance = np.cov(X, rowvar=False, ddof=1)
        scale = max(float(np.trace(covariance)) / covariance.shape[0], 1.0)
        self.covariance_ = 0.5 * (covariance + covariance.T) + self.ridge * scale * np.eye(X.shape[1])
        self.precision_ = np.linalg.pinv(self.covariance_)
        centered = X - self.location_
        self.distances_ = np.einsum("ij,jk,ik->i", centered, self.precision_, centered)
        self.X_fit_ = X
        return self


class EmpiricalPCA:
    def __init__(self, n_components: int):
        self.n_components = int(n_components)

    def fit(self, X: np.ndarray) -> "EmpiricalPCA":
        X = np.asarray(X, dtype=float)
        self.location_ = np.mean(X, axis=0)
        _, singular_values, Vt = np.linalg.svd(X - self.location_, full_matrices=False)
        self.components_ = Vt[: self.n_components]
        self.eigenvalues_ = singular_values[: self.n_components] ** 2 / max(X.shape[0] - 1, 1)
        return self

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        scores = (X - self.location_) @ self.components_.T
        return self.location_ + scores @ self.components_

    def orthogonal_distances(self, X: np.ndarray) -> np.ndarray:
        return np.linalg.norm(np.asarray(X) - self.reconstruct(X), axis=1)


class ScatterPCA:
    """Small benchmark adapter for a pre-fitted scatter estimator."""

    def __init__(self, n_components: int, estimator: Any):
        self.n_components = int(n_components)
        self.estimator = estimator

    def fit(self, X: np.ndarray) -> "ScatterPCA":
        self.estimator.fit(X)
        self.location_ = np.asarray(self.estimator.location_, dtype=float)
        values, vectors = np.linalg.eigh(np.asarray(self.estimator.covariance_, dtype=float))
        order = np.argsort(values)[::-1]
        self.eigenvalues_ = values[order][: self.n_components]
        self.components_ = vectors[:, order[: self.n_components]].T
        return self

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        scores = (X - self.location_) @ self.components_.T
        return self.location_ + scores @ self.components_

    def orthogonal_distances(self, X: np.ndarray) -> np.ndarray:
        return np.linalg.norm(np.asarray(X) - self.reconstruct(X), axis=1)


def fit_matrix_normal_mle(X: np.ndarray, max_iter: int = 80, tol: float = 1e-7) -> dict[str, np.ndarray]:
    X = np.asarray(X, dtype=float)
    n, r, c = X.shape
    location = np.mean(X, axis=0)
    centered = X - location
    row_covariance = np.eye(r)
    column_covariance = np.eye(c)
    for _ in range(max_iter):
        old = np.kron(column_covariance, row_covariance)
        column_precision = np.linalg.pinv(column_covariance)
        row_covariance = sum(E @ column_precision @ E.T for E in centered) / (n * c)
        row_covariance = 0.5 * (row_covariance + row_covariance.T) + 1e-8 * np.eye(r)
        scale = float(np.trace(row_covariance) / r)
        row_covariance /= scale

        row_precision = np.linalg.pinv(row_covariance)
        column_covariance = sum(E.T @ row_precision @ E for E in centered) / (n * r)
        column_covariance = 0.5 * (column_covariance + column_covariance.T) + 1e-8 * np.eye(c)
        column_covariance *= scale
        current = np.kron(column_covariance, row_covariance)
        if np.linalg.norm(current - old, ord="fro") <= tol * max(np.linalg.norm(old, ord="fro"), 1.0):
            break
    row_precision = np.linalg.pinv(row_covariance)
    column_precision = np.linalg.pinv(column_covariance)
    distances = np.array(
        [np.trace(column_precision @ E.T @ row_precision @ E) for E in centered],
        dtype=float,
    )
    return {
        "location": location,
        "row_covariance": row_covariance,
        "column_covariance": column_covariance,
        "distances": distances,
    }


# ---------------------------------------------------------------------------
# Scatter benchmarks


def scatter_factories(scenario: str, profile: Profile) -> list[tuple[str, Callable[[], Any], str]]:
    has_missing = scenario in {
        "cellwise errors + missing",
        "high-dimensional mixed contamination",
    }
    fast = dict(
        quality="fast",
        n_init=profile.subset_starts,
        n_best=3,
        initial_c_steps=1,
        max_iter=50,
        random_state=0,
        scale_correction="none",
    )
    mrcd = dict(
        quality="fast",
        n_init=profile.subset_starts,
        n_best=3,
        initial_c_steps=1,
        max_iter=45,
        random_state=0,
    )
    common = [
        ("Empirical covariance", lambda: EmpiricalScatter(impute=has_missing), "non-robust baseline"),
        ("StudentTScatter", lambda: rc.StudentTScatter(df=3, alpha=0.08, max_iter=180, missing_values="median" if has_missing else "raise"), "diffuse heavy tails"),
        ("RegularizedCauchy", lambda: rc.RegularizedCauchy(alpha=0.12, max_iter=180, missing_values="median" if has_missing else "raise"), "very heavy tails and small samples"),
        ("RegularizedTyler", lambda: rc.RegularizedTyler(alpha=0.15, max_iter=250, scale_correction="radial_median", missing_values="median" if has_missing else "raise"), "elliptical shape with shrinkage"),
    ]
    high_dimensional = scenario in {
        "high-dimensional row outliers",
        "high-dimensional mixed contamination",
    }
    if not high_dimensional:
        common.insert(1, ("FastMCD", lambda: rc.FastMCD(missing_values="median" if has_missing else "raise", **fast), "separable rowwise outliers"))
        common.insert(2, ("MRCD", lambda: rc.MRCD(missing_values="median" if has_missing else "raise", **mrcd), "regularized high-breakdown subset"))
        if scenario in {"rowwise outliers", "heavy-tailed elliptical"}:
            common.insert(3, ("DetS", lambda: rc.DetS(max_iter=80, missing_values="median" if has_missing else "raise"), "deterministic high-breakdown S-estimator"))
            common.insert(4, ("DetMM", lambda: rc.DetMM(efficiency=0.95, max_iter=80, missing_values="median" if has_missing else "raise"), "S-started MM refinement for higher Gaussian efficiency"))
    else:
        common.insert(1, ("MRCD", lambda: rc.MRCD(missing_values="median" if has_missing else "raise", **mrcd), "rowwise outliers with p >= n"))
    if scenario == "cellwise errors + missing":
        common.append(("CellMCD", lambda: rc.CellMCD(max_iter=profile.cell_max_iter, min_samples_per_feature=None), "cellwise contamination and missing values"))
    if scenario == "high-dimensional mixed contamination":
        q = min(4, profile.high_n - 1, profile.high_p - 1)
        common.append((
            "CellRCov",
            lambda: rc.CellRCov(
                n_components=q,
                residual_shrinkage="auto",
                shrinkage_grid=(0.25, 0.5, 0.75, 1.0),
                cv_splits=3,
                cell_pca=rc.CellPCA(
                    n_components=q,
                    max_iter=profile.cell_max_iter,
                    tol=1e-4,
                ),
                score_estimator=rc.FastMCD(
                    quality="fast",
                    n_init=profile.subset_starts,
                    n_best=3,
                    initial_c_steps=1,
                    max_iter=45,
                    random_state=0,
                    scale_correction="none",
                ),
                random_state=0,
            ),
            "high-dimensional cellwise/casewise contamination and missing values",
        ))
    return common


def run_scatter_benchmarks(profile: Profile, repeats: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scenarios: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]] = []

    rng = np.random.default_rng(seed)
    covariance = toeplitz_covariance(profile.scatter_p)
    clean = rng.multivariate_normal(np.zeros(profile.scatter_p), covariance, size=profile.scatter_n)
    rowwise, row_labels = add_rowwise_outliers(rng, clean, 0.15)
    scenarios.append(("rowwise outliers", rowwise, covariance, np.zeros(profile.scatter_p), row_labels, None))

    rng = np.random.default_rng(seed + 1)
    heavy = sample_multivariate_t(rng, profile.scatter_n, covariance, df=3.0)
    scenarios.append(("heavy-tailed elliptical", heavy, covariance, np.zeros(profile.scatter_p), None, None))

    rng = np.random.default_rng(seed + 2)
    high_cov = toeplitz_covariance(profile.high_p, rho=0.35)
    high_clean = rng.multivariate_normal(np.zeros(profile.high_p), high_cov, size=profile.high_n)
    high, high_labels = add_rowwise_outliers(rng, high_clean, 0.12, magnitude=10.0)
    scenarios.append(("high-dimensional row outliers", high, high_cov, np.zeros(profile.high_p), high_labels, None))

    rng = np.random.default_rng(seed + 3)
    cell_clean = rng.multivariate_normal(np.zeros(profile.scatter_p), covariance, size=profile.scatter_n)
    cell_data, cell_labels, _ = add_cellwise_errors(rng, cell_clean, contamination=0.10, missing=0.03, magnitude=7.0)
    scenarios.append(("cellwise errors + missing", cell_data, covariance, np.zeros(profile.scatter_p), cell_labels.any(axis=1), cell_labels))

    rng = np.random.default_rng(seed + 4)
    q = min(4, profile.high_n - 1, profile.high_p - 1)
    raw_loadings = rng.normal(size=(profile.high_p, q))
    loadings, _ = np.linalg.qr(raw_loadings)
    strengths = np.linspace(3.2, 1.6, q)
    residual_scales = np.linspace(0.18, 0.38, profile.high_p)
    mixed_covariance = (loadings * strengths**2) @ loadings.T + np.diag(residual_scales**2)
    mixed_clean = rng.multivariate_normal(
        np.zeros(profile.high_p), mixed_covariance, size=profile.high_n
    )
    mixed, mixed_cell_labels, _ = add_cellwise_errors(
        rng, mixed_clean, contamination=0.045, missing=0.05, magnitude=8.0
    )
    mixed, mixed_row_labels = add_rowwise_outliers(
        rng, mixed, fraction=0.11, magnitude=6.0
    )
    scenarios.append((
        "high-dimensional mixed contamination",
        mixed,
        mixed_covariance,
        np.zeros(profile.high_p),
        mixed_row_labels,
        mixed_cell_labels,
    ))

    for scenario, X, truth_cov, truth_loc, row_labels, cell_labels in scenarios:
        for repeat in range(repeats):
            for name, factory, note in scatter_factories(scenario, profile):
                base = _base_row(
                    family="scatter",
                    scenario=scenario,
                    method=name,
                    n=X.shape[0],
                    p=X.shape[1],
                    repeat=repeat,
                )
                try:
                    estimator, seconds, peak = _measure(lambda: factory().fit(X))
                    base["seconds"] = seconds
                    base["python_peak_mb"] = peak
                    base["covariance_error"] = relative_frobenius(estimator.covariance_, truth_cov)
                    base["location_error"] = normalized_location_error(estimator.location_, truth_loc, truth_cov)
                    if row_labels is not None and not (scenario == "cellwise errors + missing" and name == "CellMCD"):
                        if hasattr(estimator, "distances_") and np.size(estimator.distances_) == X.shape[0]:
                            scores = np.asarray(estimator.distances_, dtype=float)
                        else:
                            X_score, _ = median_impute(X)
                            scores = mahalanobis_squared(X_score, estimator.location_, estimator.covariance_)
                        base["row_outlier_auc"] = binary_auc(row_labels, scores)
                    if cell_labels is not None and hasattr(estimator, "standardized_residuals_"):
                        observed = np.isfinite(X)
                        base["cell_outlier_auc"] = binary_auc(
                            cell_labels[observed], np.abs(np.asarray(estimator.standardized_residuals_)[observed])
                        )
                    base["notes"] = note
                    rows.append(base)
                except Exception as exc:
                    rows.append(_failure_row(
                        family="scatter",
                        scenario=scenario,
                        method=name,
                        n=X.shape[0],
                        p=X.shape[1],
                        repeat=repeat,
                        exc=exc,
                    ))
    return rows


# ---------------------------------------------------------------------------
# Kernel outlier-detection benchmark


def make_curved_manifold(
    rng: np.random.Generator,
    n_inliers: int,
    n_outliers: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = rng.uniform(-2.5, 2.5, n_inliers)
    inliers = np.column_stack([
        x,
        0.55 * x**2 + 0.08 * rng.normal(size=n_inliers),
    ])

    xo = rng.uniform(-1.8, 1.8, n_outliers)
    yo = rng.uniform(0.2, 2.0, n_outliers)
    close = np.abs(yo - 0.55 * xo**2) < 0.4
    while np.any(close):
        yo[close] = rng.uniform(0.2, 2.0, np.count_nonzero(close))
        close = np.abs(yo - 0.55 * xo**2) < 0.4
    outliers = np.column_stack([xo, yo])

    X = np.vstack([inliers, outliers])
    labels = np.r_[np.zeros(n_inliers, dtype=bool), np.ones(n_outliers, dtype=bool)]
    return X, labels


def run_kernel_benchmarks(profile: Profile, repeats: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed + 10)
    n_inliers = profile.scatter_n
    n_outliers = max(20, int(round(0.20 * n_inliers)))
    X, labels = make_curved_manifold(rng, n_inliers, n_outliers)

    subset = dict(
        contamination=n_outliers / X.shape[0],
        n_init=profile.kernel_subset_starts,
        n_best=3,
        initial_c_steps=2,
        max_iter=35,
        random_state=0,
    )
    methods: list[tuple[str, Callable[[], Any], str]] = [
        (
            "MRCD",
            lambda: rc.MRCD(**subset).fit(X),
            "linear robust subset baseline",
        ),
        (
            "KMRCD(linear)",
            lambda: rc.KMRCD(kernel="linear", regularization="auto", **subset).fit(X),
            "kernel formulation with linear geometry",
        ),
        (
            "KMRCD(RBF)",
            lambda: rc.KMRCD(kernel="rbf", gamma=2.0, **subset).fit(X),
            "nonlinear feature-space subset fit; gamma fixed for this scenario",
        ),
    ]

    rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        for name, fit_method, note in methods:
            base = _base_row(
                family="kernel outlier detection",
                scenario="curved manifold + off-manifold rows",
                method=name,
                n=X.shape[0],
                p=X.shape[1],
                repeat=repeat,
            )
            try:
                model, seconds, peak = _measure(fit_method)
                base["seconds"] = seconds
                base["python_peak_mb"] = peak
                base["row_outlier_auc"] = binary_auc(labels, model.distances_)
                base["notes"] = note
                rows.append(base)
            except Exception as exc:
                rows.append(_failure_row(
                    family="kernel outlier detection",
                    scenario="curved manifold + off-manifold rows",
                    method=name,
                    n=X.shape[0],
                    p=X.shape[1],
                    repeat=repeat,
                    exc=exc,
                ))
    return rows


# ---------------------------------------------------------------------------
# PCA benchmarks


def _pca_cell_scores(X: np.ndarray, reconstruction: np.ndarray, observed: np.ndarray) -> np.ndarray:
    residual = np.where(observed, X - reconstruction, np.nan)
    scales = 1.4826 * np.nanmedian(np.abs(residual - np.nanmedian(residual, axis=0)), axis=0)
    fallback = np.nanstd(residual, axis=0, ddof=1)
    scales = np.where(np.isfinite(scales) & (scales > 1e-8), scales, fallback)
    scales = np.where(np.isfinite(scales) & (scales > 1e-8), scales, 1.0)
    return np.abs(residual / scales)


def run_pca_benchmarks(profile: Profile, repeats: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    q = 3

    rng = np.random.default_rng(seed + 20)
    clean, truth_basis = make_low_rank(rng, profile.pca_n, profile.pca_p, q)
    rowwise = clean.copy()
    count = max(1, int(round(0.15 * profile.pca_n)))
    outlier_rows = rng.choice(profile.pca_n, size=count, replace=False)
    # Add variation mostly outside the clean subspace.
    orth = rng.normal(size=(profile.pca_p, q))
    orth -= truth_basis.T @ (truth_basis @ orth)
    orth, _ = np.linalg.qr(orth)
    rowwise[outlier_rows] += rng.normal(size=(count, q)) @ orth[:, :q].T * 7.0
    row_labels = np.zeros(profile.pca_n, dtype=bool)
    row_labels[outlier_rows] = True

    subset_args = dict(
        quality="fast", n_init=profile.subset_starts, n_best=3,
        initial_c_steps=1, max_iter=50, random_state=0,
    )
    row_methods: list[tuple[str, Callable[[], Any], str]] = [
        ("Empirical PCA", lambda: EmpiricalPCA(q), "non-robust baseline"),
        ("RobustPCA(FastMCD)", lambda: rc.RobustPCA(q, estimator=rc.FastMCD(scale_correction="none", **subset_args)), "separable rowwise outliers"),
        ("RobustPCA(MRCD)", lambda: rc.RobustPCA(q, estimator=rc.MRCD(**subset_args)), "regularized high-breakdown scatter PCA"),
        ("RobustPCA(StudentT)", lambda: rc.RobustPCA(q, estimator=rc.StudentTScatter(df=3, alpha=0.08, max_iter=180)), "diffuse heavy tails"),
        ("RobustPCA(Cauchy)", lambda: rc.RobustPCA(q, estimator=rc.RegularizedCauchy(alpha=0.12, max_iter=180)), "very heavy tails"),
        (
            "DensityPowerRobustPCA",
            lambda: rc.DensityPowerRobustPCA(
                n_components=q,
                alpha=0.30,
                max_iter=profile.pca_max_iter,
                tol=5e-4,
            ),
            "direct Gaussian DPD low-rank fit",
        ),
    ]

    for repeat in range(repeats):
        for name, factory, note in row_methods:
            base = _base_row(family="pca", scenario="rowwise low-rank outliers", method=name, n=rowwise.shape[0], p=rowwise.shape[1], repeat=repeat)
            try:
                model, seconds, peak = _measure(lambda: factory().fit(rowwise))
                base["seconds"] = seconds
                base["python_peak_mb"] = peak
                base["subspace_error"] = projection_error(model.components_, truth_basis)
                base["row_outlier_auc"] = binary_auc(row_labels, model.orthogonal_distances(rowwise))
                base["notes"] = note
                rows.append(base)
            except Exception as exc:
                rows.append(_failure_row(family="pca", scenario="rowwise low-rank outliers", method=name, n=rowwise.shape[0], p=rowwise.shape[1], repeat=repeat, exc=exc))

    rng = np.random.default_rng(seed + 21)
    clean, truth_basis = make_low_rank(rng, profile.pca_n, profile.pca_p, q)
    damaged, cell_labels, missing_mask = add_cellwise_errors(
        rng, clean, contamination=0.055, missing=0.035, magnitude=9.0
    )
    imputed, _ = median_impute(damaged)
    observed = np.isfinite(damaged)

    def make_empirical() -> Any:
        return EmpiricalPCA(q).fit(imputed)

    def make_cauchy() -> Any:
        return rc.RobustPCA(q, estimator=rc.RegularizedCauchy(alpha=0.12, max_iter=180)).fit(imputed)

    def make_cellmcd() -> Any:
        return ScatterPCA(q, rc.CellMCD(max_iter=profile.cell_max_iter, min_samples_per_feature=None)).fit(damaged)

    def make_cellpca() -> Any:
        return rc.CellPCA(n_components=q, max_iter=profile.pca_max_iter).fit(damaged)

    def make_dpd_pca() -> Any:
        return rc.DensityPowerRobustPCA(
            n_components=q,
            alpha=0.35,
            max_iter=profile.pca_max_iter,
            tol=5e-4,
        ).fit(imputed)

    cell_methods: list[tuple[str, Callable[[], Any], str]] = [
        ("Median-imputed PCA", make_empirical, "simple baseline"),
        ("RobustPCA(Cauchy, imputed)", make_cauchy, "rowwise/heavy-tail robustness after imputation"),
        ("DensityPowerRobustPCA, imputed", make_dpd_pca, "direct DPD low-rank fit after median imputation"),
        ("CellMCD scatter PCA", make_cellmcd, "cellwise robust scatter followed by eigendecomposition"),
        ("CellPCA", make_cellpca, "joint cellwise and casewise low-rank fit"),
    ]

    for repeat in range(repeats):
        for name, fit_method, note in cell_methods:
            base = _base_row(family="pca", scenario="cellwise low-rank + missing", method=name, n=damaged.shape[0], p=damaged.shape[1], repeat=repeat)
            try:
                model, seconds, peak = _measure(fit_method)
                base["seconds"] = seconds
                base["python_peak_mb"] = peak
                base["subspace_error"] = projection_error(model.components_, truth_basis)
                if isinstance(model, rc.CellwiseRobustPCA):
                    reconstruction = np.asarray(model.fitted_values_, dtype=float)
                    cell_scores = np.abs(np.asarray(model.standardized_residuals_, dtype=float))
                elif isinstance(model, rc.DensityPowerRobustPCA):
                    reconstruction = model.reconstruct(imputed)
                    cell_scores = 1.0 - model.cell_weights(imputed)
                else:
                    reconstruction = model.reconstruct(imputed)
                    cell_scores = _pca_cell_scores(damaged, reconstruction, observed)
                valid_cells = observed & ~missing_mask
                base["cell_outlier_auc"] = binary_auc(cell_labels[valid_cells], cell_scores[valid_cells])
                base["missing_reconstruction_mae"] = float(np.mean(np.abs(reconstruction[missing_mask] - clean[missing_mask])))
                base["notes"] = note
                rows.append(base)
            except Exception as exc:
                rows.append(_failure_row(family="pca", scenario="cellwise low-rank + missing", method=name, n=damaged.shape[0], p=damaged.shape[1], repeat=repeat, exc=exc))

    rng = np.random.default_rng(seed + 22)
    sparse_p = max(36, 2 * profile.pca_p)
    sparse_clean, sparse_truth = make_sparse_low_rank(
        rng, profile.pca_n, sparse_p, q
    )
    sparse_damaged, sparse_cell_labels, sparse_missing = add_cellwise_errors(
        rng, sparse_clean, contamination=0.045, missing=0.03, magnitude=8.0
    )
    sparse_imputed, _ = median_impute(sparse_damaged)
    sparse_observed = np.isfinite(sparse_damaged)

    sparse_methods: list[tuple[str, Callable[[], Any], str]] = [
        (
            "Median-imputed PCA",
            lambda: EmpiricalPCA(q).fit(sparse_imputed),
            "dense non-robust baseline",
        ),
        (
            "CellPCA",
            lambda: rc.CellPCA(
                n_components=q, max_iter=profile.pca_max_iter, tol=5e-4
            ).fit(sparse_damaged),
            "dense cellwise/casewise robust subspace",
        ),
        (
            "SparseCellPCA",
            lambda: rc.SparseCellPCA(
                n_components=q,
                alpha=0.055,
                l1_ratio=1.0,
                sparsity_threshold=0.02,
                max_iter=min(profile.pca_max_iter, 60),
                loading_max_iter=45,
                tol=5e-4,
            ).fit(sparse_damaged),
            "cellwise/casewise robustness with exact-zero elastic-net loadings",
        ),
    ]

    for repeat in range(repeats):
        for name, fit_method, note in sparse_methods:
            base = _base_row(
                family="pca",
                scenario="sparse cellwise low-rank + missing",
                method=name,
                n=sparse_damaged.shape[0],
                p=sparse_damaged.shape[1],
                repeat=repeat,
            )
            try:
                model, seconds, peak = _measure(fit_method)
                base["seconds"] = seconds
                base["python_peak_mb"] = peak
                base["subspace_error"] = projection_error(
                    model.components_, sparse_truth
                )
                if isinstance(model, rc.CellwiseRobustPCA):
                    reconstruction = np.asarray(model.fitted_values_, dtype=float)
                    cell_scores = np.abs(
                        np.asarray(model.standardized_residuals_, dtype=float)
                    )
                else:
                    reconstruction = model.reconstruct(sparse_imputed)
                    cell_scores = _pca_cell_scores(
                        sparse_damaged, reconstruction, sparse_observed
                    )
                valid_cells = sparse_observed & ~sparse_missing
                base["cell_outlier_auc"] = binary_auc(
                    sparse_cell_labels[valid_cells], cell_scores[valid_cells]
                )
                base["missing_reconstruction_mae"] = float(
                    np.mean(
                        np.abs(
                            reconstruction[sparse_missing]
                            - sparse_clean[sparse_missing]
                        )
                    )
                )
                precision, recall, f1, sparsity = loading_support_metrics(
                    model.components_, sparse_truth
                )
                base["loading_support_precision"] = precision
                base["loading_support_recall"] = recall
                base["loading_support_f1"] = f1
                base["loading_sparsity"] = sparsity
                base["notes"] = note
                rows.append(base)
            except Exception as exc:
                rows.append(_failure_row(
                    family="pca",
                    scenario="sparse cellwise low-rank + missing",
                    method=name,
                    n=sparse_damaged.shape[0],
                    p=sparse_damaged.shape[1],
                    repeat=repeat,
                    exc=exc,
                ))
    return rows


# ---------------------------------------------------------------------------
# Matrix-valued covariance benchmark


def run_matrix_benchmarks(profile: Profile, repeats: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed + 40)
    r, c = 4, 6
    row_cov = toeplitz_covariance(r, rho=0.45)
    col_cov = toeplitz_covariance(c, rho=0.65)
    clean = matrix_normal_sample(rng, profile.matrix_n, row_cov, col_cov)
    X = clean.copy()
    count = max(1, int(round(0.16 * profile.matrix_n)))
    bad = rng.choice(profile.matrix_n, size=count, replace=False)
    for i in bad:
        row = int(rng.integers(r))
        start = int(rng.integers(max(c - 2, 1)))
        X[i, row, start : start + 3] += rng.choice([-1.0, 1.0]) * 4.0
    labels = np.zeros(profile.matrix_n, dtype=bool)
    labels[bad] = True
    truth = np.kron(col_cov, row_cov)

    methods: list[tuple[str, Callable[[], dict[str, Any]], str]] = [
        ("All-sample matrix-normal MLE", lambda: fit_matrix_normal_mle(X), "non-robust separable baseline"),
        (
            "MMCD",
            lambda: rc.MMCD(
                contamination=0.18,
                quality="fast",
                n_init=profile.subset_starts,
                n_best=3,
                initial_c_steps=1,
                max_iter=35,
                flip_flop_max_iter=45,
                random_state=0,
            ).fit(X),
            "rowwise contamination in matrix-valued observations",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        for name, fit_method, note in methods:
            base = _base_row(family="matrix covariance", scenario="localized matrix faults", method=name, n=X.shape[0], p=r * c, repeat=repeat)
            try:
                model, seconds, peak = _measure(fit_method)
                if isinstance(model, dict):
                    estimate = np.kron(model["column_covariance"], model["row_covariance"])
                    distances = model["distances"]
                else:
                    estimate = model.kronecker_covariance()
                    distances = model.mahalanobis(X)
                base["seconds"] = seconds
                base["python_peak_mb"] = peak
                base["matrix_covariance_error"] = relative_frobenius(estimate, truth)
                base["row_outlier_auc"] = binary_auc(labels, distances)
                base["notes"] = note
                rows.append(base)
            except Exception as exc:
                rows.append(_failure_row(family="matrix covariance", scenario="localized matrix faults", method=name, n=X.shape[0], p=r * c, repeat=repeat, exc=exc))
    return rows


# ---------------------------------------------------------------------------
# Sparse precision benchmark


def run_graph_benchmarks(profile: Profile, repeats: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # Heavy-tailed elliptical graph: spatial signs are designed for radial
    # heavy tails and estimate a trace-normalized shape precision.
    rng = np.random.default_rng(seed + 59)
    truth_precision, truth_covariance = make_sparse_precision(profile.graph_p)
    heavy = sample_multivariate_t(
        rng, profile.graph_n, truth_covariance, df=2.0
    )
    radial_rows = rng.choice(
        profile.graph_n,
        size=max(1, int(round(0.12 * profile.graph_n))),
        replace=False,
    )
    heavy[radial_rows] *= rng.uniform(4.0, 12.0, size=radial_rows.size)[:, None]
    truth_adjacency = np.abs(truth_precision) > 1e-12
    np.fill_diagonal(truth_adjacency, False)
    truth_partial = partial_correlation(truth_precision)
    heavy_kwargs = dict(
        alpha=0.08,
        max_iter=profile.graph_max_iter,
        edge_tolerance=1e-6,
    )
    heavy_methods: list[tuple[str, Callable[[], Any], str]] = [
        (
            "Empirical graphical lasso",
            lambda: rc.RobustGraphicalLasso(
                scatter_estimator="empirical", **heavy_kwargs
            ).fit(heavy),
            "non-robust Gaussian baseline",
        ),
        (
            "Cauchy graphical lasso",
            lambda: rc.RobustGraphicalLasso(
                scatter_estimator=rc.RegularizedCauchy(alpha=0.12, max_iter=180),
                **heavy_kwargs,
            ).fit(heavy),
            "robust-scatter graphical lasso",
        ),
        (
            "Spatial-sign graphical lasso",
            lambda: rc.SGLASSO(**heavy_kwargs).fit(heavy),
            "native spatial-sign shape graph for heavy-tailed elliptical data",
        ),
    ]
    for repeat in range(repeats):
        for name, fit_method, note in heavy_methods:
            base = _base_row(
                family="sparse precision",
                scenario="heavy-tailed elliptical graph",
                method=name,
                n=heavy.shape[0],
                p=heavy.shape[1],
                repeat=repeat,
            )
            try:
                model, seconds, peak = _measure(fit_method)
                precision, recall, f1, n_edges = graph_metrics(
                    model.adjacency_, truth_adjacency
                )
                base["seconds"] = seconds
                base["python_peak_mb"] = peak
                base["precision_error"] = relative_frobenius(
                    model.partial_correlation_, truth_partial
                )
                base["edge_precision"] = precision
                base["edge_recall"] = recall
                base["edge_f1"] = f1
                base["n_edges"] = n_edges
                base["notes"] = note
                rows.append(base)
            except Exception as exc:
                rows.append(_failure_row(
                    family="sparse precision",
                    scenario="heavy-tailed elliptical graph",
                    method=name,
                    n=heavy.shape[0],
                    p=heavy.shape[1],
                    repeat=repeat,
                    exc=exc,
                ))

    # Mixed bad cells and missing values: spatial signs are intentionally not
    # included because one damaged coordinate can rotate an entire sign vector.
    rng = np.random.default_rng(seed + 60)
    truth_precision, truth_covariance = make_sparse_precision(profile.graph_p)
    X = sample_multivariate_t(rng, profile.graph_n, truth_covariance, df=4.0)
    damaged, _, _ = add_cellwise_errors(
        rng, X, contamination=0.05, missing=0.02, magnitude=9.0
    )
    imputed, _ = median_impute(damaged)
    truth_adjacency = np.abs(truth_precision) > 1e-12
    np.fill_diagonal(truth_adjacency, False)
    truth_partial = partial_correlation(truth_precision)

    graph_kwargs = dict(
        # A common fixed penalty isolates the effect of the scatter estimate.
        # Penalty-path selection is benchmarked separately by the package.
        alpha=0.05,
        max_iter=profile.graph_max_iter,
        edge_tolerance=1e-6,
    )
    methods: list[tuple[str, Callable[[], Any], str]] = [
        (
            "Empirical graphical lasso",
            lambda: rc.RobustGraphicalLasso(
                scatter_estimator="empirical", **graph_kwargs
            ).fit(imputed),
            "non-robust baseline after median imputation",
        ),
        (
            "Cauchy graphical lasso",
            lambda: rc.RobustGraphicalLasso(
                scatter_estimator=rc.RegularizedCauchy(alpha=0.12, max_iter=180),
                **graph_kwargs,
            ).fit(imputed),
            "heavy-tail robust scatter after imputation",
        ),
        (
            "CellMCD graphical lasso",
            lambda: rc.RobustGraphicalLasso(
                scatter_estimator=rc.CellMCD(
                    max_iter=profile.cell_max_iter,
                    min_samples_per_feature=None,
                ),
                **graph_kwargs,
            ).fit(damaged),
            "cellwise contamination and missing values",
        ),
    ]

    for repeat in range(repeats):
        for name, fit_method, note in methods:
            base = _base_row(
                family="sparse precision",
                scenario="heavy tails + bad cells",
                method=name,
                n=damaged.shape[0],
                p=damaged.shape[1],
                repeat=repeat,
            )
            try:
                model, seconds, peak = _measure(fit_method)
                precision, recall, f1, n_edges = graph_metrics(
                    model.adjacency_, truth_adjacency
                )
                base["seconds"] = seconds
                base["python_peak_mb"] = peak
                base["precision_error"] = relative_frobenius(
                    model.partial_correlation_, truth_partial
                )
                base["edge_precision"] = precision
                base["edge_recall"] = recall
                base["edge_f1"] = f1
                base["n_edges"] = n_edges
                base["notes"] = note
                rows.append(base)
            except Exception as exc:
                rows.append(_failure_row(
                    family="sparse precision",
                    scenario="heavy tails + bad cells",
                    method=name,
                    n=damaged.shape[0],
                    p=damaged.shape[1],
                    repeat=repeat,
                    exc=exc,
                ))
    return rows


# ---------------------------------------------------------------------------
# Output and generated documentation


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["family"]), str(row["scenario"]), str(row["method"])), []).append(row)
    output: list[dict[str, Any]] = []
    numeric = [field for field in CSV_FIELDS if field not in {
        "family", "scenario", "method", "status", "repeat", "notes"
    }]
    for (family, scenario, method), group in groups.items():
        ok = [row for row in group if row.get("status") == "ok"]
        aggregate = {
            "family": family,
            "scenario": scenario,
            "method": method,
            "status": "ok" if ok else "failed",
            "notes": next((str(row.get("notes", "")) for row in ok if row.get("notes")), str(group[0].get("notes", ""))),
        }
        for field in numeric:
            values = []
            for row in ok:
                try:
                    value = float(row.get(field, ""))
                except Exception:
                    continue
                if np.isfinite(value):
                    values.append(value)
            aggregate[field] = float(np.median(values)) if values else ""
        output.append(aggregate)
    return sorted(output, key=lambda row: (row["family"], row["scenario"], row["method"]))


def _format(value: Any, digits: int = 3) -> str:
    if value == "" or value is None:
        return "—"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(value):
        return "—"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) < 0.001 and value != 0:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def _rst_table(headers: list[str], table_rows: list[list[str]], widths: str | None = None) -> str:
    lines = [".. list-table::", "   :header-rows: 1"]
    if widths:
        lines.append(f"   :widths: {widths}")
    lines.append("")
    lines.append("   * - " + headers[0])
    for header in headers[1:]:
        lines.append("     - " + header)
    for row in table_rows:
        lines.append("   * - " + row[0])
        for cell in row[1:]:
            lines.append("     - " + cell)
    return "\n".join(lines) + "\n"


def write_rst(path: Path, rows: list[dict[str, Any]], profile_name: str, repeats: int) -> None:
    aggregated = _aggregate(rows)
    lines = [
        ".. This file is generated by benchmarks/compare_methods.py.",
        ".. Do not edit benchmark values by hand.",
        "",
        "Benchmark snapshot",
        "------------------",
        "",
        f"These tables were generated with the ``{profile_name}`` profile and {repeats} run(s) per method.",
        "Lower errors and runtimes are better; AUROC and edge F1 are better when higher.",
        "When requested, the CSV records Python ``tracemalloc`` peak memory; it excludes native C++ and BLAS allocations and is not shown in the table below.",
        "",
    ]

    scatter = [row for row in aggregated if row["family"] == "scatter"]
    lines.extend(["Scatter and covariance", "~~~~~~~~~~~~~~~~~~~~~~", ""])
    table = []
    for row in scatter:
        table.append([
            str(row["scenario"]), str(row["method"]), _format(row["covariance_error"]),
            _format(row["row_outlier_auc"]), _format(row["cell_outlier_auc"]),
            _format(row["seconds"]), str(row["status"]),
        ])
    lines.append(_rst_table(
        ["Scenario", "Method", "Covariance error", "Row AUROC", "Cell AUROC", "Seconds", "Status"],
        table,
        "22 25 13 10 10 9 8",
    ))

    kernel_rows = [row for row in aggregated if row["family"] == "kernel outlier detection"]
    lines.extend(["Nonlinear kernel outlier detection", "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", ""])
    table = [[
        str(row["method"]), _format(row["row_outlier_auc"]),
        _format(row["seconds"]), str(row["notes"]), str(row["status"]),
    ] for row in kernel_rows]
    lines.append(_rst_table(
        ["Method", "Outlier AUROC", "Seconds", "Role", "Status"],
        table,
        "24 12 9 36 8",
    ))

    pca = [row for row in aggregated if row["family"] == "pca"]
    lines.extend(["Principal subspaces", "~~~~~~~~~~~~~~~~~~~", ""])
    table = []
    for row in pca:
        table.append([
            str(row["scenario"]), str(row["method"]), _format(row["subspace_error"]),
            _format(row["row_outlier_auc"]), _format(row["cell_outlier_auc"]),
            _format(row["missing_reconstruction_mae"]),
            _format(row["loading_support_f1"]),
            _format(row["loading_sparsity"]),
            _format(row["seconds"]),
        ])
    lines.append(_rst_table(
        ["Scenario", "Method", "Subspace error", "Row AUROC", "Cell AUROC", "Missing MAE", "Support F1", "Sparsity", "Seconds"],
        table,
        "20 22 10 8 8 10 9 8 8",
    ))

    matrix_rows = [row for row in aggregated if row["family"] == "matrix covariance"]
    lines.extend(["Matrix-valued observations", "~~~~~~~~~~~~~~~~~~~~~~~~~~", ""])
    table = [[
        str(row["method"]), _format(row["matrix_covariance_error"]),
        _format(row["row_outlier_auc"]), _format(row["seconds"]), str(row["notes"]),
    ] for row in matrix_rows]
    lines.append(_rst_table(
        ["Method", "Kronecker covariance error", "Outlier AUROC", "Seconds", "Role"],
        table,
        "24 18 12 9 30",
    ))

    graph_rows = [row for row in aggregated if row["family"] == "sparse precision"]
    lines.extend(["Sparse conditional-dependence graphs", "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", ""])
    table = [[
        str(row["method"]), _format(row["precision_error"]), _format(row["edge_precision"]),
        _format(row["edge_recall"]), _format(row["edge_f1"]), _format(row["n_edges"], digits=0),
        _format(row["seconds"]),
    ] for row in graph_rows]
    lines.append(_rst_table(
        ["Method", "Partial-correlation error", "Edge precision", "Edge recall", "Edge F1", "Edges", "Seconds"],
        table,
        "24 17 11 10 9 7 8",
    ))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def print_rows(rows: list[dict[str, Any]]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="quick")
    parser.add_argument("--repeats", type=int, default=None, help="Default: 1. Use 3 or more for stable timing comparisons.")
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--measure-python-memory", action="store_true", help="Record tracemalloc peak memory. This slows Python-heavy methods and excludes native allocations.")
    parser.add_argument("--families", nargs="+", choices=["scatter", "kernel", "pca", "matrix", "graph"], default=["scatter", "kernel", "pca", "matrix", "graph"])
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--rst", type=Path, default=None)
    args = parser.parse_args()

    global _MEASURE_PYTHON_MEMORY
    _MEASURE_PYTHON_MEMORY = bool(args.measure_python_memory)

    profile = PROFILES[args.profile]
    repeats = args.repeats if args.repeats is not None else 1
    if repeats < 1:
        raise ValueError("repeats must be at least one")

    rows: list[dict[str, Any]] = []
    if "scatter" in args.families:
        rows.extend(run_scatter_benchmarks(profile, repeats, args.seed))
    if "kernel" in args.families:
        rows.extend(run_kernel_benchmarks(profile, repeats, args.seed))
    if "pca" in args.families:
        rows.extend(run_pca_benchmarks(profile, repeats, args.seed))
    if "matrix" in args.families:
        rows.extend(run_matrix_benchmarks(profile, repeats, args.seed))
    if "graph" in args.families:
        rows.extend(run_graph_benchmarks(profile, repeats, args.seed))

    print_rows(rows)
    if args.csv is not None:
        write_csv(args.csv, rows)
    if args.rst is not None:
        write_rst(args.rst, rows, args.profile, repeats)


if __name__ == "__main__":
    main()
