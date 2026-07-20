# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Optional native numerical kernels with exact NumPy fallbacks."""

from __future__ import annotations

from typing import Literal

import numpy as np

_EXPECTED_NATIVE_API = 2

_native_import_error: BaseException | None = None
try:  # pragma: no cover - availability depends on the local build
    from . import _robustcov_cpp as _cpp
except Exception as exc:  # pragma: no cover
    _cpp = None
    _native_import_error = exc
else:  # pragma: no cover - exercised by native and stale-build subprocess tests
    native_api = getattr(_cpp, "__robustcov_native_api__", None)
    if native_api != _EXPECTED_NATIVE_API:
        found = "missing" if native_api is None else repr(native_api)
        _native_import_error = RuntimeError(
            "incompatible robustcov native extension: "
            f"Python expects native API {_EXPECTED_NATIVE_API}, found {found}. "
            "Remove stale editable/build artifacts and rebuild the extension."
        )
        _cpp = None

Backend = Literal["auto", "python", "cpp"]


def native_available() -> bool:
    """Return whether the optional compiled extension can be imported."""
    return _cpp is not None


def require_native(feature: str = "This operation"):
    """Return the native module or raise an actionable runtime error."""
    if _cpp is None:
        detail = f" Reason: {_native_import_error}" if _native_import_error is not None else ""
        raise RuntimeError(
            f"{feature} requires the robustcov native extension, but it is unavailable."
            f"{detail} Install a supported binary wheel, remove stale build/editable "
            "artifacts and rebuild with a C++17 compiler, or choose an "
            "estimator/backend with a NumPy fallback."
        ) from _native_import_error
    return _cpp


def resolve_backend(backend: Backend) -> str:
    if backend not in {"auto", "python", "cpp"}:
        raise ValueError("backend must be 'auto', 'python', or 'cpp'")
    if backend == "cpp" and _cpp is None:
        raise RuntimeError(
            "backend='cpp' requested, but the robustcov native extension is unavailable"
        )
    return "cpp" if backend == "auto" and _cpp is not None else backend




def joint_diagonalize_symmetric_batch(
    matrices: np.ndarray,
    *,
    max_sweeps: int = 100,
    tol: float = 1e-10,
    backend: Backend = "auto",
):
    """Jointly diagonalize symmetric matrices through Python or C++."""
    matrices = np.asarray(matrices, dtype=np.float64, order="C")
    selected = resolve_backend(backend)
    if selected == "cpp" and not hasattr(_cpp, "joint_diagonalize_symmetric"):
        if backend == "cpp":
            raise RuntimeError(
                "the installed robustcov native extension does not provide "
                "joint diagonalization; remove stale build artifacts and rebuild"
            )
        selected = "python"
    if backend == "auto" and selected == "cpp":
        if matrices.ndim != 3:
            selected = "python"
        else:
            n_matrices, n_features, _ = matrices.shape
            # Python/NumPy is competitive for tiny collections.  The native
            # Jacobi sweeps clear the 1.5x complete-SOBI gate for p >= 8.
            if n_features < 8 or n_matrices * n_features < 64:
                selected = "python"
    if selected == "cpp":
        rotation, diagonalized, info = _cpp.joint_diagonalize_symmetric(
            matrices, int(max_sweeps), float(tol)
        )
        return (
            np.asarray(rotation, dtype=np.float64),
            np.asarray(diagonalized, dtype=np.float64),
            dict(info),
        )
    return None

def mahalanobis_squared_batch(
    X: np.ndarray,
    location: np.ndarray,
    precision: np.ndarray,
    *,
    backend: Backend = "auto",
) -> np.ndarray:
    """Squared vector Mahalanobis distances using NumPy or the native kernel.

    ``backend='auto'`` uses C++ only for workloads large enough to amortize the
    extension-call overhead. The cutoff is intentionally conservative and is
    validated by ``benchmarks/native_port_gate.py``.
    """
    X = np.asarray(X, dtype=np.float64, order="C")
    location = np.asarray(location, dtype=np.float64, order="C")
    precision = np.asarray(precision, dtype=np.float64, order="C")
    selected = resolve_backend(backend)
    if backend == "auto" and selected == "cpp":
        n, p = int(X.shape[0]), int(X.shape[1])
        # Calibrated against the 1.5x native-port gate. The extension wins for
        # very small batches (lower dispatch/planning overhead), low-dimensional
        # data, and sufficiently large p=16..32 batches. NumPy remains faster for
        # many medium-sized dense BLAS workloads.
        small_batch_limit = max(1, 25_600 // max(1, p * p))
        use_cpp = (
            p <= 4
            or n <= small_batch_limit
            or (16 <= p <= 32 and n * p >= 500_000)
        )
        if not use_cpp:
            selected = "python"
    if selected == "cpp":
        return np.asarray(
            _cpp.mahalanobis2_batch(X, location, precision),
            dtype=np.float64,
        )
    centered = X - location
    return np.maximum(
        np.einsum("ij,jk,ik->i", centered, precision, centered, optimize=True),
        0.0,
    )

def matrix_mahalanobis2_batch(
    X: np.ndarray,
    location: np.ndarray,
    row_precision: np.ndarray,
    column_precision: np.ndarray,
    *,
    backend: Backend = "auto",
) -> np.ndarray:
    """Squared matrix Mahalanobis distances using Python or the native kernel."""
    X = np.asarray(X, dtype=np.float64, order="C")
    location = np.asarray(location, dtype=np.float64, order="C")
    row_precision = np.asarray(row_precision, dtype=np.float64, order="C")
    column_precision = np.asarray(column_precision, dtype=np.float64, order="C")
    selected = resolve_backend(backend)
    if backend == "auto" and selected == "cpp":
        n, r, c = map(int, X.shape)
        estimated_operations = n * r * c * (r + c)
        # NumPy's optimized contractions are faster for medium-sized batches.
        # The native path clears the 1.5x gate once the matrix workload is large
        # enough to amortize dispatch and temporary-buffer costs.
        if estimated_operations < 25_000_000:
            selected = "python"
    if selected == "cpp":
        return np.asarray(
            _cpp.matrix_mahalanobis2_batch(
                X, location, row_precision, column_precision
            ),
            dtype=np.float64,
        )
    residuals = X - location
    transformed = np.einsum(
        "ab,nbc,cd->nad", row_precision, residuals, column_precision,
        optimize=True,
    )
    return np.maximum(
        np.einsum("nij,nij->n", residuals, transformed, optimize=True), 0.0
    )


def weighted_tucker_scores_2d(
    X: np.ndarray,
    weights: np.ndarray,
    center: np.ndarray,
    row_components: np.ndarray,
    column_components: np.ndarray,
    *,
    ridge: float = 1e-8,
    backend: Backend = "auto",
) -> np.ndarray:
    """Weighted Tucker core scores for matrix-valued samples."""
    X = np.asarray(X, dtype=np.float64, order="C")
    weights = np.asarray(weights, dtype=np.float64, order="C")
    center = np.asarray(center, dtype=np.float64, order="C")
    row_components = np.asarray(row_components, dtype=np.float64, order="C")
    column_components = np.asarray(column_components, dtype=np.float64, order="C")
    selected = resolve_backend(backend)
    if selected == "cpp":
        return np.asarray(
            _cpp.weighted_tucker_scores_2d(
                X,
                weights,
                center,
                row_components,
                column_components,
                float(ridge),
            ),
            dtype=np.float64,
        )

    n, n_rows, n_columns = X.shape
    rank_rows = row_components.shape[1]
    rank_columns = column_components.shape[1]
    q = rank_rows * rank_columns
    scores = np.zeros((n, rank_rows, rank_columns), dtype=np.float64)
    identity = np.eye(q, dtype=np.float64)
    design = np.einsum(
        "au,bv->abuv", row_components, column_components, optimize=True
    ).reshape(n_rows, n_columns, q)
    centered = X - center
    for i in range(n):
        w = weights[i].reshape(-1)
        Z = design.reshape(-1, q)
        gram = Z.T @ (w[:, None] * Z) + float(ridge) * identity
        rhs = Z.T @ (w * centered[i].reshape(-1))
        scores[i] = np.linalg.solve(gram, rhs).reshape(rank_rows, rank_columns)
    return scores


__all__ = [
    "native_available",
    "require_native",
    "resolve_backend",
    "joint_diagonalize_symmetric_batch",
    "mahalanobis_squared_batch",
    "matrix_mahalanobis2_batch",
    "weighted_tucker_scores_2d",
]
