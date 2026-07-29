# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Joint diagonalization and blind-source-separation evaluation utilities.

The routines in this module deliberately separate numerical diagonalization from
estimator policy.  ICA and SOBI both need to compare matrices only up to source
permutation, sign, and scale; the metrics below make those indeterminacies
explicit instead of relying on fragile elementwise comparisons.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._native import joint_diagonalize_symmetric_batch


def _as_symmetric_stack(matrices: np.ndarray) -> np.ndarray:
    arrays = np.asarray(matrices, dtype=np.float64)
    if arrays.ndim == 2:
        arrays = arrays[None, :, :]
    if arrays.ndim != 3 or arrays.shape[1] != arrays.shape[2]:
        raise ValueError("matrices must have shape (n_matrices, p, p) or (p, p)")
    if arrays.shape[0] < 1 or arrays.shape[1] < 1:
        raise ValueError("at least one non-empty square matrix is required")
    if not np.all(np.isfinite(arrays)):
        raise ValueError("matrices must contain only finite values")
    return 0.5 * (arrays + np.swapaxes(arrays, 1, 2))


def off_diagonal_energy(matrices: np.ndarray) -> float:
    """Return the total squared off-diagonal energy of symmetric matrices."""

    arrays = _as_symmetric_stack(matrices)
    diagonals = np.zeros_like(arrays)
    indices = np.arange(arrays.shape[1])
    diagonals[:, indices, indices] = arrays[:, indices, indices]
    return float(np.sum((arrays - diagonals) ** 2))


def joint_diagonalize_symmetric(
    matrices: np.ndarray,
    *,
    max_sweeps: int = 100,
    tol: float = 1e-10,
    backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | bool]]:
    """Approximately jointly diagonalize real symmetric matrices.

    A Jacobi sweep chooses the optimal plane rotation for every coordinate pair
    with respect to the sum of squared diagonal entries.  The returned rotation
    ``V`` is orthogonal and the transformed matrices satisfy
    ``D[k] = V.T @ matrices[k] @ V``.

    Parameters
    ----------
    matrices : array-like of shape (n_matrices, p, p)
        Real symmetric matrices to diagonalize.
    max_sweeps : int, default=100
        Maximum number of complete Jacobi sweeps.
    tol : float, default=1e-10
        Stop when every sine rotation in a sweep is below this value.
    backend : {'auto', 'python', 'cpp'}, default='auto'
        Numerical backend. ``auto`` uses C++ only after the workload clears
        the complete-SOBI acceleration gate.

    Returns
    -------
    rotation : ndarray of shape (p, p)
        Orthogonal joint diagonalizer.
    diagonalized : ndarray of shape (n_matrices, p, p)
        Jointly rotated matrices.
    info : dict
        Convergence flag, number of sweeps, and initial/final off-diagonal
        energies.
    """

    if backend not in {"auto", "python", "cpp"}:
        raise ValueError("backend must be 'auto', 'python', or 'cpp'")
    if not isinstance(max_sweeps, (int, np.integer)) or max_sweeps < 1:
        raise ValueError("max_sweeps must be a positive integer")
    if not np.isscalar(tol) or not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be a positive finite number")

    diagonalized = _as_symmetric_stack(matrices).copy()
    native_result = joint_diagonalize_symmetric_batch(
        diagonalized,
        max_sweeps=int(max_sweeps),
        tol=float(tol),
        backend=backend,
    )
    if native_result is not None:
        return native_result
    n_matrices, n_features, _ = diagonalized.shape
    del n_matrices
    rotation = np.eye(n_features, dtype=np.float64)
    initial_energy = off_diagonal_energy(diagonalized)
    converged = False
    sweeps = 0

    for sweep in range(1, int(max_sweeps) + 1):
        largest_sine = 0.0
        for left in range(n_features - 1):
            for right in range(left + 1, n_features):
                diagonal_difference = (
                    diagonalized[:, left, left] - diagonalized[:, right, right]
                )
                twice_cross = 2.0 * diagonalized[:, left, right]
                gram = np.array(
                    [
                        [
                            float(diagonal_difference @ diagonal_difference),
                            float(diagonal_difference @ twice_cross),
                        ],
                        [
                            float(diagonal_difference @ twice_cross),
                            float(twice_cross @ twice_cross),
                        ],
                    ],
                    dtype=np.float64,
                )
                eigenvalues, eigenvectors = np.linalg.eigh(gram)
                cosine_twice, sine_twice = eigenvectors[:, int(np.argmax(eigenvalues))]
                if cosine_twice < 0.0:
                    cosine_twice = -cosine_twice
                    sine_twice = -sine_twice
                angle = 0.5 * np.arctan2(sine_twice, cosine_twice)
                cosine = float(np.cos(angle))
                sine = float(np.sin(angle))
                if abs(sine) <= tol:
                    continue

                plane = np.array([[cosine, -sine], [sine, cosine]])
                pair = [left, right]
                diagonalized[:, :, pair] = diagonalized[:, :, pair] @ plane
                diagonalized[:, pair, :] = np.einsum(
                    "ab,kbj->kaj", plane.T, diagonalized[:, pair, :], optimize=True
                )
                rotation[:, pair] = rotation[:, pair] @ plane
                largest_sine = max(largest_sine, abs(sine))

        sweeps = sweep
        if largest_sine <= tol:
            converged = True
            break

    diagonalized = 0.5 * (
        diagonalized + np.swapaxes(diagonalized, 1, 2)
    )
    final_energy = off_diagonal_energy(diagonalized)
    info: dict[str, float | int | bool] = {
        "converged": converged,
        "n_sweeps": sweeps,
        "initial_off_diagonal_energy": initial_energy,
        "off_diagonal_energy": final_energy,
    }
    return rotation, diagonalized, info


def gain_matrix(unmixing: np.ndarray, mixing: np.ndarray) -> np.ndarray:
    """Return the source gain matrix ``unmixing @ mixing``."""

    unmixing = np.asarray(unmixing, dtype=np.float64)
    mixing = np.asarray(mixing, dtype=np.float64)
    if unmixing.ndim != 2 or mixing.ndim != 2:
        raise ValueError("unmixing and mixing must be 2D arrays")
    if unmixing.shape[1] != mixing.shape[0]:
        raise ValueError("unmixing and mixing have incompatible shapes")
    result = unmixing @ mixing
    if result.shape[0] != result.shape[1]:
        raise ValueError("BSS metrics require the same number of recovered and true sources")
    if not np.all(np.isfinite(result)):
        raise ValueError("gain matrix contains non-finite values")
    return result


def amari_index(unmixing: np.ndarray, mixing: np.ndarray) -> float:
    """Return the scale/permutation-invariant Amari separation index.

    The index is zero for exact recovery up to source permutation and scaling.
    """

    absolute_gain = np.abs(gain_matrix(unmixing, mixing))
    n_sources = absolute_gain.shape[0]
    if n_sources == 1:
        return 0.0
    row_max = np.max(absolute_gain, axis=1)
    column_max = np.max(absolute_gain, axis=0)
    tiny = np.finfo(np.float64).tiny
    if np.any(row_max <= tiny) or np.any(column_max <= tiny):
        return float("inf")
    row_term = np.sum(np.sum(absolute_gain / row_max[:, None], axis=1) - 1.0)
    column_term = np.sum(
        np.sum(absolute_gain / column_max[None, :], axis=0) - 1.0
    )
    return float((row_term + column_term) / (2.0 * n_sources * (n_sources - 1)))


def minimum_distance_index(unmixing: np.ndarray, mixing: np.ndarray) -> float:
    """Return the minimum-distance index for an estimated unmixing matrix.

    For each recovered row the optimal scaling is solved analytically, while a
    Hungarian assignment handles the unknown source permutation.  The result is
    zero for exact source recovery and is normalized to lie in ``[0, 1]`` up to
    floating-point error.
    """

    gain = gain_matrix(unmixing, mixing)
    n_sources = gain.shape[0]
    if n_sources == 1:
        return 0.0
    row_norm_squared = np.sum(gain * gain, axis=1)
    if np.any(row_norm_squared <= np.finfo(np.float64).tiny):
        return 1.0
    squared_alignment = (gain * gain) / row_norm_squared[:, None]
    row_indices, column_indices = linear_sum_assignment(-squared_alignment)
    residual = float(
        np.sum(1.0 - squared_alignment[row_indices, column_indices])
    )
    residual = max(residual, 0.0)
    return float(np.sqrt(residual / (n_sources - 1)))


def canonicalize_unmixing(
    unmixing: np.ndarray,
    *,
    mixing: np.ndarray | None = None,
    order: np.ndarray | list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Choose deterministic source order and signs for an unmixing matrix.

    Parameters
    ----------
    unmixing : ndarray of shape (n_sources, n_features)
        Estimated unmixing matrix.
    mixing : ndarray, optional
        Matching mixing matrix.  If omitted, the Moore--Penrose inverse of the
        unmixing matrix is used.
    order : array-like, optional
        Explicit component permutation.  If omitted, components are ordered by
        the feature index of their largest absolute mixing loading and then by
        decreasing loading magnitude.
    """

    unmixing = np.asarray(unmixing, dtype=np.float64)
    if unmixing.ndim != 2 or not np.all(np.isfinite(unmixing)):
        raise ValueError("unmixing must be a finite 2D array")
    if mixing is None:
        mixing = np.linalg.pinv(unmixing)
    mixing = np.asarray(mixing, dtype=np.float64)
    if mixing.shape != (unmixing.shape[1], unmixing.shape[0]):
        raise ValueError("mixing has incompatible shape")

    n_components = unmixing.shape[0]
    if order is None:
        peak_rows = np.argmax(np.abs(mixing), axis=0)
        peak_values = np.max(np.abs(mixing), axis=0)
        order_array = np.lexsort((-peak_values, peak_rows))
    else:
        order_array = np.asarray(order, dtype=int)
        if order_array.shape != (n_components,) or set(order_array.tolist()) != set(
            range(n_components)
        ):
            raise ValueError("order must be a permutation of component indices")

    unmixing = unmixing[order_array].copy()
    mixing = mixing[:, order_array].copy()
    peak_rows = np.argmax(np.abs(mixing), axis=0)
    signs = np.sign(mixing[peak_rows, np.arange(n_components)])
    signs[signs == 0.0] = 1.0
    unmixing *= signs[:, None]
    mixing *= signs[None, :]
    return unmixing, mixing, order_array, signs
