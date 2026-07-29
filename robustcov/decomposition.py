# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Low-rank-plus-sparse matrix decomposition.

This module implements Principal Component Pursuit (PCP), the canonical convex
program often called robust PCA in the matrix-decomposition literature.  It is
distinct from :class:`robustcov.RobustPCA`, which diagonalizes a robust scatter
estimate and provides rowwise score/orthogonal-distance diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._estimator import EstimatorMixin


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    """Apply elementwise soft thresholding."""

    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def _singular_value_threshold(
    matrix: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Apply singular-value thresholding and return the retained spectrum."""

    left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    shrunk = np.maximum(singular_values - threshold, 0.0)
    rank = int(np.count_nonzero(shrunk > 0.0))
    if rank == 0:
        return np.zeros_like(matrix), 0, shrunk
    low_rank = (left[:, :rank] * shrunk[:rank]) @ right[:rank]
    return low_rank, rank, shrunk


@dataclass(frozen=True)
class PCPHistoryStep:
    """Diagnostics for one inexact augmented-Lagrangian iteration."""

    iteration: int
    rank: int
    sparse_fraction: float
    relative_residual: float
    objective: float
    mu: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "iteration": int(self.iteration),
            "rank": int(self.rank),
            "sparse_fraction": float(self.sparse_fraction),
            "relative_residual": float(self.relative_residual),
            "objective": float(self.objective),
            "mu": float(self.mu),
        }


@dataclass
class PrincipalComponentPursuit(EstimatorMixin):
    r"""Decompose a matrix into low-rank and entrywise-sparse components.

    Principal Component Pursuit solves

    .. math::

       \min_{L,S}\; \lVert L\rVert_* + \lambda\lVert S\rVert_1
       \quad\text{subject to}\quad X=L+S.

    The implementation uses the inexact augmented Lagrange multiplier algorithm
    of Lin, Chen, and Ma.  The default ``lambda_`` is
    ``1 / sqrt(max(n_samples, n_features))``, matching the canonical PCP
    prescription.

    This estimator is appropriate when the observed matrix is plausibly the
    sum of an incoherent low-rank signal and sparse, arbitrarily large *cellwise*
    corruption.  It is not a covariance estimator, does not model dense noise,
    and has no canonical out-of-sample sparse-decomposition rule.

    Parameters
    ----------
    lambda_ : float or None, default=None
        Weight on the entrywise L1 norm.  ``None`` uses
        ``1 / sqrt(max(n_samples, n_features))``.
    mu : float or None, default=None
        Initial augmented-Lagrangian penalty.  ``None`` uses
        ``1.25 / ||X||_2``.
    rho : float, default=1.5
        Multiplicative penalty increase after each iteration.
    mu_max_factor : float, default=1e7
        Upper bound on ``mu`` relative to its initial value.
    max_iter : int, default=1000
        Maximum number of inexact ALM iterations.
    tol : float, default=1e-7
        Relative Frobenius reconstruction tolerance.
    sparse_tol : float, default=1e-10
        Relative threshold used to define ``sparse_support_`` diagnostics.
    store_history : bool, default=True
        Store per-iteration convergence diagnostics.

    Notes
    -----
    Exact recovery results require assumptions including low-rank incoherence,
    sufficiently low rank, and sufficiently sparse/dispersed corruption.  This
    numerical implementation does not verify those assumptions.
    """

    lambda_: float | None = None
    mu: float | None = None
    rho: float = 1.5
    mu_max_factor: float = 1e7
    max_iter: int = 1000
    tol: float = 1e-7
    sparse_tol: float = 1e-10
    store_history: bool = True

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> "PrincipalComponentPursuit":
        """Fit the low-rank-plus-sparse decomposition of ``X``."""

        del y
        matrix = self._validate_input(X)
        self._validate_parameters()

        n_samples, n_features = matrix.shape
        lambda_value = (
            1.0 / np.sqrt(float(max(matrix.shape)))
            if self.lambda_ is None
            else float(self.lambda_)
        )
        frobenius_norm = float(np.linalg.norm(matrix, ord="fro"))
        spectral_norm = float(np.linalg.norm(matrix, ord=2))
        infinity_norm = float(np.max(np.abs(matrix)))

        if frobenius_norm == 0.0:
            return self._fit_zero(matrix, lambda_value)

        mu_initial = (
            1.25 / spectral_norm if self.mu is None else float(self.mu)
        )
        mu_value = mu_initial
        mu_max = mu_initial * float(self.mu_max_factor)
        dual_norm = max(spectral_norm, infinity_norm / lambda_value)
        dual = matrix / dual_norm
        low_rank = np.zeros_like(matrix)
        sparse = np.zeros_like(matrix)
        history: list[PCPHistoryStep] = []
        converged = False
        relative_residual = np.inf
        objective = np.inf
        final_rank = 0

        for iteration in range(1, int(self.max_iter) + 1):
            low_rank, final_rank, shrunk_singular_values = (
                _singular_value_threshold(
                    matrix - sparse + dual / mu_value,
                    1.0 / mu_value,
                )
            )
            sparse = _soft_threshold(
                matrix - low_rank + dual / mu_value,
                lambda_value / mu_value,
            )
            residual = matrix - low_rank - sparse
            relative_residual = float(
                np.linalg.norm(residual, ord="fro") / frobenius_norm
            )
            sparse_fraction = float(
                np.mean(
                    np.abs(sparse)
                    > self._support_threshold(matrix, sparse)
                )
            )
            objective = float(
                np.sum(shrunk_singular_values)
                + lambda_value * np.sum(np.abs(sparse))
            )
            if self.store_history:
                history.append(
                    PCPHistoryStep(
                        iteration=iteration,
                        rank=final_rank,
                        sparse_fraction=sparse_fraction,
                        relative_residual=relative_residual,
                        objective=objective,
                        mu=mu_value,
                    )
                )
            if relative_residual <= self.tol:
                converged = True
                break
            dual = dual + mu_value * residual
            mu_value = min(mu_value * self.rho, mu_max)

        residual = matrix - low_rank - sparse
        self._store_result(
            matrix=matrix,
            low_rank=low_rank,
            sparse=sparse,
            residual=residual,
            lambda_value=lambda_value,
            mu_initial=mu_initial,
            mu_final=mu_value,
            n_iter=iteration,
            converged=converged,
            objective=objective,
            history=history,
        )
        return self

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> np.ndarray:
        """Fit the decomposition and return the recovered low-rank matrix."""

        return self.fit(X, y).low_rank_.copy()

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project new rows onto the fitted low-rank row space.

        This is an ordinary linear projection using ``components_``.  It does
        not solve a new PCP problem and does not estimate a sparse component for
        the new rows.
        """

        self._check_fitted()
        matrix = np.asarray(X, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("X must be a 2D array")
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has an incompatible number of features: "
                f"got {matrix.shape[1]}, expected {self.n_features_in_}"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("X must contain only finite values")
        return matrix @ self.components_.T

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        """Map low-rank coordinates back to feature space."""

        self._check_fitted()
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 2:
            raise ValueError("scores must be a 2D array")
        if scores.shape[1] != self.rank_:
            raise ValueError(
                "scores have an incompatible number of components: "
                f"got {scores.shape[1]}, expected {self.rank_}"
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError("scores must contain only finite values")
        return scores @ self.components_

    def history_records(self) -> list[dict[str, float | int]]:
        """Return convergence history as serializable dictionaries."""

        self._check_fitted()
        return [step.as_dict() for step in self.history_]

    def decomposition_summary(self) -> dict[str, float | int | bool]:
        """Return compact fitted decomposition diagnostics."""

        self._check_fitted()
        return {
            "rank": int(self.rank_),
            "n_sparse": int(self.n_sparse_),
            "sparse_fraction": float(self.sparse_fraction_),
            "relative_residual": float(self.reconstruction_error_),
            "objective": float(self.objective_),
            "n_iter": int(self.n_iter_),
            "converged": bool(self.converged_),
            "lambda": float(self.lambda_value_),
        }

    def _store_result(
        self,
        *,
        matrix: np.ndarray,
        low_rank: np.ndarray,
        sparse: np.ndarray,
        residual: np.ndarray,
        lambda_value: float,
        mu_initial: float,
        mu_final: float,
        n_iter: int,
        converged: bool,
        objective: float,
        history: list[PCPHistoryStep],
    ) -> None:
        left, singular_values, right = np.linalg.svd(
            low_rank,
            full_matrices=False,
        )
        if singular_values.size:
            rank_tolerance = (
                max(matrix.shape)
                * np.finfo(float).eps
                * max(float(singular_values[0]), 1.0)
            )
        else:
            rank_tolerance = 0.0
        rank = int(np.count_nonzero(singular_values > rank_tolerance))
        threshold = self._support_threshold(matrix, sparse)
        support = np.abs(sparse) > threshold

        self.low_rank_ = low_rank
        self.sparse_ = sparse
        self.residual_ = residual
        self.sparse_support_ = support
        self.cell_outlier_scores_ = np.abs(sparse)
        self.row_outlier_scores_ = np.linalg.norm(sparse, axis=1)
        self.column_outlier_scores_ = np.linalg.norm(sparse, axis=0)
        self.rank_ = rank
        self.singular_values_ = singular_values[:rank].copy()
        self.components_ = right[:rank].copy()
        self.scores_ = (
            left[:, :rank] * singular_values[:rank]
            if rank
            else np.zeros((matrix.shape[0], 0), dtype=float)
        )
        self.n_sparse_ = int(np.count_nonzero(support))
        self.sparse_fraction_ = float(np.mean(support))
        self.reconstruction_error_ = float(
            np.linalg.norm(residual, ord="fro")
            / max(np.linalg.norm(matrix, ord="fro"), np.finfo(float).tiny)
        )
        self.objective_ = float(objective)
        self.lambda_value_ = float(lambda_value)
        self.mu_initial_ = float(mu_initial)
        self.mu_final_ = float(mu_final)
        self.n_iter_ = int(n_iter)
        self.converged_ = bool(converged)
        self.history_ = tuple(history)
        self.n_samples_in_, self.n_features_in_ = matrix.shape

    def _fit_zero(
        self,
        matrix: np.ndarray,
        lambda_value: float,
    ) -> "PrincipalComponentPursuit":
        zeros = np.zeros_like(matrix)
        self._store_result(
            matrix=matrix,
            low_rank=zeros.copy(),
            sparse=zeros.copy(),
            residual=zeros.copy(),
            lambda_value=lambda_value,
            mu_initial=1.0 if self.mu is None else float(self.mu),
            mu_final=1.0 if self.mu is None else float(self.mu),
            n_iter=0,
            converged=True,
            objective=0.0,
            history=[],
        )
        return self

    def _support_threshold(
        self,
        matrix: np.ndarray,
        sparse: np.ndarray,
    ) -> float:
        scale = max(
            float(np.max(np.abs(matrix))),
            float(np.max(np.abs(sparse))),
            1.0,
        )
        return float(self.sparse_tol) * scale

    def _validate_parameters(self) -> None:
        if self.lambda_ is not None and (
            not np.isscalar(self.lambda_) or float(self.lambda_) <= 0.0
        ):
            raise ValueError("lambda_ must be None or a positive scalar")
        if self.mu is not None and (
            not np.isscalar(self.mu) or float(self.mu) <= 0.0
        ):
            raise ValueError("mu must be None or a positive scalar")
        if not np.isscalar(self.rho) or float(self.rho) <= 1.0:
            raise ValueError("rho must be greater than 1")
        if (
            not np.isscalar(self.mu_max_factor)
            or float(self.mu_max_factor) < 1.0
        ):
            raise ValueError("mu_max_factor must be at least 1")
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not np.isscalar(self.tol) or float(self.tol) <= 0.0:
            raise ValueError("tol must be positive")
        if not np.isscalar(self.sparse_tol) or float(self.sparse_tol) < 0.0:
            raise ValueError("sparse_tol must be non-negative")
        if not isinstance(self.store_history, (bool, np.bool_)):
            raise TypeError("store_history must be boolean")

    @staticmethod
    def _validate_input(X: np.ndarray) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("X must be a 2D array")
        if matrix.shape[0] < 2 or matrix.shape[1] < 2:
            raise ValueError("X must contain at least two rows and two columns")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("X must contain only finite values")
        return matrix

    def _check_fitted(self) -> None:
        if not hasattr(self, "low_rank_"):
            raise AttributeError("PrincipalComponentPursuit is not fitted")


PCP = PrincipalComponentPursuit

__all__ = ["PrincipalComponentPursuit", "PCP", "PCPHistoryStep"]
