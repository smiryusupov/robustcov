# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""High-dimensional covariance estimation under cellwise contamination.

This module implements the covariance decomposition used by cellRCov:

* robust marginal standardization;
* a cellwise- and casewise-robust low-rank fit;
* robust covariance estimation for the fitted scores;
* a weighted covariance estimate for the residual component;
* diagonal-target ridge shrinkage of the residual covariance.

The reference cellRCov method uses robust M-scales, the reference cellPCA
implementation, DetMCD in score space, robust parallel analysis for rank
selection, and cross-validation for the residual shrinkage parameter.
``robustcov`` reuses its package-native ``CellPCA`` and ``FastMCD``
implementations.  It supports the same decomposition and a fixed-weight
cross-validation rule for shrinkage, but does not claim numerical identity
with the reference R code.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable

import numpy as np

from ._utils import check_array
from .cellpca import CellPCA, CellwiseRobustPCA, _robust_center_scale
from .covariance import BaseRobustCovariance, FastMCD


_EPS = np.finfo(np.float64).eps


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (matrix + matrix.T)


def _positive_definite(matrix: np.ndarray, minimum_eigenvalue: float) -> np.ndarray:
    matrix = _symmetrize(matrix)
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, float(minimum_eigenvalue))
    result = (vectors * values) @ vectors.T
    return _symmetrize(result)


def _weighted_residual_covariance(
    residuals: np.ndarray,
    observed: np.ndarray,
    cell_weights: np.ndarray,
    case_weights: np.ndarray,
    *,
    minimum_eigenvalue: float,
) -> tuple[np.ndarray, float]:
    """Return a PSD weighted residual covariance and its normalization.

    The cell weights multiply a residual on both sides of its outer product,
    while the case weight multiplies the complete outer product.  A scalar
    correction restores the clean-data scale when some cells are missing or
    downweighted.  This mirrors the role of the effective-pair normalization
    in cellRCov while keeping the finite-sample matrix positive semidefinite.
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    cell_weights = np.asarray(cell_weights, dtype=np.float64)
    case_weights = np.asarray(case_weights, dtype=np.float64)

    safe_residuals = np.where(observed, residuals, 0.0)
    effective_cell_weights = np.where(observed, cell_weights, 0.0)
    weighted = (
        np.sqrt(np.maximum(case_weights, 0.0))[:, None]
        * effective_cell_weights
        * safe_residuals
    )
    n = max(residuals.shape[0], 1)
    raw = weighted.T @ weighted / n

    effective = (
        np.maximum(case_weights, 0.0)[:, None]
        * effective_cell_weights**2
    )
    normalization = float(np.mean(np.sum(effective, axis=0) / n))
    normalization = max(normalization, np.sqrt(_EPS))
    covariance = raw / normalization
    covariance = _positive_definite(covariance, minimum_eigenvalue)
    return covariance, normalization


def _regularize_residual(
    covariance: np.ndarray,
    shrinkage: float,
    minimum_eigenvalue: float,
) -> np.ndarray:
    covariance = _symmetrize(covariance)
    diagonal = np.diag(np.maximum(np.diag(covariance), minimum_eigenvalue))
    result = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    return _positive_definite(result, minimum_eigenvalue)


def _validate_shrinkage_grid(values: Iterable[float]) -> np.ndarray:
    grid = np.asarray(tuple(values), dtype=np.float64)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("shrinkage_grid must contain at least one value")
    if not np.isfinite(grid).all() or np.any((grid < 0.0) | (grid > 1.0)):
        raise ValueError("shrinkage_grid values must be finite and in [0, 1]")
    return np.unique(grid)


class CellwiseRegularizedCovariance(BaseRobustCovariance):
    """Cellwise- and casewise-robust covariance for high-dimensional data.

    Parameters
    ----------
    n_components : int, default=5
        Rank of the robust low-dimensional fitted component.  It must be less
        than both the sample size and the feature count.
    residual_shrinkage : {"auto"} or float, default="auto"
        Ridge weight applied to the residual covariance.  A float in ``[0, 1]``
        uses that fixed value.  ``"auto"`` uses fixed-weight cross-validation
        over ``shrinkage_grid``.
    shrinkage_grid : iterable of float, optional
        Candidate ridge weights for automatic selection.  The default is
        ``0, 0.1, ..., 1``.
    cv_splits : int, default=5
        Number of row splits used by automatic shrinkage selection.
    cell_pca : CellPCA instance or None, default=None
        Optional configured low-rank estimator.  When supplied, its
        ``n_components`` must agree with ``n_components``.
    score_estimator : covariance estimator or None, default=None
        Robust covariance estimator fitted to the CellPCA scores.  The default
        is ``FastMCD(support_fraction=0.75)``.
    min_eigenvalue : float, default=1e-6
        Eigenvalue floor for residual and final covariance matrices in robustly
        standardized coordinates.
    random_state : int, default=0
        Seed used for the default score estimator and cross-validation splits.
    store_diagnostics : bool, default=True
        Store training cell/case weights, fitted values, corrections, and the
        two distance components.

    Notes
    -----
    The estimated standardized covariance is

    ``fitted_covariance_ + residual_covariance_regularized_``.

    The fitted term is obtained by mapping a robust score covariance back
    through the CellPCA loading matrix.  The residual term is a weighted outer-
    product covariance of robustly imputed residuals and is shrunk toward its
    diagonal.  The final estimate is transformed back to the original units.

    This implementation follows the cellRCov decomposition but uses the
    package-native CellPCA and FastMCD starts.  Rank selection is explicit rather
    than implementing the paper's robust parallel-analysis procedure.
    """

    def __init__(
        self,
        n_components: int = 5,
        *,
        residual_shrinkage: str | float = "auto",
        shrinkage_grid: Iterable[float] | None = None,
        cv_splits: int = 5,
        cell_pca: CellwiseRobustPCA | None = None,
        score_estimator: Any | None = None,
        min_eigenvalue: float = 1e-6,
        random_state: int = 0,
        store_diagnostics: bool = True,
    ):
        super().__init__(
            assume_centered=False,
            store_precision=True,
            scale_correction="none",
            tail_diagnostics=False,
            missing_values="native",
        )
        self.n_components = n_components
        self.residual_shrinkage = residual_shrinkage
        self.shrinkage_grid = shrinkage_grid
        self.cv_splits = cv_splits
        self.cell_pca = cell_pca
        self.score_estimator = score_estimator
        self.min_eigenvalue = min_eigenvalue
        self.random_state = random_state
        self.store_diagnostics = store_diagnostics

    def _validate_parameters(self, n: int, p: int) -> None:
        if isinstance(self.n_components, (bool, np.bool_)) or not isinstance(
            self.n_components, (int, np.integer)
        ):
            raise TypeError("n_components must be an integer")
        if not 1 <= int(self.n_components) < min(n, p):
            raise ValueError(
                "n_components must be between 1 and min(n_samples, n_features) - 1"
            )
        if isinstance(self.residual_shrinkage, str):
            if self.residual_shrinkage != "auto":
                raise ValueError("residual_shrinkage must be 'auto' or a float in [0, 1]")
        else:
            value = float(self.residual_shrinkage)
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("residual_shrinkage must be 'auto' or a float in [0, 1]")
        if isinstance(self.cv_splits, (bool, np.bool_)) or int(self.cv_splits) < 2:
            raise ValueError("cv_splits must be at least 2")
        if int(self.cv_splits) > n:
            raise ValueError("cv_splits cannot exceed n_samples")
        if not np.isfinite(self.min_eigenvalue) or float(self.min_eigenvalue) <= 0.0:
            raise ValueError("min_eigenvalue must be positive and finite")
        if not isinstance(self.store_diagnostics, (bool, np.bool_)):
            raise TypeError("store_diagnostics must be a boolean")
        if self.cell_pca is not None and int(self.cell_pca.n_components) != int(
            self.n_components
        ):
            raise ValueError("cell_pca.n_components must equal n_components")

    def _make_cell_pca(self) -> CellwiseRobustPCA:
        if self.cell_pca is None:
            return CellPCA(n_components=int(self.n_components))
        return copy.deepcopy(self.cell_pca)

    def _make_score_estimator(self) -> Any:
        if self.score_estimator is None:
            return FastMCD(
                support_fraction=0.75,
                quality="fast",
                n_init=100,
                n_best=5,
                initial_c_steps=2,
                max_iter=80,
                random_state=int(self.random_state),
                scale_correction="none",
            )
        return copy.deepcopy(self.score_estimator)

    def _select_shrinkage(
        self,
        residuals: np.ndarray,
        observed: np.ndarray,
        cell_weights: np.ndarray,
        case_weights: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        if not isinstance(self.residual_shrinkage, str):
            value = float(self.residual_shrinkage)
            return value, np.asarray([value]), np.asarray([np.nan])

        grid = _validate_shrinkage_grid(
            np.linspace(0.0, 1.0, 11)
            if self.shrinkage_grid is None
            else self.shrinkage_grid
        )
        rng = np.random.default_rng(int(self.random_state))
        folds = np.array_split(rng.permutation(residuals.shape[0]), int(self.cv_splits))
        scores = np.zeros(grid.size, dtype=np.float64)
        counts = np.zeros(grid.size, dtype=np.int64)
        all_rows = np.arange(residuals.shape[0])

        for validation in folds:
            if validation.size == 0:
                continue
            training = np.setdiff1d(all_rows, validation, assume_unique=True)
            if training.size < 2:
                continue
            train_cov, _ = _weighted_residual_covariance(
                residuals[training],
                observed[training],
                cell_weights[training],
                case_weights[training],
                minimum_eigenvalue=float(self.min_eigenvalue),
            )
            validation_cov, _ = _weighted_residual_covariance(
                residuals[validation],
                observed[validation],
                cell_weights[validation],
                case_weights[validation],
                minimum_eigenvalue=float(self.min_eigenvalue),
            )
            for index, value in enumerate(grid):
                regularized = _regularize_residual(
                    train_cov, float(value), float(self.min_eigenvalue)
                )
                scores[index] += float(
                    np.linalg.norm(regularized - validation_cov, ord="fro") ** 2
                )
                counts[index] += 1

        valid = counts > 0
        if not np.any(valid):
            return 0.5, grid, np.full(grid.size, np.nan)
        mean_scores = np.full(grid.size, np.inf)
        mean_scores[valid] = scores[valid] / counts[valid]
        best = int(np.argmin(mean_scores))
        return float(grid[best]), grid, mean_scores

    def fit(self, X: Any, y: Any | None = None) -> "CellwiseRegularizedCovariance":
        """Fit the robust low-rank plus regularized-residual covariance."""
        del y
        X = check_array(X, allow_nan=True)
        n, p = X.shape
        self._validate_parameters(n, p)
        observed = np.isfinite(X)
        if np.any(observed.sum(axis=0) < 2):
            bad = np.flatnonzero(observed.sum(axis=0) < 2)
            raise ValueError(
                "each feature must contain at least two finite values; failed columns: "
                + ", ".join(map(str, bad.tolist()))
            )
        if np.any(observed.sum(axis=1) == 0):
            bad = np.flatnonzero(observed.sum(axis=1) == 0)
            raise ValueError(
                "every row must contain at least one finite value; failed rows: "
                + ", ".join(map(str, bad.tolist()))
            )

        marginal_center, marginal_scale = _robust_center_scale(X)
        Z = (X - marginal_center) / marginal_scale

        cell_pca = self._make_cell_pca().fit(Z)
        scores = np.asarray(cell_pca.scores_, dtype=np.float64)
        score_estimator = self._make_score_estimator().fit(scores)

        loadings = np.asarray(cell_pca.loadings_, dtype=np.float64)
        score_covariance = _symmetrize(score_estimator.covariance_)
        fitted_covariance = _symmetrize(loadings @ score_covariance @ loadings.T)

        fitted = np.asarray(cell_pca.fitted_values_, dtype=np.float64)
        residuals = np.where(observed, Z - fitted, 0.0)
        cell_weights = np.asarray(cell_pca.cell_weights_, dtype=np.float64)
        case_weights = np.asarray(cell_pca.case_weights_, dtype=np.float64)
        residual_covariance, normalization = _weighted_residual_covariance(
            residuals,
            observed,
            cell_weights,
            case_weights,
            minimum_eigenvalue=float(self.min_eigenvalue),
        )
        shrinkage, grid, cv_scores = self._select_shrinkage(
            residuals, observed, cell_weights, case_weights
        )
        residual_regularized = _regularize_residual(
            residual_covariance, shrinkage, float(self.min_eigenvalue)
        )
        standardized_covariance = _positive_definite(
            fitted_covariance + residual_regularized,
            float(self.min_eigenvalue),
        )

        standardized_location = (
            np.asarray(cell_pca.center_, dtype=np.float64)
            + np.asarray(score_estimator.location_, dtype=np.float64) @ cell_pca.components_
        )
        covariance = (
            marginal_scale[:, None]
            * standardized_covariance
            * marginal_scale[None, :]
        )
        covariance = _positive_definite(
            covariance,
            float(self.min_eigenvalue) * float(np.min(marginal_scale) ** 2),
        )
        location = marginal_center + marginal_scale * standardized_location
        precision = np.linalg.inv(covariance)

        self.n_samples_in_ = n
        self.n_features_in_ = p
        self.n_components_ = int(self.n_components)
        self.location_ = location
        self.covariance_ = covariance
        self.precision_ = precision
        self.shape_ = covariance.copy()
        self.marginal_center_ = marginal_center
        self.marginal_scale_ = marginal_scale
        self.standardized_location_ = standardized_location
        self.standardized_covariance_ = standardized_covariance
        self.standardized_precision_ = np.linalg.inv(standardized_covariance)
        self.fitted_covariance_ = fitted_covariance
        self.residual_covariance_ = residual_covariance
        self.residual_covariance_regularized_ = residual_regularized
        self.residual_precision_ = np.linalg.inv(residual_regularized)
        self.score_covariance_ = score_covariance
        self.score_precision_ = np.linalg.pinv(score_covariance)
        self.residual_shrinkage_ = float(shrinkage)
        self.shrinkage_grid_ = grid
        self.shrinkage_cv_scores_ = cv_scores
        self.residual_normalization_ = float(normalization)
        self.cell_pca_ = cell_pca
        self.score_estimator_ = score_estimator
        self.components_ = np.asarray(cell_pca.components_, dtype=np.float64)
        self.loadings_ = loadings
        self.explained_variance_ = np.asarray(
            cell_pca.explained_variance_, dtype=np.float64
        )
        self.explained_variance_ratio_ = np.asarray(
            cell_pca.explained_variance_ratio_, dtype=np.float64
        )

        diagnostics = self._diagnostics_from_standardized(Z)
        self.distances_ = diagnostics["distances"]
        self.subspace_distances_ = diagnostics["subspace_distances"]
        self.residual_distances_ = diagnostics["residual_distances"]
        if self.store_diagnostics:
            self.fitted_values_ = marginal_center + marginal_scale * fitted
            self.residuals_ = np.where(observed, X - self.fitted_values_, np.nan)
            self.standardized_residuals_ = np.asarray(
                cell_pca.standardized_residuals_, dtype=np.float64
            )
            self.cell_weights_ = cell_weights
            self.case_weights_ = case_weights
            self.cell_outlier_mask_ = np.asarray(
                cell_pca.cell_outlier_mask_, dtype=bool
            )
            self.case_outlier_mask_ = np.asarray(
                cell_pca.case_outlier_mask_, dtype=bool
            )
            self.missing_mask_ = ~observed
            self.max_cell_residuals_ = np.asarray(cell_pca.max_cell_residuals_, dtype=np.float64)
            self.case_deviations_ = np.asarray(cell_pca.case_deviations_, dtype=np.float64)
            self.cell_cutoff_ = float(cell_pca.cell_cutoff_)
            self.case_cutoff_ = float(cell_pca.case_cutoff_)
            self.corrected_data_ = (
                marginal_center + marginal_scale * cell_pca.corrected_data_
            )
            self.imputed_data_ = marginal_center + marginal_scale * cell_pca.imputed_data_
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "precision_"):
            raise RuntimeError("CellRCov is not fitted")

    def _standardize_new(self, X: Any) -> np.ndarray:
        self._check_fitted()
        X = check_array(X, allow_nan=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )
        if np.any(np.isfinite(X).sum(axis=1) == 0):
            raise ValueError("every row must contain at least one finite value")
        return (X - self.marginal_center_) / self.marginal_scale_

    def _diagnostics_from_standardized(self, Z: np.ndarray) -> dict[str, np.ndarray]:
        diagnostics = self.cell_pca_.cellwise_diagnostics(Z)
        corrected = np.asarray(diagnostics["corrected_data"], dtype=np.float64)
        fitted = np.asarray(diagnostics["fitted_values"], dtype=np.float64)
        scores = np.asarray(diagnostics["scores"], dtype=np.float64)
        residuals = corrected - fitted

        centered_scores = scores - self.score_estimator_.location_
        subspace = np.einsum(
            "ij,jk,ik->i",
            centered_scores,
            self.score_precision_,
            centered_scores,
        )
        residual = np.einsum(
            "ij,jk,ik->i", residuals, self.residual_precision_, residuals
        )
        centered = corrected - self.standardized_location_
        distances = np.einsum(
            "ij,jk,ik->i",
            centered,
            self.standardized_precision_,
            centered,
        )
        result = dict(diagnostics)
        result.update(
            corrected_standardized=corrected,
            distances=distances,
            subspace_distances=subspace,
            residual_distances=residual,
        )
        return result

    def cellwise_diagnostics(self, X: Any) -> dict[str, np.ndarray]:
        """Return cell/case diagnostics and covariance distance components."""
        Z = self._standardize_new(X)
        result = self._diagnostics_from_standardized(Z)
        result["corrected_data"] = (
            self.marginal_center_
            + self.marginal_scale_ * result["corrected_standardized"]
        )
        result["fitted_values"] = (
            self.marginal_center_
            + self.marginal_scale_ * result["fitted_values"]
        )
        return result

    def mahalanobis(self, X: Any) -> np.ndarray:
        """Squared Mahalanobis distances after robust cell correction."""
        return self.cellwise_diagnostics(X)["distances"]

    def subspace_distances(self, X: Any) -> np.ndarray:
        """Squared robust distances within the fitted low-rank subspace."""
        return self.cellwise_diagnostics(X)["subspace_distances"]

    def residual_distances(self, X: Any) -> np.ndarray:
        """Squared distances of corrected residuals outside the fitted subspace."""
        return self.cellwise_diagnostics(X)["residual_distances"]

    def outlier_map(self, X: Any) -> np.ndarray:
        """Return subspace and residual squared distances as two columns."""
        diagnostics = self.cellwise_diagnostics(X)
        return np.column_stack(
            [diagnostics["subspace_distances"], diagnostics["residual_distances"]]
        )

    def transform(self, X: Any) -> np.ndarray:
        """Correct missing/outlying cells using the fitted low-rank model."""
        return self.cellwise_diagnostics(X)["corrected_data"]


CellRCov = CellwiseRegularizedCovariance
CellwiseRobustCovariance = CellwiseRegularizedCovariance


__all__ = [
    "CellwiseRegularizedCovariance",
    "CellRCov",
    "CellwiseRobustCovariance",
]
