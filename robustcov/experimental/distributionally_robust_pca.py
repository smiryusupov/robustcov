# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Weighted-Wasserstein distributionally robust principal component analysis.

The estimator in this module is intentionally experimental.  It solves a genuine
weighted-Wasserstein worst-case reconstruction problem *over a deterministic
candidate path*.  The candidate path follows the adaptive DRO-PCA construction of
Xu, Wood, and Yang (2026); candidate selection can use either the exact scalar-dual
worst-case risk or the paper's tractable spectral surrogate.

The finite path is not claimed to be a global optimizer of the non-convex
Grassmann problem.  The fitted objective, ambiguity radius, transport geometry,
and exact/surrogate risks are exposed so that this limitation is inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from scipy.optimize import brentq

from .._estimator import EstimatorMixin
from ..pca import _as_2d_finite_array, _deterministic_eigenvector_signs, _symmetrize


GeometryName = Literal["residual", "pca_block", "identity"]
FormulationName = Literal["exact", "surrogate"]
CenterName = Literal["mean", "median"]


def _leading_basis(matrix: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(_symmetrize(matrix))
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = _deterministic_eigenvector_signs(vectors[:, order])
    return vectors[:, :rank], values


def _projector(basis: np.ndarray) -> np.ndarray:
    return _symmetrize(basis @ basis.T)


def _spd_factors(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = _symmetrize(np.asarray(matrix, dtype=float))
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transport_matrix must be a square 2D array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("transport_matrix must contain only finite values")
    values, vectors = np.linalg.eigh(matrix)
    scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    tolerance = 100.0 * np.finfo(float).eps * scale
    if float(values[0]) <= tolerance:
        raise ValueError("transport_matrix must be symmetric positive definite")
    root = (vectors * np.sqrt(values)) @ vectors.T
    inverse_root = (vectors * (1.0 / np.sqrt(values))) @ vectors.T
    inverse = (vectors * (1.0 / values)) @ vectors.T
    return _symmetrize(root), _symmetrize(inverse_root), _symmetrize(inverse)


def _exact_wasserstein_risk(
    covariance: np.ndarray,
    residual_projector: np.ndarray,
    transport_root: np.ndarray,
    transport_inverse_root: np.ndarray,
    radius: float,
) -> tuple[float, float]:
    """Evaluate the exact scalar dual of the weighted-Wasserstein robust risk.

    Returns
    -------
    risk : float
        Exact worst-case expected squared reconstruction loss.
    dual_lambda : float
        Numerically minimizing dual multiplier. ``inf`` when ``radius == 0``.
    """
    covariance = _symmetrize(covariance)
    residual_projector = _symmetrize(residual_projector)
    nominal = max(float(np.trace(covariance @ residual_projector)), 0.0)
    if radius == 0.0:
        return nominal, float("inf")

    exposure_operator = _symmetrize(
        transport_inverse_root @ residual_projector @ transport_inverse_root
    )
    exposure_values, exposure_vectors = np.linalg.eigh(exposure_operator)
    exposure_values = np.maximum(exposure_values, 0.0)
    rho = float(exposure_values[-1])
    if rho <= np.finfo(float).tiny:
        return nominal, float("inf")

    transported_covariance = _symmetrize(transport_root @ covariance @ transport_root)
    transported_diagonal = np.diag(
        exposure_vectors.T @ transported_covariance @ exposure_vectors
    )
    transported_diagonal = np.maximum(transported_diagonal, 0.0)

    active = (exposure_values > np.finfo(float).eps * rho) & (
        transported_diagonal > np.finfo(float).tiny
    )
    s = exposure_values[active]
    a = transported_diagonal[active]
    if s.size == 0:
        return nominal, rho

    radius_sq = radius * radius

    def derivative(lam: float) -> float:
        return radius_sq - float(np.sum(a * s * s / np.square(lam - s)))

    margin = max(1e-12 * max(rho, 1.0), np.spacing(rho) * 16.0)
    lower = rho + margin
    lower_derivative = derivative(lower)
    if lower_derivative >= 0.0:
        dual_lambda = lower
    else:
        upper = max(2.0 * lower, lower + 1.0)
        for _ in range(256):
            if derivative(upper) > 0.0:
                break
            upper *= 2.0
        else:  # pragma: no cover - defensive overflow guard
            raise RuntimeError("failed to bracket the Wasserstein dual optimum")
        dual_lambda = float(brentq(derivative, lower, upper, xtol=1e-13, rtol=1e-13))

    resolvent = float(np.sum(a * s / (dual_lambda - s)))
    risk = dual_lambda * (radius_sq + resolvent)
    return max(float(risk), nominal), dual_lambda


def _surrogate_objective(
    covariance: np.ndarray,
    residual_projector: np.ndarray,
    transport_inverse_root: np.ndarray,
    radius: float,
) -> tuple[float, float, float]:
    nominal = max(float(np.trace(covariance @ residual_projector)), 0.0)
    exposure = _symmetrize(
        transport_inverse_root @ residual_projector @ transport_inverse_root
    )
    rho = max(float(np.linalg.eigvalsh(exposure)[-1]), 0.0)
    objective = np.sqrt(nominal) + radius * np.sqrt(rho)
    return float(objective), nominal, rho


@dataclass
class DistributionallyRobustPCA(EstimatorMixin):
    """PCA for weighted-Wasserstein distribution shift.

    The estimator minimizes a weighted-Wasserstein worst-case reconstruction
    criterion over a deterministic path of candidate subspaces.  With
    ``formulation="exact"`` (the default), candidate selection uses the exact
    scalar dual of the worst-case risk.  ``formulation="surrogate"`` selects by
    the spectral upper-bound criterion of Xu, Wood, and Yang (2026).

    This is distributional robustness, not an outlier-resistance synonym.  The
    ambiguity set protects against distributions within a weighted type-2
    Wasserstein radius of the centered empirical law.  Heavy-tail or gross-cell
    contamination robustness should normally be handled by the package's robust
    scatter or cellwise PCA estimators instead.

    Parameters
    ----------
    n_components : int, default=2
        Target principal-subspace dimension. Must satisfy
        ``1 <= n_components < n_features``.
    radius : float or {"sqrt_n"}, default="sqrt_n"
        Weighted-Wasserstein ambiguity radius. A numeric value is used exactly.
        ``"sqrt_n"`` applies the transparent scale-equivariant heuristic
        ``radius_scale * sqrt(mean_variance / n_samples)``. This heuristic is
        *not* the paper's full RWPI calibration.
    radius_scale : float, default=1.0
        Multiplier for ``radius="sqrt_n"``.
    transport_geometry : {"residual", "pca_block", "identity"}, default="residual"
        Adaptive transport geometry. ``"residual"`` makes directions with large
        variance outside the initial PCA block cheap for the adversary;
        ``"pca_block"`` does the same for directions already visible inside the
        PCA block. ``"identity"`` is the homogeneous control and must recover
        ordinary PCA.
    transport_matrix : array-like, optional
        Explicit symmetric positive-definite transport matrix. When supplied it
        overrides ``transport_geometry``. The matrix is normalized so the mean
        diagonal of its inverse is one; multiplying it by a positive scalar does
        not change the ambiguity geometry.
    geometry_ridge : float, default=0.05
        Strictly positive ridge fraction used in adaptive inverse-variance geometry.
        The absolute ridge is this fraction times the mean empirical variance.
    formulation : {"exact", "surrogate"}, default="exact"
        Objective used to select among path candidates. ``"exact"`` evaluates
        the genuine weighted-Wasserstein worst-case risk through its scalar
        dual. ``"surrogate"`` uses its tractable spectral upper bound.
    path_grid : iterable of float, optional
        Nonnegative path multipliers. The default is a deterministic logarithmic
        grid including zero. The path is a reproducible restricted search, not a
        claim of global Grassmann optimization.
    center : {"mean", "median"}, default="mean"
        Training location. Mean centering matches the cited formulation; median
        centering is a robustcov adaptation for contaminated reference samples.
    store_scores : bool, default=True
        Store training scores and reconstruction errors.

    Notes
    -----
    The homogeneous identity geometry is deliberately retained as a diagnostic:
    its exact and surrogate objectives rank subspaces exactly as ordinary PCA.
    A fit that changes under identity geometry indicates an implementation bug.
    """

    n_components: int = 2
    radius: float | Literal["sqrt_n"] = "sqrt_n"
    radius_scale: float = 1.0
    transport_geometry: GeometryName = "residual"
    transport_matrix: np.ndarray | None = None
    geometry_ridge: float = 0.05
    formulation: FormulationName = "exact"
    path_grid: Iterable[float] | None = None
    center: CenterName = "mean"
    store_scores: bool = True

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "DistributionallyRobustPCA":
        del y
        X = _as_2d_finite_array(X)
        n_samples, n_features = X.shape
        self._validate_parameters(n_features)

        if self.center == "mean":
            location = np.mean(X, axis=0)
        else:
            location = np.median(X, axis=0)
        centered = X - location
        covariance = _symmetrize(centered.T @ centered / n_samples)
        mean_variance = float(np.trace(covariance) / n_features)
        if not np.isfinite(mean_variance) or mean_variance <= np.finfo(float).tiny:
            raise ValueError("X must contain non-constant variation")

        initial_basis, all_eigenvalues = _leading_basis(covariance, self.n_components)
        initial_projector = _projector(initial_basis)
        residual_projector = np.eye(n_features) - initial_projector

        raw_transport_matrix, geometry_name, geometry_source = self._build_transport_geometry(
            covariance, initial_projector, residual_projector, mean_variance
        )
        _, _, raw_transport_inverse = _spd_factors(raw_transport_matrix)
        transport_normalization = float(np.trace(raw_transport_inverse) / n_features)
        if transport_normalization <= np.finfo(float).tiny:
            raise ValueError("transport geometry has a degenerate inverse scale")
        transport_matrix = _symmetrize(transport_normalization * raw_transport_matrix)
        transport_root, transport_inverse_root, transport_inverse = _spd_factors(
            transport_matrix
        )
        radius = self._resolve_radius(n_samples, mean_variance)
        normalized_inverse = transport_inverse

        grid = self._resolve_path_grid()
        candidates: list[tuple[float, str, np.ndarray]] = []
        for gamma in grid:
            candidate_matrix = (
                covariance
                + gamma * radius * np.sqrt(mean_variance) * normalized_inverse
            )
            basis, _ = _leading_basis(candidate_matrix, self.n_components)
            candidates.append((float(gamma), "path", basis))
        candidates.append((0.0, "ordinary_pca", initial_basis))

        if np.allclose(transport_matrix, np.diag(np.diag(transport_matrix))):
            inverse_weights = np.diag(transport_inverse)
            indices = np.argsort(inverse_weights)[::-1][: self.n_components]
            coordinate_basis = np.eye(n_features)[:, indices]
            candidates.append((float("nan"), "coordinate", coordinate_basis))

        unique_candidates: list[tuple[float, str, np.ndarray]] = []
        seen: list[np.ndarray] = []
        for gamma, source, basis in candidates:
            projector = _projector(basis)
            if any(np.linalg.norm(projector - previous, ord="fro") <= 1e-10 for previous in seen):
                continue
            seen.append(projector)
            unique_candidates.append((gamma, source, basis))

        records = []
        for gamma, source, basis in unique_candidates:
            projector = _projector(basis)
            residual = np.eye(n_features) - projector
            surrogate, nominal, exposure = _surrogate_objective(
                covariance, residual, transport_inverse_root, radius
            )
            exact_risk, dual_lambda = _exact_wasserstein_risk(
                covariance,
                residual,
                transport_root,
                transport_inverse_root,
                radius,
            )
            selection_value = exact_risk if self.formulation == "exact" else surrogate
            records.append(
                {
                    "gamma": gamma,
                    "source": source,
                    "basis": basis,
                    "selection_value": float(selection_value),
                    "exact_risk": float(exact_risk),
                    "surrogate": float(surrogate),
                    "surrogate_risk_bound": float(surrogate * surrogate),
                    "nominal_risk": float(nominal),
                    "residual_exposure": float(exposure),
                    "dual_lambda": float(dual_lambda),
                }
            )

        selected_index = min(range(len(records)), key=lambda index: records[index]["selection_value"])
        selected = records[selected_index]
        basis = np.asarray(selected["basis"], dtype=float)
        basis = _deterministic_eigenvector_signs(basis)
        projector = _projector(basis)
        residual = np.eye(n_features) - projector

        eigenvalues = np.einsum("ij,ji->i", basis.T @ covariance, basis)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        order = np.argsort(eigenvalues)[::-1]
        basis = basis[:, order]
        eigenvalues = eigenvalues[order]
        components = basis.T
        projector = _projector(basis)
        residual = np.eye(n_features) - projector

        # Re-evaluate after deterministic ordering for fitted diagnostics.
        surrogate, nominal, exposure = _surrogate_objective(
            covariance, residual, transport_inverse_root, radius
        )
        exact_risk, dual_lambda = _exact_wasserstein_risk(
            covariance, residual, transport_root, transport_inverse_root, radius
        )

        self.n_features_in_ = n_features
        self.n_samples_seen_ = n_samples
        self.n_components_ = self.n_components
        self.location_ = location
        self.mean_ = location
        self.covariance_ = covariance
        self.initial_components_ = initial_basis.T
        self.initial_projector_ = initial_projector
        self.components_ = components
        self.projector_ = projector
        self.residual_projector_ = residual
        self.explained_variance_ = eigenvalues
        total_variance = float(np.trace(covariance))
        self.explained_variance_ratio_ = eigenvalues / total_variance
        self.singular_values_ = np.sqrt(np.maximum(eigenvalues * n_samples, 0.0))
        self.transport_geometry_ = geometry_name
        self.geometry_source_ = geometry_source
        self.raw_transport_matrix_ = raw_transport_matrix
        self.transport_normalization_ = transport_normalization
        self.transport_matrix_ = transport_matrix
        self.transport_root_ = transport_root
        self.transport_inverse_root_ = transport_inverse_root
        self.transport_inverse_ = transport_inverse
        self.geometry_ridge_ = self.geometry_ridge * mean_variance
        self.radius_ = radius
        self.radius_calibration_ = (
            "user_supplied" if isinstance(self.radius, (int, float, np.floating)) else "heuristic_sqrt_n"
        )
        self.formulation_ = self.formulation
        self.optimizer_ = "deterministic_candidate_path"
        self.optimization_scope_ = "finite_path_not_global_grassmann"
        self.global_optimum_claim_ = False
        self.identity_control_ = geometry_name == "identity"
        self.selected_candidate_index_ = selected_index
        self.selected_gamma_ = selected["gamma"]
        self.selected_candidate_source_ = selected["source"]
        self.nominal_reconstruction_risk_ = nominal
        self.residual_exposure_ = exposure
        self.surrogate_objective_ = surrogate
        self.surrogate_risk_bound_ = surrogate * surrogate
        self.exact_worst_case_risk_ = exact_risk
        self.dual_lambda_ = dual_lambda
        self.objective_ = exact_risk if self.formulation == "exact" else surrogate
        self.candidate_results_ = tuple(
            {key: value for key, value in record.items() if key != "basis"}
            for record in records
        )
        self.ambiguity_set_ = {
            "type": "weighted_wasserstein_2",
            "radius": radius,
            "transport_geometry": geometry_name,
            "center": self.center,
            "nominal_distribution": "centered_empirical",
        }

        if geometry_name == "identity" and np.linalg.norm(projector - initial_projector, ord="fro") > 1e-8:
            raise RuntimeError("identity Wasserstein geometry must recover ordinary PCA")

        if self.store_scores:
            self.scores_ = centered @ components.T
            residuals = centered - self.scores_ @ components
            self.reconstruction_errors_ = np.einsum("ij,ij->i", residuals, residuals)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have shape (n_samples, {self.n_features_in_})")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return (X - self.location_) @ self.components_.T

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        self._check_fitted()
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 2 or scores.shape[1] != self.n_components_:
            raise ValueError(f"scores must have shape (n_samples, {self.n_components_})")
        return scores @ self.components_ + self.location_

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        scores = self.transform(X)
        residuals = X - self.inverse_transform(scores)
        return np.einsum("ij,ij->i", residuals, residuals)

    def score(self, X: np.ndarray, y: np.ndarray | None = None) -> float:
        del y
        return -float(np.mean(self.reconstruction_error(X)))

    def exact_worst_case_risk(self, components: np.ndarray | None = None) -> float:
        """Evaluate the exact weighted-Wasserstein worst-case reconstruction risk."""
        self._check_fitted()
        if components is None:
            return float(self.exact_worst_case_risk_)
        components = np.asarray(components, dtype=float)
        if components.shape != (self.n_components_, self.n_features_in_):
            raise ValueError(
                f"components must have shape ({self.n_components_}, {self.n_features_in_})"
            )
        gram = components @ components.T
        if not np.allclose(gram, np.eye(self.n_components_), rtol=1e-7, atol=1e-9):
            raise ValueError("components rows must be orthonormal")
        residual = np.eye(self.n_features_in_) - components.T @ components
        risk, _ = _exact_wasserstein_risk(
            self.covariance_,
            residual,
            self.transport_root_,
            self.transport_inverse_root_,
            self.radius_,
        )
        return risk

    def _build_transport_geometry(
        self,
        covariance: np.ndarray,
        initial_projector: np.ndarray,
        residual_projector: np.ndarray,
        mean_variance: float,
    ) -> tuple[np.ndarray, str, str]:
        n_features = covariance.shape[0]
        if self.transport_matrix is not None:
            matrix = np.asarray(self.transport_matrix, dtype=float)
            if matrix.shape != (n_features, n_features):
                raise ValueError(
                    f"transport_matrix must have shape ({n_features}, {n_features})"
                )
            _spd_factors(matrix)
            return _symmetrize(matrix), "custom", "custom"

        if self.transport_geometry == "identity":
            return np.eye(n_features), "identity", "identity"

        block = residual_projector if self.transport_geometry == "residual" else initial_projector
        variances = np.diag(block @ covariance @ block)
        variances = np.maximum(variances, 0.0)
        ridge = self.geometry_ridge * mean_variance
        denominator = variances + ridge
        if np.any(denominator <= np.finfo(float).tiny):
            raise ValueError("adaptive transport geometry is numerically singular")
        matrix = np.diag(1.0 / denominator)
        return matrix, self.transport_geometry, "adaptive_inverse_variance"

    def _resolve_radius(self, n_samples: int, mean_variance: float) -> float:
        if isinstance(self.radius, str):
            if self.radius != "sqrt_n":
                raise ValueError("radius must be nonnegative or 'sqrt_n'")
            return float(self.radius_scale * np.sqrt(mean_variance / n_samples))
        radius = float(self.radius)
        if not np.isfinite(radius) or radius < 0.0:
            raise ValueError("radius must be finite and nonnegative")
        return radius

    def _resolve_path_grid(self) -> np.ndarray:
        if self.path_grid is None:
            return np.array(
                [0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.75, 1.25, 2.0, 3.5, 6.0, 10.0, 20.0],
                dtype=float,
            )
        grid = np.asarray(tuple(self.path_grid), dtype=float)
        if grid.ndim != 1 or grid.size == 0:
            raise ValueError("path_grid must be a non-empty one-dimensional iterable")
        if not np.all(np.isfinite(grid)) or np.any(grid < 0.0):
            raise ValueError("path_grid values must be finite and nonnegative")
        if not np.any(grid == 0.0):
            grid = np.concatenate(([0.0], grid))
        return np.unique(grid)

    def _validate_parameters(self, n_features: int) -> None:
        if not isinstance(self.n_components, (int, np.integer)):
            raise TypeError("n_components must be an integer")
        if not 1 <= int(self.n_components) < n_features:
            raise ValueError("n_components must satisfy 1 <= n_components < n_features")
        if self.transport_geometry not in {"residual", "pca_block", "identity"}:
            raise ValueError("transport_geometry must be 'residual', 'pca_block', or 'identity'")
        if self.formulation not in {"exact", "surrogate"}:
            raise ValueError("formulation must be 'exact' or 'surrogate'")
        if self.center not in {"mean", "median"}:
            raise ValueError("center must be 'mean' or 'median'")
        if isinstance(self.radius, (bool, np.bool_)):
            raise TypeError("radius must be a nonnegative float or 'sqrt_n'")
        if isinstance(self.radius_scale, (bool, np.bool_)):
            raise TypeError("radius_scale must be a nonnegative float")
        if not np.isfinite(self.radius_scale) or self.radius_scale < 0.0:
            raise ValueError("radius_scale must be finite and nonnegative")
        if not np.isfinite(self.geometry_ridge) or self.geometry_ridge <= 0.0:
            raise ValueError("geometry_ridge must be finite and strictly positive")

    def _check_fitted(self) -> None:
        if not hasattr(self, "components_"):
            raise RuntimeError("DistributionallyRobustPCA is not fitted")


WassersteinRobustPCA = DistributionallyRobustPCA


__all__ = ["DistributionallyRobustPCA", "WassersteinRobustPCA"]
