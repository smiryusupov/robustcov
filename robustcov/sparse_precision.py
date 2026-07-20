# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Robust sparse precision-matrix estimation.

The module combines a robust scatter estimate with an off-diagonal graphical
lasso penalty.  The sparse optimization is solved with an ADMM algorithm and
therefore does not require scikit-learn at runtime.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ._estimator import EstimatorMixin


_EPS = np.finfo(float).eps


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _as_fit_array(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if X.shape[0] < 2:
        raise ValueError("X must contain at least two samples")
    if X.shape[1] < 2:
        raise ValueError("X must contain at least two features")
    if np.any(np.isinf(X)):
        raise ValueError("X must not contain infinite values")
    return X


def _as_finite_array(X: np.ndarray, *, n_features: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if X.shape[1] != n_features:
        raise ValueError(f"X has {X.shape[1]} features, expected {n_features}")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")
    return X


class _EmpiricalScatter:
    """Small internal baseline used by examples and comparisons."""

    def fit(self, X: np.ndarray) -> "_EmpiricalScatter":
        if np.isnan(X).any():
            raise ValueError("the empirical scatter baseline does not accept missing values")
        self.location_ = np.mean(X, axis=0)
        covariance = np.cov(X, rowvar=False, ddof=1)
        self.covariance_ = np.atleast_2d(covariance)
        return self


def _default_scatter_estimator() -> Any:
    from .m_estimators import RegularizedCauchy

    return RegularizedCauchy(alpha=0.10)


def _resolve_scatter_estimator(estimator: Any | str | None) -> Any:
    if estimator is None:
        return _default_scatter_estimator()
    if isinstance(estimator, str):
        if estimator != "empirical":
            raise ValueError("scatter_estimator string must be 'empirical'")
        return _EmpiricalScatter()
    return deepcopy(estimator)


def _regularize_spd(matrix: np.ndarray, relative_floor: float) -> tuple[np.ndarray, float]:
    """Return an SPD matrix using a scale-relative eigenvalue floor."""
    matrix = _symmetrize(np.asarray(matrix, dtype=float))
    values, vectors = np.linalg.eigh(matrix)
    scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    negative_tolerance = 1000.0 * _EPS * scale
    if float(np.min(values)) < -negative_tolerance:
        raise ValueError("scatter matrix must be positive semidefinite")
    floor = max(relative_floor * scale, np.finfo(float).tiny)
    values = np.maximum(values, floor)
    regularized = (vectors * values) @ vectors.T
    return _symmetrize(regularized), float(floor)


def _soft_threshold(
    matrix: np.ndarray,
    threshold: float,
    *,
    penalize_diagonal: bool = False,
) -> np.ndarray:
    result = np.sign(matrix) * np.maximum(np.abs(matrix) - threshold, 0.0)
    if not penalize_diagonal:
        np.fill_diagonal(result, np.diag(matrix))
    return _symmetrize(result)


def _sparse_spd_projection(matrix: np.ndarray, floor: float) -> np.ndarray:
    """Restore positive definiteness while preserving the zero pattern.

    If the thresholded ADMM variable is slightly indefinite, shrink only its
    off-diagonal entries toward zero.  This keeps all exact zeros exact and
    leaves the diagonal unchanged apart from a positive floor.
    """
    matrix = _symmetrize(np.asarray(matrix, dtype=float))
    diagonal = np.maximum(np.diag(matrix), floor)
    off_diagonal = matrix - np.diag(np.diag(matrix))
    base = np.diag(diagonal)

    candidate = base + off_diagonal
    if np.linalg.eigvalsh(candidate)[0] > floor:
        return candidate

    weight = 1.0
    for _ in range(80):
        weight *= 0.8
        candidate = base + weight * off_diagonal
        if np.linalg.eigvalsh(candidate)[0] > floor:
            return _symmetrize(candidate)
    return base


def _objective(
    scatter: np.ndarray,
    theta: np.ndarray,
    alpha: float,
    *,
    penalize_diagonal: bool = False,
) -> float:
    sign, logdet = np.linalg.slogdet(theta)
    if sign <= 0:
        return float("inf")
    penalized = theta if penalize_diagonal else theta - np.diag(np.diag(theta))
    return float(
        np.sum(scatter * theta) - logdet + alpha * np.sum(np.abs(penalized))
    )


def _solve_graphical_lasso_admm(
    scatter: np.ndarray,
    *,
    alpha: float,
    rho: float,
    max_iter: int,
    abs_tol: float,
    rel_tol: float,
    adaptive_rho: bool,
    penalize_diagonal: bool = False,
    initial: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    p = scatter.shape[0]
    if alpha == 0.0:
        precision = np.linalg.inv(scatter)
        precision = _symmetrize(precision)
        info = {
            "converged": True,
            "n_iter": 0,
            "objective_path": np.array([
                _objective(
                    scatter, precision, 0.0,
                    penalize_diagonal=penalize_diagonal,
                )
            ]),
            "primal_residual_path": np.array([0.0]),
            "dual_residual_path": np.array([0.0]),
            "rho_path": np.array([rho]),
        }
        return precision, info, (precision.copy(), precision.copy(), np.zeros_like(precision))

    if initial is None:
        theta = np.linalg.inv(scatter + alpha * np.eye(p))
        z = theta.copy()
        u = np.zeros_like(theta)
    else:
        theta, z, u = (np.asarray(item, dtype=float).copy() for item in initial)

    objective_path: list[float] = []
    primal_path: list[float] = []
    dual_path: list[float] = []
    rho_path: list[float] = []
    converged = False

    for iteration in range(1, max_iter + 1):
        # Theta update: solve rho*Theta - Theta^{-1} = rho*(Z-U) - S.
        a = _symmetrize(rho * (z - u) - scatter)
        values, vectors = np.linalg.eigh(a)
        updated_values = (values + np.sqrt(values**2 + 4.0 * rho)) / (2.0 * rho)
        theta = _symmetrize((vectors * updated_values) @ vectors.T)

        z_previous = z
        z = _soft_threshold(
            theta + u,
            alpha / rho,
            penalize_diagonal=penalize_diagonal,
        )
        u = u + theta - z

        primal = float(np.linalg.norm(theta - z, ord="fro"))
        dual = float(rho * np.linalg.norm(z - z_previous, ord="fro"))
        eps_primal = p * abs_tol + rel_tol * max(
            np.linalg.norm(theta, ord="fro"), np.linalg.norm(z, ord="fro")
        )
        eps_dual = p * abs_tol + rel_tol * np.linalg.norm(rho * u, ord="fro")

        objective_path.append(
            _objective(
                scatter, theta, alpha,
                penalize_diagonal=penalize_diagonal,
            )
        )
        primal_path.append(primal)
        dual_path.append(dual)
        rho_path.append(rho)

        if primal <= eps_primal and dual <= eps_dual:
            converged = True
            break

        if adaptive_rho:
            old_rho = rho
            if primal > 10.0 * dual and rho < 1e6:
                rho *= 2.0
            elif dual > 10.0 * primal and rho > 1e-6:
                rho /= 2.0
            if rho != old_rho:
                u *= old_rho / rho

    floor = max(
        np.finfo(float).tiny,
        100.0 * _EPS * max(np.linalg.norm(z, ord=2), np.finfo(float).tiny),
    )
    precision = _sparse_spd_projection(z, floor)
    info = {
        "converged": converged,
        "n_iter": iteration,
        "objective_path": np.asarray(objective_path, dtype=float),
        "primal_residual_path": np.asarray(primal_path, dtype=float),
        "dual_residual_path": np.asarray(dual_path, dtype=float),
        "rho_path": np.asarray(rho_path, dtype=float),
    }
    return precision, info, (theta.copy(), z.copy(), u.copy())


def _partial_correlations(precision: np.ndarray) -> np.ndarray:
    diagonal = np.sqrt(np.maximum(np.diag(precision), np.finfo(float).tiny))
    partial = -precision / np.outer(diagonal, diagonal)
    np.fill_diagonal(partial, 1.0)
    return _symmetrize(partial)


@dataclass
class RobustGraphicalLasso(EstimatorMixin):
    r"""Sparse precision estimation from a robust scatter matrix.

    The estimator first fits ``scatter_estimator`` and then solves

    .. math::

       \min_{\Theta \succ 0}
       \operatorname{tr}(S\Theta) - \log\det(\Theta)
       + \alpha\sum_{i\ne j}|\Theta_{ij}|,

    where ``S`` is the fitted robust scatter matrix.  The optimization is
    performed by ADMM and penalizes only off-diagonal precision entries.

    Parameters
    ----------
    alpha : float or {"ebic"}, default="ebic"
        Off-diagonal :math:`\ell_1` penalty. ``"ebic"`` evaluates a geometric
        penalty path and selects the model minimizing the extended Bayesian
        information criterion.
    scatter_estimator : object, {"empirical"}, or None, default=None
        Estimator copied and fitted on ``X``. It must expose ``covariance_`` and
        may expose ``location_``. ``None`` uses
        ``RegularizedCauchy(alpha=0.10)``. The ``"empirical"`` option is mainly
        intended as a non-robust baseline in examples.
    standardize : bool, default=True
        Solve the penalized problem on the robust correlation matrix and map
        the precision estimate back to the original feature scales.
    n_alphas : int, default=20
        Number of penalties evaluated when ``alpha="ebic"``.
    alpha_min_ratio : float, default=0.02
        Smallest path penalty relative to the largest off-diagonal entry of the
        working scatter matrix.
    ebic_gamma : float, default=0.5
        Additional high-dimensional EBIC penalty. Values between 0 and 1 are
        common; larger values favor sparser graphs.
    rho : float, default=1.0
        Initial ADMM penalty parameter.
    max_iter : int, default=300
        Maximum ADMM iterations per penalty value.
    abs_tol, rel_tol : float, default=1e-5, 1e-4
        Absolute and relative ADMM stopping tolerances.
    adaptive_rho : bool, default=True
        Adapt ``rho`` from primal and dual residuals.
    scatter_floor : float, default=1e-8
        Relative eigenvalue floor applied to the fitted scatter before sparse
        precision estimation.
    edge_tolerance : float, default=1e-8
        Precision entries with absolute value at most this threshold are not
        counted as graph edges. When ``standardize=True``, the threshold is
        applied to the standardized precision so graph support is invariant to
        feature measurement units.

    Notes
    -----
    Robustness comes from ``scatter_estimator``. This class is a robust-scatter
    graphical lasso, not the spatial-sign SGLASSO or robust CLIME algorithms.
    Shape-only scatter estimators may recover a useful graph, but the selected
    penalty depends on their scale convention unless ``standardize=True``.
    """

    alpha: float | str = "ebic"
    scatter_estimator: Any | str | None = None
    standardize: bool = True
    n_alphas: int = 20
    alpha_min_ratio: float = 0.02
    ebic_gamma: float = 0.5
    rho: float = 1.0
    max_iter: int = 300
    abs_tol: float = 1e-5
    rel_tol: float = 1e-4
    adaptive_rho: bool = True
    scatter_floor: float = 1e-8
    edge_tolerance: float = 1e-8

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "RobustGraphicalLasso":
        """Fit the robust scatter estimate and sparse precision matrix."""
        del y
        X = _as_fit_array(X)
        self._validate_parameters()

        estimator = _resolve_scatter_estimator(self.scatter_estimator)
        estimator.fit(X)
        if not hasattr(estimator, "covariance_"):
            raise AttributeError("scatter_estimator must expose covariance_ after fit")

        p = X.shape[1]
        raw_scatter = np.asarray(estimator.covariance_, dtype=float)
        if raw_scatter.shape != (p, p):
            raise ValueError(
                "scatter_estimator covariance_ has incompatible shape: "
                f"got {raw_scatter.shape}, expected {(p, p)}"
            )
        if not np.all(np.isfinite(raw_scatter)):
            raise ValueError("scatter_estimator covariance_ must contain only finite values")
        if self.standardize:
            raw_diagonal = np.diag(raw_scatter)
            if np.any(raw_diagonal < 0.0):
                raise ValueError(
                    "scatter_estimator covariance_ must have a non-negative diagonal"
                )
            positive = raw_diagonal > 0.0
            if not np.any(positive):
                raise ValueError(
                    "scatter_estimator covariance_ has no positive feature variance"
                )
            variance_scale = max(
                float(np.max(raw_diagonal[positive])), np.finfo(float).tiny
            )
            constant_variance = max(
                self.scatter_floor * variance_scale, np.finfo(float).tiny
            )
            adjusted_diagonal = np.where(
                positive, raw_diagonal, constant_variance
            )
            scales = np.sqrt(adjusted_diagonal)
            raw_working_scatter = raw_scatter / np.outer(scales, scales)
            constant_features = ~positive
            if np.any(constant_features):
                raw_working_scatter = raw_working_scatter.copy()
                raw_working_scatter[constant_features, :] = 0.0
                raw_working_scatter[:, constant_features] = 0.0
                raw_working_scatter[constant_features, constant_features] = 1.0
            working_scatter, floor = _regularize_spd(
                raw_working_scatter, self.scatter_floor
            )
            scatter = _symmetrize(working_scatter * np.outer(scales, scales))
        else:
            scales = np.ones(p)
            scatter, floor = _regularize_spd(raw_scatter, self.scatter_floor)
            raw_working_scatter = raw_scatter
            working_scatter = scatter

        if hasattr(estimator, "location_"):
            location = np.asarray(estimator.location_, dtype=float)
        else:
            location = np.nanmean(X, axis=0)
        if location.shape != (p,) or not np.all(np.isfinite(location)):
            raise ValueError("scatter_estimator location_ must be a finite vector of length p")

        if self.alpha == "ebic":
            alphas = self._alpha_grid(working_scatter)
            scores: list[float] = []
            edge_counts: list[int] = []
            models: list[tuple[np.ndarray, dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
            state = None
            for alpha in alphas:
                precision_working, info, state = _solve_graphical_lasso_admm(
                    working_scatter,
                    alpha=float(alpha),
                    rho=self.rho,
                    max_iter=self.max_iter,
                    abs_tol=self.abs_tol,
                    rel_tol=self.rel_tol,
                    adaptive_rho=self.adaptive_rho,
                    penalize_diagonal=False,
                    initial=state,
                )
                edges = self._count_edges(precision_working)
                scores.append(self._ebic(working_scatter, precision_working, X.shape[0], edges))
                edge_counts.append(edges)
                models.append((precision_working, info, state))
            best = int(np.argmin(scores))
            alpha_selected = float(alphas[best])
            precision_working, info, _ = models[best]
            self.alphas_ = np.asarray(alphas)
            self.ebic_scores_ = np.asarray(scores)
            self.path_n_edges_ = np.asarray(edge_counts, dtype=int)
            self.best_alpha_index_ = best
        else:
            alpha_selected = float(self.alpha)
            precision_working, info, _ = _solve_graphical_lasso_admm(
                working_scatter,
                alpha=alpha_selected,
                rho=self.rho,
                max_iter=self.max_iter,
                abs_tol=self.abs_tol,
                rel_tol=self.rel_tol,
                adaptive_rho=self.adaptive_rho,
                penalize_diagonal=False,
            )

        inverse_scales = 1.0 / scales
        precision = precision_working * np.outer(inverse_scales, inverse_scales)
        precision = _symmetrize(precision)
        covariance = _symmetrize(np.linalg.inv(precision))
        partial = _partial_correlations(precision)
        adjacency_basis = precision_working if self.standardize else precision
        adjacency = np.abs(adjacency_basis) > self.edge_tolerance
        np.fill_diagonal(adjacency, False)

        self.scatter_estimator_ = estimator
        self.location_ = location
        self.raw_scatter_ = _symmetrize(raw_scatter)
        self.scatter_ = scatter
        self.raw_working_scatter_ = _symmetrize(raw_working_scatter)
        self.working_scatter_ = working_scatter
        self.scale_ = scales
        self.constant_features_ = (
            constant_features.copy() if self.standardize else np.zeros(p, dtype=bool)
        )
        self.scatter_floor_ = floor
        self.alpha_ = alpha_selected
        self.precision_ = precision
        self.covariance_ = covariance
        self.partial_correlation_ = partial
        self.adjacency_ = adjacency
        self.n_edges_ = int(np.count_nonzero(np.triu(adjacency, 1)))
        self.graph_density_ = float(2 * self.n_edges_ / (p * (p - 1)))
        self.conditional_coefficients_ = self._conditional_coefficients(precision)
        self.converged_ = bool(info["converged"])
        self.n_iter_ = int(info["n_iter"])
        self.objective_path_ = info["objective_path"]
        self.primal_residual_path_ = info["primal_residual_path"]
        self.dual_residual_path_ = info["dual_residual_path"]
        self.rho_path_ = info["rho_path"]
        self.n_samples_in_ = X.shape[0]
        self.n_features_in_ = p
        return self

    def _validate_parameters(self) -> None:
        if isinstance(self.alpha, str):
            if self.alpha != "ebic":
                raise ValueError("alpha string must be 'ebic'")
        elif not np.isscalar(self.alpha) or not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be a non-negative finite number or 'ebic'")
        if not isinstance(self.standardize, (bool, np.bool_)):
            raise TypeError("standardize must be a boolean")
        if not isinstance(self.adaptive_rho, (bool, np.bool_)):
            raise TypeError("adaptive_rho must be a boolean")
        if not isinstance(self.n_alphas, (int, np.integer)) or self.n_alphas < 2:
            raise ValueError("n_alphas must be an integer of at least 2")
        if not np.isfinite(self.alpha_min_ratio) or not (0 < self.alpha_min_ratio <= 1):
            raise ValueError("alpha_min_ratio must be in (0, 1]")
        if not np.isfinite(self.ebic_gamma) or self.ebic_gamma < 0:
            raise ValueError("ebic_gamma must be non-negative")
        for name in ("rho", "abs_tol", "rel_tol", "scatter_floor"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not np.isfinite(self.edge_tolerance) or self.edge_tolerance < 0:
            raise ValueError("edge_tolerance must be a non-negative finite number")

    def _alpha_grid(self, scatter: np.ndarray) -> np.ndarray:
        off = scatter - np.diag(np.diag(scatter))
        alpha_max = float(np.max(np.abs(off)))
        scatter_scale = max(float(np.max(np.abs(scatter))), np.finfo(float).tiny)
        if alpha_max <= 100.0 * _EPS * scatter_scale:
            alpha_max = 1e-3 * scatter_scale
        alpha_min = max(
            alpha_max * self.alpha_min_ratio,
            np.finfo(float).tiny,
        )
        return np.geomspace(alpha_max, alpha_min, self.n_alphas)

    def _count_edges(self, precision: np.ndarray) -> int:
        return int(np.count_nonzero(np.abs(np.triu(precision, 1)) > self.edge_tolerance))

    def _ebic(self, scatter: np.ndarray, precision: np.ndarray, n: int, edges: int) -> float:
        sign, logdet = np.linalg.slogdet(precision)
        if sign <= 0:
            return float("inf")
        negative_twice_loglik = n * (float(np.sum(scatter * precision)) - float(logdet))
        p = precision.shape[0]
        return float(
            negative_twice_loglik
            + edges * np.log(max(n, 2))
            + 4.0 * self.ebic_gamma * edges * np.log(max(p, 2))
        )

    @staticmethod
    def _conditional_coefficients(precision: np.ndarray) -> np.ndarray:
        coefficients = -precision / np.diag(precision)[:, None]
        np.fill_diagonal(coefficients, 0.0)
        return coefficients

    def _check_fitted(self) -> None:
        if not hasattr(self, "precision_"):
            raise AttributeError("RobustGraphicalLasso is not fitted yet")

    def mahalanobis(self, X: np.ndarray) -> np.ndarray:
        """Return squared Mahalanobis distances from the fitted robust graph."""
        self._check_fitted()
        X = _as_finite_array(X, n_features=self.n_features_in_)
        centered = X - self.location_
        return np.einsum("ij,jk,ik->i", centered, self.precision_, centered)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return Gaussian log scores under the fitted location and precision."""
        distances = self.mahalanobis(X)
        sign, logdet = np.linalg.slogdet(self.precision_)
        if sign <= 0:  # pragma: no cover - guarded by the SPD projection
            raise RuntimeError("fitted precision matrix is not positive definite")
        constant = self.n_features_in_ * np.log(2.0 * np.pi)
        return 0.5 * (logdet - constant - distances)

    def edge_list(
        self,
        feature_names: Sequence[str] | None = None,
        *,
        min_abs_partial_correlation: float = 0.0,
    ) -> list[tuple[str | int, str | int, float]]:
        """Return graph edges sorted by absolute partial correlation."""
        self._check_fitted()
        if not np.isfinite(min_abs_partial_correlation) or min_abs_partial_correlation < 0:
            raise ValueError("min_abs_partial_correlation must be non-negative")
        if feature_names is None:
            names: list[str | int] = list(range(self.n_features_in_))
        else:
            if len(feature_names) != self.n_features_in_:
                raise ValueError("feature_names must have one entry per feature")
            names = list(feature_names)

        edges: list[tuple[str | int, str | int, float]] = []
        for i in range(self.n_features_in_):
            for j in range(i + 1, self.n_features_in_):
                value = float(self.partial_correlation_[i, j])
                if self.adjacency_[i, j] and abs(value) >= min_abs_partial_correlation:
                    edges.append((names[i], names[j], value))
        edges.sort(key=lambda edge: abs(edge[2]), reverse=True)
        return edges


SparseRobustPrecision = RobustGraphicalLasso


def _stable_vector_norm(vector: np.ndarray) -> float:
    vector = np.asarray(vector, dtype=float)
    scale = float(np.max(np.abs(vector), initial=0.0))
    if scale == 0.0:
        return 0.0
    return float(scale * np.sqrt(np.sum((vector / scale) ** 2)))


def _stable_row_norms(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    scales = np.max(np.abs(matrix), axis=1)
    norms = np.zeros(matrix.shape[0], dtype=float)
    nonzero = scales > 0.0
    normalized = matrix[nonzero] / scales[nonzero, None]
    norms[nonzero] = scales[nonzero] * np.sqrt(np.sum(normalized * normalized, axis=1))
    return norms


def _spatial_data_scale(X: np.ndarray, location: np.ndarray) -> float:
    centered = np.asarray(X, dtype=float) - np.asarray(location, dtype=float)
    radii = _stable_row_norms(centered)
    return max(float(np.max(radii, initial=0.0)), np.finfo(float).tiny)


def _spatial_median(
    X: np.ndarray,
    *,
    tol: float = 1e-8,
    max_iter: int = 300,
    zero_tolerance: float = 1e-12,
) -> tuple[np.ndarray, int, bool]:
    """Return the spatial median using scale-relative stopping tolerances."""
    X = np.asarray(X, dtype=float)
    current = np.median(X, axis=0)
    data_scale = _spatial_data_scale(X, current)
    zero_floor = max(zero_tolerance * data_scale, np.finfo(float).tiny)
    step_tolerance = tol * data_scale
    converged = False
    iteration = 0
    for iteration in range(1, max_iter + 1):
        differences = X - current
        distances = _stable_row_norms(differences)
        coincident = distances <= zero_floor
        if np.any(coincident):
            candidate = np.mean(X[coincident], axis=0)
            remaining = X[~coincident] - candidate
            norms = _stable_row_norms(remaining)
            if remaining.size == 0:
                current = candidate
                converged = True
                break
            subgradient = np.sum(
                remaining / np.maximum(norms[:, None], zero_floor), axis=0
            )
            if _stable_vector_norm(subgradient) <= int(np.count_nonzero(coincident)):
                current = candidate
                converged = True
                break
            distances = np.maximum(_stable_row_norms(X - current), zero_floor)
        weights = 1.0 / np.maximum(distances, zero_floor)
        updated = np.sum(weights[:, None] * X, axis=0) / np.sum(weights)
        if _stable_vector_norm(updated - current) <= step_tolerance:
            current = updated
            converged = True
            break
        current = updated
    return np.asarray(current, dtype=float), int(iteration), bool(converged)


def _spatial_sign_covariance(
    X: np.ndarray,
    location: np.ndarray,
    *,
    zero_tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, int]:
    centered = np.asarray(X, dtype=float) - np.asarray(location, dtype=float)
    radii = _stable_row_norms(centered)
    data_scale = max(float(np.max(radii, initial=0.0)), np.finfo(float).tiny)
    zero_floor = max(zero_tolerance * data_scale, np.finfo(float).tiny)
    nonzero = radii > zero_floor
    signs = np.zeros_like(centered)
    signs[nonzero] = centered[nonzero] / radii[nonzero, None]
    covariance = signs.T @ signs / X.shape[0]
    return _symmetrize(covariance), signs, int(np.count_nonzero(~nonzero))


@dataclass
class SpatialSignGraphicalLasso(EstimatorMixin):
    r"""Sparse shape-precision estimation from spatial signs.

    The estimator calculates the sample spatial-sign covariance matrix

    .. math::

       \widehat S
       = \frac{1}{n}\sum_{i=1}^{n}
         U(x_i-\widehat\mu)U(x_i-\widehat\mu)^T,
       \qquad U(z)=z/\lVert z\rVert_2,

    where :math:`\widehat\mu` is the spatial median, and solves a graphical
    lasso problem with :math:`p\widehat S` as the working scatter matrix.
    Under the high-dimensional elliptical assumptions used by Lu and Feng
    (2025), this targets the precision of the trace-normalized shape matrix,
    up to the scale that is irrelevant for graph support and partial
    correlations.

    Parameters
    ----------
    alpha : float or {"ebic"}, default="ebic"
        :math:`\ell_1` penalty. ``"ebic"`` selects a value from a geometric
        path using a spatial-sign pseudo-likelihood EBIC.
    penalize_diagonal : bool, default=True
        Penalize diagonal entries as in the published SGLASSO objective.
        Set to ``False`` for the more common off-diagonal-only graphical-lasso
        convention.
    missing_values : {"raise", "median"}, default="raise"
        The spatial-sign theory assumes complete rows. ``"median"`` enables a
        practical coordinate-median imputation before fitting but does not
        inherit the paper's guarantees.
    n_alphas, alpha_min_ratio, ebic_gamma : int, float, float
        Penalty-path and EBIC settings used when ``alpha="ebic"``.
    rho, max_iter, abs_tol, rel_tol, adaptive_rho :
        ADMM solver settings.
    scatter_floor : float, default=1e-8
        Relative eigenvalue floor applied to :math:`p\widehat S` before the
        sparse optimization.
    edge_tolerance : float, default=1e-8
        Absolute precision threshold used to define graph edges.
    spatial_median_tol : float, default=1e-8
        Relative convergence tolerance for the spatial median.
    spatial_median_max_iter : int, default=300
        Maximum safeguarded Weiszfeld iterations.
    zero_tolerance : float, default=1e-12
        Relative radius below which a centered observation has the zero spatial
        sign. The effective absolute threshold is stored in
        ``effective_zero_tolerance_``.

    Notes
    -----
    This class implements the spatial-sign graphical-lasso objective. The
    paper selects the penalty with an independent validation sample; the EBIC
    path provided here is package-specific. Absolute covariance scale is not
    identified by spatial signs, so ``covariance_`` is a normalized shape
    matrix and ``precision_`` is its inverse up to a common scalar.
    """

    alpha: float | str = "ebic"
    penalize_diagonal: bool = True
    missing_values: str = "raise"
    n_alphas: int = 20
    alpha_min_ratio: float = 0.02
    ebic_gamma: float = 0.5
    rho: float = 1.0
    max_iter: int = 300
    abs_tol: float = 1e-5
    rel_tol: float = 1e-4
    adaptive_rho: bool = True
    scatter_floor: float = 1e-8
    edge_tolerance: float = 1e-8
    spatial_median_tol: float = 1e-8
    spatial_median_max_iter: int = 300
    zero_tolerance: float = 1e-12

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "SpatialSignGraphicalLasso":
        """Fit the spatial median, sign covariance, and sparse shape precision."""
        del y
        X = _as_fit_array(X)
        self._validate_parameters()
        X_fit = self._prepare_fit_array(X)
        n, p = X_fit.shape

        location, median_iterations, median_converged = _spatial_median(
            X_fit,
            tol=self.spatial_median_tol,
            max_iter=self.spatial_median_max_iter,
            zero_tolerance=self.zero_tolerance,
        )
        sign_covariance, sign_vectors, zero_count = _spatial_sign_covariance(
            X_fit,
            location,
            zero_tolerance=self.zero_tolerance,
        )
        if zero_count == n:
            raise ValueError(
                "all observations coincide; spatial-sign precision is undefined"
            )
        raw_working = p * sign_covariance
        working_scatter, floor = _regularize_spd(raw_working, self.scatter_floor)
        spatial_scale = _spatial_data_scale(X_fit, location)

        if self.alpha == "ebic":
            alphas = self._alpha_grid(working_scatter)
            scores: list[float] = []
            edge_counts: list[int] = []
            models: list[
                tuple[np.ndarray, dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray]]
            ] = []
            state = None
            for alpha in alphas:
                precision, info, state = _solve_graphical_lasso_admm(
                    working_scatter,
                    alpha=float(alpha),
                    rho=self.rho,
                    max_iter=self.max_iter,
                    abs_tol=self.abs_tol,
                    rel_tol=self.rel_tol,
                    adaptive_rho=self.adaptive_rho,
                    penalize_diagonal=self.penalize_diagonal,
                    initial=state,
                )
                edges = self._count_edges(precision)
                scores.append(self._ebic(working_scatter, precision, n, edges))
                edge_counts.append(edges)
                models.append((precision, info, state))
            best = int(np.argmin(scores))
            alpha_selected = float(alphas[best])
            precision, info, _ = models[best]
            self.alphas_ = np.asarray(alphas, dtype=float)
            self.ebic_scores_ = np.asarray(scores, dtype=float)
            self.path_n_edges_ = np.asarray(edge_counts, dtype=int)
            self.best_alpha_index_ = best
        else:
            alpha_selected = float(self.alpha)
            precision, info, _ = _solve_graphical_lasso_admm(
                working_scatter,
                alpha=alpha_selected,
                rho=self.rho,
                max_iter=self.max_iter,
                abs_tol=self.abs_tol,
                rel_tol=self.rel_tol,
                adaptive_rho=self.adaptive_rho,
                penalize_diagonal=self.penalize_diagonal,
            )

        precision = _symmetrize(precision)
        covariance = _symmetrize(np.linalg.inv(precision))
        partial = _partial_correlations(precision)
        adjacency = np.abs(precision) > self.edge_tolerance
        np.fill_diagonal(adjacency, False)

        self.location_ = location
        self.spatial_median_ = location.copy()
        self.spatial_median_n_iter_ = median_iterations
        self.spatial_median_converged_ = median_converged
        self.spatial_sign_covariance_ = sign_covariance
        self.sign_vectors_ = sign_vectors
        self.zero_sign_count_ = zero_count
        self.spatial_scale_ = spatial_scale
        self.effective_zero_tolerance_ = max(
            self.zero_tolerance * spatial_scale, np.finfo(float).tiny
        )
        self.raw_working_scatter_ = raw_working
        self.working_scatter_ = working_scatter
        self.scatter_floor_ = floor
        self.alpha_ = alpha_selected
        self.precision_ = precision
        self.shape_precision_ = precision
        self.covariance_ = covariance
        self.shape_ = covariance
        self.partial_correlation_ = partial
        self.adjacency_ = adjacency
        self.n_edges_ = int(np.count_nonzero(np.triu(adjacency, 1)))
        self.graph_density_ = float(2 * self.n_edges_ / (p * (p - 1)))
        self.conditional_coefficients_ = RobustGraphicalLasso._conditional_coefficients(
            precision
        )
        self.converged_ = bool(info["converged"])
        self.n_iter_ = int(info["n_iter"])
        self.objective_path_ = info["objective_path"]
        self.primal_residual_path_ = info["primal_residual_path"]
        self.dual_residual_path_ = info["dual_residual_path"]
        self.rho_path_ = info["rho_path"]
        self.n_samples_in_ = n
        self.n_features_in_ = p
        return self

    def _prepare_fit_array(self, X: np.ndarray) -> np.ndarray:
        missing = np.isnan(X)
        if missing.any():
            if self.missing_values == "raise":
                raise ValueError(
                    "SpatialSignGraphicalLasso does not accept missing values unless "
                    "missing_values='median'"
                )
            medians = np.nanmedian(X, axis=0)
            if not np.all(np.isfinite(medians)):
                raise ValueError("every feature needs at least one finite value")
            self.imputation_values_ = medians
            return np.where(missing, medians, X)
        self.imputation_values_ = np.median(X, axis=0)
        return np.asarray(X, dtype=float)

    def _validate_parameters(self) -> None:
        if isinstance(self.alpha, str):
            if self.alpha != "ebic":
                raise ValueError("alpha string must be 'ebic'")
        elif not np.isscalar(self.alpha) or not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be a non-negative finite number or 'ebic'")
        if not isinstance(self.penalize_diagonal, (bool, np.bool_)):
            raise TypeError("penalize_diagonal must be a boolean")
        if self.missing_values not in {"raise", "median"}:
            raise ValueError("missing_values must be 'raise' or 'median'")
        if not isinstance(self.adaptive_rho, (bool, np.bool_)):
            raise TypeError("adaptive_rho must be a boolean")
        if not isinstance(self.n_alphas, (int, np.integer)) or self.n_alphas < 2:
            raise ValueError("n_alphas must be an integer of at least 2")
        if not np.isfinite(self.alpha_min_ratio) or not (0 < self.alpha_min_ratio <= 1):
            raise ValueError("alpha_min_ratio must be in (0, 1]")
        if not np.isfinite(self.ebic_gamma) or self.ebic_gamma < 0:
            raise ValueError("ebic_gamma must be non-negative")
        for name in (
            "rho", "abs_tol", "rel_tol", "scatter_floor",
            "spatial_median_tol", "zero_tolerance",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        for name in ("max_iter", "spatial_median_max_iter"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.edge_tolerance) or self.edge_tolerance < 0:
            raise ValueError("edge_tolerance must be a non-negative finite number")

    def _alpha_grid(self, scatter: np.ndarray) -> np.ndarray:
        off = scatter - np.diag(np.diag(scatter))
        alpha_max = float(np.max(np.abs(off)))
        scatter_scale = max(float(np.max(np.abs(scatter))), np.finfo(float).tiny)
        if alpha_max <= 100.0 * _EPS * scatter_scale:
            alpha_max = 1e-3 * scatter_scale
        alpha_min = max(
            alpha_max * self.alpha_min_ratio,
            np.finfo(float).tiny,
        )
        return np.geomspace(alpha_max, alpha_min, self.n_alphas)

    def _count_edges(self, precision: np.ndarray) -> int:
        return int(np.count_nonzero(np.abs(np.triu(precision, 1)) > self.edge_tolerance))

    def _ebic(self, scatter: np.ndarray, precision: np.ndarray, n: int, edges: int) -> float:
        sign, logdet = np.linalg.slogdet(precision)
        if sign <= 0:
            return float("inf")
        negative_twice_pseudo_loglik = n * (
            float(np.sum(scatter * precision)) - float(logdet)
        )
        p = precision.shape[0]
        return float(
            negative_twice_pseudo_loglik
            + edges * np.log(max(n, 2))
            + 4.0 * self.ebic_gamma * edges * np.log(max(p, 2))
        )

    def _check_fitted(self) -> None:
        if not hasattr(self, "precision_"):
            raise AttributeError("SpatialSignGraphicalLasso is not fitted yet")

    def _prepare_new_array(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )
        if np.any(np.isinf(X)):
            raise ValueError("X must not contain infinite values")
        if np.isnan(X).any():
            if self.missing_values != "median":
                raise ValueError("X contains missing values")
            X = np.where(np.isnan(X), self.imputation_values_, X)
        return X

    def mahalanobis(self, X: np.ndarray) -> np.ndarray:
        """Return squared shape distances under the fitted precision."""
        self._check_fitted()
        X = self._prepare_new_array(X)
        centered = X - self.location_
        return np.einsum("ij,jk,ik->i", centered, self.precision_, centered)

    def shape_distances(self, X: np.ndarray) -> np.ndarray:
        """Alias for :meth:`mahalanobis`, emphasizing unidentified scale."""
        return self.mahalanobis(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return Gaussian-shape pseudo log scores, defined up to radial scale."""
        distances = self.mahalanobis(X)
        sign, logdet = np.linalg.slogdet(self.precision_)
        if sign <= 0:  # pragma: no cover
            raise RuntimeError("fitted precision matrix is not positive definite")
        constant = self.n_features_in_ * np.log(2.0 * np.pi)
        return 0.5 * (logdet - constant - distances)

    def edge_list(
        self,
        feature_names: Sequence[str] | None = None,
        *,
        min_abs_partial_correlation: float = 0.0,
    ) -> list[tuple[str | int, str | int, float]]:
        """Return graph edges sorted by absolute partial correlation."""
        self._check_fitted()
        if not np.isfinite(min_abs_partial_correlation) or min_abs_partial_correlation < 0:
            raise ValueError("min_abs_partial_correlation must be non-negative")
        if feature_names is None:
            names: list[str | int] = list(range(self.n_features_in_))
        else:
            if len(feature_names) != self.n_features_in_:
                raise ValueError("feature_names must have one entry per feature")
            names = list(feature_names)
        edges: list[tuple[str | int, str | int, float]] = []
        for i in range(self.n_features_in_):
            for j in range(i + 1, self.n_features_in_):
                value = float(self.partial_correlation_[i, j])
                if self.adjacency_[i, j] and abs(value) >= min_abs_partial_correlation:
                    edges.append((names[i], names[j], value))
        edges.sort(key=lambda edge: abs(edge[2]), reverse=True)
        return edges


SGLASSO = SpatialSignGraphicalLasso
SpatialSignSparsePrecision = SpatialSignGraphicalLasso
