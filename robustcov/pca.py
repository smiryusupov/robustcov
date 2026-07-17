# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Robust principal component analysis from robust scatter estimates."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_2d_finite_array(X: np.ndarray, *, name: str = "X") -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if X.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two samples")
    if X.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one feature")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} must contain only finite values")
    return X


def _symmetrize(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def _default_estimator() -> Any:
    from .m_estimators import RegularizedCauchy

    return RegularizedCauchy(alpha=0.10)


def _deterministic_eigenvector_signs(eigenvectors: np.ndarray) -> np.ndarray:
    """Orient each eigenvector so its largest loading is non-negative."""
    eigenvectors = np.asarray(eigenvectors, dtype=float).copy()
    if eigenvectors.size == 0:
        return eigenvectors

    columns = np.arange(eigenvectors.shape[1])
    max_rows = np.argmax(np.abs(eigenvectors), axis=0)
    signs = np.sign(eigenvectors[max_rows, columns])
    signs[signs == 0.0] = 1.0
    eigenvectors *= signs
    return eigenvectors


@dataclass
class RobustPCA:
    """Principal component analysis driven by a robust scatter estimator.

    ``RobustPCA`` fits an existing ``robustcov`` estimator (or any compatible
    estimator exposing ``covariance_`` after ``fit``), eigendecomposes its
    robust scatter matrix, and provides PCA-style projection and reconstruction.
    It also exposes the two complementary diagnostics used in robust subspace
    analysis: score distance within the retained subspace and orthogonal
    distance from that subspace.

    This class implements robust *scatter PCA*. It is intentionally distinct
    from low-rank-plus-sparse matrix decomposition methods that are also called
    "robust PCA", and from the full projection-pursuit ROBPCA algorithm.

    Parameters
    ----------
    n_components : int, float, or None, default=None
        Number of retained components. If an integer, it must be between 1 and
        ``n_features``. If a float in ``(0, 1]``, the smallest number of
        components reaching that fraction of robust explained variance is
        selected. ``None`` retains all components.
    estimator : object, optional
        Estimator copied and fitted through ``estimator.fit(X)``. It must expose
        ``covariance_`` and may expose ``location_``. If omitted,
        ``RegularizedCauchy(alpha=0.10)`` is used.
    whiten : bool, default=False
        Divide projected scores by the square roots of their robust eigenvalues.
    ridge : float, default=1e-10
        Positive relative eigenvalue floor. The floor is ``ridge`` times the
        larger of the maximum raw eigenvalue and one.
    store_scores : bool, default=True
        Store training projections and distance diagnostics after fitting.

    Notes
    -----
    For shape-only estimators such as an unscaled Tyler estimator, component
    directions and explained-variance ratios are meaningful, while absolute
    eigenvalue scales depend on the estimator's scale convention.
    """

    n_components: int | float | None = None
    estimator: Any | None = None
    whiten: bool = False
    ridge: float = 1e-10
    store_scores: bool = True

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "RobustPCA":
        """Fit a robust principal subspace.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training observations.
        y : ignored, optional
            Present for sklearn-style compatibility.
        """
        del y

        X = _as_2d_finite_array(X)
        self._validate_parameters()

        estimator = _default_estimator() if self.estimator is None else deepcopy(self.estimator)
        estimator.fit(X)

        if not hasattr(estimator, "covariance_"):
            raise AttributeError("estimator must expose covariance_ after fit")

        covariance = np.asarray(estimator.covariance_, dtype=float)
        expected_shape = (X.shape[1], X.shape[1])
        if covariance.shape != expected_shape:
            raise ValueError(
                "estimator covariance_ has incompatible shape: "
                f"got {covariance.shape}, expected {expected_shape}"
            )
        if not np.all(np.isfinite(covariance)):
            raise ValueError("estimator covariance_ must contain only finite values")
        covariance = _symmetrize(covariance)

        if hasattr(estimator, "location_"):
            location = np.asarray(estimator.location_, dtype=float)
        else:
            location = np.mean(X, axis=0)
        if location.shape != (X.shape[1],):
            raise ValueError("estimator location_ has incompatible shape")
        if not np.all(np.isfinite(location)):
            raise ValueError("estimator location_ must contain only finite values")

        raw_eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(raw_eigenvalues)[::-1]
        raw_eigenvalues = raw_eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        eigenvectors = _deterministic_eigenvector_signs(eigenvectors)

        scale = max(float(raw_eigenvalues[0]), 1.0)
        eigenvalue_floor = self.ridge * scale
        all_eigenvalues = np.maximum(raw_eigenvalues, eigenvalue_floor)
        regularized_covariance = (eigenvectors * all_eigenvalues) @ eigenvectors.T
        regularized_covariance = _symmetrize(regularized_covariance)

        n_components = self._resolve_n_components(all_eigenvalues, X.shape[1])
        components = eigenvectors[:, :n_components].T
        eigenvalues = all_eigenvalues[:n_components]
        total_variance = float(np.sum(all_eigenvalues))

        self.estimator_ = estimator
        self.location_ = location
        self.mean_ = location
        self.raw_covariance_ = covariance
        self.covariance_ = regularized_covariance
        self.raw_eigenvalues_ = raw_eigenvalues
        self.all_eigenvalues_ = all_eigenvalues
        self.eigenvalue_floor_ = float(eigenvalue_floor)
        self.components_ = components
        self.eigenvalues_ = eigenvalues
        self.explained_variance_ = eigenvalues.copy()
        self.explained_variance_ratio_ = eigenvalues / total_variance
        self.singular_values_ = np.sqrt(eigenvalues * max(X.shape[0] - 1, 1))
        self.n_components_ = n_components
        self.n_samples_in_ = X.shape[0]
        self.n_features_in_ = X.shape[1]
        self.noise_variance_ = (
            float(np.mean(all_eigenvalues[n_components:]))
            if n_components < X.shape[1]
            else 0.0
        )

        if self.store_scores:
            self.scores_ = self.transform(X)
            self.score_distances_ = self.score_distances(X)
            self.orthogonal_distances_ = self.orthogonal_distances(X)
        else:
            for attribute in ("scores_", "score_distances_", "orthogonal_distances_"):
                if hasattr(self, attribute):
                    delattr(self, attribute)

        return self

    def _validate_parameters(self) -> None:
        if not isinstance(self.whiten, (bool, np.bool_)):
            raise TypeError("whiten must be a boolean")
        if not isinstance(self.store_scores, (bool, np.bool_)):
            raise TypeError("store_scores must be a boolean")
        if not np.isscalar(self.ridge) or not np.isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError("ridge must be a positive finite number")

        n_components = self.n_components
        if n_components is None:
            return
        if isinstance(n_components, (bool, np.bool_)):
            raise TypeError("n_components must be an int, float, or None")
        if isinstance(n_components, (int, np.integer)):
            if n_components < 1:
                raise ValueError("integer n_components must be at least 1")
            return
        if isinstance(n_components, (float, np.floating)):
            if not np.isfinite(n_components) or not (0.0 < n_components <= 1.0):
                raise ValueError("float n_components must be in (0, 1]")
            return
        raise TypeError("n_components must be an int, float, or None")

    def _resolve_n_components(self, eigenvalues: np.ndarray, n_features: int) -> int:
        n_components = self.n_components
        if n_components is None:
            return n_features
        if isinstance(n_components, (int, np.integer)):
            if n_components > n_features:
                raise ValueError(
                    f"n_components={n_components} exceeds n_features={n_features}"
                )
            return int(n_components)

        threshold = float(n_components)
        if threshold == 1.0:
            return n_features
        cumulative = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        return int(np.searchsorted(cumulative, threshold, side="left") + 1)

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "components_"):
            raise AttributeError("RobustPCA is not fitted yet")

    def _check_features(self, X: np.ndarray, *, name: str = "X") -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"{name} must be a 2D array")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"{name} has {X.shape[1]} features, expected {self.n_features_in_}"
            )
        if not np.all(np.isfinite(X)):
            raise ValueError(f"{name} must contain only finite values")
        return X

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        X = self._check_features(X)
        return (X - self.location_) @ self.components_.T

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project observations onto the fitted robust principal components."""
        scores = self._raw_scores(X)
        if self.whiten:
            scores = scores / np.sqrt(self.eigenvalues_)
        return scores

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fit the robust principal subspace and project the training data."""
        return self.fit(X, y).transform(X)

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        """Map component scores back to the original feature space."""
        self._check_is_fitted()
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 2:
            raise ValueError("scores must be a 2D array")
        if scores.shape[1] != self.n_components_:
            raise ValueError(
                f"scores has {scores.shape[1]} components, expected {self.n_components_}"
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError("scores must contain only finite values")
        if self.whiten:
            scores = scores * np.sqrt(self.eigenvalues_)
        return scores @ self.components_ + self.location_

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Project and reconstruct observations using the retained subspace."""
        return self.inverse_transform(self.transform(X))

    def score_distances(self, X: np.ndarray) -> np.ndarray:
        """Return distances from the center within the retained subspace.

        The squared score distance is the sum of squared unwhitened component
        scores divided by their corresponding robust eigenvalues.
        """
        scores = self._raw_scores(X)
        d2 = np.sum((scores * scores) / self.eigenvalues_, axis=1)
        return np.sqrt(np.maximum(d2, 0.0))

    def orthogonal_distances(self, X: np.ndarray) -> np.ndarray:
        """Return Euclidean distances from observations to the retained subspace."""
        X = self._check_features(X)
        centered = X - self.location_
        scores = centered @ self.components_.T
        residuals = centered - scores @ self.components_
        return np.linalg.norm(residuals, axis=1)

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Return per-sample squared reconstruction errors."""
        distances = self.orthogonal_distances(X)
        return distances * distances

    def outlier_map(self, X: np.ndarray) -> np.ndarray:
        """Return score and orthogonal distances for robust outlier mapping.

        Returns
        -------
        distances : ndarray of shape (n_samples, 2)
            Column 0 contains score distances and column 1 contains orthogonal
            distances. Keeping both axes explicit avoids imposing an arbitrary
            combined anomaly score.
        """
        return np.column_stack(
            [self.score_distances(X), self.orthogonal_distances(X)]
        )
