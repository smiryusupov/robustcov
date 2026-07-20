# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Robust independent component analysis based on two scatter matrices."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from ._estimator import EstimatorMixin
from ._utils import check_array
from .joint_diagonalization import canonicalize_unmixing
from .m_estimators import StudentTScatter


def _symmetric_eigendecomposition(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("scatter matrix must be square")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("scatter matrix must contain only finite values")
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def _sample_pair_differences(
    X: np.ndarray, max_pairs: int, random_state: int | None
) -> np.ndarray:
    n_samples = X.shape[0]
    total_pairs = n_samples * (n_samples - 1) // 2
    if total_pairs <= max_pairs:
        left, right = np.triu_indices(n_samples, k=1)
    else:
        rng = np.random.default_rng(random_state)
        selected: set[tuple[int, int]] = set()
        while len(selected) < max_pairs:
            left_batch = rng.integers(0, n_samples, size=max_pairs)
            right_batch = rng.integers(0, n_samples, size=max_pairs)
            for left_index, right_index in zip(left_batch, right_batch, strict=True):
                if left_index == right_index:
                    continue
                if left_index > right_index:
                    left_index, right_index = right_index, left_index
                selected.add((int(left_index), int(right_index)))
                if len(selected) == max_pairs:
                    break
        pairs = np.asarray(sorted(selected), dtype=int)
        left, right = pairs[:, 0], pairs[:, 1]
    return (X[left] - X[right]) / np.sqrt(2.0)


class TwoScatterICA(EstimatorMixin):
    """Independent component analysis driven by robust scatter geometry.

    The observations are first whitened with ``scatter_estimator``.  A second
    scatter matrix is then diagonalized in the whitened coordinates.  The
    default second scatter is a winsorized radial fourth-moment matrix, a robust
    analogue of the FOBI/ICS construction.  A second robust scatter estimator
    may be supplied instead.

    Parameters
    ----------
    n_components : int or None, default=None
        Number of recovered components. ``None`` retains all features.
    scatter_estimator : object or None, default=None
        Affine-equivariant scatter estimator used for centering and whitening.
        The default is an unregularized Student-t M-scatter estimator.
    second_scatter_estimator : object or None, default=None
        Optional estimator fitted to whitened observations.  When omitted, a
        clipped radial fourth-moment scatter is used.
    radial_clip_quantile : float, default=0.95
        Upper winsorization quantile for radial squared norms.
    symmetrize : bool, default=False
        Use pairwise differences for the second scatter.  This removes source
        skewness and gives the scatter construction the independent-components
        property for a wider class of source distributions.
    max_pairs : int, default=20000
        Maximum number of pairwise differences used when ``symmetrize=True``.
    random_state : int or None, default=0
        Seed for pair subsampling.
    eigenvalue_floor : float, default=1e-12
        Relative floor used while whitening.
    """

    def __init__(
        self,
        n_components=None,
        scatter_estimator=None,
        second_scatter_estimator=None,
        radial_clip_quantile=0.95,
        symmetrize=False,
        max_pairs=20000,
        random_state=0,
        eigenvalue_floor=1e-12,
    ):
        self.n_components = n_components
        self.scatter_estimator = scatter_estimator
        self.second_scatter_estimator = second_scatter_estimator
        self.radial_clip_quantile = radial_clip_quantile
        self.symmetrize = symmetrize
        self.max_pairs = max_pairs
        self.random_state = random_state
        self.eigenvalue_floor = eigenvalue_floor

    def _validate_parameters(self, n_features: int, n_samples: int) -> int:
        if self.n_components is None:
            n_components = n_features
        elif isinstance(self.n_components, (int, np.integer)) and not isinstance(
            self.n_components, (bool, np.bool_)
        ):
            n_components = int(self.n_components)
        else:
            raise TypeError("n_components must be an integer or None")
        if not 1 <= n_components <= min(n_features, n_samples - 1):
            raise ValueError("n_components must be between 1 and min(n_features, n_samples - 1)")
        if not np.isscalar(self.radial_clip_quantile) or not (
            0.5 < float(self.radial_clip_quantile) <= 1.0
        ):
            raise ValueError("radial_clip_quantile must be in (0.5, 1]")
        if not isinstance(self.symmetrize, (bool, np.bool_)):
            raise TypeError("symmetrize must be a boolean")
        if not isinstance(self.max_pairs, (int, np.integer)) or self.max_pairs < 1:
            raise ValueError("max_pairs must be a positive integer")
        if not np.isscalar(self.eigenvalue_floor) or self.eigenvalue_floor <= 0:
            raise ValueError("eigenvalue_floor must be positive")
        return n_components

    def fit(self, X, y=None):
        del y
        X = check_array(X)
        n_samples, n_features = X.shape
        n_components = self._validate_parameters(n_features, n_samples)

        scatter_estimator = (
            StudentTScatter(
                df=3.0,
                alpha=0.0,
                max_iter=300,
                tol=1e-7,
                warn_on_nonconvergence=False,
            )
            if self.scatter_estimator is None
            else deepcopy(self.scatter_estimator)
        )
        scatter_estimator.fit(X)
        if not hasattr(scatter_estimator, "covariance_"):
            raise AttributeError("scatter_estimator must expose covariance_ after fit")
        covariance = np.asarray(scatter_estimator.covariance_, dtype=np.float64)
        if covariance.shape != (n_features, n_features):
            raise ValueError("scatter_estimator covariance_ has incompatible shape")
        location = np.asarray(
            getattr(scatter_estimator, "location_", np.mean(X, axis=0)),
            dtype=np.float64,
        )
        if location.shape != (n_features,):
            raise ValueError("scatter_estimator location_ has incompatible shape")

        eigenvalues, eigenvectors = _symmetric_eigendecomposition(covariance)
        scale = max(float(eigenvalues[0]), np.finfo(np.float64).tiny)
        floor = max(float(self.eigenvalue_floor) * scale, np.finfo(np.float64).tiny)
        if eigenvalues[n_components - 1] <= floor:
            raise ValueError(
                "whitening scatter is rank deficient for the requested n_components"
            )
        retained_values = np.maximum(eigenvalues[:n_components], floor)
        retained_vectors = eigenvectors[:, :n_components]
        whitening = retained_vectors.T / np.sqrt(retained_values)[:, None]
        dewhitening = retained_vectors * np.sqrt(retained_values)[None, :]
        centered = X - location
        whitened = centered @ whitening.T

        if self.second_scatter_estimator is None:
            second_data = (
                _sample_pair_differences(
                    whitened, int(self.max_pairs), self.random_state
                )
                if self.symmetrize
                else whitened
            )
            radial_squared = np.sum(second_data * second_data, axis=1)
            positive = radial_squared[radial_squared > np.finfo(np.float64).tiny]
            if positive.size == 0:
                raise ValueError("the second scatter is undefined for coincident samples")
            clip = float(np.quantile(positive, float(self.radial_clip_quantile)))
            # Bound every observation's matrix contribution.  Writing
            # z = r u, the update is min(r^2, clip) u u.T rather than the
            # unbounded r^2 * z z.T form.
            contribution_scale = np.minimum(
                1.0, clip / np.maximum(radial_squared, np.finfo(np.float64).tiny)
            )
            second_scatter = (
                second_data.T @ (contribution_scale[:, None] * second_data)
            ) / second_data.shape[0]
            second_estimator = None
        else:
            second_estimator = deepcopy(self.second_scatter_estimator)
            second_estimator.fit(whitened)
            if not hasattr(second_estimator, "covariance_"):
                raise AttributeError(
                    "second_scatter_estimator must expose covariance_ after fit"
                )
            second_scatter = np.asarray(
                second_estimator.covariance_, dtype=np.float64
            )
        second_scatter = 0.5 * (second_scatter + second_scatter.T)
        second_eigenvalues, rotation = _symmetric_eigendecomposition(second_scatter)

        unmixing = rotation.T @ whitening
        mixing = np.linalg.pinv(unmixing)
        unmixing, mixing, permutation, signs = canonicalize_unmixing(
            unmixing,
            mixing=mixing,
            order=np.arange(n_components),
        )
        sources = centered @ unmixing.T

        self.scatter_estimator_ = scatter_estimator
        self.second_scatter_estimator_ = second_estimator
        self.location_ = location
        self.mean_ = location
        self.covariance_ = 0.5 * (covariance + covariance.T)
        self.whitening_ = whitening
        self.dewhitening_ = dewhitening
        self.second_scatter_ = second_scatter
        self.component_eigenvalues_ = second_eigenvalues
        self.unmixing_ = unmixing
        self.components_ = unmixing
        self.mixing_ = mixing
        self.sources_ = sources
        self.permutation_ = permutation
        self.signs_ = signs
        self.n_components_ = n_components
        self.n_features_in_ = n_features
        self.n_samples_in_ = n_samples
        self.eigenvalue_floor_ = floor
        return self

    def _check_fitted(self):
        if not hasattr(self, "unmixing_"):
            raise AttributeError("TwoScatterICA is not fitted yet")

    def transform(self, X):
        self._check_fitted()
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has an incompatible number of features")
        return (X - self.location_) @ self.unmixing_.T

    def fit_transform(self, X, y=None):
        return self.fit(X, y).sources_

    def inverse_transform(self, sources):
        self._check_fitted()
        sources = np.asarray(sources, dtype=np.float64)
        if sources.ndim != 2 or sources.shape[1] != self.n_components_:
            raise ValueError("sources has incompatible shape")
        if not np.all(np.isfinite(sources)):
            raise ValueError("sources must contain only finite values")
        return sources @ self.mixing_.T + self.location_
