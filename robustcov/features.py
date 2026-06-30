# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Robust geometry tools for learned feature representations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_2d_array(X: np.ndarray, *, name: str = "X") -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} must contain only finite values")
    return X


def _symmetrize(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def _spd_inverse_and_invsqrt(
    covariance: np.ndarray,
    *,
    ridge: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return precision and inverse square root with eigenvalue clipping."""
    covariance = _symmetrize(np.asarray(covariance, dtype=float))

    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix")

    evals, evecs = np.linalg.eigh(covariance)
    scale = max(float(np.max(evals)), 1.0)
    floor = ridge * scale
    evals = np.maximum(evals, floor)

    precision = (evecs * (1.0 / evals)) @ evecs.T
    invsqrt = (evecs * (1.0 / np.sqrt(evals))) @ evecs.T

    return _symmetrize(precision), _symmetrize(invsqrt)


def _default_estimator() -> Any:
    """Default robust scatter estimator for feature geometry."""
    from .m_estimators import RegularizedCauchy

    return RegularizedCauchy(alpha=0.10)


@dataclass
class FeatureGeometry:
    """Robust geometry layer for learned feature matrices.

    ``FeatureGeometry`` is a light wrapper around existing robust scatter
    estimators.  It accepts an array of learned features and exposes robust
    Mahalanobis scores, whitening, pairwise distances, and RBF-style kernels.

    The class does not train representation models.  It assumes that features
    have already been produced by a model, encoder, embedding system, or other
    feature extractor.

    Parameters
    ----------
    estimator : object, optional
        Fitted through ``estimator.fit(X)``.  The estimator should expose
        ``covariance_`` after fitting and may expose ``location_``.  If omitted,
        ``RegularizedCauchy(alpha=0.10)`` is used.
    ridge : float, default=1e-10
        Relative eigenvalue floor used when forming the precision and whitening
        matrices.
    """

    estimator: Any | None = None
    ridge: float = 1e-10

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "FeatureGeometry":
        """Fit robust feature geometry.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : ignored, optional
            Present for sklearn-style compatibility.
        """
        del y

        X = _as_2d_array(X)
        estimator = _default_estimator() if self.estimator is None else deepcopy(self.estimator)
        estimator.fit(X)

        if not hasattr(estimator, "covariance_"):
            raise AttributeError("estimator must expose covariance_ after fit")

        covariance = _symmetrize(np.asarray(estimator.covariance_, dtype=float))

        if hasattr(estimator, "location_"):
            location = np.asarray(estimator.location_, dtype=float)
        else:
            location = np.mean(X, axis=0)

        if location.shape != (X.shape[1],):
            raise ValueError("estimator location_ has incompatible shape")

        precision, invsqrt = _spd_inverse_and_invsqrt(covariance, ridge=self.ridge)

        self.estimator_ = estimator
        self.location_ = location
        self.covariance_ = covariance
        self.precision_ = precision
        self.whitening_ = invsqrt
        self.n_features_in_ = X.shape[1]

        return self

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "precision_"):
            raise AttributeError("FeatureGeometry is not fitted yet")

    def _check_features(self, X: np.ndarray, *, name: str = "X") -> np.ndarray:
        self._check_is_fitted()
        X = _as_2d_array(X, name=name)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"{name} has {X.shape[1]} features, expected {self.n_features_in_}"
            )
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Center and robustly whiten feature vectors."""
        X = self._check_features(X)
        return (X - self.location_) @ self.whitening_

    def squared_mahalanobis(self, X: np.ndarray) -> np.ndarray:
        """Return squared robust Mahalanobis distances to the fitted center."""
        X = self._check_features(X)
        Z = X - self.location_
        return np.einsum("ij,jk,ik->i", Z, self.precision_, Z)

    def mahalanobis_scores(self, X: np.ndarray) -> np.ndarray:
        """Return robust Mahalanobis distances to the fitted center."""
        d2 = self.squared_mahalanobis(X)
        return np.sqrt(np.maximum(d2, 0.0))

    def pairwise_squared_distances(
        self,
        X: np.ndarray,
        Y: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return pairwise squared distances in the fitted robust metric."""
        X = self._check_features(X, name="X")
        Y = X if Y is None else self._check_features(Y, name="Y")

        Xw = X @ self.whitening_
        Yw = Y @ self.whitening_

        x2 = np.sum(Xw * Xw, axis=1)[:, None]
        y2 = np.sum(Yw * Yw, axis=1)[None, :]
        D2 = x2 + y2 - 2.0 * Xw @ Yw.T
        return np.maximum(D2, 0.0)

    def rbf_kernel(
        self,
        X: np.ndarray,
        Y: np.ndarray | None = None,
        *,
        length_scale: float = 1.0,
    ) -> np.ndarray:
        """Return an RBF kernel induced by the fitted robust feature metric."""
        if length_scale <= 0:
            raise ValueError("length_scale must be positive")

        D2 = self.pairwise_squared_distances(X, Y)
        return np.exp(-0.5 * D2 / (length_scale * length_scale))


@dataclass
class ClassConditionalFeatureGeometry:
    """Robust class-conditional geometry for labeled feature matrices.

    This class fits one ``FeatureGeometry`` object per class.  It is useful for
    Lee-style class-conditional Mahalanobis workflows on learned features, while
    allowing empirical covariance to be replaced by robust scatter estimators.

    Parameters
    ----------
    estimator : object, optional
        Base estimator copied and fitted separately within each class.  If
        omitted, ``RegularizedCauchy(alpha=0.10)`` is used.
    ridge : float, default=1e-10
        Relative eigenvalue floor used when forming each class precision matrix.
    min_samples_per_class : int, default=2
        Minimum number of samples required for each class.
    """

    estimator: Any | None = None
    ridge: float = 1e-10
    min_samples_per_class: int = 2

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassConditionalFeatureGeometry":
        """Fit one robust feature geometry per class."""
        X = _as_2d_array(X)
        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y have incompatible lengths")
        if self.min_samples_per_class < 2:
            raise ValueError("min_samples_per_class must be at least 2")

        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError("at least two classes are required")

        geometries = []

        for cls in classes:
            mask = y == cls
            n_cls = int(mask.sum())

            if n_cls < self.min_samples_per_class:
                raise ValueError(
                    f"class {cls!r} has {n_cls} samples; "
                    f"expected at least {self.min_samples_per_class}"
                )

            geom = FeatureGeometry(
                estimator=self.estimator,
                ridge=self.ridge,
            ).fit(X[mask])
            geometries.append(geom)

        self.classes_ = classes
        self.geometries_ = geometries
        self.n_features_in_ = X.shape[1]

        return self

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "geometries_"):
            raise AttributeError("ClassConditionalFeatureGeometry is not fitted yet")

    def _check_features(self, X: np.ndarray, *, name: str = "X") -> np.ndarray:
        self._check_is_fitted()
        X = _as_2d_array(X, name=name)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"{name} has {X.shape[1]} features, expected {self.n_features_in_}"
            )
        return X

    def class_squared_mahalanobis(self, X: np.ndarray) -> np.ndarray:
        """Return squared Mahalanobis distances to every class geometry.

        Returns
        -------
        distances : ndarray of shape (n_samples, n_classes)
            Entry ``distances[i, j]`` is the squared robust distance from sample
            ``i`` to class ``j``.
        """
        X = self._check_features(X)
        cols = [geom.squared_mahalanobis(X) for geom in self.geometries_]
        return np.column_stack(cols)

    def class_mahalanobis_scores(self, X: np.ndarray) -> np.ndarray:
        """Return Mahalanobis distances to every class geometry."""
        D2 = self.class_squared_mahalanobis(X)
        return np.sqrt(np.maximum(D2, 0.0))

    def ood_scores(self, X: np.ndarray) -> np.ndarray:
        """Return distance-to-nearest-class scores for OOD-style diagnostics."""
        D2 = self.class_squared_mahalanobis(X)
        return np.sqrt(np.maximum(np.min(D2, axis=1), 0.0))

    def predict_nearest_class(self, X: np.ndarray) -> np.ndarray:
        """Predict the nearest class under robust class-conditional geometry."""
        D2 = self.class_squared_mahalanobis(X)
        idx = np.argmin(D2, axis=1)
        return self.classes_[idx]
