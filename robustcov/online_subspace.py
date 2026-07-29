# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Experimental online robust subspace tracking.

The implementation is a practical robustcov composition: it combines a robust
batch subspace estimate, projected-residual cell cleaning, row-outlier rejection,
a bounded recent-sample buffer, and smoothed mini-batch subspace updates.  It is
inspired by robust subspace-tracking and online outlier-robust PCA research, but
it is not an implementation of NORST and does not inherit NORST's guarantees.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._estimator import EstimatorMixin
from .pca import RobustPCA


def _as_2d_finite_array(
    X: Any,
    *,
    name: str = "X",
    min_samples: int = 1,
) -> np.ndarray:
    values = np.asarray(X, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if values.shape[0] < min_samples:
        raise ValueError(
            f"{name} must contain at least {min_samples} sample"
            f"{'s' if min_samples != 1 else ''}"
        )
    if values.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two features")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _robust_scale(values: np.ndarray, *, floor: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    center = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - center), axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    iqr_scale = (q75 - q25) / 1.349
    std_scale = np.std(values, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > floor), scale, iqr_scale)
    scale = np.where(np.isfinite(scale) & (scale > floor), scale, std_scale)
    return np.maximum(np.where(np.isfinite(scale), scale, floor), floor)


def _orient_components(components: np.ndarray) -> np.ndarray:
    components = np.asarray(components, dtype=float).copy()
    if components.size == 0:
        return components
    columns = np.arange(components.shape[0])
    rows = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[columns, rows])
    signs[signs == 0.0] = 1.0
    components *= signs[:, None]
    return components


def _principal_angles_degrees(
    components_a: np.ndarray,
    components_b: np.ndarray,
) -> np.ndarray:
    singular_values = np.linalg.svd(
        np.asarray(components_a, dtype=float)
        @ np.asarray(components_b, dtype=float).T,
        compute_uv=False,
    )
    singular_values = np.clip(singular_values, 0.0, 1.0)
    return np.degrees(np.arccos(singular_values))


def _blend_subspaces(
    old_components: np.ndarray,
    new_components: np.ndarray,
    adaptation_rate: float,
) -> np.ndarray:
    if adaptation_rate >= 1.0:
        return _orient_components(new_components)
    old_projector = old_components.T @ old_components
    new_projector = new_components.T @ new_components
    blended = (1.0 - adaptation_rate) * old_projector + adaptation_rate * new_projector
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (blended + blended.T))
    order = np.argsort(eigenvalues)[::-1][: old_components.shape[0]]
    return _orient_components(eigenvectors[:, order].T)


@dataclass
class OnlineSubspaceUpdate:
    """Diagnostics returned by :meth:`OnlineRobustSubspaceTracker.update`."""

    n_batch_samples: int
    n_accepted: int
    n_rejected: int
    n_cell_corrections: int
    update_attempted: bool
    update_performed: bool
    change_detected: bool
    candidate_max_angle: float
    subspace_version: int
    mean_anomaly_score: float
    median_anomaly_score: float
    anomaly_scores: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    sample_outlier_mask: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=bool)
    )
    cell_outlier_mask: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=bool)
    )

    def as_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        result: dict[str, Any] = {
            "n_batch_samples": int(self.n_batch_samples),
            "n_accepted": int(self.n_accepted),
            "n_rejected": int(self.n_rejected),
            "n_cell_corrections": int(self.n_cell_corrections),
            "update_attempted": bool(self.update_attempted),
            "update_performed": bool(self.update_performed),
            "change_detected": bool(self.change_detected),
            "candidate_max_angle": float(self.candidate_max_angle),
            "subspace_version": int(self.subspace_version),
            "mean_anomaly_score": float(self.mean_anomaly_score),
            "median_anomaly_score": float(self.median_anomaly_score),
        }
        if include_arrays:
            result.update(
                {
                    "anomaly_scores": self.anomaly_scores.tolist(),
                    "sample_outlier_mask": self.sample_outlier_mask.tolist(),
                    "cell_outlier_mask": self.cell_outlier_mask.tolist(),
                }
            )
        return result


@dataclass
class OnlineRobustSubspaceTracker(EstimatorMixin):
    """Track a slowly changing low-dimensional subspace in streaming data.

    The tracker fits an initial :class:`~robustcov.RobustPCA` model, scores each
    incoming observation against the current subspace, replaces isolated large
    projected residuals by their subspace reconstruction, rejects observations
    that look like dense row outliers, and periodically refits a robust candidate
    subspace on a bounded recent-sample buffer.  Candidate and current projectors
    are interpolated to avoid abrupt redefinition of normality.

    This is an experimental robustcov workflow inspired by projected-residual
    robust subspace tracking and online outlier-robust PCA.  It is **not** NORST:
    it does not solve an l1 projected-compressive-sensing problem and it does not
    carry NORST's support-recovery or tracking-delay guarantees.

    Parameters
    ----------
    n_components : int, default=2
        Number of tracked components.  Must be smaller than the feature count.
    estimator : object, optional
        Scatter estimator passed to every robust PCA fit.  It is copied before
        fitting.  The default is RobustPCA's regularized Cauchy estimator.
    update_interval : int, default=64
        Number of accepted observations between candidate subspace updates.
    buffer_size : int, default=256
        Maximum number of recent cleaned observations retained for updates.
    adaptation_rate : float, default=0.5
        Weight in ``(0, 1]`` assigned to each accepted candidate projector and
        candidate location.  Smaller values adapt more gradually.
    residual_quantile : float, default=0.99
        Initial empirical quantile for orthogonal-residual and score-distance
        screening thresholds.
    threshold_scale : float, default=1.5
        Multiplicative margin applied to fitted screening thresholds.
    cell_threshold : float, default=8.0
        Standardized projected-residual magnitude used to mark an isolated cell
        as corrupted before candidate fitting.
    max_cell_fraction : float, default=0.25
        Observations with more than this fraction of marked cells are rejected
        as dense row outliers rather than repaired.
    change_detection_angle : float, default=5.0
        Candidate/current largest principal angle, in degrees, reported as a
        detected subspace change.
    max_update_angle : float, default=45.0
        Candidate updates with a larger principal angle are rejected.  This is a
        slow-change safeguard, not a statistical test.
    ridge : float, default=1e-10
        Relative eigenvalue floor passed to RobustPCA.
    history_size : int, default=100
        Maximum number of update diagnostics retained.  Zero disables history.

    Notes
    -----
    The initial batch should represent acceptable operation and be large enough
    to identify the requested subspace.  The method assumes that genuine change
    is gradual enough to enter the recent-sample buffer before the slow-change
    safeguard is exceeded.  It is not appropriate when outliers are dense but
    indistinguishable from a new subspace, or when exact sparse-support recovery
    is required.
    """

    n_components: int = 2
    estimator: Any | None = None
    update_interval: int = 64
    buffer_size: int = 256
    adaptation_rate: float = 0.5
    residual_quantile: float = 0.99
    threshold_scale: float = 1.5
    cell_threshold: float = 8.0
    max_cell_fraction: float = 0.25
    change_detection_angle: float = 5.0
    max_update_angle: float = 45.0
    ridge: float = 1e-10
    history_size: int = 100

    def _validate_parameters(self) -> None:
        if isinstance(self.n_components, (bool, np.bool_)) or not isinstance(
            self.n_components, (int, np.integer)
        ):
            raise TypeError("n_components must be an integer")
        if self.n_components < 1:
            raise ValueError("n_components must be at least 1")
        for name in ("update_interval", "buffer_size"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(f"{name} must be an integer")
            if value < 2:
                raise ValueError(f"{name} must be at least 2")
        if self.buffer_size < self.update_interval:
            raise ValueError("buffer_size must be at least update_interval")
        for name, lower, upper, closed_upper in (
            ("adaptation_rate", 0.0, 1.0, True),
            ("residual_quantile", 0.5, 1.0, False),
            ("max_cell_fraction", 0.0, 1.0, False),
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
                raise TypeError(f"{name} must be a real number")
            value = float(value)
            valid_upper = value <= upper if closed_upper else value < upper
            if not np.isfinite(value) or not (value > lower and valid_upper):
                bracket = "]" if closed_upper else ")"
                raise ValueError(f"{name} must be in ({lower}, {upper}{bracket}")
        for name in (
            "threshold_scale",
            "cell_threshold",
            "change_detection_angle",
            "max_update_angle",
            "ridge",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
                raise TypeError(f"{name} must be a positive real number")
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{name} must be a positive finite number")
        if self.change_detection_angle > self.max_update_angle:
            raise ValueError(
                "change_detection_angle cannot exceed max_update_angle"
            )
        if isinstance(self.history_size, (bool, np.bool_)) or not isinstance(
            self.history_size, (int, np.integer)
        ):
            raise TypeError("history_size must be an integer")
        if self.history_size < 0:
            raise ValueError("history_size must be non-negative")

    def _new_model(self) -> RobustPCA:
        estimator = None if self.estimator is None else deepcopy(self.estimator)
        return RobustPCA(
            n_components=int(self.n_components),
            estimator=estimator,
            ridge=float(self.ridge),
            store_scores=True,
        )

    def fit(self, X: Any, y: Any = None) -> "OnlineRobustSubspaceTracker":
        """Fit the initial robust subspace and screening scales."""
        del y
        self._validate_parameters()
        X = _as_2d_finite_array(
            X,
            min_samples=max(int(self.n_components) + 2, 4),
        )
        if self.n_components >= X.shape[1]:
            raise ValueError("n_components must be smaller than n_features")

        model = self._new_model().fit(X)
        self.initial_model_ = model
        self.components_ = model.components_.copy()
        self.location_ = model.location_.copy()
        self.mean_ = self.location_
        self.explained_variance_ = model.explained_variance_.copy()
        self.n_components_ = int(model.n_components_)
        self.n_features_in_ = int(X.shape[1])
        self.n_samples_in_ = int(X.shape[0])

        residuals = self._residual_matrix(X)
        feature_floor = np.maximum(
            1e-8 * np.maximum(np.std(X, axis=0), 1.0),
            np.finfo(float).eps,
        )
        self.cell_residual_scale_ = _robust_scale(
            residuals,
            floor=feature_floor,
        )
        residual_scores = self._standardized_residual_scores_from(residuals)
        score_distances = self._score_distances_from(X)
        self.residual_threshold_ = self._scaled_quantile(residual_scores)
        self.score_threshold_ = self._scaled_quantile(score_distances)

        initial_inliers = (
            (residual_scores <= self.residual_threshold_)
            | (score_distances <= self.score_threshold_)
        )
        if np.sum(initial_inliers) < self.n_components_ + 2:
            initial_inliers = np.ones(X.shape[0], dtype=bool)
        self.buffer_ = X[initial_inliers][-int(self.buffer_size) :].copy()
        self.samples_since_update_ = 0
        self.n_seen_ = int(X.shape[0])
        self.n_accepted_ = int(np.sum(initial_inliers))
        self.n_rejected_ = int(X.shape[0] - np.sum(initial_inliers))
        self.n_cell_corrections_ = 0
        self.n_updates_ = 0
        self.subspace_version_ = 0
        self.history_: list[OnlineSubspaceUpdate] = []
        self.last_update_: OnlineSubspaceUpdate | None = None
        return self

    def _scaled_quantile(self, values: np.ndarray) -> float:
        threshold = float(np.quantile(values, float(self.residual_quantile)))
        magnitude = max(float(np.max(np.abs(values), initial=0.0)), 1.0)
        floor = 100.0 * np.finfo(float).eps * magnitude
        return max(float(self.threshold_scale) * threshold, floor)

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "components_"):
            raise AttributeError("OnlineRobustSubspaceTracker is not fitted yet")

    def _check_X(self, X: Any, *, name: str = "X") -> np.ndarray:
        self._check_is_fitted()
        values = _as_2d_finite_array(X, name=name)
        if values.shape[1] != self.n_features_in_:
            raise ValueError(
                f"{name} has {values.shape[1]} features, expected "
                f"{self.n_features_in_}"
            )
        return values

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        return (X - self.location_) @ self.components_.T

    def _reconstruct_from(self, X: np.ndarray) -> np.ndarray:
        return self._raw_scores(X) @ self.components_ + self.location_

    def _residual_matrix(self, X: np.ndarray) -> np.ndarray:
        return X - self._reconstruct_from(X)

    def _standardized_residual_scores_from(
        self,
        residuals: np.ndarray,
    ) -> np.ndarray:
        standardized = residuals / self.cell_residual_scale_
        return np.sqrt(np.mean(standardized * standardized, axis=1))

    def _score_distances_from(self, X: np.ndarray) -> np.ndarray:
        raw = self._raw_scores(X)
        eigenvalues = np.maximum(
            self.explained_variance_,
            np.finfo(float).tiny,
        )
        return np.sqrt(np.sum((raw * raw) / eigenvalues, axis=1))

    def transform(self, X: Any) -> np.ndarray:
        """Project observations onto the current tracked subspace."""
        return self._raw_scores(self._check_X(X))

    def inverse_transform(self, scores: Any) -> np.ndarray:
        """Map component scores back to the current feature space."""
        self._check_is_fitted()
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 2:
            raise ValueError("scores must be a 2D array")
        if scores.shape[1] != self.n_components_:
            raise ValueError(
                f"scores has {scores.shape[1]} components, expected "
                f"{self.n_components_}"
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError("scores must contain only finite values")
        return scores @ self.components_ + self.location_

    def reconstruct(self, X: Any) -> np.ndarray:
        """Reconstruct observations using the current tracked subspace."""
        values = self._check_X(X)
        return self._reconstruct_from(values)

    def residuals(self, X: Any) -> np.ndarray:
        """Return projected residual vectors under the current subspace."""
        values = self._check_X(X)
        return self._residual_matrix(values)

    def anomaly_scores(self, X: Any) -> np.ndarray:
        """Return dimensionless scores; values above one are screened as unusual."""
        values = self._check_X(X)
        residual_scores = self._standardized_residual_scores_from(
            self._residual_matrix(values)
        )
        score_distances = self._score_distances_from(values)
        return np.maximum(
            residual_scores / self.residual_threshold_,
            score_distances / self.score_threshold_,
        )

    def score_samples(self, X: Any) -> np.ndarray:
        """Return negative anomaly scores; larger values indicate normality."""
        return -self.anomaly_scores(X)

    def decision_function(self, X: Any) -> np.ndarray:
        """Return signed inlier margins; positive values indicate normality."""
        return 1.0 - self.anomaly_scores(X)

    def predict(self, X: Any) -> np.ndarray:
        """Return ``1`` for screened inliers and ``-1`` for row outliers."""
        return np.where(self.decision_function(X) >= 0.0, 1, -1)

    def _screen_and_clean(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        reconstruction = self._reconstruct_from(X)
        residuals = X - reconstruction
        standardized = np.abs(residuals) / self.cell_residual_scale_
        above_threshold = standardized > float(self.cell_threshold)
        max_cells = max(
            1,
            int(np.floor(float(self.max_cell_fraction) * self.n_features_in_)),
        )
        cell_mask = np.zeros_like(above_threshold, dtype=bool)
        for row in range(X.shape[0]):
            candidates = np.flatnonzero(above_threshold[row])
            if candidates.size == 0:
                continue
            if candidates.size > max_cells:
                order = np.argsort(standardized[row, candidates])[::-1]
                candidates = candidates[order[:max_cells]]
            cell_mask[row, candidates] = True

        cleaned = X.copy()
        cleaned[cell_mask] = reconstruction[cell_mask]
        cleaned_residual_scores = self._standardized_residual_scores_from(
            cleaned - reconstruction
        )
        score_distances = self._score_distances_from(cleaned)
        row_mask = (
            (cleaned_residual_scores > self.residual_threshold_)
            & (score_distances > self.score_threshold_)
        )
        anomaly_scores = np.maximum(
            cleaned_residual_scores / self.residual_threshold_,
            score_distances / self.score_threshold_,
        )
        return cleaned, cell_mask, row_mask, anomaly_scores

    def _append_buffer(self, accepted: np.ndarray) -> None:
        if accepted.size == 0:
            return
        self.buffer_ = np.vstack([self.buffer_, accepted])
        if self.buffer_.shape[0] > self.buffer_size:
            self.buffer_ = self.buffer_[-int(self.buffer_size) :]

    def _candidate_update(self) -> tuple[bool, float, bool]:
        if self.buffer_.shape[0] < max(self.n_components_ + 2, self.update_interval):
            return False, float("nan"), False

        candidate = self._new_model().fit(self.buffer_)
        angles = _principal_angles_degrees(
            self.components_,
            candidate.components_,
        )
        max_angle = float(np.max(angles))
        change_detected = bool(max_angle >= float(self.change_detection_angle))
        if max_angle > float(self.max_update_angle):
            return False, max_angle, change_detected

        rate = float(self.adaptation_rate)
        self.components_ = _blend_subspaces(
            self.components_,
            candidate.components_,
            rate,
        )
        self.location_ = (
            (1.0 - rate) * self.location_ + rate * candidate.location_
        )
        self.mean_ = self.location_
        self.explained_variance_ = (
            (1.0 - rate) * self.explained_variance_
            + rate * candidate.explained_variance_
        )

        residuals = self._residual_matrix(self.buffer_)
        feature_floor = np.maximum(
            1e-8 * np.maximum(np.std(self.buffer_, axis=0), 1.0),
            np.finfo(float).eps,
        )
        candidate_scale = _robust_scale(residuals, floor=feature_floor)
        self.cell_residual_scale_ = (
            (1.0 - rate) * self.cell_residual_scale_ + rate * candidate_scale
        )
        candidate_residual_threshold = self._scaled_quantile(
            self._standardized_residual_scores_from(residuals)
        )
        candidate_score_threshold = self._scaled_quantile(
            self._score_distances_from(self.buffer_)
        )
        self.residual_threshold_ = (
            (1.0 - rate) * self.residual_threshold_
            + rate * candidate_residual_threshold
        )
        self.score_threshold_ = (
            (1.0 - rate) * self.score_threshold_
            + rate * candidate_score_threshold
        )
        self.n_updates_ += 1
        self.subspace_version_ += 1
        return True, max_angle, change_detected

    def update(self, X: Any) -> OnlineSubspaceUpdate:
        """Screen a batch, update the recent buffer, and possibly adapt."""
        values = self._check_X(X)
        cleaned, cell_mask, row_mask, anomaly_scores = self._screen_and_clean(values)
        accepted = cleaned[~row_mask]
        self._append_buffer(accepted)

        n_accepted = int(accepted.shape[0])
        n_rejected = int(np.sum(row_mask))
        n_cell_corrections = int(np.sum(cell_mask & ~row_mask[:, None]))
        self.n_seen_ += int(values.shape[0])
        self.n_accepted_ += n_accepted
        self.n_rejected_ += n_rejected
        self.n_cell_corrections_ += n_cell_corrections
        self.samples_since_update_ += n_accepted

        update_attempted = self.samples_since_update_ >= self.update_interval
        update_performed = False
        candidate_angle = float("nan")
        change_detected = False
        if update_attempted:
            update_performed, candidate_angle, change_detected = (
                self._candidate_update()
            )
            self.samples_since_update_ = 0

        result = OnlineSubspaceUpdate(
            n_batch_samples=int(values.shape[0]),
            n_accepted=n_accepted,
            n_rejected=n_rejected,
            n_cell_corrections=n_cell_corrections,
            update_attempted=bool(update_attempted),
            update_performed=bool(update_performed),
            change_detected=bool(change_detected),
            candidate_max_angle=float(candidate_angle),
            subspace_version=int(self.subspace_version_),
            mean_anomaly_score=float(np.mean(anomaly_scores)),
            median_anomaly_score=float(np.median(anomaly_scores)),
            anomaly_scores=anomaly_scores,
            sample_outlier_mask=row_mask,
            cell_outlier_mask=cell_mask,
        )
        self.last_update_ = result
        self._record_history(result)
        return result

    def partial_fit(
        self,
        X: Any,
        y: Any = None,
    ) -> "OnlineRobustSubspaceTracker":
        """Update the tracker and return ``self`` for incremental pipelines."""
        del y
        self.update(X)
        return self

    def _record_history(self, result: OnlineSubspaceUpdate) -> None:
        if self.history_size == 0:
            return
        self.history_.append(result)
        if len(self.history_) > self.history_size:
            del self.history_[: len(self.history_) - self.history_size]

    def history_records(self, *, include_arrays: bool = False) -> list[dict[str, Any]]:
        """Return retained update diagnostics as dictionaries."""
        self._check_is_fitted()
        return [
            item.as_dict(include_arrays=include_arrays)
            for item in self.history_
        ]
