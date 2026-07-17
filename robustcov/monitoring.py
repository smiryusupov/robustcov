# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Robust rolling monitoring of multivariate location, scatter, and subspaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .geometry import affine_invariant_distance, det_normalize
from .pca import RobustPCA


_MONITOR_METRICS = (
    "location_shift",
    "scale_shift",
    "shape_shift",
    "max_subspace_angle",
    "score_distance_shift",
    "orthogonal_distance_shift",
    "combined_outlier_fraction",
)


def _as_2d_finite_array(X: np.ndarray, *, name: str = "X") -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if X.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one sample")
    if X.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one feature")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} must contain only finite values")
    return X


def _robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 100.0 * np.finfo(float).eps:
        q25, q75 = np.quantile(values, [0.25, 0.75])
        scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale <= 100.0 * np.finfo(float).eps:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 100.0 * np.finfo(float).eps:
        scale = max(abs(center), 1.0) * 1e-12
    return center, scale


def _principal_angles_degrees(
    components_a: np.ndarray,
    components_b: np.ndarray,
) -> np.ndarray:
    cross = np.asarray(components_a, dtype=float) @ np.asarray(
        components_b, dtype=float
    ).T
    singular_values = np.linalg.svd(cross, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    return np.degrees(np.arccos(singular_values))


@dataclass
class SubspaceDriftResult:
    """Structured result returned by :meth:`RobustSubspaceMonitor.update`.

    Aggregate drift metrics describe the complete rolling window.  The distance
    arrays and ``batch_*`` fractions describe only the newly supplied batch.
    This separation makes the monitor useful for both alerting and record-level
    inspection.
    """

    ready: bool
    n_batch_samples: int
    n_window_samples: int
    location_shift: float = float("nan")
    scale_shift: float = float("nan")
    shape_shift: float = float("nan")
    max_subspace_angle: float = float("nan")
    mean_subspace_angle: float = float("nan")
    score_distance_shift: float = float("nan")
    orthogonal_distance_shift: float = float("nan")
    score_outlier_fraction: float = float("nan")
    orthogonal_outlier_fraction: float = float("nan")
    combined_outlier_fraction: float = float("nan")
    batch_score_outlier_fraction: float = float("nan")
    batch_orthogonal_outlier_fraction: float = float("nan")
    batch_combined_outlier_fraction: float = float("nan")
    raw_alarm: bool = False
    alarm: bool = False
    consecutive_alarms: int = 0
    thresholds: dict[str, float] = field(default_factory=dict)
    exceeded: dict[str, bool] = field(default_factory=dict)
    principal_angles: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    score_distances: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    orthogonal_distances: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    sample_outlier_mask: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=bool)
    )

    @property
    def metrics(self) -> dict[str, float]:
        """Return the aggregate rolling-window metrics."""
        return {
            "location_shift": float(self.location_shift),
            "scale_shift": float(self.scale_shift),
            "shape_shift": float(self.shape_shift),
            "max_subspace_angle": float(self.max_subspace_angle),
            "score_distance_shift": float(self.score_distance_shift),
            "orthogonal_distance_shift": float(self.orthogonal_distance_shift),
            "combined_outlier_fraction": float(self.combined_outlier_fraction),
        }

    def as_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        """Return a JSON-friendly dictionary representation."""
        result: dict[str, Any] = {
            "ready": bool(self.ready),
            "n_batch_samples": int(self.n_batch_samples),
            "n_window_samples": int(self.n_window_samples),
            **self.metrics,
            "mean_subspace_angle": float(self.mean_subspace_angle),
            "score_outlier_fraction": float(self.score_outlier_fraction),
            "orthogonal_outlier_fraction": float(
                self.orthogonal_outlier_fraction
            ),
            "batch_score_outlier_fraction": float(
                self.batch_score_outlier_fraction
            ),
            "batch_orthogonal_outlier_fraction": float(
                self.batch_orthogonal_outlier_fraction
            ),
            "batch_combined_outlier_fraction": float(
                self.batch_combined_outlier_fraction
            ),
            "raw_alarm": bool(self.raw_alarm),
            "alarm": bool(self.alarm),
            "consecutive_alarms": int(self.consecutive_alarms),
            "thresholds": dict(self.thresholds),
            "exceeded": dict(self.exceeded),
        }
        if include_arrays:
            result.update(
                {
                    "principal_angles": self.principal_angles.tolist(),
                    "score_distances": self.score_distances.tolist(),
                    "orthogonal_distances": self.orthogonal_distances.tolist(),
                    "sample_outlier_mask": self.sample_outlier_mask.tolist(),
                }
            )
        return result

    def summary(self) -> str:
        """Return a compact human-readable monitoring summary."""
        if not self.ready:
            return (
                "Robust subspace monitor warming up: "
                f"{self.n_window_samples} observations in the rolling window."
            )

        triggered = [name for name, value in self.exceeded.items() if value]
        status = "ALARM" if self.alarm else ("warning" if self.raw_alarm else "stable")
        details = ", ".join(triggered) if triggered else "no calibrated metric exceeded"
        return (
            f"Robust subspace monitor: {status}; {details}. "
            f"max_angle={self.max_subspace_angle:.3f} deg, "
            f"location_shift={self.location_shift:.3f}, "
            f"orthogonal_shift={self.orthogonal_distance_shift:.3f}, "
            f"outlier_fraction={self.combined_outlier_fraction:.3f}."
        )


@dataclass
class RobustSubspaceMonitor:
    """Monitor rolling multivariate drift relative to a robust reference model.

    The monitor fits one frozen :class:`~robustcov.RobustPCA` reference model and
    compares each full rolling window with that reference.  Incoming samples are
    always scored against the frozen reference before the current-window model
    is fitted, which prevents observed drift from silently redefining normality.

    The resulting diagnostics deliberately remain decomposed: location, global
    scale, covariance shape, principal-subspace rotation, score-distance shift,
    orthogonal-distance shift, and the fraction of reference-distance outliers.
    A calibrated alarm can be delayed until several consecutive windows exceed
    at least one selected metric.

    Parameters
    ----------
    n_components : int, float, or None, default=0.95
        Robust principal-subspace dimension.  Float values use the reference
        explained-variance threshold; current windows retain that resolved
        integer dimension.
    estimator : object, optional
        Robust scatter estimator accepted by :class:`~robustcov.RobustPCA`.
        The estimator is copied before every fit.  If omitted, RobustPCA's
        default ``RegularizedCauchy(alpha=0.10)`` is used.
    window_size : int, default=256
        Number of recent observations used by the current-window model.
    calibration_windows : int, default=16
        Number of reference windows used to calibrate aggregate thresholds.
        Contiguous windows are sampled from the reference period.
    threshold_quantile : float, default=0.99
        Quantile of calibration-window metrics used as the alarm threshold.
    sample_quantile : float, default=0.99
        Reference quantile used for individual score- and orthogonal-distance
        outlier flags.
    threshold_scale : float, default=1.0
        Multiplicative safety margin applied to aggregate calibration thresholds.
        Values above one reduce false alarms when the calibration period is short.
    alarm_patience : int, default=1
        Number of consecutive raw alarms required before ``alarm=True``.
    alarm_metrics : iterable of str, optional
        Metrics allowed to trigger alarms.  Defaults to all calibrated metrics.
    ridge : float, default=1e-10
        Relative eigenvalue floor passed to RobustPCA.
    random_state : int or None, default=0
        Seed controlling sampled calibration windows.
    history_size : int, default=100
        Maximum number of update results retained in ``history_``.  Set to zero
        to disable history storage.

    Notes
    -----
    Calibration assumes the reference period is representative of acceptable
    operation.  The thresholds are empirical diagnostics, not universal
    hypothesis tests.  For autocorrelated time series, choose ``window_size`` to
    reflect the operational horizon and inspect false alarms on a held-out
    stable period.
    """

    n_components: int | float | None = 0.95
    estimator: Any | None = None
    window_size: int = 256
    calibration_windows: int = 16
    threshold_quantile: float = 0.99
    sample_quantile: float = 0.99
    threshold_scale: float = 1.0
    alarm_patience: int = 1
    alarm_metrics: Iterable[str] | None = None
    ridge: float = 1e-10
    random_state: int | None = 0
    history_size: int = 100

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> "RobustSubspaceMonitor":
        """Fit the frozen robust reference model and calibrate thresholds."""
        del y
        X = _as_2d_finite_array(X)
        self._validate_parameters(X.shape[0])

        reference_model = RobustPCA(
            n_components=self.n_components,
            estimator=self.estimator,
            ridge=self.ridge,
            store_scores=False,
        ).fit(X)

        reference_score_distances = reference_model.score_distances(X)
        reference_orthogonal_distances = reference_model.orthogonal_distances(X)
        score_center, score_scale = _robust_center_scale(reference_score_distances)
        orthogonal_center, orthogonal_scale = _robust_center_scale(
            reference_orthogonal_distances
        )

        score_threshold = float(
            np.quantile(reference_score_distances, self.sample_quantile)
        )
        orthogonal_threshold = float(
            np.quantile(reference_orthogonal_distances, self.sample_quantile)
        )

        self.reference_model_ = reference_model
        self.reference_location_ = reference_model.location_.copy()
        self.reference_covariance_ = reference_model.covariance_.copy()
        self.reference_components_ = reference_model.components_.copy()
        self.reference_score_distances_ = reference_score_distances
        self.reference_orthogonal_distances_ = reference_orthogonal_distances
        self.score_distance_center_ = score_center
        self.score_distance_scale_ = score_scale
        self.orthogonal_distance_center_ = orthogonal_center
        self.orthogonal_distance_scale_ = orthogonal_scale
        self.score_distance_threshold_ = score_threshold
        self.orthogonal_distance_threshold_ = orthogonal_threshold
        self.n_features_in_ = X.shape[1]
        self.n_reference_samples_ = X.shape[0]
        self.n_components_ = reference_model.n_components_
        self.subspace_rotation_available_ = self.n_components_ < self.n_features_in_
        self.orthogonal_distance_available_ = self.n_components_ < self.n_features_in_
        if not self.orthogonal_distance_available_:
            self.orthogonal_distance_threshold_ = float("inf")
        self.alarm_metrics_ = self._resolve_alarm_metrics()

        calibration = {name: [] for name in _MONITOR_METRICS}
        rng = np.random.default_rng(self.random_state)
        max_start = X.shape[0] - self.window_size
        starts = rng.integers(
            0,
            max_start + 1,
            size=self.calibration_windows,
            endpoint=False,
        )

        failures: list[str] = []
        for start in starts:
            window = X[int(start) : int(start) + self.window_size]
            try:
                current_model = self._fit_current_model(window)
                metrics, _ = self._window_metrics(window, current_model)
            except Exception as exc:  # pragma: no cover - estimator-specific failures
                failures.append(f"{type(exc).__name__}: {exc}")
                continue
            for name in _MONITOR_METRICS:
                value = metrics[name]
                if np.isfinite(value):
                    calibration[name].append(float(value))

        minimum_successes = max(3, self.calibration_windows // 2)
        successes = max((len(values) for values in calibration.values()), default=0)
        if successes < minimum_successes:
            detail = failures[-1] if failures else "no finite calibration metrics"
            raise RuntimeError(
                "too few calibration windows could be fitted; "
                f"got {successes}, need at least {minimum_successes}. Last failure: {detail}"
            )

        thresholds: dict[str, float] = {}
        calibration_arrays: dict[str, np.ndarray] = {}
        for name, values in calibration.items():
            array = np.asarray(values, dtype=float)
            calibration_arrays[name] = array
            if array.size == 0:
                thresholds[name] = float("inf")
            else:
                threshold = float(np.quantile(array, self.threshold_quantile))
                thresholds[name] = max(self.threshold_scale * threshold, 0.0)

        if not self.subspace_rotation_available_:
            thresholds["max_subspace_angle"] = float("inf")
        if not self.orthogonal_distance_available_:
            thresholds["orthogonal_distance_shift"] = float("inf")

        self.thresholds_ = thresholds
        self.calibration_metrics_ = calibration_arrays
        self.reset()
        return self

    def _validate_parameters(self, n_reference_samples: int) -> None:
        if (
            isinstance(self.window_size, (bool, np.bool_))
            or not isinstance(self.window_size, (int, np.integer))
            or self.window_size < 2
        ):
            raise ValueError("window_size must be an integer of at least 2")
        if self.window_size >= n_reference_samples:
            raise ValueError(
                "window_size must be smaller than the number of reference samples"
            )
        if (
            isinstance(self.calibration_windows, (bool, np.bool_))
            or not isinstance(self.calibration_windows, (int, np.integer))
            or self.calibration_windows < 3
        ):
            raise ValueError("calibration_windows must be an integer of at least 3")
        if not np.isfinite(self.threshold_quantile) or not (
            0.5 <= self.threshold_quantile <= 1.0
        ):
            raise ValueError("threshold_quantile must be in [0.5, 1]")
        if not np.isfinite(self.sample_quantile) or not (
            0.5 <= self.sample_quantile < 1.0
        ):
            raise ValueError("sample_quantile must be in [0.5, 1)")
        if not np.isscalar(self.threshold_scale) or not np.isfinite(
            self.threshold_scale
        ) or self.threshold_scale <= 0:
            raise ValueError("threshold_scale must be a positive finite number")
        if (
            isinstance(self.alarm_patience, (bool, np.bool_))
            or not isinstance(self.alarm_patience, (int, np.integer))
            or self.alarm_patience < 1
        ):
            raise ValueError("alarm_patience must be an integer of at least 1")
        if not np.isscalar(self.ridge) or not np.isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError("ridge must be a positive finite number")
        if (
            isinstance(self.history_size, (bool, np.bool_))
            or not isinstance(self.history_size, (int, np.integer))
            or self.history_size < 0
        ):
            raise ValueError("history_size must be a non-negative integer")

    def _resolve_alarm_metrics(self) -> tuple[str, ...]:
        if self.alarm_metrics is None:
            return _MONITOR_METRICS
        if isinstance(self.alarm_metrics, str):
            raise TypeError("alarm_metrics must be an iterable of metric names, not a string")
        metrics = tuple(self.alarm_metrics)
        unknown = sorted(set(metrics) - set(_MONITOR_METRICS))
        if unknown:
            raise ValueError(
                "unknown alarm_metrics: " + ", ".join(unknown)
            )
        if not metrics:
            raise ValueError("alarm_metrics cannot be empty")
        return metrics

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "reference_model_"):
            raise AttributeError("RobustSubspaceMonitor is not fitted yet")

    def _check_batch(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = _as_2d_finite_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )
        return X

    def _fit_current_model(self, X: np.ndarray) -> RobustPCA:
        return RobustPCA(
            n_components=self.n_components_,
            estimator=self.estimator,
            ridge=self.ridge,
            store_scores=False,
        ).fit(X)

    def _window_metrics(
        self,
        window: np.ndarray,
        current_model: RobustPCA,
    ) -> tuple[dict[str, float], np.ndarray]:
        reference = self.reference_model_
        delta = current_model.location_ - reference.location_
        location_squared = float(
            delta @ np.linalg.solve(reference.covariance_, delta)
        )
        location_shift = float(np.sqrt(max(location_squared, 0.0)))

        sign_ref, logdet_ref = np.linalg.slogdet(reference.covariance_)
        sign_cur, logdet_cur = np.linalg.slogdet(current_model.covariance_)
        if sign_ref <= 0 or sign_cur <= 0:
            raise ValueError("regularized covariance matrices must be positive definite")
        scale_shift = float(
            abs(logdet_cur - logdet_ref) / self.n_features_in_
        )

        shape_shift = float(
            affine_invariant_distance(
                det_normalize(reference.covariance_),
                det_normalize(current_model.covariance_),
            )
            / np.sqrt(self.n_features_in_)
        )

        if self.subspace_rotation_available_:
            angles = _principal_angles_degrees(
                reference.components_, current_model.components_
            )
            max_angle = float(np.max(angles))
            mean_angle = float(np.mean(angles))
        else:
            angles = np.zeros(self.n_components_, dtype=float)
            max_angle = 0.0
            mean_angle = 0.0

        score_distances = reference.score_distances(window)
        orthogonal_distances = reference.orthogonal_distances(window)
        score_shift = float(
            (np.median(score_distances) - self.score_distance_center_)
            / self.score_distance_scale_
        )
        if self.orthogonal_distance_available_:
            orthogonal_shift = float(
                (
                    np.median(orthogonal_distances)
                    - self.orthogonal_distance_center_
                )
                / self.orthogonal_distance_scale_
            )
        else:
            orthogonal_shift = float("nan")

        score_outliers = score_distances > self.score_distance_threshold_
        if self.orthogonal_distance_available_:
            orthogonal_outliers = (
                orthogonal_distances > self.orthogonal_distance_threshold_
            )
        else:
            orthogonal_outliers = np.zeros(window.shape[0], dtype=bool)
        combined_outliers = score_outliers | orthogonal_outliers

        metrics = {
            "location_shift": location_shift,
            "scale_shift": scale_shift,
            "shape_shift": shape_shift,
            "max_subspace_angle": max_angle,
            "score_distance_shift": score_shift,
            "orthogonal_distance_shift": orthogonal_shift,
            "score_outlier_fraction": float(np.mean(score_outliers)),
            "orthogonal_outlier_fraction": float(np.mean(orthogonal_outliers)),
            "combined_outlier_fraction": float(np.mean(combined_outliers)),
            "mean_subspace_angle": mean_angle,
        }
        return metrics, angles

    def _candidate_window(self, X: np.ndarray) -> np.ndarray:
        if self.window_.size == 0:
            combined = X
        else:
            combined = np.vstack([self.window_, X])
        if combined.shape[0] > self.window_size:
            combined = combined[-self.window_size :]
        return combined

    def _evaluate(self, X: np.ndarray, *, commit: bool) -> SubspaceDriftResult:
        X = self._check_batch(X)
        candidate = self._candidate_window(X)

        batch_scores = self.reference_model_.score_distances(X)
        batch_orthogonal = self.reference_model_.orthogonal_distances(X)
        batch_score_outliers = batch_scores > self.score_distance_threshold_
        if self.orthogonal_distance_available_:
            batch_orthogonal_outliers = (
                batch_orthogonal > self.orthogonal_distance_threshold_
            )
        else:
            batch_orthogonal_outliers = np.zeros(X.shape[0], dtype=bool)
        batch_combined = batch_score_outliers | batch_orthogonal_outliers

        if candidate.shape[0] < self.window_size:
            result = SubspaceDriftResult(
                ready=False,
                n_batch_samples=X.shape[0],
                n_window_samples=candidate.shape[0],
                batch_score_outlier_fraction=float(np.mean(batch_score_outliers)),
                batch_orthogonal_outlier_fraction=float(
                    np.mean(batch_orthogonal_outliers)
                ),
                batch_combined_outlier_fraction=float(np.mean(batch_combined)),
                thresholds=dict(self.thresholds_),
                exceeded={name: False for name in self.alarm_metrics_},
                score_distances=batch_scores,
                orthogonal_distances=batch_orthogonal,
                sample_outlier_mask=batch_combined,
            )
            if commit:
                self.window_ = candidate
                self.current_model_ = None
                self.last_result_ = result
                self._record_history(result)
            return result

        current_model = self._fit_current_model(candidate)
        metrics, angles = self._window_metrics(candidate, current_model)
        exceeded = {
            name: bool(
                np.isfinite(metrics[name])
                and metrics[name] > self.thresholds_[name]
            )
            for name in self.alarm_metrics_
        }
        raw_alarm = any(exceeded.values())
        prospective_consecutive = self.consecutive_alarms_ + 1 if raw_alarm else 0
        alarm = prospective_consecutive >= self.alarm_patience

        result = SubspaceDriftResult(
            ready=True,
            n_batch_samples=X.shape[0],
            n_window_samples=candidate.shape[0],
            location_shift=metrics["location_shift"],
            scale_shift=metrics["scale_shift"],
            shape_shift=metrics["shape_shift"],
            max_subspace_angle=metrics["max_subspace_angle"],
            mean_subspace_angle=metrics["mean_subspace_angle"],
            score_distance_shift=metrics["score_distance_shift"],
            orthogonal_distance_shift=metrics["orthogonal_distance_shift"],
            score_outlier_fraction=metrics["score_outlier_fraction"],
            orthogonal_outlier_fraction=metrics[
                "orthogonal_outlier_fraction"
            ],
            combined_outlier_fraction=metrics["combined_outlier_fraction"],
            batch_score_outlier_fraction=float(np.mean(batch_score_outliers)),
            batch_orthogonal_outlier_fraction=float(
                np.mean(batch_orthogonal_outliers)
            ),
            batch_combined_outlier_fraction=float(np.mean(batch_combined)),
            raw_alarm=raw_alarm,
            alarm=alarm,
            consecutive_alarms=prospective_consecutive,
            thresholds=dict(self.thresholds_),
            exceeded=exceeded,
            principal_angles=angles,
            score_distances=batch_scores,
            orthogonal_distances=batch_orthogonal,
            sample_outlier_mask=batch_combined,
        )

        if commit:
            self.window_ = candidate
            self.current_model_ = current_model
            self.consecutive_alarms_ = prospective_consecutive
            self.last_result_ = result
            self._record_history(result)
        return result

    def evaluate(self, X: np.ndarray) -> SubspaceDriftResult:
        """Preview the result of adding a batch without modifying monitor state."""
        return self._evaluate(X, commit=False)

    def update(self, X: np.ndarray) -> SubspaceDriftResult:
        """Score a batch, update the rolling window, and return drift diagnostics."""
        return self._evaluate(X, commit=True)

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> "RobustSubspaceMonitor":
        """Update the monitor and return ``self`` for incremental API compatibility."""
        del y
        self.update(X)
        return self

    def reset(self) -> "RobustSubspaceMonitor":
        """Clear rolling state while preserving the fitted frozen reference."""
        self._check_is_fitted()
        self.window_ = np.empty((0, self.n_features_in_), dtype=float)
        self.current_model_ = None
        self.last_result_ = None
        self.history_: list[SubspaceDriftResult] = []
        self.consecutive_alarms_ = 0
        return self

    def _record_history(self, result: SubspaceDriftResult) -> None:
        if self.history_size == 0:
            return
        self.history_.append(result)
        if len(self.history_) > self.history_size:
            del self.history_[: len(self.history_) - self.history_size]

    def history_records(self, *, include_arrays: bool = False) -> list[dict[str, Any]]:
        """Return retained update results as dictionaries."""
        self._check_is_fitted()
        return [
            result.as_dict(include_arrays=include_arrays)
            for result in self.history_
        ]
