# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Conformal calibration for anomaly and monitoring scores.

The calibrator is intentionally score-model agnostic.  Fit an anomaly detector,
subspace monitor, or any other scoring rule on data that are separate from the
calibration scores, then use :class:`ConformalAlertCalibrator` to turn new scores
into finite-sample conformal p-values.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._estimator import EstimatorMixin


_VALID_TAILS = {"upper", "lower"}


def _as_score_vector(scores: Any, *, name: str) -> np.ndarray:
    """Validate calibration scores and return a one-dimensional float array."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array of scores")
    if values.size < 1:
        raise ValueError(f"{name} must contain at least one score")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite scores")
    return values


def _as_query_scores(scores: Any) -> tuple[np.ndarray, bool]:
    """Validate scalar or one-dimensional query scores."""
    values = np.asarray(scores, dtype=float)
    scalar = values.ndim == 0
    if scalar:
        values = values.reshape(1)
    elif values.ndim != 1:
        raise ValueError("scores must be a scalar or one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must contain only finite values")
    return values, scalar


class ConformalAlertCalibrator(EstimatorMixin):
    r"""Convert arbitrary anomaly scores into conformal p-values and alerts.

    Parameters
    ----------
    alpha : float, default=0.05
        Target marginal false-alert level.  Alerts are produced when the
        conservative conformal p-value is less than or equal to ``alpha``.
    tail : {"upper", "lower"}, default="upper"
        Direction in which scores become more anomalous.  Use ``"upper"`` for
        distances, losses, residual risks, and most anomaly scores.  Use
        ``"lower"`` when small scores are more anomalous.

    Notes
    -----
    The fitted calibration scores and future inlier scores must be exchangeable
    for the usual finite-sample marginal guarantee.  The score-producing model
    must be fitted without using the calibration observations.  Time series,
    adaptive monitoring, covariate shift, or repeated decisions generally need
    a design that goes beyond ordinary split conformal calibration.

    P-values use conservative deterministic tie handling:

    .. math::

       p(x) = \frac{1 + \#\{i: s_i \ge s(x)\}}{n + 1}

    for an upper-tail score, with the inequality reversed for a lower-tail
    score.  Consequently, the smallest attainable p-value is ``1 / (n + 1)``.

    Calibration sets containing unusually large upper-tail scores tend to make
    the resulting p-values more conservative, but contamination does not remove
    the need to state and assess the calibration assumptions.
    """

    def __init__(self, alpha: float = 0.05, tail: str = "upper"):
        self.alpha = alpha
        self.tail = tail

    def _validate_parameters(self) -> tuple[float, str]:
        if isinstance(self.alpha, (bool, np.bool_)):
            raise TypeError("alpha must be a real number in (0, 1)")
        try:
            alpha = float(self.alpha)
        except (TypeError, ValueError) as exc:
            raise TypeError("alpha must be a real number in (0, 1)") from exc
        if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")

        if not isinstance(self.tail, str):
            raise TypeError("tail must be 'upper' or 'lower'")
        tail = self.tail.lower()
        if tail not in _VALID_TAILS:
            raise ValueError("tail must be 'upper' or 'lower'")
        return alpha, tail

    def fit(self, scores: Any, y: Any = None):
        """Fit the calibration distribution.

        Parameters
        ----------
        scores : array-like of shape (n_calibration,)
            Scores from a held-out reference/calibration set.  The score model
            should already be fitted on separate training data.
        y : ignored
            Included for sklearn-style pipeline compatibility.
        """
        del y
        alpha, tail = self._validate_parameters()
        values = _as_score_vector(scores, name="scores")

        self.alpha_ = alpha
        self.tail_ = tail
        self.calibration_scores_ = values.copy()
        self.sorted_calibration_scores_ = np.sort(values)
        self.n_calibration_ = int(values.size)
        self.min_p_value_ = 1.0 / (self.n_calibration_ + 1.0)
        self.resolution_limited_ = bool(alpha < self.min_p_value_)

        # At most k calibration scores may be at least/as small as a strictly
        # more-extreme query score while still satisfying p <= alpha.
        k = int(np.floor(alpha * (self.n_calibration_ + 1.0)))
        self.max_extreme_calibration_count_ = max(0, k - 1)
        if k < 1:
            self.threshold_ = (
                float("inf") if tail == "upper" else float("-inf")
            )
        elif tail == "upper":
            self.threshold_ = float(
                self.sorted_calibration_scores_[self.n_calibration_ - k]
            )
        else:
            self.threshold_ = float(self.sorted_calibration_scores_[k - 1])
        return self

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "sorted_calibration_scores_"):
            raise AttributeError("ConformalAlertCalibrator is not fitted yet")

    def p_values(self, scores: Any):
        """Return conservative conformal p-values for new scores.

        A scalar input returns a float.  A one-dimensional input returns an
        array with the same length.
        """
        self._check_is_fitted()
        values, scalar = _as_query_scores(scores)
        sorted_scores = self.sorted_calibration_scores_
        n = self.n_calibration_

        if self.tail_ == "upper":
            extreme_counts = n - np.searchsorted(
                sorted_scores, values, side="left"
            )
        else:
            extreme_counts = np.searchsorted(
                sorted_scores, values, side="right"
            )
        result = (1.0 + extreme_counts.astype(float)) / (n + 1.0)
        if scalar:
            return float(result[0])
        return result

    def predict_alerts(self, scores: Any):
        """Return boolean alert decisions for new scores."""
        p_values = self.p_values(scores)
        if np.isscalar(p_values):
            return bool(p_values <= self.alpha_)
        return np.asarray(p_values <= self.alpha_, dtype=bool)

    def predict(self, scores: Any):
        """Return sklearn-style labels: ``1`` for normal and ``-1`` for alert."""
        alerts = self.predict_alerts(scores)
        if np.isscalar(alerts):
            return -1 if alerts else 1
        return np.where(alerts, -1, 1)

    def decision_function(self, scores: Any):
        """Return signed p-value margins; positive values indicate normality."""
        p_values = self.p_values(scores)
        return p_values - self.alpha_

    def calibration_summary(self) -> dict[str, Any]:
        """Return JSON-friendly fitted calibration diagnostics."""
        self._check_is_fitted()
        return {
            "alpha": float(self.alpha_),
            "tail": self.tail_,
            "n_calibration": int(self.n_calibration_),
            "min_p_value": float(self.min_p_value_),
            "resolution_limited": bool(self.resolution_limited_),
            "threshold": float(self.threshold_),
            "tie_handling": "conservative",
        }
