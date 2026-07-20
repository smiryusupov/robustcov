# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy

import numpy as np
from scipy.stats import chi2

from .covariance import FastMCD, RegularizedTyler
from ._estimator import EstimatorMixin
from .m_estimators import RegularizedCauchy
from ._utils import check_array


class RobustOutlierDetector(EstimatorMixin):
    """Small sklearn-style outlier detector using robust Mahalanobis distances."""

    def __init__(self, estimator=None, threshold="chi2", alpha=0.975, contamination=None):
        self.estimator = estimator
        self.threshold = threshold
        self.alpha = alpha
        self.contamination = contamination

    def fit(self, X, y=None):
        base_estimator = FastMCD(random_state=0) if self.estimator is None else self.estimator
        allow_nan = getattr(base_estimator, "missing_values", "raise") == "median"
        X = check_array(X, allow_nan=allow_nan)
        self.estimator_ = copy.deepcopy(base_estimator).fit(X)
        self.n_features_in_ = X.shape[1]
        d = self.estimator_.distances_
        if self.contamination is not None:
            contamination = float(self.contamination)
            if not (0.0 <= contamination < 0.5):
                raise ValueError("contamination must be in [0, 0.5)")
            self.threshold_ = float(np.quantile(d, 1.0 - contamination))
        elif self.threshold == "chi2":
            self.threshold_ = float(chi2.ppf(self.alpha, self.n_features_in_))
        elif self.threshold in {"empirical", "tail_calibrated"}:
            self.threshold_ = float(np.quantile(d, self.alpha))
        elif isinstance(self.threshold, (int, float)):
            self.threshold_ = float(self.threshold)
        else:
            raise ValueError("threshold must be 'chi2', 'empirical', 'tail_calibrated', or a number")
        self.distances_ = d
        self.labels_ = np.where(d <= self.threshold_, 1, -1)
        return self

    def score_samples(self, X):
        """Return negative robust squared distances; larger values are more normal."""
        return -np.asarray(self.estimator_.mahalanobis(X), dtype=float)

    def decision_function(self, X):
        """Return signed inlier margins; positive values indicate inliers."""
        d = np.asarray(self.estimator_.mahalanobis(X), dtype=float)
        return self.threshold_ - d

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0.0, 1, -1)

    def fit_predict(self, X, y=None):
        return self.fit(X, y=y).labels_


class AutoRobustAnomalyDetector(EstimatorMixin):
    """Simple ensemble detector built from robust covariance estimators.

    Each estimator contributes a robust squared-distance score. Scores are normalized
    by a fitted quantile and averaged. This is deliberately lightweight; it is meant
    for practical comparison/diagnostics rather than a full AutoML search.
    """

    def __init__(self, estimators=None, threshold="empirical", alpha=0.975, contamination=None, normalize_quantile=0.90):
        self.estimators = estimators
        self.threshold = threshold
        self.alpha = alpha
        self.contamination = contamination
        self.normalize_quantile = normalize_quantile

    def _default_estimators(self):
        contamination = self.contamination
        return [
            FastMCD(quality="fast", random_state=0, contamination=contamination),
            RegularizedTyler(alpha=0.05, scale_correction="radial_median"),
            RegularizedCauchy(alpha=0.05, scale_correction="radial_median"),
        ]

    def fit(self, X, y=None):
        X = check_array(X, allow_nan=False)
        self.n_features_in_ = X.shape[1]
        base_estimators = self._default_estimators() if self.estimators is None else self.estimators
        self.estimators_ = []
        scores = []
        self.normalizers_ = []
        for est in base_estimators:
            fitted = copy.deepcopy(est).fit(X)
            d = np.asarray(fitted.distances_, dtype=float)
            norm = float(np.quantile(d, self.normalize_quantile))
            if not np.isfinite(norm) or norm <= 0:
                norm = float(np.median(d[d > 0])) if np.any(d > 0) else 1.0
            self.estimators_.append(fitted)
            self.normalizers_.append(norm)
            scores.append(d / norm)
        self.scores_by_estimator_ = np.vstack(scores)
        self.score_ = np.mean(self.scores_by_estimator_, axis=0)
        if self.contamination is not None:
            q = 1.0 - float(self.contamination)
            self.threshold_ = float(np.quantile(self.score_, q))
        elif self.threshold == "empirical":
            self.threshold_ = float(np.quantile(self.score_, self.alpha))
        elif self.threshold == "chi2":
            # Ensemble scores are normalized, so this is only a rough fallback.
            self.threshold_ = float(np.quantile(self.score_, self.alpha))
        elif isinstance(self.threshold, (int, float)):
            self.threshold_ = float(self.threshold)
        else:
            raise ValueError("threshold must be 'empirical', 'chi2', or a number")
        self.labels_ = np.where(self.score_ <= self.threshold_, 1, -1)
        return self

    def decision_function(self, X):
        X = check_array(X, allow_nan=False)
        scores = []
        for est, norm in zip(self.estimators_, self.normalizers_):
            scores.append(np.asarray(est.mahalanobis(X), dtype=float) / norm)
        return np.mean(np.vstack(scores), axis=0)

    def predict(self, X):
        score = self.decision_function(X)
        return np.where(score <= self.threshold_, 1, -1)

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_
