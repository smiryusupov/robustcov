# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Robust reference distributions for SHAP and LIME tabular explainers.

The adapters in this module do not reimplement SHAP or LIME. They fit one of
RobustCov's scatter estimators, build a contamination-resistant reference
sample, and pass the resulting background or covariance geometry to the
upstream explainer.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from scipy.stats import chi2

from ._estimator import EstimatorMixin
from ._utils import check_array
from .metrics import RobustInputMetric


def _default_explanation_estimator():
    from .m_estimators import RegularizedCauchy

    return RegularizedCauchy(
        alpha=0.10,
        scale_correction="radial_median",
        warn_on_nonconvergence=False,
    )


def _require_shap():
    try:
        import shap
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "SHAP is required for make_shap_explainer; install robustcov[explain]"
        ) from exc
    return shap


def _require_lime_tabular():
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "LIME is required for make_lime_tabular_explainer; "
            "install robustcov[explain]"
        ) from exc
    return LimeTabularExplainer


def _systematic_subset(order: np.ndarray, size: int) -> np.ndarray:
    """Select evenly spaced ranks without introducing another random sampler."""
    if size >= order.size:
        return order.copy()
    positions = np.floor(np.linspace(0, order.size, size, endpoint=False)).astype(int)
    return order[positions]


class RobustExplanationReference(EstimatorMixin):
    """Fit a robust tabular reference distribution for explanation methods.

    Parameters
    ----------
    estimator : object, optional
        Robust scatter estimator exposing ``fit`` and fitted ``covariance_`` or
        ``precision_``. The default is a regularized Cauchy scatter estimator,
        which also works when the number of features is close to or exceeds the
        number of observations.
    max_samples : int or None, default=100
        Maximum number of representative rows retained in ``background_``.
        ``None`` keeps every row accepted by the robust support rule.
    support_alpha : float, default=0.975
        Chi-square probability used to define a central robust support when the
        fitted estimator does not expose a boolean ``support_`` mask.
    ridge : float, default=1e-10
        Relative numerical ridge used when forming a positive-semidefinite
        precision matrix.
    copy_estimator : bool, default=True
        Deep-copy the estimator before fitting.

    Notes
    -----
    The fitted object exposes the robust ``location_``, ``covariance_``, and
    ``precision_`` together with a representative ``background_`` matrix. SHAP
    can use the background directly or the location/covariance pair. LIME uses
    the background for perturbation statistics and the precision for locality.
    """

    def __init__(
        self,
        estimator: Any | None = None,
        *,
        max_samples: int | None = 100,
        support_alpha: float = 0.975,
        ridge: float = 1e-10,
        copy_estimator: bool = True,
    ):
        self.estimator = estimator
        self.max_samples = max_samples
        self.support_alpha = support_alpha
        self.ridge = ridge
        self.copy_estimator = copy_estimator

    def fit(self, X, y=None):
        del y
        feature_names = list(X.columns) if hasattr(X, "columns") else None
        X = check_array(X, allow_nan=False)
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError("X must contain at least two rows")
        if self.max_samples is not None and int(self.max_samples) < 2:
            raise ValueError("max_samples must be at least 2 or None")
        support_alpha = float(self.support_alpha)
        if not np.isfinite(support_alpha) or not (0.5 < support_alpha < 1.0):
            raise ValueError("support_alpha must be between 0.5 and 1")
        ridge = float(self.ridge)
        if not np.isfinite(ridge) or ridge < 0.0:
            raise ValueError("ridge must be a finite non-negative number")

        estimator = (
            _default_explanation_estimator()
            if self.estimator is None
            else (deepcopy(self.estimator) if self.copy_estimator else self.estimator)
        )
        metric = RobustInputMetric(
            estimator=estimator,
            copy_estimator=False,
            ridge=ridge,
        ).fit(X)

        centered = X - metric.location_
        distances = np.einsum(
            "ij,jk,ik->i", centered, metric.precision_, centered
        )
        fitted_support = getattr(metric.estimator_, "support_", None)
        if fitted_support is not None:
            support = np.asarray(fitted_support, dtype=bool)
            if support.shape != (n_samples,):
                raise ValueError("fitted estimator support_ has incompatible shape")
        else:
            cutoff = float(chi2.ppf(support_alpha, n_features))
            support = distances <= cutoff

        minimum = min(n_samples, max(2, n_features + 1))
        if int(np.count_nonzero(support)) < minimum:
            nearest = np.argsort(distances, kind="stable")[:minimum]
            support = np.zeros(n_samples, dtype=bool)
            support[nearest] = True

        accepted = np.flatnonzero(support)
        accepted = accepted[np.argsort(distances[accepted], kind="stable")]
        background_size = accepted.size if self.max_samples is None else min(
            int(self.max_samples), accepted.size
        )
        selected = _systematic_subset(accepted, background_size)

        self.estimator_ = metric.estimator_
        self.metric_ = metric
        self.location_ = np.asarray(metric.location_, dtype=float)
        self.covariance_ = np.asarray(metric.covariance_, dtype=float)
        self.precision_ = np.asarray(metric.precision_, dtype=float)
        self.distances_ = np.asarray(distances, dtype=float)
        self.support_ = support
        self.background_indices_ = np.asarray(selected, dtype=int)
        self.background_ = np.asarray(X[selected], dtype=float, order="C")
        self.n_samples_in_ = n_samples
        self.n_features_in_ = n_features
        self.support_fraction_ = float(np.mean(support))
        if feature_names is not None:
            self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        return self

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "background_"):
            raise RuntimeError("RobustExplanationReference is not fitted")

    def mean_covariance(self) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of the robust location and covariance."""
        self._check_is_fitted()
        return self.location_.copy(), self.covariance_.copy()


def _coerce_reference(
    reference,
    *,
    estimator=None,
    max_samples: int | None = 100,
    support_alpha: float = 0.975,
    ridge: float = 1e-10,
) -> RobustExplanationReference:
    if isinstance(reference, RobustExplanationReference):
        reference._check_is_fitted()
        return reference
    return RobustExplanationReference(
        estimator=estimator,
        max_samples=max_samples,
        support_alpha=support_alpha,
        ridge=ridge,
    ).fit(reference)


def make_shap_explainer(
    model,
    reference,
    *,
    estimator=None,
    max_samples: int | None = 100,
    support_alpha: float = 0.975,
    ridge: float = 1e-10,
    correlation_dependent: bool = False,
    algorithm: str = "auto",
    **kwargs,
):
    """Create a SHAP explainer from a robust reference distribution.

    For general models, the default passes ``background_`` through SHAP's
    independent masker. For linear models, ``correlation_dependent=True`` uses
    SHAP's linear imputation masker with the robust location and covariance.
    """
    robust_reference = _coerce_reference(
        reference,
        estimator=estimator,
        max_samples=max_samples,
        support_alpha=support_alpha,
        ridge=ridge,
    )
    shap = _require_shap()
    feature_names = getattr(robust_reference, "feature_names_in_", None)
    if feature_names is not None and "feature_names" not in kwargs:
        kwargs["feature_names"] = list(feature_names)

    if correlation_dependent:
        if algorithm not in {"auto", "linear"}:
            raise ValueError(
                "correlation_dependent SHAP requires algorithm='auto' or 'linear'"
            )
        masker = shap.maskers.Impute(
            {
                "mean": robust_reference.location_,
                "cov": robust_reference.covariance_,
            },
            method="linear",
        )
        explainer = shap.LinearExplainer(model, masker, **kwargs)
        explainer.robust_reference_ = robust_reference
        return explainer

    masker = shap.maskers.Independent(
        robust_reference.background_,
        max_samples=robust_reference.background_.shape[0],
    )
    explainer = shap.Explainer(model, masker, algorithm=algorithm, **kwargs)
    explainer.robust_reference_ = robust_reference
    return explainer


class _ScaledMahalanobisDistance:
    def __init__(self, precision: np.ndarray):
        self.precision = np.asarray(precision, dtype=float)

    def __call__(self, left, right) -> float:
        difference = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
        squared = float(difference @ self.precision @ difference)
        return float(np.sqrt(max(squared, 0.0)))


class RobustLimeTabularExplainer:
    """Thin LIME wrapper using a robust background and Mahalanobis locality.

    This adapter targets dense continuous tabular features. It delegates all
    neighborhood generation and local surrogate fitting to LIME; RobustCov only
    supplies contamination-resistant perturbation statistics and a full-matrix
    distance metric.
    """

    def __init__(self, reference: RobustExplanationReference, **kwargs):
        reference._check_is_fitted()
        LimeTabularExplainer = _require_lime_tabular()
        categorical = kwargs.get("categorical_features")
        if categorical is not None and len(categorical):
            raise ValueError(
                "RobustLimeTabularExplainer currently supports continuous features only"
            )
        if kwargs.get("discretize_continuous", False):
            raise ValueError(
                "discretize_continuous must be False for the robust full-matrix metric"
            )
        kwargs["discretize_continuous"] = False
        if "random_state" not in kwargs:
            kwargs["random_state"] = 0
        if "feature_names" not in kwargs and hasattr(reference, "feature_names_in_"):
            kwargs["feature_names"] = list(reference.feature_names_in_)

        explainer = LimeTabularExplainer(reference.background_, **kwargs)
        scale = np.asarray(explainer.scaler.scale_, dtype=float)
        scaled_precision = (
            scale[:, None] * reference.precision_ * scale[None, :]
        )

        self.reference_ = reference
        self.background_ = reference.background_
        self.scaled_precision_ = scaled_precision
        self.distance_metric_ = _ScaledMahalanobisDistance(scaled_precision)
        self.explainer_ = explainer

    def explain_instance(self, data_row, predict_fn, *, distance_metric=None, **kwargs):
        """Delegate to LIME, using robust Mahalanobis locality by default."""
        metric = self.distance_metric_ if distance_metric is None else distance_metric
        return self.explainer_.explain_instance(
            data_row,
            predict_fn,
            distance_metric=metric,
            **kwargs,
        )

    def __getattr__(self, name):
        explainer = self.__dict__.get("explainer_")
        if explainer is None:
            raise AttributeError(name)
        return getattr(explainer, name)


def make_lime_tabular_explainer(
    reference,
    *,
    estimator=None,
    max_samples: int | None = 100,
    support_alpha: float = 0.975,
    ridge: float = 1e-10,
    **kwargs,
) -> RobustLimeTabularExplainer:
    """Create a LIME tabular explainer from a robust reference distribution."""
    robust_reference = _coerce_reference(
        reference,
        estimator=estimator,
        max_samples=max_samples,
        support_alpha=support_alpha,
        ridge=ridge,
    )
    return RobustLimeTabularExplainer(robust_reference, **kwargs)


__all__ = [
    "RobustExplanationReference",
    "RobustLimeTabularExplainer",
    "make_shap_explainer",
    "make_lime_tabular_explainer",
]
