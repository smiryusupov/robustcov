from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import chi2


@dataclass
class RobustDiagnosticReport:
    """Compact diagnostic report for a fitted robust covariance estimator."""

    estimator_name: str
    n_samples: int
    n_features: int
    alpha: float
    threshold: float
    detected_fraction: float
    radial_kurtosis: float
    condition_number: float
    max_distance: float
    median_distance: float
    qq_tail_deviation: float
    support_fraction: float | None
    effective_support_fraction: float | None
    recommendations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            'estimator_name': self.estimator_name,
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'alpha': self.alpha,
            'threshold': self.threshold,
            'detected_fraction': self.detected_fraction,
            'radial_kurtosis': self.radial_kurtosis,
            'condition_number': self.condition_number,
            'max_distance': self.max_distance,
            'median_distance': self.median_distance,
            'qq_tail_deviation': self.qq_tail_deviation,
            'support_fraction': self.support_fraction,
            'effective_support_fraction': self.effective_support_fraction,
            'recommendations': list(self.recommendations),
        }

    def summary(self) -> str:
        lines = [
            f'Robust diagnostic report: {self.estimator_name}',
            f'n_samples={self.n_samples}, n_features={self.n_features}, alpha={self.alpha:.3f}',
            f'detected_fraction={self.detected_fraction:.4f}, threshold={self.threshold:.4f}',
            f'radial_kurtosis={self.radial_kurtosis:.4f}, qq_tail_deviation={self.qq_tail_deviation:.4f}',
            f'condition_number={self.condition_number:.4g}, max_distance={self.max_distance:.4f}',
        ]
        if self.support_fraction is not None:
            lines.append(f'support_fraction={self.support_fraction:.4f}')
        if self.effective_support_fraction is not None:
            lines.append(f'effective_support_fraction={self.effective_support_fraction:.4f}')
        if self.recommendations:
            lines.append('recommendations:')
            lines.extend(f'  - {r}' for r in self.recommendations)
        else:
            lines.append('recommendations: no major warnings from simple diagnostics')
        return '\n'.join(lines)


def _radial_kurtosis_from_distances(d2: np.ndarray, p: int) -> float:
    if d2.size == 0 or p <= 0:
        return float('nan')
    return float(np.mean(np.asarray(d2, dtype=float) ** 2) / (p * (p + 2.0)))


def _qq_tail_deviation(d2: np.ndarray, p: int, tail_fraction: float = 0.10) -> float:
    d2 = np.sort(np.asarray(d2, dtype=float))
    n = d2.size
    if n < 5 or p <= 0:
        return float('nan')
    probs = (np.arange(1, n + 1) - 0.5) / n
    theo = chi2.ppf(probs, p)
    k = max(1, int(np.ceil(tail_fraction * n)))
    tail_obs = d2[-k:]
    tail_theo = theo[-k:]
    denom = np.maximum(tail_theo, 1e-12)
    return float(np.median(tail_obs / denom))


def diagnostic_report(estimator, X=None, alpha: float = 0.975) -> RobustDiagnosticReport:
    """Create a practical diagnostic report for a fitted robustcov estimator.

    The report is intentionally lightweight and heuristic. It helps users notice
    heavy tails, high rejection rates, ill-conditioning, and support-size issues.
    """
    p = int(getattr(estimator, 'n_features_in_', 0))
    if X is None:
        d2 = np.asarray(getattr(estimator, 'distances_', []), dtype=float)
    else:
        d2 = np.asarray(estimator.mahalanobis(X), dtype=float)
        if p == 0:
            p = np.asarray(X).shape[1]
    n = int(d2.size)
    threshold = float(chi2.ppf(alpha, p)) if p > 0 else float('nan')
    detected_fraction = float(np.mean(d2 > threshold)) if n else float('nan')
    radial_kurt = float(getattr(estimator, 'radial_kurtosis_', _radial_kurtosis_from_distances(d2, p)))
    qq_tail = _qq_tail_deviation(d2, p)
    cov = np.asarray(getattr(estimator, 'covariance_', np.eye(max(p, 1))), dtype=float)
    try:
        condition = float(np.linalg.cond(cov))
    except Exception:
        condition = float('inf')
    support = getattr(estimator, 'support_', None)
    support_fraction = None
    if support is not None:
        support_fraction = float(np.asarray(support, dtype=bool).mean())
    effective_support_fraction = getattr(estimator, 'effective_support_fraction_', None)
    if effective_support_fraction is not None:
        effective_support_fraction = float(effective_support_fraction)

    recommendations: list[str] = []
    if radial_kurt > 10:
        recommendations.append('Very high radial kurtosis: inspect outliers visually and prefer empirical/tail-calibrated thresholds over Gaussian chi-square thresholds.')
    elif radial_kurt > 3:
        recommendations.append('Heavy tails detected: compare FastMCD with RegularizedTyler and inspect QQ diagnostics.')
    if detected_fraction > max(0.10, 1.5 * (1.0 - alpha)):
        recommendations.append('Large detected fraction for the chosen alpha: consider threshold="empirical" or set a contamination prior.')
    if condition > 1e8:
        recommendations.append('Covariance is ill-conditioned: try RegularizedTyler, dimensionality reduction, or stronger regularization.')
    if support_fraction is not None and support_fraction > 0.90 and radial_kurt > 3:
        recommendations.append('Most observations are retained despite heavy tails: contamination may be diffuse or clustered; MCD assumptions may be weak.')
    if support_fraction is not None and support_fraction < 0.50:
        recommendations.append('Small final support: check whether support_fraction/contamination is too aggressive.')
    if qq_tail > 2.0:
        recommendations.append('QQ tail deviation is large: tail behavior is far from Gaussian; use empirical thresholds for anomaly detection.')

    return RobustDiagnosticReport(
        estimator_name=type(estimator).__name__,
        n_samples=n,
        n_features=p,
        alpha=float(alpha),
        threshold=threshold,
        detected_fraction=detected_fraction,
        radial_kurtosis=radial_kurt,
        condition_number=condition,
        max_distance=float(np.max(d2)) if n else float('nan'),
        median_distance=float(np.median(d2)) if n else float('nan'),
        qq_tail_deviation=qq_tail,
        support_fraction=support_fraction,
        effective_support_fraction=effective_support_fraction,
        recommendations=recommendations,
    )
