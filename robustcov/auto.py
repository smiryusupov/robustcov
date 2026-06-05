from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .m_estimators import RegularizedCauchy, StudentTScatter
from .covariance import RegularizedTyler
from .diagnostics import diagnostic_report
from ._utils import check_array


def _trace_normalize_cov(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    S = 0.5 * (S + S.T)
    p = S.shape[0]
    tr = float(np.trace(S))
    if not np.isfinite(tr) or tr <= 1e-12:
        return np.eye(p)
    return S * (p / tr)


def _scatter_distance(A: np.ndarray, B: np.ndarray) -> float:
    A = _trace_normalize_cov(A)
    B = _trace_normalize_cov(B)
    denom = max(1e-12, np.linalg.norm(A, ord="fro"))
    return float(np.linalg.norm(A - B, ord="fro") / denom)


@dataclass
class ScatterCandidateResult:
    name: str
    score: float
    diagnostic_score: float
    stability_score: float
    converged: bool
    n_iter: int
    condition_number: float
    radial_kurtosis: float
    finite: bool
    estimator: object


class AutoRobustScatter:
    """Lightweight unsupervised selector for robust scatter estimators.

    ``selection='diagnostic'`` fits each candidate once and scores finite covariance,
    convergence, condition number, and log-scaled radial kurtosis.

    ``selection='stability'`` additionally refits candidates on split-sample subsets
    and penalizes scatter matrices that move too much. This is slower, but it is more
    useful in small-sample, high-dimensional, heavy-tailed settings.
    """

    def __init__(
        self,
        candidates=None,
        criterion=None,
        selection="stability",
        max_condition=1e6,
        prefer_converged=True,
        n_splits=3,
        subsample_fraction=0.7,
        random_state=0,
        stability_weight=2.0,
    ):
        if criterion is not None:
            selection = criterion
        if selection not in {"diagnostic", "stability"}:
            raise ValueError("selection must be 'diagnostic' or 'stability'")
        self.candidates = candidates
        self.selection = selection
        self.criterion = selection
        self.max_condition = float(max_condition)
        self.prefer_converged = bool(prefer_converged)
        self.n_splits = int(n_splits)
        self.subsample_fraction = float(subsample_fraction)
        self.random_state = int(random_state)
        self.stability_weight = float(stability_weight)
        if self.n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if not (0.2 <= self.subsample_fraction <= 1.0):
            raise ValueError("subsample_fraction must be in [0.2, 1.0]")

    def _default_candidates(self, n, p):
        alpha_hi = 0.20 if p >= n else 0.10
        alpha_mid = 0.10 if p >= n else 0.05
        return [
            RegularizedCauchy(alpha=alpha_hi, max_iter=700, damping=0.5, tol=1e-5, warn_on_nonconvergence=False),
            RegularizedCauchy(alpha=alpha_mid, max_iter=700, damping=0.5, tol=1e-5, warn_on_nonconvergence=False),
            StudentTScatter(df=2, alpha=alpha_hi, max_iter=700, damping=0.5, tol=1e-5, warn_on_nonconvergence=False),
            StudentTScatter(df=3, alpha=alpha_mid, max_iter=700, damping=0.5, tol=1e-5, warn_on_nonconvergence=False),
            RegularizedTyler(alpha=alpha_hi, scale_correction="radial_median", max_iter=700),
        ]

    def _clone_candidate(self, cand):
        est = copy.deepcopy(cand)
        if hasattr(est, "warn_on_nonconvergence"):
            est.warn_on_nonconvergence = False
        return est

    def _diagnostic_score(self, est):
        finite = bool(np.isfinite(est.covariance_).all())
        try:
            cond = float(np.linalg.cond(est.covariance_))
        except Exception:
            cond = float("inf")
        radial_kurt = float(getattr(est, "radial_kurtosis_", np.nan))
        if not np.isfinite(radial_kurt):
            radial_kurt = float("inf")
        converged = bool(getattr(est, "converged_", True))
        log_cond = np.log10(max(cond, 1.0)) if np.isfinite(cond) else 12.0
        log_tail = np.log1p(max(radial_kurt, 0.0)) if np.isfinite(radial_kurt) else 12.0
        score = log_cond * log_cond + 0.30 * log_tail
        if not finite:
            score += 1e6
        if not converged and self.prefer_converged:
            score += 5.0
        if cond > self.max_condition:
            score += 20.0
        return float(score), cond, radial_kurt, finite, converged

    def _stability_score(self, cand, X, full_est):
        n = X.shape[0]
        m = max(3, int(round(self.subsample_fraction * n)))
        if m >= n:
            return 0.0
        rng = np.random.default_rng(self.random_state)
        distances = []
        failures = 0
        for _ in range(self.n_splits):
            idx = rng.choice(n, size=m, replace=False)
            try:
                sub_est = self._clone_candidate(cand).fit(X[idx])
                distances.append(_scatter_distance(full_est.covariance_, sub_est.covariance_))
                if not bool(getattr(sub_est, "converged_", True)):
                    failures += 1
            except Exception:
                failures += 1
                distances.append(10.0)
        return float(np.median(distances) + failures)

    def fit(self, X, y=None):
        X = check_array(X, allow_nan=False)
        n, p = X.shape
        self.n_samples_in_, self.n_features_in_ = n, p
        candidates = self._default_candidates(n, p) if self.candidates is None else self.candidates
        results = []
        for cand in candidates:
            name = type(cand).__name__
            try:
                est = self._clone_candidate(cand).fit(X)
                diag_score, cond, rk, finite, converged = self._diagnostic_score(est)
                stability = self._stability_score(cand, X, est) if self.selection == "stability" else 0.0
                score = diag_score + self.stability_weight * stability
                results.append(ScatterCandidateResult(
                    name=name,
                    score=float(score),
                    diagnostic_score=float(diag_score),
                    stability_score=float(stability),
                    converged=converged,
                    n_iter=int(getattr(est, "n_iter_", -1)),
                    condition_number=cond,
                    radial_kurtosis=rk,
                    finite=finite,
                    estimator=est,
                ))
            except Exception:
                results.append(ScatterCandidateResult(
                    name=name,
                    score=1e9,
                    diagnostic_score=1e9,
                    stability_score=1e9,
                    converged=False,
                    n_iter=-1,
                    condition_number=float("inf"),
                    radial_kurtosis=float("inf"),
                    finite=False,
                    estimator=None,
                ))
        if not results:
            raise ValueError("No candidate estimators were provided")
        best = min(results, key=lambda r: r.score)
        if best.estimator is None:
            raise RuntimeError("All AutoRobustScatter candidates failed")
        self.candidate_results_ = results
        self.best_result_ = best
        self.estimator_ = best.estimator
        # Backward/ergonomic alias: many users naturally look for best_estimator_.
        self.best_estimator_ = best.estimator
        self.best_estimator_name_ = best.name
        self.estimator_name_ = best.name
        for attr in [
            "location_", "shape_", "covariance_", "precision_", "distances_", "scale_",
            "radial_kurtosis_", "tail_index_", "n_iter_", "converged_", "weights_",
        ]:
            if hasattr(self.estimator_, attr):
                setattr(self, attr, getattr(self.estimator_, attr))
        return self

    def mahalanobis(self, X):
        return self.estimator_.mahalanobis(X)

    def report(self):
        return diagnostic_report(self.estimator_)

    def summary(self) -> str:
        lines = [f"AutoRobustScatter selected: {self.best_estimator_name_} ({self.selection})", "candidates:"]
        for r in self.candidate_results_:
            lines.append(
                f"  - {r.name}: score={r.score:.3f}, diagnostic={r.diagnostic_score:.3f}, "
                f"stability={r.stability_score:.3f}, converged={r.converged}, "
                f"n_iter={r.n_iter}, cond={r.condition_number:.4g}, radial_kurtosis={r.radial_kurtosis:.4g}"
            )
        return "\n".join(lines)
