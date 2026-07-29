"""Shared covariance/scatter benchmark method catalog.

The gallery benchmarks intentionally compare only estimators that solve the same
statistical task.  This module keeps the method lists, applicability rules, and
labels consistent across accuracy, timing, and documentation scripts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import robustcov as rc


Applicability = Callable[[int, int], tuple[bool, str]]
Factory = Callable[[], object]


@dataclass(frozen=True)
class CovarianceBenchmarkMethod:
    name: str
    family: str
    factory: Factory
    applicable: Applicability
    note: str = ""
    experimental: bool = False


def _always(n: int, p: int) -> tuple[bool, str]:
    del n, p
    return True, ""


def _full_rank(n: int, p: int) -> tuple[bool, str]:
    ok = n > p
    return ok, "requires n > p" if not ok else ""


def _mcd(n: int, p: int) -> tuple[bool, str]:
    # The default raw support is about 75% of n and must contain more rows than
    # columns for a nonsingular classical MCD covariance.
    ok = int(0.75 * n) > p
    return ok, "requires default MCD support h > p" if not ok else ""


def _dets(n: int, p: int) -> tuple[bool, str]:
    ok = int((n + 1) // 2) > p
    return ok, "requires ceil(n / 2) > p" if not ok else ""


def _sklearn_methods() -> list[CovarianceBenchmarkMethod]:
    methods: list[CovarianceBenchmarkMethod] = []
    try:
        from sklearn.covariance import EmpiricalCovariance, LedoitWolf, MinCovDet, OAS
    except Exception:
        return methods
    methods.extend(
        [
            CovarianceBenchmarkMethod(
                "sklearn Empirical",
                "classical / shrinkage",
                lambda: EmpiricalCovariance(),
                _always,
                "non-robust reference",
            ),
            CovarianceBenchmarkMethod(
                "sklearn LedoitWolf",
                "classical / shrinkage",
                lambda: LedoitWolf(),
                _always,
                "linear shrinkage reference",
            ),
            CovarianceBenchmarkMethod(
                "sklearn OAS",
                "classical / shrinkage",
                lambda: OAS(),
                _always,
                "oracle-approximating shrinkage reference",
            ),
            CovarianceBenchmarkMethod(
                "sklearn MinCovDet",
                "high-breakdown subset",
                lambda: MinCovDet(random_state=0, support_fraction=None),
                _mcd,
                "classical MCD reference",
            ),
        ]
    )
    return methods


def covariance_methods(
    *,
    purpose: str = "accuracy",
    include_experimental: bool = True,
    include_selector: bool = True,
    include_sklearn: bool = True,
) -> list[CovarianceBenchmarkMethod]:
    """Return the shared covariance benchmark catalog.

    Parameters
    ----------
    purpose:
        ``"accuracy"`` uses more conservative iteration limits. ``"speed"``
        uses representative but bounded settings so the complete catalog can be
        timed in CI and local quick runs.
    include_experimental:
        Include the explicitly labelled Hellinger prototype.
    include_selector:
        Include :class:`AutoRobustScatter` as a workflow-level timing/accuracy
        row.  Its runtime includes fitting and selecting several candidates.
    include_sklearn:
        Add sklearn baselines when sklearn is installed.
    """
    if purpose not in {"accuracy", "speed"}:
        raise ValueError("purpose must be 'accuracy' or 'speed'")

    fast = purpose == "speed"
    m_iter = 180 if fast else 500
    tyler_iter = 180 if fast else 400
    mrcd_starts = 8 if fast else 20
    mrcd_best = 3 if fast else 5
    det_iter = 35 if fast else 80

    methods = [
        CovarianceBenchmarkMethod(
            "robustcov FastMCD",
            "high-breakdown subset",
            lambda: rc.FastMCD(
                quality="fast",
                n_init=60 if fast else 120,
                n_best=5,
                random_state=0,
                scale_correction="none",
            ),
            _mcd,
            "native classical MCD",
        ),
        CovarianceBenchmarkMethod(
            "robustcov MRCD",
            "high-breakdown regularized subset",
            lambda: rc.MRCD(
                quality="fast",
                n_init=mrcd_starts,
                n_best=mrcd_best,
                max_iter=50 if fast else 100,
                random_state=0,
            ),
            _always,
            "regularized subset estimator for p close to or above n",
        ),
        CovarianceBenchmarkMethod(
            "robustcov DetS",
            "deterministic high-breakdown",
            lambda: rc.DetS(
                initial_steps=1 if fast else 2,
                n_best=1 if fast else 2,
                max_iter=det_iter,
                tail_diagnostics=False,
            ),
            _dets,
            "deterministic S-estimator",
        ),
        CovarianceBenchmarkMethod(
            "robustcov DetMM",
            "deterministic high-breakdown",
            lambda: rc.DetMM(
                initial_steps=1 if fast else 2,
                n_best=1 if fast else 2,
                max_iter=det_iter,
                tail_diagnostics=False,
            ),
            _dets,
            "high-breakdown start with efficient MM refinement",
        ),
        CovarianceBenchmarkMethod(
            "robustcov TylerShape",
            "elliptical shape",
            lambda: rc.TylerShape(
                scale_correction="radial_median",
                max_iter=tyler_iter,
                tol=1e-6,
            ),
            _full_rank,
            "unregularized affine-equivariant shape",
        ),
        CovarianceBenchmarkMethod(
            "robustcov RegularizedTyler",
            "regularized elliptical shape",
            lambda: rc.RegularizedTyler(
                alpha=0.10,
                scale_correction="radial_median",
                max_iter=tyler_iter,
                tol=1e-6,
            ),
            _always,
            "regularized Tyler shape",
        ),
        CovarianceBenchmarkMethod(
            "robustcov KLRegularizedTyler",
            "regularized elliptical shape",
            lambda: rc.KLRegularizedTyler(
                alpha=0.10,
                scale_correction="radial_median",
                max_iter=tyler_iter,
                tol=1e-6,
            ),
            _always,
            "KL-labelled regularized Tyler interface",
        ),
        CovarianceBenchmarkMethod(
            "robustcov WieselTyler",
            "regularized elliptical shape",
            lambda: rc.WieselTyler(
                alpha=0.10,
                scale_correction="radial_median",
                max_iter=tyler_iter,
                tol=1e-6,
            ),
            _always,
            "Wiesel-labelled regularized Tyler interface",
        ),
        CovarianceBenchmarkMethod(
            "robustcov StudentT(df=3)",
            "heavy-tail M-scatter",
            lambda: rc.StudentTScatter(
                df=3,
                alpha=0.05,
                max_iter=m_iter,
                damping=0.7,
                tol=1e-5,
                warn_on_nonconvergence=False,
            ),
            _always,
            "Student-t radial weights",
        ),
        CovarianceBenchmarkMethod(
            "robustcov RegularizedCauchy",
            "heavy-tail M-scatter",
            lambda: rc.RegularizedCauchy(
                alpha=0.10,
                max_iter=m_iter,
                damping=0.7,
                tol=1e-5,
                warn_on_nonconvergence=False,
            ),
            _always,
            "redescending Cauchy-like radial weights",
        ),
    ]

    if include_experimental:
        methods.append(
            CovarianceBenchmarkMethod(
                "robustcov HellingerTyler (experimental)",
                "experimental elliptical shape",
                lambda: rc.HellingerRegularizedTyler(
                    alpha=0.10,
                    scale_correction="radial_median",
                    max_iter=120 if fast else 180,
                    tol=1e-6,
                    warn_on_nonconvergence=False,
                ),
                _always,
                "experimental square-root shrinkage prototype",
                experimental=True,
            )
        )

    if include_selector:
        methods.append(
            CovarianceBenchmarkMethod(
                "robustcov AutoRobustScatter",
                "automatic selection workflow",
                lambda: rc.AutoRobustScatter(
                    selection="diagnostic",
                    random_state=0,
                ),
                _always,
                "runtime includes fitting and selecting multiple candidates",
            )
        )

    if include_sklearn:
        methods.extend(_sklearn_methods())
    return methods


def method_names(**kwargs) -> list[str]:
    return [method.name for method in covariance_methods(**kwargs)]
