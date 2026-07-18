# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Matrix Minimum Covariance Determinant estimation.

This module implements a practical Matrix MCD estimator for samples whose
observations are matrices.  It preserves the row/column structure instead of
vectorizing each observation and estimating an unrestricted covariance matrix.
The subset search is approximate: multiple central and randomized elemental
starts are screened with short concentration steps, then the best candidates
are polished to convergence.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import chi2

from ._utils import radial_kurtosis
from .covariance import ConvergenceWarning


_EPS = np.finfo(np.float64).eps


def _check_matrix_sample(
    X, *, allow_nan: bool = False, name: str = "X", min_samples: int = 2
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError(f"{name} must have shape (n_samples, n_rows, n_columns)")
    if X.shape[0] < int(min_samples):
        unit = "observation" if int(min_samples) == 1 else "observations"
        raise ValueError(
            f"{name} must contain at least {int(min_samples)} matrix {unit}"
        )
    if X.shape[1] < 1 or X.shape[2] < 1:
        raise ValueError(f"{name} matrices must have at least one row and one column")
    if allow_nan:
        if np.isinf(X).any():
            raise ValueError(f"{name} contains infinity")
    elif not np.isfinite(X).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return np.asarray(X, dtype=np.float64, order="C")


def _median_impute_matrices(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.nanmedian(X, axis=0)
    if not np.isfinite(values).all():
        raise ValueError("Each matrix cell must contain at least one finite value")
    output = X.copy()
    missing = np.isnan(output)
    if missing.any():
        output[missing] = np.broadcast_to(values, output.shape)[missing]
    return output, values


def _symmetrize(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def _regularize_spd(A: np.ndarray, ridge: float, *, name: str) -> np.ndarray:
    A = _symmetrize(np.asarray(A, dtype=np.float64))
    d = A.shape[0]
    average = float(np.trace(A) / d)
    scale = max(abs(average), np.linalg.norm(A, ord="fro") / max(d, 1), 1.0)
    if ridge > 0:
        A = A + float(ridge) * scale * np.eye(d)
    values, vectors = np.linalg.eigh(A)
    floor = 64.0 * _EPS * scale * max(d, 1)
    if values[0] <= floor:
        if ridge <= 0:
            raise np.linalg.LinAlgError(
                f"{name} is singular; increase ridge or use a larger support"
            )
        values = np.maximum(values, max(float(ridge) * scale, floor))
        A = (vectors * values) @ vectors.T
    return _symmetrize(A)


def _precision_logdet(A: np.ndarray) -> tuple[np.ndarray, float]:
    values, vectors = np.linalg.eigh(_symmetrize(A))
    if values[0] <= 0 or not np.isfinite(values).all():
        raise np.linalg.LinAlgError("covariance factor is not positive definite")
    precision = (vectors * (1.0 / values)) @ vectors.T
    return _symmetrize(precision), float(np.log(values).sum())


def _normalize_factors(
    row_covariance: np.ndarray,
    column_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fix Kronecker scale ambiguity by setting det(row covariance) to one."""
    sign, logdet = np.linalg.slogdet(row_covariance)
    if sign <= 0:
        raise np.linalg.LinAlgError("row covariance is not positive definite")
    factor = float(np.exp(logdet / row_covariance.shape[0]))
    row_covariance = row_covariance / factor
    column_covariance = column_covariance * factor
    return _symmetrize(row_covariance), _symmetrize(column_covariance)


def _matrix_objective(row_covariance: np.ndarray, column_covariance: np.ndarray) -> float:
    _, row_logdet = np.linalg.slogdet(row_covariance)
    _, column_logdet = np.linalg.slogdet(column_covariance)
    r = row_covariance.shape[0]
    c = column_covariance.shape[0]
    return float(c * row_logdet + r * column_logdet)


def _matrix_distances(
    X: np.ndarray,
    location: np.ndarray,
    row_precision: np.ndarray,
    column_precision: np.ndarray,
) -> np.ndarray:
    residuals = X - location
    distances = np.empty(X.shape[0], dtype=np.float64)
    for i, residual in enumerate(residuals):
        transformed = row_precision @ residual @ column_precision
        distances[i] = float(np.sum(residual * transformed))
    return np.maximum(distances, 0.0)


def _flip_flop_mle(
    X: np.ndarray,
    *,
    ridge: float,
    max_iter: int,
    tol: float,
    initial_row: np.ndarray | None = None,
    initial_column: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bool, float]:
    """Fit a matrix-normal mean and Kronecker covariance by flip-flop updates."""
    n, r, c = X.shape
    location = X.mean(axis=0)
    residuals = X - location

    row_covariance = (
        np.eye(r, dtype=np.float64)
        if initial_row is None
        else _regularize_spd(initial_row, ridge, name="initial row covariance")
    )
    column_covariance = (
        np.eye(c, dtype=np.float64)
        if initial_column is None
        else _regularize_spd(initial_column, ridge, name="initial column covariance")
    )
    row_covariance, column_covariance = _normalize_factors(
        row_covariance, column_covariance
    )

    previous = np.inf
    converged = False
    objective = np.inf
    iterations = 0
    for iterations in range(1, int(max_iter) + 1):
        column_precision, _ = _precision_logdet(column_covariance)
        row_update = np.zeros((r, r), dtype=np.float64)
        for residual in residuals:
            row_update += residual @ column_precision @ residual.T
        row_update /= float(n * c)
        row_update = _regularize_spd(row_update, ridge, name="row covariance")

        row_precision, _ = _precision_logdet(row_update)
        column_update = np.zeros((c, c), dtype=np.float64)
        for residual in residuals:
            column_update += residual.T @ row_precision @ residual
        column_update /= float(n * r)
        column_update = _regularize_spd(
            column_update, ridge, name="column covariance"
        )

        row_update, column_update = _normalize_factors(row_update, column_update)
        objective = _matrix_objective(row_update, column_update)
        relative = abs(previous - objective) / max(1.0, abs(previous), abs(objective))
        row_covariance, column_covariance = row_update, column_update
        if np.isfinite(previous) and relative <= tol:
            converged = True
            break
        previous = objective

    return (
        location,
        row_covariance,
        column_covariance,
        iterations,
        converged,
        objective,
    )


def _consistency_factor(retained_fraction: float, dimension: int) -> float:
    if retained_fraction >= 1.0 - 1e-15:
        return 1.0
    quantile = chi2.ppf(retained_fraction, dimension)
    denominator = chi2.cdf(quantile, dimension + 2)
    if not np.isfinite(denominator) or denominator <= 0:
        return 1.0
    return float(retained_fraction / denominator)


def _apply_global_scale(
    row_covariance: np.ndarray,
    column_covariance: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Multiplying one factor scales the full Kronecker covariance by ``scale``.
    return row_covariance.copy(), _symmetrize(column_covariance * float(scale))


def _smallest_indices(values: np.ndarray, count: int) -> np.ndarray:
    indices = np.argpartition(values, count - 1)[:count]
    return np.sort(indices.astype(np.int64, copy=False))


def _minimum_elemental_size(n_rows: int, n_columns: int) -> int:
    # Matrix-normal MLE existence depends on the ratio of the two dimensions.
    # This is the elemental-size rule used as a practical lower bound in the
    # FastMMCD literature, with three observations as an absolute minimum.
    ratio_term = n_rows / n_columns + n_columns / n_rows
    return max(3, int(np.floor(ratio_term)) + 2)


class MatrixMinimumCovarianceDeterminant:
    """Robust location and Kronecker covariance for matrix-valued samples.

    Parameters
    ----------
    support_fraction : float or None, default=None
        Fraction of observations retained by the raw subset estimator.  ``None``
        requests the maximum-breakdown value of one half, increased when needed
        for the matrix-normal covariance factors to be estimable.
    contamination : float or None, default=None
        Alternative parameterization as the expected contaminated fraction.
        Specify either ``support_fraction`` or ``contamination``.
    quality : {"fast", "balanced", "high"}, default="fast"
        Preset controlling the number of initial subsets and candidates.
    n_init, n_best, initial_c_steps, max_iter : int or None
        Explicit overrides for the subset search.
    flip_flop_max_iter : int, default=100
        Maximum matrix-normal MLE updates within each full C-step.
    flip_flop_tol : float, default=1e-7
        Relative objective tolerance for flip-flop updates.
    ridge : float, default=1e-8
        Small trace-relative ridge used for numerical stability.  Setting it to
        zero preserves exact matrix-affine equivariance when all updates remain
        positive definite.
    reweight : bool, default=True
        Refit on observations whose raw matrix Mahalanobis distance is below a
        chi-square cutoff.
    reweight_alpha : float, default=0.975
        Chi-square probability used for reweighting and prediction defaults.
    adapt_support : bool, default=True
        Increase an undersized requested support to the elemental lower bound.
    missing_values : {"raise", "median"}, default="raise"
        ``"median"`` imputes each matrix cell by its median across observations.
    random_state : int, default=0
        Seed for randomized elemental starts.

    Notes
    -----
    For observations ``X[i]`` with shape ``(r, c)``, the fitted vectorized
    covariance is ``kron(column_covariance_, row_covariance_)``.  It is not
    materialized automatically because its dimensions are ``(r*c, r*c)``.
    """

    _QUALITY_PRESETS = {
        "fast": {"n_init": 50, "n_best": 5, "initial_c_steps": 2, "max_iter": 50},
        "balanced": {"n_init": 200, "n_best": 10, "initial_c_steps": 2, "max_iter": 100},
        "high": {"n_init": 500, "n_best": 20, "initial_c_steps": 3, "max_iter": 150},
    }

    def __init__(
        self,
        support_fraction=None,
        contamination=None,
        *,
        quality="fast",
        n_init=None,
        n_best=None,
        initial_c_steps=None,
        max_iter=None,
        flip_flop_max_iter=100,
        flip_flop_initial_iter=2,
        flip_flop_tol=1e-7,
        tol=1e-7,
        ridge=1e-8,
        reweight=True,
        reweight_alpha=0.975,
        adapt_support=True,
        missing_values="raise",
        tail_diagnostics=True,
        random_state=0,
    ):
        if quality not in self._QUALITY_PRESETS:
            raise ValueError("quality must be one of 'fast', 'balanced', or 'high'")
        if support_fraction is not None and contamination is not None:
            raise ValueError("Specify either support_fraction or contamination, not both")
        if support_fraction is not None and not (0.5 <= float(support_fraction) <= 1.0):
            raise ValueError("support_fraction must be in [0.5, 1]")
        if contamination is not None and not (0.0 <= float(contamination) < 0.5):
            raise ValueError("contamination must be in [0, 0.5)")
        if int(flip_flop_max_iter) < 1 or int(flip_flop_initial_iter) < 1:
            raise ValueError("flip-flop iteration limits must be positive")
        if float(flip_flop_tol) <= 0 or float(tol) <= 0:
            raise ValueError("tolerances must be positive")
        if float(ridge) < 0:
            raise ValueError("ridge must be non-negative")
        if not (0.5 < float(reweight_alpha) < 1.0):
            raise ValueError("reweight_alpha must be between 0.5 and 1")
        if missing_values not in {"raise", "median"}:
            raise ValueError("missing_values must be 'raise' or 'median'")

        preset = self._QUALITY_PRESETS[quality]
        self.support_fraction = support_fraction
        self.contamination = contamination
        self.quality = quality
        self.n_init = preset["n_init"] if n_init is None else int(n_init)
        self.n_best = preset["n_best"] if n_best is None else int(n_best)
        self.initial_c_steps = (
            preset["initial_c_steps"] if initial_c_steps is None else int(initial_c_steps)
        )
        self.max_iter = preset["max_iter"] if max_iter is None else int(max_iter)
        if self.n_init < 1 or self.n_best < 1 or self.initial_c_steps < 0 or self.max_iter < 1:
            raise ValueError("subset-search iteration counts must be positive")
        self.flip_flop_max_iter = int(flip_flop_max_iter)
        self.flip_flop_initial_iter = int(flip_flop_initial_iter)
        self.flip_flop_tol = float(flip_flop_tol)
        self.tol = float(tol)
        self.ridge = float(ridge)
        self.reweight = bool(reweight)
        self.reweight_alpha = float(reweight_alpha)
        self.adapt_support = bool(adapt_support)
        self.missing_values = missing_values
        self.tail_diagnostics = bool(tail_diagnostics)
        self.random_state = int(random_state)

    def _fit_subset(self, X: np.ndarray, support: np.ndarray, *, initial: bool):
        return _flip_flop_mle(
            X[support],
            ridge=self.ridge,
            max_iter=(self.flip_flop_initial_iter if initial else self.flip_flop_max_iter),
            tol=self.flip_flop_tol,
        )

    def _c_steps(
        self,
        X: np.ndarray,
        support: np.ndarray,
        *,
        steps: int,
        initial: bool,
    ):
        h = self.h_
        model = self._fit_subset(X, support, initial=initial)
        location, row_cov, col_cov, _, _, objective = model
        objective_path = [float(objective)]
        converged = False
        iterations = 0

        for iterations in range(1, int(steps) + 1):
            row_precision, _ = _precision_logdet(row_cov)
            column_precision, _ = _precision_logdet(col_cov)
            distances = _matrix_distances(
                X, location, row_precision, column_precision
            )
            new_support = _smallest_indices(distances, h)
            if np.array_equal(new_support, support):
                converged = True
                break
            candidate = self._fit_subset(X, new_support, initial=initial)
            candidate_objective = float(candidate[-1])
            allowance = 1e-8 * max(1.0, abs(objective))
            if candidate_objective > objective + allowance:
                # Numerical ridge and incomplete flip-flop iterations can break
                # exact C-step monotonicity.  Keep the better previous iterate.
                break
            support = new_support
            location, row_cov, col_cov, _, _, objective = candidate
            objective_path.append(float(objective))
            if abs(objective_path[-2] - objective_path[-1]) <= self.tol * max(
                1.0, abs(objective_path[-2])
            ):
                converged = True
                break

        return {
            "support": support,
            "location": location,
            "row_covariance": row_cov,
            "column_covariance": col_cov,
            "objective": float(objective),
            "objective_path": objective_path,
            "n_iter": iterations,
            "converged": converged,
        }

    def fit(self, X, y=None):
        X = _check_matrix_sample(
            X, allow_nan=self.missing_values == "median"
        )
        if self.missing_values == "median":
            X, self.impute_values_ = _median_impute_matrices(X)

        n, r, c = X.shape
        self.n_samples_in_ = n
        self.n_rows_in_ = r
        self.n_columns_in_ = c
        self.n_features_in_ = r * c
        self.matrix_shape_in_ = (r, c)
        self.elemental_size_ = _minimum_elemental_size(r, c)
        if n < self.elemental_size_:
            raise ValueError(
                "Too few matrix observations for the requested row/column dimensions: "
                f"need at least {self.elemental_size_}, got {n}"
            )

        if self.contamination is not None:
            fraction = 1.0 - float(self.contamination)
        elif self.support_fraction is not None:
            fraction = float(self.support_fraction)
        else:
            fraction = 0.5
        requested_h = max(1, int(np.floor(fraction * n)))
        if requested_h < self.elemental_size_:
            if not self.adapt_support:
                raise ValueError(
                    "Requested support is too small for the matrix dimensions; "
                    f"need at least {self.elemental_size_} observations"
                )
            requested_h = self.elemental_size_
        self.h_ = min(requested_h, n)
        self.effective_support_fraction_ = self.h_ / float(n)

        rng = np.random.default_rng(self.random_state)
        start_size = min(self.h_, self.elemental_size_)
        median_matrix = np.median(X, axis=0)
        central_distance = np.sum((X - median_matrix) ** 2, axis=(1, 2))
        starts = [_smallest_indices(central_distance, start_size)]
        for _ in range(max(0, self.n_init - 1)):
            starts.append(np.sort(rng.choice(n, size=start_size, replace=False)))

        screened = []
        for start in starts:
            try:
                if start.size < self.h_:
                    seed_model = self._fit_subset(X, start, initial=True)
                    row_precision, _ = _precision_logdet(seed_model[1])
                    column_precision, _ = _precision_logdet(seed_model[2])
                    seed_distances = _matrix_distances(
                        X, seed_model[0], row_precision, column_precision
                    )
                    support = _smallest_indices(seed_distances, self.h_)
                else:
                    support = start
                result = self._c_steps(
                    X,
                    support,
                    steps=self.initial_c_steps,
                    initial=True,
                )
                if np.isfinite(result["objective"]):
                    screened.append(result)
            except (np.linalg.LinAlgError, FloatingPointError):
                continue

        if not screened:
            raise RuntimeError(
                "MMCD could not construct a positive-definite initial model; "
                "increase ridge or use more observations"
            )
        screened.sort(key=lambda item: item["objective"])

        polished = []
        for result in screened[: min(self.n_best, len(screened))]:
            try:
                polished.append(
                    self._c_steps(
                        X,
                        result["support"],
                        steps=self.max_iter,
                        initial=False,
                    )
                )
            except (np.linalg.LinAlgError, FloatingPointError):
                continue
        if not polished:
            raise RuntimeError("MMCD polishing failed for every candidate subset")
        best = min(polished, key=lambda item: item["objective"])

        raw_factor = _consistency_factor(
            self.effective_support_fraction_, self.n_features_in_
        )
        raw_row, raw_column = _apply_global_scale(
            best["row_covariance"], best["column_covariance"], raw_factor
        )
        raw_row_precision, _ = _precision_logdet(raw_row)
        raw_column_precision, _ = _precision_logdet(raw_column)
        raw_distances = _matrix_distances(
            X, best["location"], raw_row_precision, raw_column_precision
        )
        raw_support = np.zeros(n, dtype=bool)
        raw_support[best["support"]] = True

        self.raw_location_ = best["location"]
        self.raw_row_covariance_ = raw_row
        self.raw_column_covariance_ = raw_column
        self.raw_row_precision_ = raw_row_precision
        self.raw_column_precision_ = raw_column_precision
        self.raw_distances_ = raw_distances
        self.raw_support_ = raw_support
        self.raw_consistency_factor_ = raw_factor
        self.raw_objective_value_ = float(best["objective"])
        self.objective_path_ = np.asarray(best["objective_path"], dtype=np.float64)
        self.n_iter_ = int(best["n_iter"])
        self.converged_ = bool(best["converged"])

        if self.reweight:
            cutoff = float(chi2.ppf(self.reweight_alpha, self.n_features_in_))
            reweighted_indices = np.flatnonzero(raw_distances <= cutoff)
            if reweighted_indices.size < self.h_:
                reweighted_indices = _smallest_indices(raw_distances, self.h_)
            final_model = _flip_flop_mle(
                X[reweighted_indices],
                ridge=self.ridge,
                max_iter=self.flip_flop_max_iter,
                tol=self.flip_flop_tol,
            )
            final_fraction = reweighted_indices.size / float(n)
            final_factor = _consistency_factor(final_fraction, self.n_features_in_)
            row_covariance, column_covariance = _apply_global_scale(
                final_model[1], final_model[2], final_factor
            )
            support = np.zeros(n, dtype=bool)
            support[reweighted_indices] = True
            location = final_model[0]
            self.reweight_threshold_ = cutoff
            self.consistency_factor_ = final_factor
        else:
            row_covariance, column_covariance = raw_row, raw_column
            support = raw_support.copy()
            location = best["location"]
            self.reweight_threshold_ = None
            self.consistency_factor_ = raw_factor

        row_precision, row_logdet = _precision_logdet(row_covariance)
        column_precision, column_logdet = _precision_logdet(column_covariance)
        distances = _matrix_distances(
            X, location, row_precision, column_precision
        )

        self.location_ = location
        self.row_covariance_ = row_covariance
        self.column_covariance_ = column_covariance
        self.row_precision_ = row_precision
        self.column_precision_ = column_precision
        self.distances_ = distances
        self.support_ = support
        self.log_determinant_ = float(c * row_logdet + r * column_logdet)
        self.objective_value_ = self.log_determinant_
        self.det_ = (
            float(np.exp(self.log_determinant_))
            if self.log_determinant_ < np.log(np.finfo(float).max)
            else float("inf")
        )
        if self.tail_diagnostics:
            self.radial_kurtosis_ = radial_kurtosis(
                self.distances_, self.n_features_in_
            )
            self.tail_index_ = self.radial_kurtosis_
        if not self.converged_:
            warnings.warn(
                "MMCD subset polishing reached the iteration limit or a numerical "
                "plateau before support convergence",
                ConvergenceWarning,
                stacklevel=2,
            )
        return self

    def _check_fitted_input(self, X) -> np.ndarray:
        if not hasattr(self, "row_precision_"):
            raise RuntimeError("Estimator is not fitted")
        X = _check_matrix_sample(
            X, allow_nan=self.missing_values == "median", min_samples=1
        )
        if X.shape[1:] != self.matrix_shape_in_:
            raise ValueError(
                f"X matrices must have shape {self.matrix_shape_in_}, got {X.shape[1:]}"
            )
        if self.missing_values == "median":
            output = X.copy()
            missing = np.isnan(output)
            if missing.any():
                output[missing] = np.broadcast_to(
                    self.impute_values_, output.shape
                )[missing]
            X = output
        return X

    def mahalanobis(self, X) -> np.ndarray:
        """Return squared matrix Mahalanobis distances."""
        X = self._check_fitted_input(X)
        return _matrix_distances(
            X,
            self.location_,
            self.row_precision_,
            self.column_precision_,
        )

    def score_samples(self, X) -> np.ndarray:
        return -0.5 * self.mahalanobis(X)

    def predict(self, X, alpha: float = 0.975) -> np.ndarray:
        if not (0.5 < float(alpha) < 1.0):
            raise ValueError("alpha must be between 0.5 and 1")
        cutoff = chi2.ppf(float(alpha), self.n_features_in_)
        return np.where(self.mahalanobis(X) <= cutoff, 1, -1)

    def fit_predict(self, X, y=None) -> np.ndarray:
        return self.fit(X, y=y).predict(X)

    def kronecker_covariance(self) -> np.ndarray:
        """Materialize covariance of column-major vectorized observations."""
        if not hasattr(self, "row_covariance_"):
            raise RuntimeError("Estimator is not fitted")
        return np.kron(self.column_covariance_, self.row_covariance_)

    def kronecker_precision(self) -> np.ndarray:
        """Materialize precision of column-major vectorized observations."""
        if not hasattr(self, "row_precision_"):
            raise RuntimeError("Estimator is not fitted")
        return np.kron(self.column_precision_, self.row_precision_)

    def whiten(self, X) -> np.ndarray:
        """Whiten matrix observations using the two covariance factors."""
        X = self._check_fitted_input(X)
        row_values, row_vectors = np.linalg.eigh(self.row_covariance_)
        column_values, column_vectors = np.linalg.eigh(self.column_covariance_)
        row_inverse_sqrt = (row_vectors * (1.0 / np.sqrt(row_values))) @ row_vectors.T
        column_inverse_sqrt = (
            column_vectors * (1.0 / np.sqrt(column_values))
        ) @ column_vectors.T
        return np.asarray(
            [
                row_inverse_sqrt @ (matrix - self.location_) @ column_inverse_sqrt
                for matrix in X
            ]
        )

    def cell_contributions(self, X) -> np.ndarray:
        """Signed additive cell contributions to squared matrix distance.

        Correlation can make individual contributions negative.  Their sum over
        every cell is exactly the squared matrix Mahalanobis distance.  These are
        quadratic-form contributions, not Shapley values.
        """
        X = self._check_fitted_input(X)
        residuals = X - self.location_
        output = np.empty_like(residuals)
        for i, residual in enumerate(residuals):
            output[i] = residual * (
                self.row_precision_ @ residual @ self.column_precision_
            )
        return output

    def row_contributions(self, X) -> np.ndarray:
        """Signed additive row contributions to squared matrix distance."""
        return self.cell_contributions(X).sum(axis=2)

    def column_contributions(self, X) -> np.ndarray:
        """Signed additive column contributions to squared matrix distance."""
        return self.cell_contributions(X).sum(axis=1)


MatrixMCD = MatrixMinimumCovarianceDeterminant
MMCD = MatrixMinimumCovarianceDeterminant
