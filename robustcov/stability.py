# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap uncertainty and stability diagnostics for robust PCA subspaces."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._estimator import EstimatorMixin


def _as_2d_finite_array(X: np.ndarray, *, name: str = "X") -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if X.shape[0] < 3:
        raise ValueError(f"{name} must contain at least three samples")
    if X.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one feature")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} must contain only finite values")
    return X


def _default_pca() -> Any:
    from .pca import RobustPCA

    return RobustPCA(n_components=0.95, store_scores=False)


def _extract_pca_state(model: Any, n_features: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not hasattr(model, "components_"):
        raise AttributeError("PCA estimator must expose components_ after fit")
    if not hasattr(model, "eigenvalues_"):
        raise AttributeError("PCA estimator must expose eigenvalues_ after fit")
    if not hasattr(model, "explained_variance_ratio_"):
        raise AttributeError(
            "PCA estimator must expose explained_variance_ratio_ after fit"
        )

    components = np.asarray(model.components_, dtype=float)
    eigenvalues = np.asarray(model.eigenvalues_, dtype=float)
    ratios = np.asarray(model.explained_variance_ratio_, dtype=float)

    if components.ndim != 2 or components.shape[1] != n_features:
        raise ValueError("PCA components_ has an incompatible shape")
    q = components.shape[0]
    if q < 1:
        raise ValueError("PCA estimator must retain at least one component")
    if eigenvalues.shape != (q,) or ratios.shape != (q,):
        raise ValueError("PCA eigenvalue attributes have incompatible shapes")
    if not (
        np.all(np.isfinite(components))
        and np.all(np.isfinite(eigenvalues))
        and np.all(np.isfinite(ratios))
    ):
        raise ValueError("PCA fitted attributes must contain only finite values")
    return components, eigenvalues, ratios


def _principal_angles(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    singular_values = np.linalg.svd(
        reference @ candidate.T,
        compute_uv=False,
    )
    singular_values = np.clip(singular_values, 0.0, 1.0)
    return np.arccos(singular_values)


def _align_signs(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    aligned = candidate.copy()
    signs = np.sign(np.sum(reference * aligned, axis=1))
    signs[signs == 0.0] = 1.0
    aligned *= signs[:, None]
    return aligned


def _align_sign_permutation(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    similarity = np.abs(candidate @ reference.T)
    candidate_rows, reference_rows = linear_sum_assignment(-similarity)
    aligned = np.empty_like(candidate)
    aligned[reference_rows] = candidate[candidate_rows]
    return _align_signs(reference, aligned)


def _align_procrustes(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    cross = reference @ candidate.T
    left, _, right_t = np.linalg.svd(cross, full_matrices=False)
    rotation = left @ right_t
    aligned = rotation @ candidate
    return _align_signs(reference, aligned)


def _set_fixed_component_count(model: Any, n_components: int) -> None:
    if hasattr(model, "n_components"):
        try:
            model.n_components = int(n_components)
        except Exception:
            pass
    if hasattr(model, "store_scores"):
        try:
            model.store_scores = False
        except Exception:
            pass


_RESAMPLING_METHODS = {
    "iid",
    "moving_block",
    "circular_block",
    "stationary",
    "cluster",
}


def _default_block_length(n_samples: int) -> int:
    """Return a conservative cube-root block-length default."""
    return max(2, min(n_samples, int(np.ceil(n_samples ** (1.0 / 3.0)))))


def _ordered_unique(values: np.ndarray) -> np.ndarray:
    """Return unique values in first-occurrence order."""
    unique, first = np.unique(values, return_index=True)
    return unique[np.argsort(first)]


def _draw_resample_indices(
    rng: np.random.Generator,
    *,
    n_samples: int,
    sample_size: int,
    method: str,
    block_length: int | None = None,
    groups: np.ndarray | None = None,
    n_clusters_to_sample: int | None = None,
) -> np.ndarray:
    """Draw one set of row indices under the requested resampling design."""
    if method == "iid":
        return rng.integers(0, n_samples, size=sample_size)

    if method == "moving_block":
        assert block_length is not None
        n_blocks = int(np.ceil(sample_size / block_length))
        starts = rng.integers(0, n_samples - block_length + 1, size=n_blocks)
        blocks = [np.arange(start, start + block_length) for start in starts]
        return np.concatenate(blocks)[:sample_size]

    if method == "circular_block":
        assert block_length is not None
        n_blocks = int(np.ceil(sample_size / block_length))
        starts = rng.integers(0, n_samples, size=n_blocks)
        offsets = np.arange(block_length)
        blocks = [(start + offsets) % n_samples for start in starts]
        return np.concatenate(blocks)[:sample_size]

    if method == "stationary":
        assert block_length is not None
        restart_probability = 1.0 / float(block_length)
        indices = np.empty(sample_size, dtype=int)
        indices[0] = int(rng.integers(0, n_samples))
        for position in range(1, sample_size):
            if rng.random() < restart_probability:
                indices[position] = int(rng.integers(0, n_samples))
            else:
                indices[position] = (indices[position - 1] + 1) % n_samples
        return indices

    if method == "cluster":
        assert groups is not None
        assert n_clusters_to_sample is not None
        unique_groups = _ordered_unique(groups)
        selected = rng.integers(0, unique_groups.size, size=n_clusters_to_sample)
        blocks = [np.flatnonzero(groups == unique_groups[index]) for index in selected]
        return np.concatenate(blocks)

    raise ValueError(f"unknown resampling method: {method}")


@dataclass
class SubspaceStability(EstimatorMixin):
    """Bootstrap stability analysis for PCA-style subspace estimators.

    The class repeatedly refits a PCA estimator on bootstrap samples, aligns
    each fitted basis with the full-data reference basis, and summarizes
    uncertainty in eigenvalues, explained-variance ratios, loadings, and the
    retained subspace. IID, block, stationary, and cluster resampling designs
    are available so the bootstrap can match the observation dependence.

    Parameters
    ----------
    pca : object, optional
        PCA-style estimator copied and fitted through ``fit(X)``. It must expose
        ``components_``, ``eigenvalues_``, and ``explained_variance_ratio_``.
        ``RobustPCA(n_components=0.95)`` is used by default.
    n_resamples : int, default=200
        Number of bootstrap refits.
    confidence_level : float, default=0.95
        Central percentile interval coverage.
    sample_fraction : float, default=1.0
        Bootstrap sample size as a fraction of the input row count. For
        ``resampling="cluster"``, the fraction applies to the number of unique
        clusters and all rows in each selected cluster are retained.
    resampling : {"iid", "moving_block", "circular_block", "stationary", "cluster"}, default="iid"
        Resampling design. Block methods preserve consecutive observations in
        the row order. ``stationary`` uses geometrically distributed circular
        block lengths. ``cluster`` resamples complete groups supplied to
        :meth:`fit`.
    block_length : int or None, default=None
        Block length for moving, circular, and stationary bootstrap sampling.
        ``None`` uses a cube-root rule, ``ceil(n_samples**(1/3))``. For the
        stationary bootstrap this is the expected block length.
    alignment : {"procrustes", "sign_permutation", "sign"}, default="procrustes"
        Rule used to orient bootstrap loading matrices before loading intervals
        are calculated. Principal angles are calculated before alignment.
    random_state : int or None, default=None
        Seed controlling bootstrap row sampling.
    min_successful_resamples : int or None, default=None
        Minimum number of successful refits required to return intervals.
        ``None`` uses the smaller of 20 and ``n_resamples``.

    Notes
    -----
    Percentile intervals describe sampling stability under the selected
    resampling design. They are not formal finite-sample confidence guarantees
    under arbitrary contamination or dependence. Near-equal eigenvalues make
    individual component loadings
    intrinsically weakly identified; principal-angle summaries remain the safer
    diagnostic in that setting.
    """

    pca: Any | None = None
    n_resamples: int = 200
    confidence_level: float = 0.95
    sample_fraction: float = 1.0
    resampling: str = "iid"
    block_length: int | None = None
    alignment: str = "procrustes"
    random_state: int | None = None
    min_successful_resamples: int | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        *,
        groups: np.ndarray | None = None,
    ) -> "SubspaceStability":
        """Fit the reference model and bootstrap its retained subspace.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Observations. Row order is treated as time order by the block and
            stationary bootstrap methods.
        y : ignored, optional
            Present for sklearn-style compatibility.
        groups : array-like of shape (n_samples,), optional
            Cluster labels required when ``resampling="cluster"``. Complete
            clusters are sampled with replacement.
        """
        del y
        X = _as_2d_finite_array(X)
        groups_array = self._validate_parameters(X.shape[0], groups)

        template = _default_pca() if self.pca is None else deepcopy(self.pca)
        if hasattr(template, "store_scores"):
            template.store_scores = False
        reference_model = deepcopy(template)
        reference_model.fit(X)
        reference, reference_values, reference_ratios = _extract_pca_state(
            reference_model,
            X.shape[1],
        )
        q = reference.shape[0]

        rng = np.random.default_rng(self.random_state)
        sample_size = max(3, int(np.ceil(self.sample_fraction * X.shape[0])))
        effective_block_length = None
        n_clusters_to_sample = None
        if self.resampling in {"moving_block", "circular_block", "stationary"}:
            effective_block_length = (
                _default_block_length(X.shape[0])
                if self.block_length is None
                else int(self.block_length)
            )
            effective_block_length = min(effective_block_length, X.shape[0])
        if self.resampling == "cluster":
            assert groups_array is not None
            n_clusters = _ordered_unique(groups_array).size
            n_clusters_to_sample = max(
                1,
                int(np.ceil(self.sample_fraction * n_clusters)),
            )

        aligned_components: list[np.ndarray] = []
        eigenvalues: list[np.ndarray] = []
        variance_ratios: list[np.ndarray] = []
        principal_angles: list[np.ndarray] = []
        projection_distances: list[float] = []
        component_similarities: list[np.ndarray] = []
        sample_sizes: list[int] = []
        failures: list[str] = []

        for _ in range(self.n_resamples):
            indices = _draw_resample_indices(
                rng,
                n_samples=X.shape[0],
                sample_size=sample_size,
                method=self.resampling,
                block_length=effective_block_length,
                groups=groups_array,
                n_clusters_to_sample=n_clusters_to_sample,
            )
            candidate_model = deepcopy(template)
            _set_fixed_component_count(candidate_model, q)
            try:
                candidate_model.fit(X[indices])
                candidate, values, ratios = _extract_pca_state(
                    candidate_model,
                    X.shape[1],
                )
                if candidate.shape[0] != q:
                    raise ValueError(
                        "bootstrap fit retained a different number of components"
                    )

                angles = _principal_angles(reference, candidate)
                aligned = self._align(reference, candidate)
                similarities = np.abs(np.sum(reference * aligned, axis=1))

                aligned_components.append(aligned)
                eigenvalues.append(values)
                variance_ratios.append(ratios)
                principal_angles.append(angles)
                projection_distances.append(
                    float(
                        np.linalg.norm(
                            reference.T @ reference - candidate.T @ candidate,
                            ord="fro",
                        )
                    )
                )
                component_similarities.append(similarities)
                sample_sizes.append(int(indices.size))
            except Exception as exc:  # bootstrap failures are recorded and skipped
                failures.append(f"{type(exc).__name__}: {exc}")

        n_successful = len(aligned_components)
        minimum_successful = (
            min(20, self.n_resamples)
            if self.min_successful_resamples is None
            else int(self.min_successful_resamples)
        )
        if n_successful < minimum_successful:
            raise RuntimeError(
                "too few successful bootstrap refits: "
                f"{n_successful} < {minimum_successful}. "
                "Use a more regularized PCA estimator or reduce the component count."
            )

        loading_samples = np.stack(aligned_components)
        eigenvalue_samples = np.stack(eigenvalues)
        ratio_samples = np.stack(variance_ratios)
        angle_samples = np.stack(principal_angles)
        projection_samples = np.asarray(projection_distances, dtype=float)
        similarity_samples = np.stack(component_similarities)

        tail = 0.5 * (1.0 - self.confidence_level)
        quantiles = [tail, 1.0 - tail]

        self.reference_model_ = reference_model
        self.components_ = reference.copy()
        self.eigenvalues_ = reference_values.copy()
        self.explained_variance_ratio_ = reference_ratios.copy()
        self.n_components_ = q
        self.n_samples_in_ = X.shape[0]
        self.n_features_in_ = X.shape[1]
        self.resampling_ = self.resampling
        self.block_length_ = effective_block_length
        self.bootstrap_sample_size_ = (
            sample_size if self.resampling != "cluster" else None
        )
        self.bootstrap_sample_sizes_ = np.asarray(sample_sizes, dtype=int)
        if self.resampling == "cluster":
            assert groups_array is not None
            self.n_clusters_in_ = _ordered_unique(groups_array).size
            self.bootstrap_cluster_count_ = int(n_clusters_to_sample)
        else:
            self.n_clusters_in_ = None
            self.bootstrap_cluster_count_ = None

        self.loading_samples_ = loading_samples
        self.eigenvalue_samples_ = eigenvalue_samples
        self.explained_variance_ratio_samples_ = ratio_samples
        self.principal_angle_samples_ = angle_samples
        self.principal_angle_degrees_ = np.degrees(angle_samples)
        self.max_principal_angle_degrees_ = np.max(
            self.principal_angle_degrees_,
            axis=1,
        )
        self.projection_distance_samples_ = projection_samples
        self.component_similarity_samples_ = similarity_samples

        loading_interval = np.quantile(loading_samples, quantiles, axis=0)
        eigenvalue_interval = np.quantile(eigenvalue_samples, quantiles, axis=0)
        ratio_interval = np.quantile(ratio_samples, quantiles, axis=0)
        angle_interval = np.quantile(
            self.principal_angle_degrees_,
            quantiles,
            axis=0,
        )

        self.loading_interval_ = loading_interval
        self.loading_interval_lower_ = loading_interval[0]
        self.loading_interval_upper_ = loading_interval[1]
        self.eigenvalue_interval_ = eigenvalue_interval
        self.eigenvalue_interval_lower_ = eigenvalue_interval[0]
        self.eigenvalue_interval_upper_ = eigenvalue_interval[1]
        self.explained_variance_ratio_interval_ = ratio_interval
        self.explained_variance_ratio_interval_lower_ = ratio_interval[0]
        self.explained_variance_ratio_interval_upper_ = ratio_interval[1]
        self.principal_angle_interval_degrees_ = angle_interval
        self.principal_angle_interval_lower_degrees_ = angle_interval[0]
        self.principal_angle_interval_upper_degrees_ = angle_interval[1]

        self.loading_standard_error_ = np.std(loading_samples, axis=0, ddof=1)
        self.eigenvalue_standard_error_ = np.std(
            eigenvalue_samples,
            axis=0,
            ddof=1,
        )
        self.explained_variance_ratio_standard_error_ = np.std(
            ratio_samples,
            axis=0,
            ddof=1,
        )
        self.stable_loading_mask_ = (
            self.loading_interval_lower_ * self.loading_interval_upper_ > 0.0
        )
        self.median_max_principal_angle_degrees_ = float(
            np.median(self.max_principal_angle_degrees_)
        )
        self.max_principal_angle_interval_degrees_ = np.quantile(
            self.max_principal_angle_degrees_,
            quantiles,
        )
        self.median_projection_distance_ = float(np.median(projection_samples))
        self.n_successful_resamples_ = n_successful
        self.min_successful_resamples_ = minimum_successful
        self.n_failed_resamples_ = len(failures)
        self.failure_messages_ = tuple(failures)

        if q > 1:
            gaps = reference_values[:-1] - reference_values[1:]
            scales = np.maximum(reference_values[:-1], np.finfo(float).eps)
            self.relative_eigenvalue_gaps_ = gaps / scales
        else:
            self.relative_eigenvalue_gaps_ = np.empty(0, dtype=float)

        return self

    def _validate_parameters(
        self,
        n_samples: int,
        groups: np.ndarray | None,
    ) -> np.ndarray | None:
        if isinstance(self.n_resamples, (bool, np.bool_)) or self.n_resamples < 1:
            raise ValueError("n_resamples must be a positive integer")
        if not isinstance(self.n_resamples, (int, np.integer)):
            raise TypeError("n_resamples must be an integer")
        if not np.isfinite(self.confidence_level) or not (
            0.0 < self.confidence_level < 1.0
        ):
            raise ValueError("confidence_level must be in (0, 1)")
        if not np.isfinite(self.sample_fraction) or not (
            0.0 < self.sample_fraction <= 1.0
        ):
            raise ValueError("sample_fraction must be in (0, 1]")
        if self.resampling not in _RESAMPLING_METHODS:
            allowed = ", ".join(sorted(_RESAMPLING_METHODS))
            raise ValueError(f"resampling must be one of: {allowed}")
        if self.alignment not in {"procrustes", "sign_permutation", "sign"}:
            raise ValueError(
                "alignment must be 'procrustes', 'sign_permutation', or 'sign'"
            )
        if self.min_successful_resamples is not None:
            if (
                isinstance(self.min_successful_resamples, (bool, np.bool_))
                or not isinstance(self.min_successful_resamples, (int, np.integer))
                or self.min_successful_resamples < 1
            ):
                raise ValueError(
                    "min_successful_resamples must be a positive integer or None"
                )
            if self.min_successful_resamples > self.n_resamples:
                raise ValueError(
                    "min_successful_resamples cannot exceed n_resamples"
                )
        if (
            self.resampling != "cluster"
            and int(np.ceil(self.sample_fraction * n_samples)) < 3
        ):
            raise ValueError("sample_fraction produces fewer than three rows")

        block_methods = {"moving_block", "circular_block", "stationary"}
        if self.block_length is not None:
            if self.resampling not in block_methods:
                raise ValueError(
                    "block_length is only valid for moving_block, "
                    "circular_block, or stationary resampling"
                )
            if (
                isinstance(self.block_length, (bool, np.bool_))
                or not isinstance(self.block_length, (int, np.integer))
            ):
                raise TypeError("block_length must be an integer or None")
            if self.block_length < 1 or self.block_length > n_samples:
                raise ValueError("block_length must be between 1 and n_samples")

        if self.resampling == "cluster":
            if groups is None:
                raise ValueError(
                    "groups must be provided when resampling='cluster'"
                )
            groups_array = np.asarray(groups)
            if groups_array.ndim != 1 or groups_array.shape[0] != n_samples:
                raise ValueError("groups must have shape (n_samples,)")
            if _ordered_unique(groups_array).size < 2:
                raise ValueError("cluster resampling requires at least two groups")
            return groups_array

        if groups is not None:
            raise ValueError(
                "groups is only used when resampling='cluster'"
            )
        return None

    def _align(self, reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
        if self.alignment == "procrustes":
            return _align_procrustes(reference, candidate)
        if self.alignment == "sign_permutation":
            return _align_sign_permutation(reference, candidate)
        return _align_signs(reference, candidate)

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "loading_samples_"):
            raise AttributeError("SubspaceStability is not fitted")

    def loading_interval(self, component: int) -> np.ndarray:
        """Return lower and upper loading bounds for one component."""
        self._check_is_fitted()
        component = int(component)
        if component < 0 or component >= self.n_components_:
            raise IndexError("component index is out of range")
        return self.loading_interval_[:, component, :].copy()

    def summary(self) -> str:
        """Return a compact human-readable stability summary."""
        self._check_is_fitted()
        stable_counts = np.sum(self.stable_loading_mask_, axis=1)
        return (
            f"SubspaceStability(n_components={self.n_components_}, "
            f"resampling={self.resampling_}, "
            f"successful={self.n_successful_resamples_}/{self.n_resamples}, "
            f"median_max_angle={self.median_max_principal_angle_degrees_:.2f} deg, "
            f"stable_loadings={stable_counts.tolist()})"
        )
