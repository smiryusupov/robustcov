import numpy as np
import pytest

import robustcov as rc
from robustcov.stability import (
    _align_procrustes,
    _draw_resample_indices,
    _principal_angles,
)


class EmpiricalScatter:
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.location_ = X.mean(axis=0)
        centered = X - self.location_
        self.covariance_ = centered.T @ centered / X.shape[0]
        return self


class DuplicateSensitivePCA:
    n_components = 2

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if np.unique(X, axis=0).shape[0] < X.shape[0]:
            raise RuntimeError("duplicate bootstrap rows")
        centered = X - X.mean(axis=0)
        values, vectors = np.linalg.eigh(centered.T @ centered / X.shape[0])
        order = np.argsort(values)[::-1][: self.n_components]
        self.components_ = vectors[:, order].T
        self.eigenvalues_ = values[order]
        self.explained_variance_ratio_ = values[order] / np.sum(values)
        return self


def make_low_rank(seed=0, n=320, p=8, q=2, noise=0.08):
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(p, q)))
    scores = rng.normal(size=(n, q)) @ np.diag(np.linspace(3.0, 1.5, q))
    X = scores @ basis.T + rng.normal(scale=noise, size=(n, p))
    return X, basis.T


def test_procrustes_alignment_recovers_rotated_basis():
    rng = np.random.default_rng(10)
    reference, _ = np.linalg.qr(rng.normal(size=(7, 3)))
    reference = reference.T
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    candidate = rotation @ reference

    aligned = _align_procrustes(reference, candidate)

    assert np.allclose(aligned, reference, atol=1e-12)
    assert np.max(_principal_angles(reference, candidate)) < 1e-7


def test_subspace_stability_shapes_intervals_and_summary():
    X, _ = make_low_rank(seed=11)
    analysis = rc.SubspaceStability(
        pca=rc.RobustPCA(n_components=2, estimator=EmpiricalScatter()),
        n_resamples=35,
        random_state=0,
        min_successful_resamples=25,
    ).fit(X)

    assert analysis.loading_samples_.shape == (35, 2, X.shape[1])
    assert analysis.eigenvalue_samples_.shape == (35, 2)
    assert analysis.principal_angle_degrees_.shape == (35, 2)
    assert analysis.loading_interval_.shape == (2, 2, X.shape[1])
    assert analysis.eigenvalue_interval_.shape == (2, 2)
    assert analysis.explained_variance_ratio_interval_.shape == (2, 2)
    assert analysis.n_successful_resamples_ == 35
    assert analysis.n_failed_resamples_ == 0
    assert "median_max_angle=" in analysis.summary()
    assert analysis.loading_interval(0).shape == (2, X.shape[1])


def test_subspace_stability_is_deterministic_with_seed():
    X, _ = make_low_rank(seed=12)
    kwargs = dict(
        pca=rc.RobustPCA(n_components=2, estimator=EmpiricalScatter()),
        n_resamples=25,
        random_state=42,
        min_successful_resamples=20,
    )
    first = rc.SubspaceStability(**kwargs).fit(X)
    second = rc.SubspaceStability(**kwargs).fit(X)

    assert np.allclose(first.loading_samples_, second.loading_samples_)
    assert np.allclose(
        first.max_principal_angle_degrees_,
        second.max_principal_angle_degrees_,
    )


def test_well_separated_subspace_has_small_bootstrap_angles():
    X, _ = make_low_rank(seed=13, n=500, noise=0.03)
    analysis = rc.SubspaceStability(
        pca=rc.RobustPCA(n_components=2, estimator=EmpiricalScatter()),
        n_resamples=40,
        random_state=1,
        min_successful_resamples=30,
    ).fit(X)

    assert analysis.median_max_principal_angle_degrees_ < 2.0
    assert analysis.max_principal_angle_interval_degrees_[1] < 4.0
    assert analysis.median_projection_distance_ < 0.1


def test_robust_stability_improves_under_row_contamination():
    X_clean, true_components = make_low_rank(seed=14, n=280, p=6, noise=0.05)
    rng = np.random.default_rng(15)
    outliers = rng.normal(scale=0.1, size=(45, 6))
    outliers[:, 3:] += rng.normal(scale=12.0, size=(45, 3))
    X = np.vstack([X_clean, outliers])

    empirical = rc.SubspaceStability(
        pca=rc.RobustPCA(n_components=2, estimator=EmpiricalScatter()),
        n_resamples=30,
        random_state=2,
        min_successful_resamples=25,
    ).fit(X)
    robust = rc.SubspaceStability(
        pca=rc.RobustPCA(
            n_components=2,
            estimator=rc.FastMCD(n_init=35, random_state=0),
        ),
        n_resamples=30,
        random_state=2,
        min_successful_resamples=25,
    ).fit(X)

    empirical_error = np.linalg.norm(
        empirical.components_.T @ empirical.components_
        - true_components.T @ true_components,
        ord="fro",
    )
    robust_error = np.linalg.norm(
        robust.components_.T @ robust.components_
        - true_components.T @ true_components,
        ord="fro",
    )

    assert robust_error < empirical_error * 0.5
    assert (
        robust.median_max_principal_angle_degrees_
        < empirical.median_max_principal_angle_degrees_
    )


def test_float_component_selection_is_frozen_across_bootstraps():
    X, _ = make_low_rank(seed=16, p=7, q=3)
    analysis = rc.SubspaceStability(
        pca=rc.RobustPCA(n_components=0.80, estimator=EmpiricalScatter()),
        n_resamples=25,
        random_state=3,
        min_successful_resamples=20,
    ).fit(X)

    assert analysis.loading_samples_.shape[1] == analysis.n_components_
    assert analysis.eigenvalue_samples_.shape[1] == analysis.n_components_


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"n_resamples": 0}, ValueError),
        ({"n_resamples": 2.5}, TypeError),
        ({"confidence_level": 1.0}, ValueError),
        ({"sample_fraction": 0.0}, ValueError),
        ({"alignment": "unknown"}, ValueError),
        ({"n_resamples": 5, "min_successful_resamples": 6}, ValueError),
    ],
)
def test_subspace_stability_parameter_validation(kwargs, error):
    X, _ = make_low_rank(seed=17, n=40)
    with pytest.raises(error):
        rc.SubspaceStability(**kwargs).fit(X)


def test_subspace_stability_reports_excessive_fit_failures():
    X, _ = make_low_rank(seed=18, n=40)
    with pytest.raises(RuntimeError, match="too few successful"):
        rc.SubspaceStability(
            pca=DuplicateSensitivePCA(),
            n_resamples=5,
            min_successful_resamples=1,
        ).fit(X)


def test_subspace_stability_unfitted_and_component_errors():
    analysis = rc.SubspaceStability()
    with pytest.raises(AttributeError, match="not fitted"):
        analysis.summary()

    X, _ = make_low_rank(seed=19)
    analysis = rc.SubspaceStability(
        pca=rc.RobustPCA(n_components=2, estimator=EmpiricalScatter()),
        n_resamples=20,
        random_state=0,
        min_successful_resamples=15,
    ).fit(X)
    with pytest.raises(IndexError):
        analysis.loading_interval(3)


def test_plot_subspace_stability(tmp_path):
    X, _ = make_low_rank(seed=20)
    analysis = rc.SubspaceStability(
        pca=rc.RobustPCA(n_components=2, estimator=EmpiricalScatter()),
        n_resamples=20,
        random_state=0,
        min_successful_resamples=15,
    ).fit(X)
    output = tmp_path / "stability.png"

    figure = rc.plot_subspace_stability(
        analysis,
        component=0,
        feature_names=[f"x{i}" for i in range(X.shape[1])],
        output_path=output,
        show=False,
    )

    assert output.exists()
    assert figure is not None


@pytest.mark.parametrize("method", ["moving_block", "circular_block", "stationary"])
def test_dependent_resampling_fit_is_deterministic(method):
    X, _ = make_low_rank(seed=21, n=90)
    kwargs = dict(
        pca=rc.RobustPCA(n_components=2, estimator=EmpiricalScatter()),
        n_resamples=20,
        resampling=method,
        block_length=7,
        random_state=123,
        min_successful_resamples=15,
    )
    first = rc.SubspaceStability(**kwargs).fit(X)
    second = rc.SubspaceStability(**kwargs).fit(X)

    assert first.resampling_ == method
    assert first.block_length_ == 7
    assert np.all(first.bootstrap_sample_sizes_ == X.shape[0])
    assert np.allclose(first.loading_samples_, second.loading_samples_)


def test_moving_block_indices_preserve_consecutive_rows_within_blocks():
    indices = _draw_resample_indices(
        np.random.default_rng(0),
        n_samples=20,
        sample_size=12,
        method="moving_block",
        block_length=4,
    )
    assert indices.shape == (12,)
    for start in range(0, 12, 4):
        assert np.all(np.diff(indices[start : start + 4]) == 1)


def test_circular_block_indices_can_wrap_at_sample_boundary():
    found_wrap = False
    for seed in range(50):
        indices = _draw_resample_indices(
            np.random.default_rng(seed),
            n_samples=8,
            sample_size=16,
            method="circular_block",
            block_length=4,
        )
        if np.any((indices[:-1] == 7) & (indices[1:] == 0)):
            found_wrap = True
            break
    assert found_wrap


def test_stationary_bootstrap_contains_runs_and_restarts():
    indices = _draw_resample_indices(
        np.random.default_rng(4),
        n_samples=30,
        sample_size=200,
        method="stationary",
        block_length=6,
    )
    consecutive = (indices[1:] - indices[:-1]) % 30 == 1
    assert np.mean(consecutive) > 0.65
    assert np.any(~consecutive)


def test_cluster_resampling_keeps_complete_groups():
    X, _ = make_low_rank(seed=22, n=48)
    groups = np.repeat(np.arange(8), 6)
    analysis = rc.SubspaceStability(
        pca=rc.RobustPCA(n_components=2, estimator=EmpiricalScatter()),
        n_resamples=20,
        sample_fraction=0.75,
        resampling="cluster",
        random_state=3,
        min_successful_resamples=15,
    ).fit(X, groups=groups)

    assert analysis.resampling_ == "cluster"
    assert analysis.n_clusters_in_ == 8
    assert analysis.bootstrap_cluster_count_ == 6
    assert analysis.bootstrap_sample_size_ is None
    assert np.all(analysis.bootstrap_sample_sizes_ == 36)


@pytest.mark.parametrize(
    "kwargs,fit_kwargs,error,match",
    [
        ({"resampling": "unknown"}, {}, ValueError, "resampling"),
        ({"resampling": "iid", "block_length": 3}, {}, ValueError, "block_length"),
        ({"resampling": "moving_block", "block_length": 0}, {}, ValueError, "block_length"),
        ({"resampling": "moving_block", "block_length": 2.5}, {}, TypeError, "block_length"),
        ({"resampling": "cluster"}, {}, ValueError, "groups"),
        ({"resampling": "iid"}, {"groups": np.arange(40)}, ValueError, "groups"),
        ({"resampling": "cluster"}, {"groups": np.zeros(40)}, ValueError, "two groups"),
    ],
)
def test_dependent_resampling_parameter_validation(kwargs, fit_kwargs, error, match):
    X, _ = make_low_rank(seed=23, n=40)
    with pytest.raises(error, match=match):
        rc.SubspaceStability(n_resamples=5, **kwargs).fit(X, **fit_kwargs)


def test_stationary_bootstrap_reflects_serial_dependence_in_eigenvalue_uncertainty():
    rng = np.random.default_rng(30)
    n_samples, n_features, n_components = 240, 6, 2
    basis, _ = np.linalg.qr(rng.normal(size=(n_features, n_components)))
    factors = np.zeros((n_samples, n_components))
    innovations = rng.normal(size=(n_samples, n_components))
    for row in range(1, n_samples):
        factors[row] = 0.93 * factors[row - 1] + innovations[row]
    X = (
        factors @ np.diag([2.0, 1.0]) @ basis.T
        + rng.normal(scale=0.15, size=(n_samples, n_features))
    )

    common = dict(
        pca=rc.RobustPCA(n_components=2, estimator=EmpiricalScatter()),
        n_resamples=50,
        confidence_level=0.90,
        random_state=7,
        min_successful_resamples=40,
    )
    iid = rc.SubspaceStability(resampling="iid", **common).fit(X)
    stationary = rc.SubspaceStability(
        resampling="stationary",
        block_length=14,
        **common,
    ).fit(X)

    assert stationary.eigenvalue_standard_error_[0] > 1.5 * iid.eigenvalue_standard_error_[0]
