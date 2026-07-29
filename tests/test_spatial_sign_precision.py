import numpy as np
import pytest

import robustcov as rc


def _chain_precision(p, strength=0.28):
    precision = np.eye(p)
    for index in range(p - 1):
        precision[index, index + 1] = -strength
        precision[index + 1, index] = -strength
    return precision


def _edge_f1(adjacency, truth):
    predicted = np.triu(np.asarray(adjacency, dtype=bool), 1)
    expected = np.triu(np.asarray(truth, dtype=bool), 1)
    tp = np.count_nonzero(predicted & expected)
    fp = np.count_nonzero(predicted & ~expected)
    fn = np.count_nonzero(~predicted & expected)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def _symmetric_sample(rng, n_pairs=80, p=6):
    half = rng.normal(size=(n_pairs, p))
    return np.vstack([half, -half])


def test_alpha_zero_matches_inverse_scaled_spatial_sign_covariance():
    rng = np.random.default_rng(1)
    X = _symmetric_sample(rng, n_pairs=100, p=5)
    signs = X / np.linalg.norm(X, axis=1)[:, None]
    working = X.shape[1] * signs.T @ signs / X.shape[0]

    model = rc.SGLASSO(alpha=0.0, scatter_floor=1e-12).fit(X)

    np.testing.assert_allclose(model.location_, 0.0, atol=1e-12)
    np.testing.assert_allclose(model.spatial_sign_covariance_, working / X.shape[1], atol=1e-12)
    np.testing.assert_allclose(model.precision_, np.linalg.inv(working), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(model.covariance_, working, rtol=1e-9, atol=1e-9)
    assert model.n_iter_ == 0
    assert model.converged_


def test_spatial_sign_fit_is_translation_equivariant():
    rng = np.random.default_rng(2)
    X = _symmetric_sample(rng, n_pairs=70, p=5)
    shift = np.array([2.0, -1.0, 0.5, 3.0, -2.0])

    first = rc.SGLASSO(alpha=0.08, max_iter=800).fit(X)
    second = rc.SGLASSO(alpha=0.08, max_iter=800).fit(X + shift)

    np.testing.assert_allclose(second.location_, first.location_ + shift, atol=1e-8)
    np.testing.assert_allclose(second.precision_, first.precision_, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(
        second.partial_correlation_, first.partial_correlation_, atol=1e-7, rtol=1e-7
    )


def test_spatial_sign_graph_is_invariant_to_pairwise_radial_rescaling():
    rng = np.random.default_rng(3)
    half = rng.normal(size=(90, 7))
    scales = np.exp(rng.normal(scale=1.4, size=half.shape[0]))
    X = np.vstack([half, -half])
    X_scaled = np.vstack([half * scales[:, None], -half * scales[:, None]])

    first = rc.SGLASSO(alpha=0.10, max_iter=900).fit(X)
    second = rc.SGLASSO(alpha=0.10, max_iter=900).fit(X_scaled)

    np.testing.assert_allclose(first.spatial_sign_covariance_, second.spatial_sign_covariance_, atol=1e-10)
    np.testing.assert_allclose(first.precision_, second.precision_, atol=2e-7, rtol=2e-7)
    np.testing.assert_array_equal(first.adjacency_, second.adjacency_)


def test_fixed_penalty_produces_sparse_spd_shape_precision():
    rng = np.random.default_rng(4)
    truth_precision = _chain_precision(10, 0.28)
    covariance = np.linalg.inv(truth_precision)
    gaussian = rng.multivariate_normal(np.zeros(10), covariance, size=280)
    X = gaussian / np.sqrt(rng.chisquare(3.0, size=280) / 3.0)[:, None]

    model = rc.SGLASSO(alpha=0.12, max_iter=1000).fit(X)

    assert model.converged_
    assert np.linalg.eigvalsh(model.precision_).min() > 0.0
    assert 4 <= model.n_edges_ < 45
    assert np.all(model.adjacency_ == model.adjacency_.T)
    assert model.shape_ is model.covariance_
    assert model.shape_precision_ is model.precision_


def test_ebic_path_records_candidates_and_selects_one():
    rng = np.random.default_rng(5)
    X = rng.standard_t(df=3.0, size=(180, 9))

    model = rc.SGLASSO(
        alpha="ebic",
        n_alphas=11,
        alpha_min_ratio=0.04,
        max_iter=700,
    ).fit(X)

    assert model.alphas_.shape == (11,)
    assert model.ebic_scores_.shape == (11,)
    assert model.path_n_edges_.shape == (11,)
    assert model.alpha_ == pytest.approx(model.alphas_[model.best_alpha_index_])
    assert np.isfinite(model.ebic_scores_).all()
    assert np.all(np.diff(model.alphas_) < 0)


def test_spatial_sign_method_improves_graph_recovery_under_heavy_tails():
    rng = np.random.default_rng(6)
    p = 12
    truth_precision = _chain_precision(p, 0.30)
    covariance = np.linalg.inv(truth_precision)
    gaussian = rng.multivariate_normal(np.zeros(p), covariance, size=220)
    X = gaussian / np.sqrt(rng.chisquare(2.5, size=220) / 2.5)[:, None]

    empirical = rc.RobustGraphicalLasso(
        alpha=0.15,
        scatter_estimator="empirical",
        max_iter=900,
    ).fit(X)
    spatial = rc.SGLASSO(alpha=0.15, max_iter=900).fit(X)

    truth = np.abs(truth_precision) > 1e-12
    np.fill_diagonal(truth, False)
    assert _edge_f1(spatial.adjacency_, truth) > _edge_f1(empirical.adjacency_, truth) + 0.10


def test_high_dimensional_fit_is_positive_definite():
    rng = np.random.default_rng(7)
    X = rng.standard_t(df=3.0, size=(45, 70))

    model = rc.SGLASSO(
        alpha=0.12,
        max_iter=700,
        scatter_floor=1e-6,
    ).fit(X)

    assert model.precision_.shape == (70, 70)
    assert np.isfinite(model.precision_).all()
    assert np.linalg.eigvalsh(model.precision_).min() > 0.0
    assert model.n_samples_in_ < model.n_features_in_


def test_missing_values_raise_by_default_and_median_mode_works():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(90, 8))
    X[::9, 2] = np.nan
    X[::11, 6] = np.nan

    with pytest.raises(ValueError, match="missing_values='median'"):
        rc.SGLASSO(alpha=0.1).fit(X)

    model = rc.SGLASSO(alpha=0.1, missing_values="median").fit(X)
    assert np.isfinite(model.precision_).all()
    scores = model.shape_distances(X[:6])
    assert scores.shape == (6,)
    assert np.isfinite(scores).all()


def test_zero_sign_observations_are_recorded():
    X = np.vstack([
        np.zeros((3, 4)),
        np.eye(4),
        -np.eye(4),
    ])
    model = rc.SGLASSO(alpha=0.0, scatter_floor=1e-8).fit(X)
    assert model.zero_sign_count_ >= 3
    assert np.allclose(model.sign_vectors_[:3], 0.0)


def test_partial_correlations_conditional_coefficients_and_edge_list():
    rng = np.random.default_rng(9)
    X = rng.standard_t(df=4.0, size=(160, 6))
    model = rc.SGLASSO(alpha=0.10, max_iter=700).fit(X)

    expected = -model.precision_ / np.sqrt(
        np.outer(np.diag(model.precision_), np.diag(model.precision_))
    )
    np.fill_diagonal(expected, 1.0)
    np.testing.assert_allclose(model.partial_correlation_, expected)

    coefficients = -model.precision_ / np.diag(model.precision_)[:, None]
    np.fill_diagonal(coefficients, 0.0)
    np.testing.assert_allclose(model.conditional_coefficients_, coefficients)

    names = [f"x{index}" for index in range(6)]
    edges = model.edge_list(names)
    assert len(edges) == model.n_edges_
    assert all(abs(edges[i][2]) >= abs(edges[i + 1][2]) for i in range(len(edges) - 1))


def test_penalize_diagonal_changes_fitted_shape_but_not_api():
    rng = np.random.default_rng(10)
    X = rng.standard_t(df=3.0, size=(220, 7))
    paper = rc.SGLASSO(alpha=0.10, penalize_diagonal=True).fit(X)
    conventional = rc.SGLASSO(alpha=0.10, penalize_diagonal=False).fit(X)

    assert not np.allclose(np.diag(paper.precision_), np.diag(conventional.precision_))
    assert np.linalg.eigvalsh(paper.precision_).min() > 0.0
    assert np.linalg.eigvalsh(conventional.precision_).min() > 0.0


def test_shape_distances_and_pseudo_scores_order_observations():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(150, 5))
    model = rc.SGLASSO(alpha=0.08).fit(X)
    probe = np.vstack([model.location_, model.location_ + 5.0])

    distances = model.mahalanobis(probe)
    np.testing.assert_allclose(distances, model.shape_distances(probe))
    assert distances[0] < distances[1]
    assert model.score_samples(probe)[0] > model.score_samples(probe)[1]


def test_aliases_are_public():
    assert rc.SGLASSO is rc.SpatialSignGraphicalLasso
    assert rc.SpatialSignSparsePrecision is rc.SpatialSignGraphicalLasso


def test_plot_partial_correlation_network_accepts_spatial_sign_model(tmp_path):
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(12)
    X = rng.standard_t(df=3.0, size=(140, 6))
    model = rc.SGLASSO(alpha=0.1).fit(X)
    output = tmp_path / "spatial_network.png"

    figure = rc.plot_partial_correlation_network(
        model,
        feature_names=[f"f{i}" for i in range(6)],
        output_path=output,
        show=False,
    )

    assert output.exists()
    assert figure is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": -0.1},
        {"alpha": "cv"},
        {"penalize_diagonal": 1},
        {"missing_values": "drop"},
        {"n_alphas": 1},
        {"alpha_min_ratio": 0.0},
        {"ebic_gamma": -0.1},
        {"rho": 0.0},
        {"max_iter": 0},
        {"scatter_floor": 0.0},
        {"edge_tolerance": -1.0},
        {"spatial_median_tol": 0.0},
        {"spatial_median_max_iter": 0},
        {"zero_tolerance": 0.0},
    ],
)
def test_invalid_parameters(kwargs):
    X = np.arange(60, dtype=float).reshape(15, 4)
    with pytest.raises((ValueError, TypeError)):
        rc.SGLASSO(**kwargs).fit(X)


def test_unfitted_and_feature_validation():
    model = rc.SGLASSO(alpha=0.1)
    with pytest.raises(AttributeError):
        model.mahalanobis(np.ones((2, 3)))

    fitted = rc.SGLASSO(alpha=0.1).fit(np.random.default_rng(13).normal(size=(40, 5)))
    with pytest.raises(ValueError):
        fitted.mahalanobis(np.ones((2, 4)))
    with pytest.raises(ValueError):
        fitted.mahalanobis(np.array([[1.0, np.nan, 2.0, 3.0, 4.0]]))
