import numpy as np
import pytest

import robustcov as rc


def _sample_matrix_normal(rng, n, mean, row_cov, col_cov):
    row_root = np.linalg.cholesky(row_cov)
    col_root = np.linalg.cholesky(col_cov)
    noise = rng.normal(size=(n, mean.shape[0], mean.shape[1]))
    return np.asarray([mean + row_root @ z @ col_root.T for z in noise])


def _fit_fast(**kwargs):
    defaults = dict(
        quality="fast",
        n_init=18,
        n_best=4,
        initial_c_steps=2,
        max_iter=30,
        flip_flop_max_iter=60,
        random_state=0,
    )
    defaults.update(kwargs)
    return rc.MMCD(**defaults)


def test_mmcd_fits_matrix_data_and_returns_spd_factors():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(70, 4, 6))
    X[:8, :2, :3] += 8.0

    est = _fit_fast(contamination=0.15).fit(X)

    assert est.location_.shape == (4, 6)
    assert est.row_covariance_.shape == (4, 4)
    assert est.column_covariance_.shape == (6, 6)
    assert np.linalg.eigvalsh(est.row_covariance_).min() > 0
    assert np.linalg.eigvalsh(est.column_covariance_).min() > 0
    assert est.raw_support_.sum() == est.h_
    assert np.isfinite(est.distances_).all()


def test_mmcd_resists_shifted_matrix_outliers():
    rng = np.random.default_rng(2)
    row = 0.3 * np.ones((4, 4)) + 0.7 * np.eye(4)
    col = 0.4 ** np.abs(np.subtract.outer(np.arange(5), np.arange(5)))
    clean = _sample_matrix_normal(rng, 100, np.zeros((4, 5)), row, col)
    outliers = _sample_matrix_normal(rng, 18, np.full((4, 5), 7.0), row, col)
    X = np.concatenate([clean, outliers])

    est = _fit_fast(contamination=0.17, n_init=30).fit(X)

    assert np.linalg.norm(est.location_) < 0.3 * np.linalg.norm(X.mean(axis=0))
    assert np.count_nonzero(est.raw_support_[100:]) <= 1
    assert np.median(est.distances_[100:]) > 5 * np.median(est.distances_[:100])


def test_mmcd_is_deterministic_for_fixed_seed():
    X = np.random.default_rng(3).standard_t(df=4, size=(75, 3, 5))
    first = _fit_fast(random_state=7).fit(X)
    second = _fit_fast(random_state=7).fit(X)

    np.testing.assert_array_equal(first.raw_support_, second.raw_support_)
    np.testing.assert_allclose(first.location_, second.location_)
    np.testing.assert_allclose(first.row_covariance_, second.row_covariance_)
    np.testing.assert_allclose(first.column_covariance_, second.column_covariance_)


def test_matrix_distance_matches_vectorized_kronecker_distance():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(80, 3, 4))
    est = _fit_fast(support_fraction=0.8, reweight=False).fit(X)

    covariance = est.kronecker_covariance()
    precision = np.linalg.inv(covariance)
    vectorized = np.asarray(
        [(x - est.location_).reshape(-1, order="F") for x in X]
    )
    expected = np.einsum("ni,ij,nj->n", vectorized, precision, vectorized)
    np.testing.assert_allclose(est.mahalanobis(X), expected, rtol=1e-8, atol=1e-9)


def test_signed_contributions_sum_to_distance():
    X = np.random.default_rng(5).normal(size=(65, 4, 3))
    est = _fit_fast(support_fraction=0.75).fit(X)

    cell = est.cell_contributions(X[:7])
    row = est.row_contributions(X[:7])
    column = est.column_contributions(X[:7])
    distance = est.mahalanobis(X[:7])

    np.testing.assert_allclose(cell.sum(axis=(1, 2)), distance)
    np.testing.assert_allclose(row.sum(axis=1), distance)
    np.testing.assert_allclose(column.sum(axis=1), distance)


def test_whitened_frobenius_norm_equals_distance():
    X = np.random.default_rng(6).normal(size=(70, 3, 5))
    est = _fit_fast(support_fraction=0.8).fit(X)
    whitened = est.whiten(X[:10])
    np.testing.assert_allclose(
        np.sum(whitened**2, axis=(1, 2)),
        est.mahalanobis(X[:10]),
        rtol=1e-8,
        atol=1e-9,
    )


def test_objective_path_does_not_increase():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(90, 4, 4))
    X[:10, :2] += 5.0
    est = _fit_fast(n_init=25).fit(X)
    assert np.all(np.diff(est.objective_path_) <= 1e-8)


def test_matrix_affine_equivariance_without_ridge():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(120, 2, 3))
    A = np.array([[1.5, 0.2], [-0.1, 0.8]])
    B = np.array([[1.2, 0.1, 0.0], [0.2, 0.9, -0.1], [0.0, 0.3, 1.1]])
    shift = np.arange(6, dtype=float).reshape(2, 3) / 4
    Y = np.asarray([A @ x @ B.T + shift for x in X])

    est_x = _fit_fast(support_fraction=0.8, ridge=0.0, reweight=False).fit(X)
    est_y = _fit_fast(support_fraction=0.8, ridge=0.0, reweight=False).fit(Y)

    np.testing.assert_array_equal(est_x.raw_support_, est_y.raw_support_)
    np.testing.assert_allclose(est_y.location_, A @ est_x.location_ @ B.T + shift, atol=1e-7)
    transform = np.kron(B, A)
    expected = transform @ est_x.kronecker_covariance() @ transform.T
    np.testing.assert_allclose(est_y.kronecker_covariance(), expected, rtol=2e-6, atol=2e-7)


def test_missing_value_cellwise_median_imputation():
    X = np.random.default_rng(9).normal(size=(60, 3, 4))
    X[0, 1, 2] = np.nan
    X[4, 2, 0] = np.nan
    est = _fit_fast(missing_values="median").fit(X)
    assert np.isfinite(est.location_).all()
    assert est.impute_values_.shape == (3, 4)
    assert np.isfinite(est.mahalanobis(X[:5])).all()


def test_predict_and_score_samples():
    X = np.random.default_rng(10).normal(size=(65, 3, 3))
    est = _fit_fast(support_fraction=0.8).fit(X)
    labels = est.predict(X)
    assert set(np.unique(labels)) <= {-1, 1}
    np.testing.assert_allclose(est.score_samples(X), -0.5 * est.mahalanobis(X))


def test_support_is_adapted_to_matrix_dimensions():
    X = np.random.default_rng(11).normal(size=(12, 2, 10))
    est = _fit_fast(support_fraction=0.5).fit(X)
    assert est.h_ >= est.elemental_size_
    assert est.effective_support_fraction_ >= 0.5


def test_support_adaptation_can_be_disabled():
    X = np.random.default_rng(12).normal(size=(12, 2, 10))
    with pytest.raises(ValueError, match="support is too small"):
        _fit_fast(support_fraction=0.5, adapt_support=False).fit(X)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"support_fraction": 0.4},
        {"contamination": 0.5},
        {"ridge": -1e-3},
        {"reweight_alpha": 0.5},
        {"quality": "unknown"},
        {"missing_values": "ignore"},
        {"n_init": 0},
    ],
)
def test_mmcd_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        _fit_fast(**kwargs)


def test_mmcd_rejects_wrong_input_shape_and_too_few_observations():
    est = _fit_fast()
    with pytest.raises(ValueError, match="shape"):
        est.fit(np.ones((20, 4)))
    with pytest.raises(ValueError, match="Too few"):
        est.fit(np.ones((3, 2, 10)))


def test_unfitted_methods_raise():
    est = _fit_fast()
    with pytest.raises(RuntimeError):
        est.mahalanobis(np.ones((3, 2, 2)))
    with pytest.raises(RuntimeError):
        est.kronecker_covariance()


def test_matrix_contribution_plot(tmp_path):
    import matplotlib.pyplot as plt

    X = np.random.default_rng(13).normal(size=(50, 3, 4))
    est = _fit_fast(support_fraction=0.8).fit(X)
    path = tmp_path / "contributions.png"
    fig = rc.plot_matrix_outlier_contributions(
        est,
        X,
        index=2,
        row_labels=["a", "b", "c"],
        column_labels=["t0", "t1", "t2", "t3"],
        output_path=path,
        show=False,
    )
    assert path.exists()
    plt.close(fig)


def test_support_fraction_one_matches_sample_mean():
    X = np.random.default_rng(14).normal(size=(45, 3, 4))
    est = _fit_fast(
        support_fraction=1.0,
        n_init=1,
        n_best=1,
        initial_c_steps=0,
        max_iter=1,
        reweight=False,
    ).fit(X)
    np.testing.assert_allclose(est.location_, X.mean(axis=0), atol=1e-12)
    assert est.raw_support_.all()
