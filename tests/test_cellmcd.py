import numpy as np
import pytest

import robustcov as rc


def make_data(seed=0, n=120, p=5, n_cells=18, missing=4):
    rng = np.random.default_rng(seed)
    covariance = 0.45 * np.ones((p, p)) + 0.55 * np.eye(p)
    clean = rng.multivariate_normal(np.zeros(p), covariance, size=n)
    contaminated = clean.copy()
    outlier_mask = np.zeros_like(contaminated, dtype=bool)
    choices = rng.choice(n * p, size=n_cells, replace=False)
    outlier_mask.flat[choices] = True
    contaminated[outlier_mask] += rng.choice([-9.0, 9.0], size=n_cells)
    missing_mask = np.zeros_like(contaminated, dtype=bool)
    available = np.flatnonzero(~outlier_mask.ravel())
    missing_choices = rng.choice(available, size=missing, replace=False)
    missing_mask.flat[missing_choices] = True
    contaminated[missing_mask] = np.nan
    return clean, contaminated, covariance, outlier_mask, missing_mask


def fitted_model(seed=0):
    _, X, _, _, _ = make_data(seed=seed)
    return rc.CellMCD(max_iter=60, tol=1e-5).fit(X)


def test_public_aliases():
    assert rc.CellMCD is rc.CellwiseMinimumCovarianceDeterminant
    assert rc.CellwiseMCD is rc.CellwiseMinimumCovarianceDeterminant


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0.49},
        {"alpha": 1.01},
        {"quantile": 0.5},
        {"quantile": 1.0},
        {"max_iter": 0},
        {"tol": 0.0},
        {"min_eigenvalue": 1e-10},
        {"min_samples_per_feature": 0.0},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        rc.CellMCD(**kwargs)


def test_fit_shapes_and_positive_definiteness():
    model = fitted_model()
    assert model.location_.shape == (5,)
    assert model.covariance_.shape == (5, 5)
    assert model.precision_.shape == (5, 5)
    assert model.cell_support_.shape == (120, 5)
    assert model.standardized_residuals_.shape == (120, 5)
    assert model.imputed_data_.shape == (120, 5)
    assert np.linalg.eigvalsh(model.covariance_).min() > 0.0
    assert np.isfinite(model.imputed_data_).all()


def test_column_support_constraint():
    model = fitted_model()
    assert np.all(model.cell_support_.sum(axis=0) >= model.h_)


def test_missing_cells_are_not_reported_as_outliers():
    _, X, _, _, missing = make_data()
    model = rc.CellMCD().fit(X)
    assert np.array_equal(model.missing_mask_, missing)
    assert not np.any(model.cell_outlier_mask_ & missing)
    assert np.isfinite(model.imputed_data_[missing]).all()


def test_objective_is_monotone_nonincreasing():
    model = fitted_model()
    differences = np.diff(model.objective_history_)
    assert np.all(differences <= 1e-7 * np.maximum(1.0, np.abs(model.objective_history_[:-1])))


def test_detects_injected_cell_outliers():
    _, X, _, truth, _ = make_data(n_cells=24)
    model = rc.CellMCD(max_iter=60).fit(X)
    recall = np.count_nonzero(model.cell_outlier_mask_ & truth) / truth.sum()
    assert recall >= 0.70


def test_covariance_improves_over_naive_zero_filled_empirical():
    _, X, covariance, _, _ = make_data(n_cells=28)
    model = rc.CellMCD(max_iter=60).fit(X)
    zero_filled = np.nan_to_num(X, nan=0.0)
    empirical = np.cov(zero_filled, rowvar=False)
    robust_error = np.linalg.norm(model.covariance_ - covariance, ord="fro")
    empirical_error = np.linalg.norm(empirical - covariance, ord="fro")
    assert robust_error < empirical_error


def test_location_and_scale_equivariance():
    _, X, _, _, _ = make_data(seed=2)
    model = rc.CellMCD(max_iter=60).fit(X)
    shift = np.array([2.0, -1.0, 0.5, 3.0, -2.0])
    scale = np.array([2.0, 0.5, 1.5, 3.0, 0.8])
    transformed = X * scale + shift
    other = rc.CellMCD(max_iter=60).fit(transformed)
    assert np.allclose(other.location_, model.location_ * scale + shift, atol=2e-4, rtol=2e-4)
    expected_cov = model.covariance_ * np.outer(scale, scale)
    assert np.allclose(other.covariance_, expected_cov, atol=4e-4, rtol=4e-4)
    assert np.array_equal(other.cell_outlier_mask_, model.cell_outlier_mask_)


def test_new_data_diagnostics_and_transform():
    model = fitted_model()
    rng = np.random.default_rng(10)
    X = rng.normal(size=(12, 5))
    X[2, 1] = 12.0
    X[4, 3] = np.nan
    diagnostics = model.cellwise_diagnostics(X)
    assert diagnostics["cell_outlier_mask"].shape == X.shape
    assert diagnostics["standardized_residuals"].shape == X.shape
    assert diagnostics["cell_outlier_mask"][2, 1]
    assert not diagnostics["cell_outlier_mask"][4, 3]
    corrected = model.transform(X)
    assert np.isfinite(corrected).all()
    assert corrected[2, 1] != X[2, 1]


def test_transform_can_keep_outliers_and_only_impute_missing():
    model = fitted_model()
    X = np.zeros((4, 5))
    X[0, 0] = 20.0
    X[1, 1] = np.nan
    transformed = model.transform(X, replace_outliers=False)
    assert transformed[0, 0] == X[0, 0]
    assert np.isfinite(transformed[1, 1])


def test_cell_scores_match_absolute_residuals():
    model = fitted_model()
    X = np.zeros((4, 5))
    scores = model.cell_scores(X)
    residuals = model.cellwise_diagnostics(X)["standardized_residuals"]
    assert np.allclose(scores, np.abs(residuals), equal_nan=True)


def test_partial_mahalanobis_is_finite_with_missing_cells():
    model = fitted_model()
    X = np.zeros((5, 5))
    X[0, :2] = np.nan
    distances = model.mahalanobis(X)
    labels = model.predict(X)
    assert distances.shape == (5,)
    assert np.isfinite(distances).all()
    assert set(np.unique(labels)).issubset({-1, 1})


def test_fit_predict_works():
    _, X, _, _, _ = make_data()
    labels = rc.CellMCD().fit_predict(X)
    assert labels.shape == (X.shape[0],)


def test_rejects_too_few_finite_cells_per_column():
    X = np.random.default_rng(0).normal(size=(50, 4))
    X[:20, 0] = np.nan
    with pytest.raises(ValueError, match="at least h finite cells"):
        rc.CellMCD(alpha=0.75).fit(X)


def test_sample_to_feature_guard_and_override():
    X = np.random.default_rng(0).normal(size=(18, 5))
    with pytest.raises(ValueError, match="n / p"):
        rc.CellMCD().fit(X)
    model = rc.CellMCD(min_samples_per_feature=None, max_iter=5).fit(X)
    assert model.covariance_.shape == (5, 5)


def test_rejects_constant_feature():
    X = np.random.default_rng(0).normal(size=(80, 4))
    X[:, 2] = 1.0
    with pytest.raises(ValueError, match="nonzero robust scale"):
        rc.CellMCD().fit(X)


def test_plot_cellwise_residual_map(tmp_path):
    pytest.importorskip("matplotlib")
    model = fitted_model()
    output = tmp_path / "cellmap.png"
    fig = rc.plot_cellwise_residual_map(model, output_path=output, show=False)
    assert fig is not None
    assert output.exists()


def test_unfitted_methods_raise():
    model = rc.CellMCD()
    X = np.zeros((4, 2))
    with pytest.raises(RuntimeError):
        model.cellwise_diagnostics(X)


def _rowwise_em_update(X, W, location, covariance, minimum_eigenvalue):
    import robustcov.cellmcd as cellmcd

    n, p = X.shape
    conditional_means = np.empty((n, p), dtype=np.float64)
    covariance_bias = np.zeros((p, p), dtype=np.float64)
    for i in range(n):
        observed = np.flatnonzero(W[i])
        missing = np.flatnonzero(~W[i])
        conditional_means[i, observed] = X[i, observed]
        mean_missing, conditional_cov = cellmcd._conditional_parameters(
            location, covariance, missing, observed, X[i, observed]
        )
        conditional_means[i, missing] = mean_missing
        if missing.size:
            covariance_bias[np.ix_(missing, missing)] += conditional_cov
    new_location = conditional_means.mean(axis=0)
    centered = conditional_means - new_location
    new_covariance = (centered.T @ centered + covariance_bias) / n
    new_covariance = cellmcd._truncate_covariance(
        new_covariance, minimum_eigenvalue
    )
    return new_location, new_covariance, conditional_means


def test_grouped_em_update_matches_rowwise_reference():
    import robustcov.cellmcd as cellmcd

    rng = np.random.default_rng(321)
    X = rng.normal(size=(48, 7))
    W = np.ones_like(X, dtype=bool)
    patterns = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 0, 1, 1],
            [0, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    W[:] = patterns[np.arange(X.shape[0]) % patterns.shape[0]]
    a = rng.normal(size=(X.shape[1], X.shape[1]))
    covariance = a @ a.T + 0.5 * np.eye(X.shape[1])
    location = rng.normal(size=X.shape[1])

    expected = _rowwise_em_update(X, W, location, covariance, 1e-7)
    actual = cellmcd._em_update(X, W, location, covariance, 1e-7)
    for expected_item, actual_item in zip(expected, actual):
        assert np.allclose(actual_item, expected_item, rtol=1e-12, atol=1e-12)


def test_grouped_cellmcd_diagnostics_match_scalar_conditionals():
    import robustcov.cellmcd as cellmcd

    model = fitted_model(seed=13)
    X = model.imputed_data_[:18].copy()
    X[1, 2] = np.nan
    X[4, 0] = np.nan
    support = np.isfinite(X)
    support[2::4, 1] = False
    support[3::5, 3] = False

    predictions, conditional_std, residuals = model._diagnostics_with_support(
        X, support
    )
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            observed = np.flatnonzero(
                support[i] & (np.arange(X.shape[1]) != j)
            )
            mean, conditional = cellmcd._conditional_parameters(
                model.location_,
                model.covariance_,
                np.array([j]),
                observed,
                X[i, observed],
            )
            expected_std = np.sqrt(
                max(float(conditional[0, 0]), np.finfo(float).tiny)
            )
            assert predictions[i, j] == pytest.approx(mean[0], rel=1e-12, abs=1e-12)
            assert conditional_std[i, j] == pytest.approx(
                expected_std, rel=1e-12, abs=1e-12
            )
            if np.isfinite(X[i, j]):
                expected_residual = (X[i, j] - mean[0]) / expected_std
                assert residuals[i, j] == pytest.approx(
                    expected_residual, rel=1e-12, abs=1e-12
                )
            else:
                assert np.isnan(residuals[i, j])

    expected_distances = np.asarray(
        [
            cellmcd._partial_distance(
                row, mask, model.location_, model.covariance_
            )
            for row, mask in zip(X, support)
        ]
    )
    actual_distances = model._partial_distances(X, support)
    assert np.allclose(actual_distances, expected_distances, rtol=1e-12, atol=1e-12)
