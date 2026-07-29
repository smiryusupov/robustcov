import numpy as np
import pytest

import robustcov as rc


def make_low_rank_data(seed=0, n=90, p=45, q=3):
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(p, q)))
    scores = rng.normal(size=(n, q)) * np.array([3.0, 2.0, 1.2])
    residual_cov = 0.18 * np.eye(p)
    clean = scores @ basis.T + rng.multivariate_normal(
        np.zeros(p), residual_cov, size=n
    )
    truth_cov = basis @ np.diag([9.0, 4.0, 1.44]) @ basis.T + residual_cov
    return rng, clean, truth_cov


def damage_data(rng, clean, cell_fraction=0.06, case_fraction=0.12, missing=0.04):
    X = clean.copy()
    n, p = X.shape
    cell_truth = np.zeros_like(X, dtype=bool)
    bad = rng.choice(X.size, size=max(1, int(cell_fraction * X.size)), replace=False)
    rows, cols = np.unravel_index(bad, X.shape)
    X[rows, cols] += rng.choice([-1.0, 1.0], size=bad.size) * 8.0
    cell_truth[rows, cols] = True

    case_truth = np.zeros(n, dtype=bool)
    case_rows = rng.choice(n, size=max(1, int(case_fraction * n)), replace=False)
    case_truth[case_rows] = True
    direction = rng.normal(size=p)
    direction /= np.linalg.norm(direction)
    X[case_rows] += 7.0 * direction

    missing_truth = np.zeros_like(X, dtype=bool)
    if missing > 0.0:
        available = np.flatnonzero(~cell_truth.ravel())
        missing_flat = rng.choice(
            available, size=max(1, int(missing * X.size)), replace=False
        )
        mr, mc = np.unravel_index(missing_flat, X.shape)
        X[mr, mc] = np.nan
        missing_truth[mr, mc] = True
    return X, cell_truth, case_truth, missing_truth


def small_model(**kwargs):
    return rc.CellRCov(
        n_components=3,
        residual_shrinkage=0.35,
        cell_pca=rc.CellPCA(n_components=3, max_iter=45, tol=1e-5),
        score_estimator=rc.FastMCD(
            support_fraction=0.75,
            quality="fast",
            n_init=30,
            n_best=3,
            initial_c_steps=1,
            max_iter=45,
            random_state=0,
            scale_correction="none",
        ),
        **kwargs,
    )


def test_aliases_are_public():
    assert rc.CellRCov is rc.CellwiseRegularizedCovariance
    assert rc.CellwiseRobustCovariance is rc.CellwiseRegularizedCovariance


def test_fit_high_dimensional_and_missing():
    rng, clean, _ = make_low_rank_data(n=55, p=70)
    X, _, _, _ = damage_data(rng, clean)
    model = small_model().fit(X)
    assert model.covariance_.shape == (70, 70)
    assert model.precision_.shape == (70, 70)
    assert np.linalg.eigvalsh(model.covariance_).min() > 0.0
    assert np.isfinite(model.covariance_).all()
    assert model.distances_.shape == (55,)


def test_covariance_decomposition_in_standardized_coordinates():
    rng, clean, _ = make_low_rank_data(n=70, p=30)
    X, _, _, _ = damage_data(rng, clean)
    model = small_model().fit(X)
    expected = model.fitted_covariance_ + model.residual_covariance_regularized_
    assert np.allclose(model.standardized_covariance_, expected, atol=2e-6)


def test_fixed_residual_shrinkage_preserves_diagonal():
    rng, clean, _ = make_low_rank_data(n=70, p=25)
    X, _, _, _ = damage_data(rng, clean)
    model = small_model().fit(X)
    assert model.residual_shrinkage_ == pytest.approx(0.35)
    assert np.allclose(
        np.diag(model.residual_covariance_),
        np.diag(model.residual_covariance_regularized_),
    )


def test_auto_shrinkage_selects_grid_value():
    rng, clean, _ = make_low_rank_data(n=72, p=28)
    X, _, _, _ = damage_data(rng, clean)
    model = rc.CellRCov(
        n_components=3,
        residual_shrinkage="auto",
        shrinkage_grid=[0.0, 0.4, 0.8, 1.0],
        cv_splits=4,
        cell_pca=rc.CellPCA(n_components=3, max_iter=35),
        score_estimator=rc.FastMCD(
            support_fraction=0.75,
            quality="fast",
            n_init=25,
            n_best=3,
            initial_c_steps=1,
            max_iter=40,
            random_state=0,
            scale_correction="none",
        ),
    ).fit(X)
    assert model.residual_shrinkage_ in {0.0, 0.4, 0.8, 1.0}
    assert model.shrinkage_cv_scores_.shape == (4,)
    assert np.isfinite(model.shrinkage_cv_scores_).all()


def test_scale_equivariance():
    rng, clean, _ = make_low_rank_data(n=75, p=22)
    X, _, _, _ = damage_data(rng, clean, missing=0.0)
    scales = np.linspace(0.5, 2.0, X.shape[1])
    first = small_model().fit(X)
    second = small_model().fit(X * scales)
    expected_cov = scales[:, None] * first.covariance_ * scales[None, :]
    assert np.allclose(second.location_, first.location_ * scales, rtol=2e-4, atol=2e-4)
    assert np.allclose(second.covariance_, expected_cov, rtol=4e-3, atol=4e-3)


def test_training_diagnostics_and_corrections():
    rng, clean, _ = make_low_rank_data(n=70, p=24)
    X, cell_truth, _, missing_truth = damage_data(rng, clean)
    model = small_model().fit(X)
    assert model.cell_weights_.shape == X.shape
    assert model.case_weights_.shape == (X.shape[0],)
    assert model.cell_outlier_mask_.shape == X.shape
    assert model.corrected_data_.shape == X.shape
    assert np.isfinite(model.corrected_data_).all()
    assert np.isfinite(model.imputed_data_).all()
    assert np.all(model.missing_mask_ == missing_truth)
    # The method should identify a substantial fraction of gross bad cells.
    assert model.cell_outlier_mask_[cell_truth].mean() > 0.65


def test_new_data_diagnostics_and_outlier_map():
    rng, clean, _ = make_low_rank_data(n=75, p=25)
    X, _, _, _ = damage_data(rng, clean)
    model = small_model().fit(X)
    new = X[:8].copy()
    diagnostics = model.cellwise_diagnostics(new)
    assert diagnostics["corrected_data"].shape == new.shape
    assert diagnostics["distances"].shape == (8,)
    assert np.isfinite(diagnostics["corrected_data"]).all()
    mapping = model.outlier_map(new)
    assert mapping.shape == (8, 2)
    assert np.all(mapping >= -1e-10)
    assert np.allclose(model.mahalanobis(new), diagnostics["distances"])


def test_transform_returns_corrected_original_units():
    rng, clean, _ = make_low_rank_data(n=70, p=20)
    X, _, _, _ = damage_data(rng, clean)
    model = small_model().fit(X)
    corrected = model.transform(X[:5])
    assert corrected.shape == (5, 20)
    assert np.isfinite(corrected).all()


def test_covariance_recovery_beats_median_empirical_on_mixed_contamination():
    rng, clean, truth = make_low_rank_data(n=85, p=40)
    X, _, _, _ = damage_data(rng, clean, cell_fraction=0.07, case_fraction=0.14)
    medians = np.nanmedian(X, axis=0)
    imputed = np.where(np.isnan(X), medians, X)
    empirical = np.cov(imputed, rowvar=False, ddof=1)
    model = small_model().fit(X)
    empirical_error = np.linalg.norm(empirical - truth, ord="fro") / np.linalg.norm(truth, ord="fro")
    robust_error = np.linalg.norm(model.covariance_ - truth, ord="fro") / np.linalg.norm(truth, ord="fro")
    assert robust_error < empirical_error * 0.65


def test_works_as_robust_pca_scatter_estimator():
    rng, clean, _ = make_low_rank_data(n=80, p=25)
    X, _, _, _ = damage_data(rng, clean, missing=0.0)
    pca = rc.RobustPCA(
        n_components=3,
        estimator=small_model(),
    ).fit(X)
    assert pca.components_.shape == (3, 25)
    assert np.isfinite(pca.transform(X[:4])).all()


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"n_components": 0}, ValueError),
        ({"residual_shrinkage": -0.1}, ValueError),
        ({"residual_shrinkage": "bad"}, ValueError),
        ({"cv_splits": 1}, ValueError),
        ({"min_eigenvalue": 0.0}, ValueError),
    ],
)
def test_invalid_parameters(kwargs, error):
    X = np.random.default_rng(0).normal(size=(20, 8))
    with pytest.raises(error):
        rc.CellRCov(**kwargs).fit(X)


def test_rejects_empty_rows_and_columns():
    X = np.random.default_rng(0).normal(size=(20, 8))
    X[:, 0] = np.nan
    with pytest.raises(ValueError, match="feature"):
        rc.CellRCov(n_components=2).fit(X)
    X = np.random.default_rng(0).normal(size=(20, 8))
    X[0] = np.nan
    with pytest.raises(ValueError, match="row"):
        rc.CellRCov(n_components=2).fit(X)
