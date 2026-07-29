import numpy as np
import pytest

import robustcov as rc


def projection_error(components, truth):
    Q1, _ = np.linalg.qr(np.asarray(components).T)
    Q2, _ = np.linalg.qr(np.asarray(truth).T)
    return np.linalg.norm(Q1 @ Q1.T - Q2 @ Q2.T, ord="fro")


def make_low_rank(seed=0, n=240, p=8, q=2, noise=0.05):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(p, q))
    basis, _ = np.linalg.qr(raw)
    scores = rng.normal(size=(n, q)) * np.linspace(2.5, 1.2, q)
    X = scores @ basis.T + rng.normal(scale=noise, size=(n, p))
    return X, basis.T


def test_alpha_zero_matches_classical_pca_subspace():
    X, _ = make_low_rank(seed=1, n=180, p=6, q=3)
    X += np.array([1.5, -0.2, 0.4, 0.0, 0.1, -0.7])

    model = rc.DensityPowerRobustPCA(
        n_components=3,
        alpha=0.0,
        location="mean",
        init="svd",
        tol=1e-10,
    ).fit(X)

    Xc = X - X.mean(axis=0)
    _, singular, Vt = np.linalg.svd(Xc, full_matrices=False)
    assert projection_error(model.components_, Vt[:3]) < 1e-8
    assert np.allclose(model.eigenvalues_, singular[:3] ** 2 / X.shape[0], rtol=1e-7)


def test_density_power_pca_recovers_contaminated_subspace():
    X_clean, truth = make_low_rank(seed=2, n=280, p=10, q=2)
    rng = np.random.default_rng(22)
    X_bad = X_clean.copy()
    rows = rng.choice(X_bad.shape[0], 55, replace=False)
    columns = rng.integers(4, X_bad.shape[1], size=rows.size)
    X_bad[rows, columns] += rng.normal(0.0, 18.0, size=rows.size)

    empirical = rc.DensityPowerRobustPCA(
        n_components=2, alpha=0.0, location="mean", init="svd"
    ).fit(X_bad)
    robust = rc.DensityPowerRobustPCA(
        n_components=2, alpha=0.35, max_iter=120, tol=1e-5
    ).fit(X_bad)

    empirical_error = projection_error(empirical.components_, truth)
    robust_error = projection_error(robust.components_, truth)
    assert robust_error < 0.25
    assert robust_error < 0.4 * empirical_error


def test_cell_weights_downweight_large_residuals():
    X, _ = make_low_rank(seed=3, n=160, p=7, q=2)
    labels = np.zeros_like(X, dtype=bool)
    labels[:12, 5] = True
    X[:12, 5] += 15.0

    model = rc.DensityPowerRobustPCA(n_components=2, alpha=0.4).fit(X)
    weights = model.cell_weights(X)

    assert weights.shape == X.shape
    assert np.median(weights[labels]) < 0.05
    assert np.median(weights[~labels]) > 0.8
    assert np.mean(model.row_outlier_scores_[:12]) > np.mean(model.row_outlier_scores_[30:])


def test_transform_inverse_and_diagnostics_shapes():
    X, _ = make_low_rank(seed=4, n=100, p=6, q=2)
    model = rc.DensityPowerRobustPCA(n_components=2, alpha=0.25).fit(X)

    scores = model.transform(X[:9])
    reconstructed = model.inverse_transform(scores)
    diagnostics = model.outlier_map(X[:9])

    assert scores.shape == (9, 2)
    assert reconstructed.shape == (9, 6)
    assert diagnostics.shape == (9, 2)
    assert np.all(diagnostics >= 0.0)
    assert model.fitted_values_.shape == X.shape
    assert model.weights_.shape == X.shape


def test_whitening_round_trip():
    X, _ = make_low_rank(seed=5, n=300, p=7, q=3)
    model = rc.DensityPowerRobustPCA(
        n_components=3, alpha=0.0, location="mean", init="svd", whiten=True
    ).fit(X)
    scores = model.transform(X)
    reconstructed = model.inverse_transform(scores)

    assert np.allclose(reconstructed, model.location_ + (X - model.location_) @ model.components_.T @ model.components_, atol=1e-7)
    assert np.all(np.isfinite(scores))


def test_outlier_map_separates_score_and_orthogonal_outliers():
    rng = np.random.default_rng(6)
    latent = rng.normal(size=(220, 2))
    X = np.column_stack([3.0 * latent[:, 0], latent[:, 1], np.zeros(220)])
    X += rng.normal(scale=0.02, size=X.shape)
    model = rc.DensityPowerRobustPCA(n_components=2, alpha=0.2).fit(X)

    regular = np.array([[0.0, 0.0, 0.0]])
    leverage = np.array([[12.0, 0.0, 0.0]])
    orthogonal = np.array([[0.0, 0.0, 5.0]])
    values = model.outlier_map(np.vstack([regular, leverage, orthogonal]))

    assert values[1, 0] > values[0, 0] + 2.0
    assert values[2, 1] > values[0, 1] + 4.0


def test_high_dimensional_fit_is_finite():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(45, 80))
    model = rc.DensityPowerRobustPCA(
        n_components=5, alpha=0.25, max_iter=30
    ).fit(X)

    assert model.components_.shape == (5, 80)
    assert np.all(np.isfinite(model.components_))
    assert np.all(model.eigenvalues_ >= 0.0)
    assert np.allclose(model.components_ @ model.components_.T, np.eye(5), atol=1e-8)


def test_deterministic_fit_and_component_orientation():
    X, _ = make_low_rank(seed=8, n=150, p=6, q=2)
    first = rc.DensityPowerRobustPCA(n_components=2, alpha=0.3).fit(X)
    second = rc.DensityPowerRobustPCA(n_components=2, alpha=0.3).fit(X)

    assert np.allclose(first.components_, second.components_)
    for component in first.components_:
        assert component[np.argmax(np.abs(component))] >= 0.0


def test_translation_and_orthogonal_equivariance_approximately_hold():
    X, _ = make_low_rank(seed=9, n=180, p=5, q=2)
    rng = np.random.default_rng(91)
    Q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    shift = rng.normal(size=5)

    original = rc.DensityPowerRobustPCA(n_components=2, alpha=0.25).fit(X)
    transformed = rc.DensityPowerRobustPCA(n_components=2, alpha=0.25).fit(X @ Q + shift)

    expected = original.components_ @ Q
    assert projection_error(transformed.components_, expected) < 0.08
    assert np.allclose(transformed.location_, original.location_ @ Q + shift, atol=1e-5)


def test_custom_location_is_used():
    X, _ = make_low_rank(seed=10, n=80, p=5, q=2)
    location = np.arange(5, dtype=float)
    model = rc.DensityPowerRobustPCA(
        n_components=2, alpha=0.2, location=location
    ).fit(X)
    assert np.array_equal(model.location_, location)


def test_objective_and_scale_diagnostics_are_finite():
    X, _ = make_low_rank(seed=11, n=140, p=7, q=2)
    model = rc.DensityPowerRobustPCA(n_components=2, alpha=0.3).fit(X)

    assert model.objective_history_.ndim == 1
    assert np.all(np.isfinite(model.objective_history_))
    assert model.residual_scale_ > 0.0
    assert model.noise_variance_ == pytest.approx(model.residual_scale_**2)
    assert 1 <= model.n_iter_ <= model.max_iter


def test_fit_transform_and_reconstruction_error():
    X, _ = make_low_rank(seed=12, n=120, p=6, q=2)
    model = rc.DensityPowerRobustPCA(n_components=2, alpha=0.2)
    scores = model.fit_transform(X)
    errors = model.reconstruction_error(X)

    assert scores.shape == (120, 2)
    assert errors.shape == (120,)
    assert np.all(errors >= 0.0)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"n_components": 0}, ValueError),
        ({"n_components": 20}, ValueError),
        ({"n_components": 2.5}, TypeError),
        ({"alpha": -0.1}, ValueError),
        ({"alpha": 1.1}, ValueError),
        ({"init": "random"}, ValueError),
        ({"winsorize": 0.0}, ValueError),
        ({"max_iter": 0}, ValueError),
        ({"inner_iter": 0}, ValueError),
        ({"tol": 0.0}, ValueError),
        ({"ridge": 0.0}, ValueError),
        ({"min_scale": 0.0}, ValueError),
        ({"whiten": "yes"}, TypeError),
    ],
)
def test_parameter_validation(kwargs, error):
    X = np.arange(60, dtype=float).reshape(12, 5)
    with pytest.raises(error):
        rc.DensityPowerRobustPCA(**kwargs).fit(X)


def test_input_validation_and_unfitted_calls():
    model = rc.DensityPowerRobustPCA()
    with pytest.raises(AttributeError, match="not fitted"):
        model.transform(np.ones((3, 2)))
    with pytest.raises(ValueError, match="finite"):
        model.fit(np.array([[1.0, np.nan], [2.0, 3.0]]))

    fitted = rc.DensityPowerRobustPCA(n_components=1).fit(np.arange(24, dtype=float).reshape(8, 3))
    with pytest.raises(ValueError, match="expected"):
        fitted.transform(np.ones((4, 2)))
    with pytest.raises(ValueError, match="n_components"):
        fitted.inverse_transform(np.ones((4, 2)))
