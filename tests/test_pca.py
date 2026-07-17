import numpy as np
import pytest

import robustcov as rc


class EmpiricalScatter:
    def __init__(self, ridge=0.0):
        self.ridge = ridge

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.location_ = X.mean(axis=0)
        Xc = X - self.location_
        self.covariance_ = Xc.T @ Xc / X.shape[0]
        self.covariance_ += self.ridge * np.eye(X.shape[1])
        return self


class InvalidScatter:
    def __init__(self, covariance, location=None):
        self.covariance = covariance
        self.location = location

    def fit(self, X):
        self.covariance_ = np.asarray(self.covariance, dtype=float)
        self.location_ = (
            np.asarray(self.location, dtype=float)
            if self.location is not None
            else np.mean(X, axis=0)
        )
        return self


def projection_distance(components_a, components_b):
    Pa = components_a.T @ components_a
    Pb = components_b.T @ components_b
    return np.linalg.norm(Pa - Pb, ord="fro")


def test_robust_pca_matches_empirical_eigendecomposition():
    rng = np.random.default_rng(100)
    X = rng.normal(size=(300, 5)) @ np.diag([3.0, 2.0, 1.0, 0.5, 0.2])

    pca = rc.RobustPCA(
        n_components=3,
        estimator=EmpiricalScatter(),
    ).fit(X)

    Xc = X - X.mean(axis=0)
    covariance = Xc.T @ Xc / X.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    expected_components = eigenvectors[:, order[:3]].T

    assert pca.components_.shape == (3, 5)
    assert np.allclose(pca.eigenvalues_, eigenvalues[order[:3]])
    assert projection_distance(pca.components_, expected_components) < 1e-10
    assert np.all(np.diff(pca.eigenvalues_) <= 0)


def test_robust_pca_full_reconstruction_and_stored_diagnostics():
    rng = np.random.default_rng(101)
    X = rng.normal(size=(120, 4))

    pca = rc.RobustPCA(estimator=EmpiricalScatter()).fit(X)
    reconstructed = pca.reconstruct(X)

    assert reconstructed.shape == X.shape
    assert np.allclose(reconstructed, X, atol=1e-10)
    assert pca.scores_.shape == X.shape
    assert pca.score_distances_.shape == (X.shape[0],)
    assert pca.orthogonal_distances_.shape == (X.shape[0],)
    assert np.max(pca.orthogonal_distances_) < 1e-10
    assert np.allclose(pca.reconstruction_error(X), 0.0, atol=1e-18)


def test_robust_pca_whitening_and_inverse_transform():
    rng = np.random.default_rng(102)
    X = rng.normal(size=(1000, 4)) @ np.array(
        [
            [2.0, 0.4, 0.0, 0.0],
            [0.0, 1.5, 0.2, 0.0],
            [0.0, 0.0, 0.8, 0.1],
            [0.0, 0.0, 0.0, 0.3],
        ]
    )

    pca = rc.RobustPCA(
        n_components=4,
        estimator=EmpiricalScatter(),
        whiten=True,
    ).fit(X)
    Z = pca.transform(X)

    assert np.allclose(np.cov(Z, rowvar=False, bias=True), np.eye(4), atol=1e-10)
    assert np.allclose(pca.inverse_transform(Z), X, atol=1e-10)


def test_robust_pca_variance_threshold_and_deterministic_signs():
    rng = np.random.default_rng(103)
    X = rng.normal(size=(400, 4)) @ np.diag([5.0, 2.0, 0.3, 0.1])

    pca_a = rc.RobustPCA(
        n_components=0.90,
        estimator=EmpiricalScatter(),
    ).fit(X)
    pca_b = rc.RobustPCA(
        n_components=0.90,
        estimator=EmpiricalScatter(),
    ).fit(X)

    assert pca_a.n_components_ == 2
    assert pca_a.explained_variance_ratio_.sum() >= 0.90
    assert np.allclose(pca_a.components_, pca_b.components_)
    for component in pca_a.components_:
        largest = np.argmax(np.abs(component))
        assert component[largest] >= 0.0


def test_robust_pca_outlier_map_separates_subspace_and_orthogonal_outliers():
    rng = np.random.default_rng(104)
    latent = rng.normal(size=(250, 2))
    X = np.column_stack([3.0 * latent[:, 0], latent[:, 1], np.zeros(250)])
    X += rng.normal(scale=0.01, size=X.shape)

    pca = rc.RobustPCA(
        n_components=2,
        estimator=EmpiricalScatter(ridge=1e-8),
    ).fit(X)

    leverage = np.array([[12.0, 0.0, 0.0]])
    orthogonal = np.array([[0.0, 0.0, 5.0]])
    regular = np.array([[0.0, 0.0, 0.0]])
    diagnostics = pca.outlier_map(np.vstack([regular, leverage, orthogonal]))

    assert diagnostics.shape == (3, 2)
    assert diagnostics[1, 0] > diagnostics[0, 0] + 2.0
    assert diagnostics[1, 1] < 0.1
    assert diagnostics[2, 1] > diagnostics[0, 1] + 4.0


def test_robust_pca_fastmcd_recovers_contaminated_subspace_better_than_empirical():
    rng = np.random.default_rng(105)
    n_clean = 360
    latent = rng.normal(size=(n_clean, 2))
    basis = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.2],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    X_clean = latent @ basis.T + rng.normal(scale=0.05, size=(n_clean, 5))
    true_components = np.linalg.svd(basis, full_matrices=False)[0].T

    contamination = rng.normal(scale=0.15, size=(50, 5))
    contamination[:, 3:] += rng.normal(loc=0.0, scale=12.0, size=(50, 2))
    X = np.vstack([X_clean, contamination])

    empirical = rc.RobustPCA(
        n_components=2,
        estimator=EmpiricalScatter(),
    ).fit(X)
    robust = rc.RobustPCA(
        n_components=2,
        estimator=rc.FastMCD(n_init=60, random_state=0),
    ).fit(X)

    empirical_error = projection_distance(empirical.components_, true_components)
    robust_error = projection_distance(robust.components_, true_components)

    assert robust_error < 0.35
    assert robust_error < empirical_error * 0.5


def test_robust_pca_regularizes_singular_covariance_consistently():
    covariance = np.diag([4.0, 1.0, 0.0, -1e-8])
    X = np.arange(40, dtype=float).reshape(10, 4)

    pca = rc.RobustPCA(
        estimator=InvalidScatter(covariance),
        ridge=1e-6,
    ).fit(X)

    assert np.min(pca.all_eigenvalues_) == pytest.approx(4e-6)
    assert np.all(np.linalg.eigvalsh(pca.covariance_) > 0.0)
    reconstructed = (
        pca.components_.T @ np.diag(pca.eigenvalues_) @ pca.components_
    )
    assert np.allclose(pca.covariance_, reconstructed)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"n_components": 0}, ValueError),
        ({"n_components": 5}, ValueError),
        ({"n_components": 1.2}, ValueError),
        ({"n_components": True}, TypeError),
        ({"ridge": 0.0}, ValueError),
        ({"whiten": "yes"}, TypeError),
    ],
)
def test_robust_pca_parameter_validation(kwargs, error):
    X = np.arange(24, dtype=float).reshape(8, 3)
    with pytest.raises(error):
        rc.RobustPCA(estimator=EmpiricalScatter(), **kwargs).fit(X)


def test_robust_pca_estimator_output_validation_and_unfitted_calls():
    X = np.arange(24, dtype=float).reshape(8, 3)

    with pytest.raises(AttributeError, match="not fitted"):
        rc.RobustPCA().transform(X)

    with pytest.raises(ValueError, match="incompatible shape"):
        rc.RobustPCA(
            estimator=InvalidScatter(np.eye(2)),
        ).fit(X)

    bad = np.eye(3)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        rc.RobustPCA(
            estimator=InvalidScatter(bad),
        ).fit(X)


def test_plot_robust_pca_outlier_map(tmp_path):
    rng = np.random.default_rng(106)
    X = rng.normal(size=(80, 3))
    pca = rc.RobustPCA(
        n_components=2,
        estimator=EmpiricalScatter(),
    ).fit(X)
    output = tmp_path / "outlier_map.png"

    fig = rc.plot_robust_pca_outlier_map(
        pca,
        labels=np.arange(X.shape[0]) % 17 == 0,
        output_path=output,
        show=False,
    )

    assert fig is not None
    assert output.exists()
    assert output.stat().st_size > 0


def test_robust_pca_high_dimensional_regularized_scatter():
    rng = np.random.default_rng(107)
    X = rng.standard_t(df=3, size=(24, 40))

    pca = rc.RobustPCA(
        n_components=8,
        estimator=rc.RegularizedCauchy(alpha=0.50, max_iter=200),
        whiten=True,
    ).fit(X)
    Z = pca.transform(X)

    assert pca.components_.shape == (8, 40)
    assert Z.shape == (24, 8)
    assert np.all(np.isfinite(Z))
    assert np.all(pca.eigenvalues_ > 0.0)


def test_robust_pca_store_scores_false_and_full_variance_threshold():
    rng = np.random.default_rng(108)
    X = rng.normal(size=(60, 4))
    pca = rc.RobustPCA(
        n_components=1.0,
        estimator=EmpiricalScatter(),
        store_scores=False,
    ).fit(X)

    assert pca.n_components_ == X.shape[1]
    assert not hasattr(pca, "scores_")
    assert not hasattr(pca, "score_distances_")
    assert not hasattr(pca, "orthogonal_distances_")
    with pytest.raises(RuntimeError, match="unavailable"):
        rc.plot_robust_pca_outlier_map(pca, show=False)
