import numpy as np
import pytest

import robustcov as rc


class FixedScatter:
    def __init__(self, covariance, location=None):
        self.covariance = np.asarray(covariance, dtype=float)
        self.location = location

    def fit(self, X):
        self.covariance_ = self.covariance.copy()
        self.location_ = (
            np.zeros(X.shape[1])
            if self.location is None
            else np.asarray(self.location, dtype=float).copy()
        )
        return self


def _chain_precision(p, strength=0.28):
    precision = np.eye(p)
    for index in range(p - 1):
        precision[index, index + 1] = -strength
        precision[index + 1, index] = -strength
    return precision


def _edge_f1(adjacency, truth):
    predicted = np.triu(adjacency, 1)
    expected = np.triu(truth, 1)
    tp = np.count_nonzero(predicted & expected)
    fp = np.count_nonzero(predicted & ~expected)
    fn = np.count_nonzero(~predicted & expected)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def test_alpha_zero_matches_inverse_scatter_with_standardization():
    covariance = np.array(
        [[4.0, 0.8, 0.0], [0.8, 2.0, 0.3], [0.0, 0.3, 1.5]]
    )
    X = np.arange(60, dtype=float).reshape(20, 3)

    model = rc.RobustGraphicalLasso(
        alpha=0.0,
        scatter_estimator=FixedScatter(covariance),
        standardize=True,
    ).fit(X)

    np.testing.assert_allclose(model.precision_, np.linalg.inv(covariance), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(model.covariance_, covariance, rtol=1e-9, atol=1e-9)
    assert model.converged_
    assert model.n_iter_ == 0


def test_fixed_penalty_produces_sparse_spd_precision():
    rng = np.random.default_rng(2)
    precision = _chain_precision(9)
    covariance = np.linalg.inv(precision)
    X = rng.multivariate_normal(np.zeros(9), covariance, size=350)

    model = rc.RobustGraphicalLasso(
        alpha=0.08,
        scatter_estimator="empirical",
        max_iter=800,
    ).fit(X)

    assert model.converged_
    assert np.linalg.eigvalsh(model.precision_).min() > 0.0
    assert model.n_edges_ < 9 * 8 // 2
    assert model.n_edges_ >= 6
    assert np.all(model.adjacency_ == model.adjacency_.T)
    assert not np.any(np.diag(model.adjacency_))


def test_ebic_path_selects_one_candidate_and_records_path():
    rng = np.random.default_rng(3)
    precision = _chain_precision(8, 0.24)
    X = rng.multivariate_normal(np.zeros(8), np.linalg.inv(precision), size=280)

    model = rc.RobustGraphicalLasso(
        alpha="ebic",
        scatter_estimator="empirical",
        n_alphas=12,
        max_iter=600,
    ).fit(X)

    assert model.alphas_.shape == (12,)
    assert model.ebic_scores_.shape == (12,)
    assert model.path_n_edges_.shape == (12,)
    assert model.alpha_ == pytest.approx(model.alphas_[model.best_alpha_index_])
    assert np.isfinite(model.ebic_scores_).all()
    assert np.all(np.diff(model.alphas_) < 0)


def test_partial_correlations_and_conditional_coefficients_are_correct():
    precision = np.array(
        [[2.0, -0.4, 0.0], [-0.4, 1.5, 0.3], [0.0, 0.3, 1.2]]
    )
    X = np.zeros((10, 3))
    model = rc.RobustGraphicalLasso(
        alpha=0.0,
        scatter_estimator=FixedScatter(np.linalg.inv(precision)),
    ).fit(X)

    expected_partial = -precision / np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    np.fill_diagonal(expected_partial, 1.0)
    np.testing.assert_allclose(model.partial_correlation_, expected_partial, atol=1e-12)

    expected_coefficients = -precision / np.diag(precision)[:, None]
    np.fill_diagonal(expected_coefficients, 0.0)
    np.testing.assert_allclose(model.conditional_coefficients_, expected_coefficients, atol=1e-12)


def test_mahalanobis_and_log_scores_match_manual_calculation():
    covariance = np.array([[2.0, 0.4], [0.4, 1.0]])
    location = np.array([1.0, -2.0])
    X_train = np.zeros((8, 2))
    X_test = np.array([[1.0, -2.0], [2.0, -1.0]])
    model = rc.RobustGraphicalLasso(
        alpha=0.0,
        scatter_estimator=FixedScatter(covariance, location),
    ).fit(X_train)

    centered = X_test - location
    expected = np.einsum("ij,jk,ik->i", centered, np.linalg.inv(covariance), centered)
    np.testing.assert_allclose(model.mahalanobis(X_test), expected)
    assert model.score_samples(X_test)[0] > model.score_samples(X_test)[1]


def test_edge_list_uses_names_threshold_and_sorting():
    precision = np.array(
        [[2.0, -0.8, 0.0], [-0.8, 2.0, 0.2], [0.0, 0.2, 1.5]]
    )
    X = np.zeros((10, 3))
    model = rc.RobustGraphicalLasso(
        alpha=0.0,
        scatter_estimator=FixedScatter(np.linalg.inv(precision)),
        edge_tolerance=1e-12,
    ).fit(X)

    edges = model.edge_list(["a", "b", "c"], min_abs_partial_correlation=0.05)
    assert edges[0][0:2] == ("a", "b")
    assert len(edges) == 2
    with pytest.raises(ValueError):
        model.edge_list(["a"])


def test_default_robust_scatter_improves_graph_recovery_under_row_outliers():
    rng = np.random.default_rng(7)
    p = 10
    truth_precision = _chain_precision(p, 0.30)
    covariance = np.linalg.inv(truth_precision)
    clean = rng.multivariate_normal(np.zeros(p), covariance, size=260)
    contaminated = clean.copy()
    contaminated[:32, ::2] += rng.normal(10.0, 1.0, size=(32, 5))

    empirical = rc.RobustGraphicalLasso(
        alpha=0.09,
        scatter_estimator="empirical",
        max_iter=800,
    ).fit(contaminated)
    robust = rc.RobustGraphicalLasso(
        alpha=0.09,
        max_iter=800,
    ).fit(contaminated)

    truth = np.abs(truth_precision) > 1e-12
    np.fill_diagonal(truth, False)
    assert _edge_f1(robust.adjacency_, truth) > _edge_f1(empirical.adjacency_, truth)


def test_cellmcd_scatter_allows_missing_training_entries():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(70, 6))
    X[::11, 2] = np.nan
    X[::13, 4] += 9.0

    model = rc.RobustGraphicalLasso(
        alpha=0.12,
        scatter_estimator=rc.CellMCD(max_iter=25, min_samples_per_feature=None),
        max_iter=500,
    ).fit(X)

    assert np.isfinite(model.precision_).all()
    assert np.linalg.eigvalsh(model.precision_).min() > 0.0


def test_plot_partial_correlation_network_writes_file(tmp_path):
    pytest.importorskip("matplotlib")
    covariance = np.array([[1.0, 0.4, 0.0], [0.4, 1.0, 0.2], [0.0, 0.2, 1.0]])
    X = np.zeros((10, 3))
    model = rc.RobustGraphicalLasso(
        alpha=0.0,
        scatter_estimator=FixedScatter(covariance),
    ).fit(X)
    output = tmp_path / "network.png"

    figure = rc.plot_partial_correlation_network(
        model,
        feature_names=["x", "y", "z"],
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
        {"n_alphas": 1},
        {"alpha_min_ratio": 0.0},
        {"ebic_gamma": -0.1},
        {"rho": 0.0},
        {"max_iter": 0},
        {"scatter_floor": 0.0},
        {"edge_tolerance": -1.0},
    ],
)
def test_invalid_parameters(kwargs):
    X = np.arange(40, dtype=float).reshape(10, 4)
    with pytest.raises((ValueError, TypeError)):
        rc.RobustGraphicalLasso(**kwargs).fit(X)


def test_unfitted_and_feature_validation():
    model = rc.RobustGraphicalLasso(alpha=0.1)
    with pytest.raises(AttributeError):
        model.mahalanobis(np.ones((2, 3)))

    fitted = rc.RobustGraphicalLasso(alpha=0.0, scatter_estimator="empirical").fit(
        np.random.default_rng(9).normal(size=(30, 4))
    )
    with pytest.raises(ValueError):
        fitted.mahalanobis(np.ones((2, 3)))
    with pytest.raises(ValueError):
        fitted.mahalanobis(np.array([[1.0, np.nan, 2.0, 3.0]]))


def test_admm_matches_sklearn_graphical_lasso_on_fixed_scatter():
    covariance_module = pytest.importorskip("sklearn.covariance")
    rng = np.random.default_rng(10)
    matrix = rng.normal(size=(6, 6))
    scatter = matrix @ matrix.T / 6.0 + np.eye(6)
    X = np.zeros((12, 6))

    model = rc.RobustGraphicalLasso(
        alpha=0.15,
        scatter_estimator=FixedScatter(scatter),
        standardize=False,
        max_iter=1500,
        abs_tol=1e-7,
        rel_tol=1e-7,
    ).fit(X)
    _, reference_precision = covariance_module.graphical_lasso(
        scatter,
        alpha=0.15,
        max_iter=1000,
        tol=1e-4,
    )

    np.testing.assert_allclose(model.precision_, reference_precision, atol=2e-4, rtol=2e-4)
