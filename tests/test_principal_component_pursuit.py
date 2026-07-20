from __future__ import annotations

import numpy as np
import pytest

import robustcov as rc


def _problem(seed: int = 0):
    rng = np.random.default_rng(seed)
    n_samples, n_features, rank = 60, 45, 3
    left, _ = np.linalg.qr(rng.normal(size=(n_samples, rank)))
    right, _ = np.linalg.qr(rng.normal(size=(n_features, rank)))
    low_rank = left @ np.diag([20.0, 14.0, 9.0]) @ right.T
    sparse = np.zeros((n_samples, n_features))
    indices = rng.choice(
        n_samples * n_features,
        size=int(0.04 * n_samples * n_features),
        replace=False,
    )
    sparse.flat[indices] = (
        rng.choice([-1.0, 1.0], size=indices.size)
        * rng.uniform(6.0, 12.0, size=indices.size)
    )
    return low_rank + sparse, low_rank, sparse, indices


def _relative_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.linalg.norm(estimate - truth, ord="fro")
        / np.linalg.norm(truth, ord="fro")
    )


def test_exact_low_rank_sparse_recovery_on_incoherent_problem():
    observed, low_rank, sparse, indices = _problem()
    estimator = rc.PrincipalComponentPursuit(tol=1e-7).fit(observed)

    assert estimator.converged_
    assert estimator.rank_ == 3
    assert _relative_error(estimator.low_rank_, low_rank) < 1e-4
    assert _relative_error(estimator.sparse_, sparse) < 1e-4
    assert np.mean(estimator.sparse_support_.flat[indices]) == 1.0
    assert np.mean(estimator.sparse_support_[sparse == 0.0]) < 1e-3
    assert estimator.reconstruction_error_ <= estimator.tol


def test_default_lambda_and_diagnostics_are_exposed():
    observed, _, _, _ = _problem(seed=1)
    estimator = rc.PCP(store_history=True).fit(observed)

    assert estimator.lambda_value_ == pytest.approx(
        1.0 / np.sqrt(max(observed.shape))
    )
    assert estimator.n_sparse_ == int(np.count_nonzero(estimator.sparse_support_))
    assert estimator.row_outlier_scores_.shape == (observed.shape[0],)
    assert estimator.column_outlier_scores_.shape == (observed.shape[1],)
    assert estimator.history_records()
    summary = estimator.decomposition_summary()
    assert summary["rank"] == estimator.rank_
    assert summary["converged"] is True


def test_fit_transform_projection_and_inverse_transform():
    observed, _, _, _ = _problem(seed=2)
    estimator = rc.PrincipalComponentPursuit().fit(observed)

    assert np.allclose(estimator.fit_transform(observed), estimator.low_rank_)
    scores = estimator.transform(observed[:5])
    reconstructed = estimator.inverse_transform(scores)
    assert scores.shape == (5, estimator.rank_)
    assert reconstructed.shape == (5, observed.shape[1])
    assert np.allclose(
        reconstructed,
        observed[:5] @ estimator.components_.T @ estimator.components_,
    )


def test_zero_matrix_has_rank_zero_and_no_sparse_entries():
    estimator = rc.PrincipalComponentPursuit().fit(np.zeros((8, 5)))

    assert estimator.converged_
    assert estimator.n_iter_ == 0
    assert estimator.rank_ == 0
    assert estimator.n_sparse_ == 0
    assert estimator.reconstruction_error_ == 0.0
    assert estimator.transform(np.ones((2, 5))).shape == (2, 0)
    assert estimator.inverse_transform(np.empty((2, 0))).shape == (2, 5)


def test_fit_is_deterministic():
    observed, _, _, _ = _problem(seed=3)
    first = rc.PrincipalComponentPursuit().fit(observed)
    second = rc.PrincipalComponentPursuit().fit(observed)

    assert np.allclose(first.low_rank_, second.low_rank_)
    assert np.allclose(first.sparse_, second.sparse_)
    assert first.history_records() == second.history_records()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lambda_": 0.0}, "lambda_"),
        ({"mu": 0.0}, "mu"),
        ({"rho": 1.0}, "rho"),
        ({"mu_max_factor": 0.5}, "mu_max_factor"),
        ({"max_iter": 0}, "max_iter"),
        ({"tol": 0.0}, "tol"),
        ({"sparse_tol": -1.0}, "sparse_tol"),
        ({"store_history": 1}, "store_history"),
    ],
)
def test_parameter_validation(kwargs, message):
    observed, _, _, _ = _problem(seed=4)
    with pytest.raises((TypeError, ValueError), match=message):
        rc.PrincipalComponentPursuit(**kwargs).fit(observed)


def test_input_and_unfitted_validation():
    estimator = rc.PrincipalComponentPursuit()
    with pytest.raises(AttributeError, match="not fitted"):
        estimator.transform(np.ones((2, 2)))
    with pytest.raises(ValueError, match="2D"):
        estimator.fit(np.ones(5))
    with pytest.raises(ValueError, match="at least two"):
        estimator.fit(np.ones((1, 5)))
    bad = np.ones((4, 3))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        estimator.fit(bad)


def test_transform_shape_and_finite_validation():
    observed, _, _, _ = _problem(seed=5)
    estimator = rc.PrincipalComponentPursuit().fit(observed)
    with pytest.raises(ValueError, match="features"):
        estimator.transform(np.ones((2, observed.shape[1] + 1)))
    bad = observed[:2].copy()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        estimator.transform(bad)
    with pytest.raises(ValueError, match="components"):
        estimator.inverse_transform(np.ones((2, estimator.rank_ + 1)))


def test_sklearn_clone_when_available():
    sklearn = pytest.importorskip("sklearn")
    estimator = rc.PrincipalComponentPursuit(lambda_=0.2, rho=1.6)
    clone = sklearn.base.clone(estimator)
    assert clone.lambda_ == 0.2
    assert clone.rho == 1.6
