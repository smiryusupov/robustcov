from __future__ import annotations

import numpy as np
import pytest

from robustcov.experimental import DistributionallyRobustPCA, WassersteinRobustPCA


def _pca_basis(X: np.ndarray, rank: int) -> np.ndarray:
    centered = X - np.mean(X, axis=0)
    covariance = centered.T @ centered / len(X)
    values, vectors = np.linalg.eigh(covariance)
    return vectors[:, np.argsort(values)[::-1][:rank]]


def _projector(basis: np.ndarray) -> np.ndarray:
    return basis @ basis.T


def _target_risk(X: np.ndarray, location: np.ndarray, basis: np.ndarray) -> float:
    centered = X - location
    residual = centered - (centered @ basis) @ basis.T
    return float(np.mean(np.einsum("ij,ij->i", residual, residual)))


def test_identity_geometry_is_exactly_the_classical_pca_control():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(180, 6)) @ np.diag([3.0, 2.2, 1.5, 0.8, 0.5, 0.2])
    radius = 0.35
    fitted = DistributionallyRobustPCA(
        n_components=2,
        radius=radius,
        transport_geometry="identity",
        formulation="exact",
    ).fit(X)
    pca = _pca_basis(X, 2)

    np.testing.assert_allclose(fitted.projector_, _projector(pca), rtol=1e-10, atol=1e-11)
    expected = (np.sqrt(fitted.nominal_reconstruction_risk_) + radius) ** 2
    np.testing.assert_allclose(fitted.exact_worst_case_risk_, expected, rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(fitted.surrogate_risk_bound_, expected, rtol=2e-11, atol=2e-12)
    assert fitted.selected_candidate_source_ == "path"
    assert fitted.selected_gamma_ == 0.0


def test_exact_dual_is_a_genuine_ambiguity_set_risk_and_below_the_surrogate_bound():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(160, 7)) @ np.diag([2.8, 2.0, 1.4, 1.0, 0.7, 0.4, 0.2])
    fitted = DistributionallyRobustPCA(
        n_components=3,
        radius=0.4,
        transport_geometry="residual",
        formulation="exact",
    ).fit(X)

    assert fitted.ambiguity_set_["type"] == "weighted_wasserstein_2"
    assert fitted.optimizer_ == "deterministic_candidate_path"
    assert fitted.optimization_scope_ == "finite_path_not_global_grassmann"
    assert fitted.global_optimum_claim_ is False
    assert fitted.exact_worst_case_risk_ >= fitted.nominal_reconstruction_risk_
    assert fitted.exact_worst_case_risk_ <= fitted.surrogate_risk_bound_ + 1e-10
    assert fitted.dual_lambda_ > fitted.residual_exposure_
    np.testing.assert_allclose(
        fitted.exact_worst_case_risk(), fitted.exact_worst_case_risk_, rtol=0, atol=0
    )


def test_adaptive_geometry_is_global_scale_and_feature_permutation_equivariant():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(220, 8)) @ np.diag([3.0, 2.4, 1.8, 1.2, 0.9, 0.6, 0.4, 0.2])
    kwargs = dict(
        n_components=3,
        radius="sqrt_n",
        radius_scale=8.0,
        transport_geometry="residual",
        formulation="exact",
    )
    base = DistributionallyRobustPCA(**kwargs).fit(X)
    tiny = DistributionallyRobustPCA(**kwargs).fit(1e-80 * X)
    np.testing.assert_allclose(tiny.projector_, base.projector_, rtol=2e-9, atol=2e-10)
    np.testing.assert_allclose(
        tiny.exact_worst_case_risk_, 1e-160 * base.exact_worst_case_risk_, rtol=2e-8, atol=1e-170
    )

    permutation = np.array([4, 0, 7, 2, 6, 1, 5, 3])
    permuted = DistributionallyRobustPCA(**kwargs).fit(X[:, permutation])
    restored = np.empty_like(permuted.projector_)
    restored[np.ix_(permutation, permutation)] = permuted.projector_
    np.testing.assert_allclose(restored, base.projector_, rtol=2e-8, atol=2e-9)


def test_exact_path_improves_held_out_risk_under_aligned_covariance_shift():
    rng = np.random.default_rng(13)
    n_train = 260
    n_target = 12000
    train_variances = np.array([6.0, 5.0, 2.5, 2.2, 1.5, 1.3, 1.1, 1.0, 0.9, 0.8])
    target_variances = np.array([4.5, 4.0, 9.0, 8.0, 1.5, 1.3, 1.1, 1.0, 0.9, 0.8])
    X_train = rng.normal(size=(n_train, 10)) * np.sqrt(train_variances)
    X_target = rng.normal(size=(n_target, 10)) * np.sqrt(target_variances)

    pca_basis = _pca_basis(X_train, 2)
    fitted = DistributionallyRobustPCA(
        n_components=2,
        radius=2.5,
        transport_geometry="residual",
        formulation="exact",
    ).fit(X_train)
    pca_risk = _target_risk(X_target, np.mean(X_train, axis=0), pca_basis)
    dro_risk = float(np.mean(fitted.reconstruction_error(X_target)))

    assert dro_risk < 0.75 * pca_risk
    assert fitted.selected_gamma_ > 0.0
    assert np.trace(fitted.projector_[2:4, 2:4]) > 1.8


def test_custom_geometry_is_scale_normalized_and_validated():
    rng = np.random.default_rng(14)
    X = rng.normal(size=(120, 5))
    G = np.array(
        [
            [2.0, 0.2, 0.0, 0.0, 0.0],
            [0.2, 1.5, 0.1, 0.0, 0.0],
            [0.0, 0.1, 1.2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.8, 0.1],
            [0.0, 0.0, 0.0, 0.1, 0.6],
        ]
    )
    a = DistributionallyRobustPCA(n_components=2, radius=0.3, transport_matrix=G).fit(X)
    b = DistributionallyRobustPCA(n_components=2, radius=0.3, transport_matrix=17.0 * G).fit(X)
    np.testing.assert_allclose(a.projector_, b.projector_, rtol=2e-9, atol=2e-10)
    np.testing.assert_allclose(a.transport_matrix_, b.transport_matrix_, rtol=2e-12, atol=2e-13)

    with pytest.raises(ValueError, match="positive definite"):
        DistributionallyRobustPCA(
            n_components=2,
            radius=0.3,
            transport_matrix=np.diag([1.0, 1.0, 1.0, 1.0, 0.0]),
        ).fit(X)


def test_transform_inverse_transform_and_sklearn_clone_protocol():
    sklearn = pytest.importorskip("sklearn")
    rng = np.random.default_rng(15)
    X = rng.normal(size=(100, 6))
    estimator = DistributionallyRobustPCA(
        n_components=2,
        radius=0.25,
        transport_geometry="pca_block",
        path_grid=(0.0, 0.5, 1.0, 2.0),
    )
    cloned = sklearn.base.clone(estimator)
    assert cloned.get_params(deep=True) == estimator.get_params(deep=True)
    fitted = cloned.fit(X)
    scores = fitted.transform(X)
    reconstructed = fitted.inverse_transform(scores)
    assert scores.shape == (100, 2)
    assert reconstructed.shape == X.shape
    np.testing.assert_allclose(
        fitted.reconstruction_error(X),
        np.sum((X - reconstructed) ** 2, axis=1),
        rtol=1e-13,
        atol=1e-14,
    )
    assert WassersteinRobustPCA is DistributionallyRobustPCA
