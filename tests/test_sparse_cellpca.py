# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

import robustcov as rc


def _sparse_low_rank(seed=0, n=100, p=36, rank=3, noise=0.10):
    rng = np.random.default_rng(seed)
    loadings = np.zeros((p, rank))
    block = 6
    starts = np.linspace(1, p - block - 1, rank, dtype=int)
    magnitudes = np.array([1.0, 0.80, 0.60, 0.45, 0.30, 0.20])
    for k, start in enumerate(starts):
        loadings[start : start + block, k] = magnitudes * rng.choice(
            [-1.0, 1.0], block
        )
    loadings, _ = np.linalg.qr(loadings)
    scores = rng.normal(size=(n, rank)) * np.linspace(3.0, 1.4, rank)
    clean = scores @ loadings.T + noise * rng.normal(size=(n, p))
    return rng, clean, loadings


def _projection_error(components, truth):
    basis, _ = np.linalg.qr(np.asarray(components).T)
    return np.linalg.norm(basis @ basis.T - truth @ truth.T, ord="fro")


def _feature_support_f1(components, truth):
    predicted = np.any(np.abs(components) > 1e-10, axis=0)
    expected = np.any(np.abs(truth.T) > 1e-10, axis=0)
    tp = int(np.count_nonzero(predicted & expected))
    fp = int(np.count_nonzero(predicted & ~expected))
    fn = int(np.count_nonzero(~predicted & expected))
    return 2.0 * tp / (2.0 * tp + fp + fn)


def test_public_aliases():
    assert rc.SparseCellPCA is rc.SparseCellwiseRobustPCA
    assert rc.SparseCasewiseCellwisePCA is rc.SparseCellwiseRobustPCA


def test_fit_produces_exact_zero_loadings_and_diagnostics():
    _, X, _ = _sparse_low_rank(1)
    model = rc.SparseCellPCA(
        n_components=3,
        alpha=0.05,
        max_iter=35,
        loading_max_iter=40,
        tol=5e-4,
    ).fit(X)

    assert model.components_.shape == (3, X.shape[1])
    assert model.loading_support_.shape == model.components_.shape
    assert np.count_nonzero(model.components_ == 0.0) > 0
    assert model.sparsity_ > 0.25
    assert np.array_equal(model.loading_support_, model.components_ != 0.0)
    assert np.all(model.n_nonzero_loadings_ == model.loading_support_.sum(axis=1))
    assert np.allclose(model.component_gram_, model.components_ @ model.components_.T)


def test_sparse_support_is_more_interpretable_than_dense_cellpca():
    _, X, truth = _sparse_low_rank(2)
    dense = rc.CellPCA(n_components=3, max_iter=35, tol=5e-4).fit(X)
    sparse = rc.SparseCellPCA(
        n_components=3,
        alpha=0.06,
        max_iter=35,
        loading_max_iter=40,
        tol=5e-4,
    ).fit(X)

    assert _feature_support_f1(sparse.components_, truth) > 0.75
    assert np.count_nonzero(sparse.components_) < 0.65 * np.count_nonzero(
        dense.components_
    )
    assert _projection_error(sparse.components_, truth) < 0.25


def test_cellwise_contamination_and_missing_values():
    rng, clean, truth_basis = _sparse_low_rank(3, n=110)
    X = clean.copy()
    cell_truth = np.zeros_like(X, dtype=bool)
    bad = rng.choice(X.size, size=int(0.05 * X.size), replace=False)
    cell_truth.flat[bad] = True
    X.flat[bad] += rng.choice([-1.0, 1.0], bad.size) * rng.uniform(
        4.0, 8.0, bad.size
    )
    case_truth = np.zeros(X.shape[0], dtype=bool)
    case_truth[:6] = True
    X[case_truth] += rng.normal(0.0, 4.0, size=(case_truth.sum(), X.shape[1]))
    missing = (rng.random(X.shape) < 0.03) & ~cell_truth
    X[missing] = np.nan

    model = rc.SparseCellPCA(
        n_components=3,
        alpha=0.055,
        max_iter=40,
        loading_max_iter=45,
        tol=5e-4,
    ).fit(X)

    valid = ~missing
    assert roc_auc_score(
        cell_truth[valid], np.abs(model.standardized_residuals_[valid])
    ) > 0.94
    assert _projection_error(model.components_, truth_basis) < 0.35
    clean_missing = missing & ~case_truth[:, None]
    assert np.mean(
        np.abs(model.fitted_values_[clean_missing] - clean[clean_missing])
    ) < 0.25
    assert np.isfinite(model.imputed_data_).all()


def test_component_specific_penalties_are_supported():
    _, X, _ = _sparse_low_rank(4)
    model = rc.SparseCellPCA(
        n_components=3,
        alpha=[0.02, 0.05, 0.09],
        max_iter=30,
        loading_max_iter=35,
        tol=7e-4,
    ).fit(X)

    assert model.alpha_.shape == (3,)
    assert np.all(np.sort(model.alpha_) == np.sort([0.02, 0.05, 0.09]))
    assert model.n_nonzero_loadings_.shape == (3,)


def test_l2_only_penalty_does_not_promise_sparsity():
    _, X, _ = _sparse_low_rank(5)
    model = rc.SparseCellPCA(
        n_components=3,
        alpha=0.05,
        l1_ratio=0.0,
        sparsity_threshold=0.0,
        max_iter=25,
        loading_max_iter=30,
        tol=8e-4,
    ).fit(X)

    assert np.count_nonzero(model.components_) == model.components_.size
    assert model.sparsity_ == 0.0


def test_transform_reconstruct_and_new_row_diagnostics():
    rng, X, _ = _sparse_low_rank(6)
    X[rng.random(X.shape) < 0.03] = np.nan
    model = rc.SparseCellPCA(
        n_components=3,
        alpha=0.05,
        max_iter=30,
        loading_max_iter=35,
        tol=7e-4,
    ).fit(X)

    scores = model.transform(X[:8])
    reconstructed = model.inverse_transform(scores)
    diagnostics = model.cellwise_diagnostics(X[:8])

    assert scores.shape == (8, 3)
    assert reconstructed.shape == (8, X.shape[1])
    assert np.isfinite(scores).all()
    assert np.isfinite(diagnostics["corrected_data"]).all()
    assert np.allclose(reconstructed, diagnostics["fitted_values"], atol=1e-6)


def test_penalty_and_objective_histories_are_recorded():
    _, X, _ = _sparse_low_rank(7)
    model = rc.SparseCellPCA(
        n_components=3,
        alpha=0.04,
        max_iter=20,
        loading_max_iter=25,
        tol=1e-3,
    ).fit(X)

    assert model.objective_history_.ndim == 1
    assert model.objective_history_.size == model.n_iter_
    assert model.penalty_history_.shape == model.objective_history_.shape
    assert model.reconstruction_loss_history_.shape == model.objective_history_.shape
    assert np.isfinite(model.objective_history_).all()


def test_deterministic_fit():
    rng, X, _ = _sparse_low_rank(8)
    X.flat[rng.choice(X.size, 80, replace=False)] += 5.0
    kwargs = dict(
        n_components=3,
        alpha=0.05,
        max_iter=25,
        loading_max_iter=30,
        tol=8e-4,
    )
    first = rc.SparseCellPCA(**kwargs).fit(X)
    second = rc.SparseCellPCA(**kwargs).fit(X)

    assert np.allclose(first.center_, second.center_)
    assert np.allclose(first.components_, second.components_)
    assert np.array_equal(first.loading_support_, second.loading_support_)


def test_alpha_zero_matches_dense_subspace_reasonably():
    _, X, _ = _sparse_low_rank(9)
    dense = rc.CellPCA(n_components=3, max_iter=30, tol=7e-4).fit(X)
    sparse = rc.SparseCellPCA(
        n_components=3,
        alpha=0.0,
        l1_ratio=1.0,
        max_iter=30,
        loading_max_iter=35,
        tol=7e-4,
    ).fit(X)

    dense_basis, _ = np.linalg.qr(dense.components_.T)
    sparse_basis, _ = np.linalg.qr(sparse.components_.T)
    difference = np.linalg.norm(
        dense_basis @ dense_basis.T - sparse_basis @ sparse_basis.T,
        ord="fro",
    )
    assert difference < 0.15


def test_invalid_parameters():
    _, X, _ = _sparse_low_rank(10, n=30, p=18)
    with pytest.raises(ValueError, match="alpha"):
        rc.SparseCellPCA(n_components=3, alpha=-0.1).fit(X)
    with pytest.raises(ValueError, match="one value per component"):
        rc.SparseCellPCA(n_components=3, alpha=[0.1, 0.2]).fit(X)
    with pytest.raises(ValueError, match="l1_ratio"):
        rc.SparseCellPCA(n_components=3, l1_ratio=1.2).fit(X)
    with pytest.raises(ValueError, match="loading_max_iter"):
        rc.SparseCellPCA(n_components=3, loading_max_iter=0).fit(X)
    with pytest.raises(ValueError, match="loading_tol"):
        rc.SparseCellPCA(n_components=3, loading_tol=0.0).fit(X)
    with pytest.raises(ValueError, match="sparsity_threshold"):
        rc.SparseCellPCA(n_components=3, sparsity_threshold=-1.0).fit(X)


def test_unfitted_and_shape_errors():
    _, X, _ = _sparse_low_rank(11)
    model = rc.SparseCellPCA(n_components=3)
    with pytest.raises(AttributeError, match="not fitted"):
        model.transform(X)
    model.fit(X)
    with pytest.raises(ValueError, match="features"):
        model.transform(X[:, :-1])


def test_plot_sparse_loadings(tmp_path):
    _, X, _ = _sparse_low_rank(12, n=70, p=24)
    model = rc.SparseCellPCA(
        n_components=3,
        alpha=0.05,
        max_iter=25,
        loading_max_iter=30,
        tol=8e-4,
    ).fit(X)
    path = tmp_path / "sparse_loadings.png"
    fig = rc.plot_sparse_cellpca_loadings(
        model,
        feature_names=[f"x{j}" for j in range(X.shape[1])],
        output_path=path,
        show=False,
    )

    assert fig is not None
    assert path.exists() and path.stat().st_size > 0


def test_plot_validation():
    _, X, _ = _sparse_low_rank(13, n=60, p=18)
    model = rc.SparseCellPCA(n_components=3, alpha=0.04, max_iter=20).fit(X)
    with pytest.raises(ValueError, match="feature_names"):
        rc.plot_sparse_cellpca_loadings(model, feature_names=["too", "short"], show=False)
    with pytest.raises(ValueError, match="invalid component"):
        rc.plot_sparse_cellpca_loadings(model, components=[9], show=False)
