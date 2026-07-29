# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

import robustcov as rc


def _low_rank_data(seed=0, n=120, p=12, rank=2, noise=0.12):
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(p, rank)))
    scores = rng.normal(size=(n, rank)) * np.linspace(3.0, 1.4, rank)
    X = scores @ basis.T + noise * rng.normal(size=(n, p))
    return rng, X, basis


def _projection_error(components, basis):
    estimated = components.T @ components
    truth = basis @ basis.T
    return np.linalg.norm(estimated - truth, ord="fro")


def test_public_aliases():
    assert rc.CellPCA is rc.CellwiseRobustPCA
    assert rc.CasewiseCellwisePCA is rc.CellwiseRobustPCA


def test_clean_fit_returns_orthonormal_components_and_scores():
    _, X, _ = _low_rank_data(1)
    model = rc.CellPCA(n_components=2, max_iter=50).fit(X)

    assert model.components_.shape == (2, X.shape[1])
    assert model.scores_.shape == (X.shape[0], 2)
    assert np.allclose(model.components_ @ model.components_.T, np.eye(2), atol=1e-8)
    assert np.isfinite(model.explained_variance_).all()
    assert np.all(model.explained_variance_[:-1] >= model.explained_variance_[1:])
    assert np.isfinite(model.fitted_values_).all()


def test_clean_subspace_is_close_to_classical_pca():
    _, X, _ = _low_rank_data(2, noise=0.08)
    robust = rc.CellPCA(n_components=2, max_iter=60).fit(X)
    classical = PCA(n_components=2).fit(X)

    difference = np.linalg.norm(
        robust.components_.T @ robust.components_
        - classical.components_.T @ classical.components_,
        ord="fro",
    )
    assert difference < 0.12


def test_combined_contamination_improves_subspace_recovery():
    rng, clean, basis = _low_rank_data(3, n=140, p=14, noise=0.10)
    X = clean.copy()
    cell_truth = np.zeros_like(X, dtype=bool)
    bad = rng.choice(X.size, size=int(0.07 * X.size), replace=False)
    cell_truth.flat[bad] = True
    X.flat[bad] += rng.choice([-1.0, 1.0], size=bad.size) * rng.uniform(5.0, 9.0, size=bad.size)
    case_truth = np.zeros(X.shape[0], dtype=bool)
    case_truth[:10] = True
    X[case_truth] += rng.normal(0.0, 6.0, size=(case_truth.sum(), X.shape[1]))

    classical = PCA(n_components=2).fit(X)
    robust = rc.CellPCA(n_components=2, max_iter=80).fit(X)

    assert _projection_error(robust.components_, basis) < 0.35
    assert _projection_error(robust.components_, basis) < 0.45 * _projection_error(
        classical.components_, basis
    )
    assert roc_auc_score(cell_truth.ravel(), np.abs(robust.standardized_residuals_).ravel()) > 0.95
    assert roc_auc_score(case_truth, robust.case_deviations_) > 0.95


def test_missing_values_are_imputed_and_transform_accepts_incomplete_rows():
    rng, X, _ = _low_rank_data(4)
    missing = rng.random(X.shape) < 0.08
    X_missing = X.copy()
    X_missing[missing] = np.nan

    model = rc.CellPCA(n_components=2).fit(X_missing)
    scores = model.transform(X_missing[:7])
    diagnostics = model.cellwise_diagnostics(X_missing[:7])

    assert scores.shape == (7, 2)
    assert np.isfinite(scores).all()
    assert np.isfinite(model.imputed_data_).all()
    assert np.isfinite(diagnostics["imputed_data"]).all()
    assert np.array_equal(model.missing_mask_, missing)
    assert np.allclose(model.imputed_data_[~missing], X_missing[~missing])


def test_corrected_data_replaces_only_missing_or_flagged_cells():
    rng, X, _ = _low_rank_data(5)
    X[8, 3] += 15.0
    X[10, 6] = np.nan
    model = rc.CellPCA(n_components=2).fit(X)

    replace = model.missing_mask_ | model.cell_outlier_mask_
    keep = ~replace
    assert np.allclose(model.corrected_data_[keep], X[keep])
    assert np.allclose(model.corrected_data_[replace], model.fitted_values_[replace])


def test_cell_and_case_weights_are_bounded():
    rng, X, _ = _low_rank_data(6)
    X.flat[rng.choice(X.size, 40, replace=False)] += 8.0
    model = rc.CellPCA(n_components=2).fit(X)

    assert model.cell_weights_.shape == X.shape
    assert model.case_weights_.shape == (X.shape[0],)
    assert np.all((0.0 <= model.cell_weights_) & (model.cell_weights_ <= 1.0))
    assert np.all((0.0 <= model.case_weights_) & (model.case_weights_ <= 1.0))


def test_fit_transform_inverse_and_reconstruct():
    _, X, _ = _low_rank_data(7)
    model = rc.CellPCA(n_components=3, store_scores=False)
    scores = model.fit_transform(X)
    reconstructed = model.inverse_transform(scores)

    assert scores.shape == (X.shape[0], 3)
    assert reconstructed.shape == X.shape
    assert np.allclose(reconstructed, model.reconstruct(X), atol=1e-7)
    assert not hasattr(model, "scores_")


def test_outlier_map_columns_match_diagnostics():
    _, X, _ = _low_rank_data(8)
    model = rc.CellPCA(n_components=2).fit(X)
    mapping = model.outlier_map()
    mapping_new = model.outlier_map(X[:5])

    assert mapping.shape == (X.shape[0], 2)
    assert np.allclose(mapping[:, 0], model.case_deviations_)
    assert np.allclose(mapping[:, 1], model.max_cell_residuals_)
    assert mapping_new.shape == (5, 2)


def test_deterministic_fit():
    rng, X, _ = _low_rank_data(9)
    X.flat[rng.choice(X.size, 30, replace=False)] += 6.0
    first = rc.CellPCA(n_components=2).fit(X)
    second = rc.CellPCA(n_components=2).fit(X)

    assert np.allclose(first.center_, second.center_)
    assert np.allclose(first.components_, second.components_)
    assert np.allclose(first.cell_weights_, second.cell_weights_)


def test_invalid_parameters_and_inputs():
    _, X, _ = _low_rank_data(10, n=20, p=6)
    with pytest.raises(ValueError, match="n_components"):
        rc.CellPCA(n_components=0).fit(X)
    with pytest.raises(ValueError, match="n_components"):
        rc.CellPCA(n_components=6).fit(X)
    with pytest.raises(ValueError, match="0 < b < c"):
        rc.CellPCA(cell_b=4.0, cell_c=2.0).fit(X)
    with pytest.raises(ValueError, match="weight_threshold"):
        rc.CellPCA(weight_threshold=1.0).fit(X)
    with pytest.raises(ValueError, match="infinity"):
        rc.CellPCA().fit(np.where(np.arange(X.size).reshape(X.shape) == 0, np.inf, X))

    all_missing_column = X.copy()
    all_missing_column[:, 2] = np.nan
    with pytest.raises(ValueError, match="at least two finite"):
        rc.CellPCA().fit(all_missing_column)

    all_missing_row = X.copy()
    all_missing_row[0] = np.nan
    with pytest.raises(ValueError, match="every row"):
        rc.CellPCA().fit(all_missing_row)


def test_unfitted_and_shape_errors():
    _, X, _ = _low_rank_data(11)
    model = rc.CellPCA(n_components=2)
    with pytest.raises(AttributeError, match="not fitted"):
        model.transform(X)
    model.fit(X)
    with pytest.raises(ValueError, match="features"):
        model.transform(X[:, :-1])
    with pytest.raises(ValueError, match="components"):
        model.inverse_transform(np.ones((3, 1)))


def test_plotting_helpers(tmp_path):
    pytest.importorskip("matplotlib")
    rng, X, _ = _low_rank_data(12, n=60, p=8)
    X.flat[rng.choice(X.size, 20, replace=False)] += 7.0
    model = rc.CellPCA(n_components=2).fit(X)
    residual_path = tmp_path / "residuals.png"
    outlier_path = tmp_path / "outlier_map.png"

    fig1 = rc.plot_cellwise_residual_map(
        model, output_path=residual_path, show=False
    )
    fig2 = rc.plot_cellpca_outlier_map(
        model, output_path=outlier_path, show=False
    )

    assert fig1 is not None and fig2 is not None
    assert residual_path.exists() and residual_path.stat().st_size > 0
    assert outlier_path.exists() and outlier_path.stat().st_size > 0


def _loop_weighted_scores(X, observed, weights, center, loadings, ridge):
    n = X.shape[0]
    q = loadings.shape[1]
    scores = np.zeros((n, q), dtype=np.float64)
    identity = np.eye(q)
    centered = np.where(observed, X, center) - center
    for i in range(n):
        w = weights[i]
        if not np.any(w > 0.0):
            continue
        gram = loadings.T @ (w[:, None] * loadings) + ridge * identity
        rhs = loadings.T @ (w * centered[i])
        scores[i] = np.linalg.solve(gram, rhs)
    return scores


def _loop_weighted_loadings(
    X, observed, weights, center, scores, previous, ridge
):
    p = X.shape[1]
    q = scores.shape[1]
    identity = np.eye(q)
    centered = np.where(observed, X, center) - center
    loadings = previous.copy()
    working_scores = scores.copy()
    for j in range(p):
        w = weights[:, j]
        if not np.any(w > 0.0):
            continue
        gram = working_scores.T @ (w[:, None] * working_scores) + ridge * identity
        rhs = working_scores.T @ (w * centered[:, j])
        loadings[j] = np.linalg.solve(gram, rhs)
    qmat, rmat = np.linalg.qr(loadings, mode="reduced")
    working_scores[:] = working_scores @ rmat.T
    return qmat, working_scores


def test_batched_weighted_normal_equations_match_loop_reference():
    import robustcov.cellpca as cellpca

    rng = np.random.default_rng(123)
    X = rng.normal(size=(37, 11))
    observed = rng.random(X.shape) > 0.12
    X = np.where(observed, X, np.nan)
    weights = rng.random(X.shape) * observed
    weights[3] = 0.0
    weights[:, 7] = 0.0
    center = rng.normal(size=X.shape[1])
    loadings, _ = np.linalg.qr(rng.normal(size=(X.shape[1], 4)))
    ridge = 1e-7

    expected_scores = _loop_weighted_scores(
        X, observed, weights, center, loadings, ridge
    )
    actual_scores = cellpca._weighted_scores(
        X, observed, weights, center, loadings, ridge
    )
    assert np.allclose(actual_scores, expected_scores, rtol=1e-12, atol=1e-12)

    expected_loadings, expected_rotated_scores = _loop_weighted_loadings(
        X,
        observed,
        weights,
        center,
        expected_scores,
        loadings,
        ridge,
    )
    actual_input_scores = actual_scores.copy()
    actual_loadings = cellpca._weighted_loadings(
        X,
        observed,
        weights,
        center,
        actual_input_scores,
        loadings,
        ridge,
    )
    assert np.allclose(actual_loadings, expected_loadings, rtol=1e-12, atol=1e-12)
    assert np.allclose(
        actual_input_scores,
        expected_rotated_scores,
        rtol=1e-12,
        atol=1e-12,
    )
