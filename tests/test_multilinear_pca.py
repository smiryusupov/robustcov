# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

import robustcov as rc
from robustcov._native import native_available


def _tensor_data(seed=0, n=80, r=8, c=10, ranks=(2, 3), noise=0.08):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(r, ranks[0])))
    V, _ = np.linalg.qr(rng.normal(size=(c, ranks[1])))
    cores = rng.normal(size=(n, *ranks))
    scales = np.linspace(3.0, 1.0, ranks[0] * ranks[1]).reshape(ranks)
    cores *= scales
    center = rng.normal(scale=0.2, size=(r, c))
    X = center + np.einsum("au,nuv,bv->nab", U, cores, V, optimize=True)
    X += noise * rng.normal(size=X.shape)
    return rng, X, U, V


def _projection_error(estimated, truth):
    return np.linalg.norm(estimated @ estimated.T - truth @ truth.T, ord="fro")


def test_public_aliases():
    assert rc.CasewiseCellwiseMultilinearPCA is rc.RobustMultilinearPCA
    assert rc.CellwiseRobustMultilinearPCA is rc.RobustMultilinearPCA


def test_clean_fit_recovers_both_mode_subspaces():
    _, X, U, V = _tensor_data(1)
    model = rc.RobustMultilinearPCA(ranks=(2, 3), max_iter=40).fit(X)
    assert model.row_components_.shape == U.shape
    assert model.column_components_.shape == V.shape
    assert np.allclose(model.row_components_.T @ model.row_components_, np.eye(2), atol=1e-8)
    assert np.allclose(model.column_components_.T @ model.column_components_, np.eye(3), atol=1e-8)
    assert _projection_error(model.row_components_, U) < 0.15
    assert _projection_error(model.column_components_, V) < 0.18
    assert model.core_scores_.shape == (X.shape[0], 2, 3)


def test_mixed_contamination_and_missing_values_are_detected():
    rng, clean, U, V = _tensor_data(2, n=95)
    X = clean.copy()
    cell_truth = np.zeros_like(X, dtype=bool)
    bad = rng.choice(X.size, size=int(0.045 * X.size), replace=False)
    cell_truth.flat[bad] = True
    X.flat[bad] += rng.choice([-1.0, 1.0], size=bad.size) * rng.uniform(5.0, 8.0, size=bad.size)
    case_truth = np.zeros(X.shape[0], dtype=bool)
    case_truth[:8] = True
    X[case_truth] += rng.normal(0.0, 4.5, size=X[case_truth].shape)
    missing = rng.random(X.shape) < 0.03
    X[missing] = np.nan

    model = rc.RobustMultilinearPCA(ranks=(2, 3), max_iter=50).fit(X)
    assert _projection_error(model.row_components_, U) < 0.35
    assert _projection_error(model.column_components_, V) < 0.4
    valid = ~missing
    assert roc_auc_score(cell_truth[valid], np.abs(model.standardized_residuals_[valid])) > 0.95
    assert roc_auc_score(case_truth, model.case_deviations_) > 0.95
    assert np.isfinite(model.imputed_data_).all()
    assert np.allclose(model.imputed_data_[valid], X[valid])


def test_transform_inverse_and_correction():
    rng, X, _, _ = _tensor_data(3)
    X[4, 2, 5] += 10.0
    X[7, 3, 4] = np.nan
    model = rc.RobustMultilinearPCA(ranks=(2, 2), max_iter=35).fit(X)
    cores = model.transform(X[:6])
    reconstructed = model.inverse_transform(cores)
    diagnostics = model.cellwise_diagnostics(X[:6])
    assert cores.shape == (6, 2, 2)
    assert reconstructed.shape == X[:6].shape
    assert np.allclose(reconstructed, diagnostics["fitted_values"], atol=1e-7)
    assert np.isfinite(model.correct(X[:6])).all()
    assert model.outlier_map().shape == (X.shape[0], 2)


def test_python_and_cpp_backends_are_equivalent():
    if not native_available():
        pytest.skip("native extension unavailable")
    rng, X, _, _ = _tensor_data(4, n=55, r=7, c=9, ranks=(2, 2))
    X.flat[rng.choice(X.size, 40, replace=False)] += 5.0
    python = rc.RobustMultilinearPCA(ranks=(2, 2), max_iter=20, backend="python").fit(X)
    cpp = rc.RobustMultilinearPCA(ranks=(2, 2), max_iter=20, backend="cpp").fit(X)
    assert np.allclose(cpp.fitted_values_, python.fitted_values_, rtol=1e-10, atol=1e-10)
    assert np.allclose(cpp.cell_weights_, python.cell_weights_, rtol=1e-10, atol=1e-10)


def test_mmcd_backend_equivalence():
    if not native_available():
        pytest.skip("native extension unavailable")
    rng = np.random.default_rng(5)
    X = rng.normal(size=(30, 3, 4))
    kwargs = dict(n_init=8, n_best=3, max_iter=8, flip_flop_max_iter=20, random_state=2)
    python = rc.MMCD(backend="python", **kwargs).fit(X)
    cpp = rc.MMCD(backend="cpp", **kwargs).fit(X)
    assert np.allclose(cpp.distances_, python.distances_, rtol=1e-9, atol=1e-9)
    assert np.array_equal(cpp.support_, python.support_)


def test_invalid_inputs_and_plotting(tmp_path):
    pytest.importorskip("matplotlib")
    _, X, _, _ = _tensor_data(6, n=35, r=6, c=7, ranks=(2, 2))
    with pytest.raises(ValueError, match="ranks"):
        rc.RobustMultilinearPCA(ranks=(6, 2)).fit(X)
    with pytest.raises(ValueError, match="shape"):
        rc.RobustMultilinearPCA().fit(X.reshape(X.shape[0], -1))
    model = rc.RobustMultilinearPCA(ranks=(2, 2), max_iter=20).fit(X)
    residual_path = tmp_path / "residual.png"
    outlier_path = tmp_path / "outlier.png"
    fig1 = rc.plot_multilinear_residual_map(model, index=0, output_path=residual_path, show=False)
    fig2 = rc.plot_multilinear_outlier_map(model, output_path=outlier_path, show=False)
    assert fig1 is not None and fig2 is not None
    assert residual_path.exists() and outlier_path.exists()
