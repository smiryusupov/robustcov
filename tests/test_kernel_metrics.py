import numpy as np
import pytest

import robustcov as rc


class EmpiricalScatter:
    def fit(self, X, y=None):
        self.location_ = np.mean(X, axis=0)
        self.covariance_ = np.cov(X, rowvar=False) + 1e-6 * np.eye(X.shape[1])
        self.precision_ = np.linalg.pinv(self.covariance_)
        return self


def test_robust_input_metric_pairwise_distances():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))
    metric = rc.RobustInputMetric(estimator=EmpiricalScatter()).fit(X)
    D2 = metric.squared_distance(X[:4], X[4:9])
    assert D2.shape == (4, 5)
    assert np.isfinite(D2).all()
    assert np.all(D2 >= 0)

    # Centering cancels for pairwise distances, so this matches the direct form.
    diff = X[0] - X[4]
    expected = diff @ metric.precision_ @ diff
    assert D2[0, 0] == pytest.approx(expected)


def test_robust_rbf_and_matern_kernel_matrices():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(10, 2))
    precision = np.array([[2.0, 0.3], [0.3, 0.8]])
    K_rbf = rc.robust_rbf_kernel(X, precision=precision, length_scale=1.2)
    K_matern = rc.robust_matern_kernel(X, precision=precision, length_scale=1.2, nu=1.5)
    assert K_rbf.shape == (10, 10)
    assert K_matern.shape == (10, 10)
    np.testing.assert_allclose(np.diag(K_rbf), 1.0)
    np.testing.assert_allclose(K_rbf, K_rbf.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(K_matern), 1.0)
    assert np.all((K_rbf >= 0) & (K_rbf <= 1))
    assert np.all((K_matern >= 0) & (K_matern <= 1))


def test_single_row_kernel_prediction_shape():
    X = np.array([[0.0, 0.0]])
    Y = np.array([[1.0, 0.0], [0.0, 2.0]])
    precision = np.eye(2)
    K = rc.robust_rbf_kernel(X, Y, precision=precision)
    assert K.shape == (1, 2)


def test_sklearn_kernel_adapter_gradient():
    pytest.importorskip("sklearn")
    from robustcov.sklearn_kernels import RobustMahalanobisRBF

    rng = np.random.default_rng(2)
    X = rng.normal(size=(8, 2))
    kernel = RobustMahalanobisRBF(precision=np.eye(2), length_scale=2.0)
    K, grad = kernel(X, eval_gradient=True)
    assert K.shape == (8, 8)
    assert grad.shape == (8, 8, 1)
    np.testing.assert_allclose(np.diag(K), 1.0)
    np.testing.assert_allclose(np.diag(grad[:, :, 0]), 0.0, atol=1e-12)


def test_gpytorch_kernel_adapter_forward():
    torch = pytest.importorskip("torch")
    pytest.importorskip("gpytorch")
    from robustcov.gpytorch_kernels import RobustMahalanobisRBFKernel

    X = torch.randn(6, 2)
    kernel = RobustMahalanobisRBFKernel(precision=np.eye(2))
    K = kernel(X).to_dense()
    assert tuple(K.shape) == (6, 6)
    torch.testing.assert_close(torch.diagonal(K), torch.ones(6, dtype=K.dtype))
