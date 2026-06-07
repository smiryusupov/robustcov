from __future__ import annotations

import math

import numpy as np

try:  # optional dependencies
    import torch
    import gpytorch
except Exception as exc:  # pragma: no cover - exercised only without torch/gpytorch
    torch = None
    gpytorch = None
    _GPYTORCH_IMPORT_ERROR = exc
else:
    _GPYTORCH_IMPORT_ERROR = None


def _require_gpytorch():
    if _GPYTORCH_IMPORT_ERROR is not None:
        raise ImportError("torch and gpytorch are required for robustcov.gpytorch_kernels") from _GPYTORCH_IMPORT_ERROR


if gpytorch is not None:
    _KernelBase = gpytorch.kernels.Kernel
else:
    _KernelBase = object


class _BaseRobustMahalanobisTorchKernel(_KernelBase):
    """Shared GPyTorch adapter for frozen robust Mahalanobis metrics."""

    has_lengthscale = True

    def __init__(self, precision, center=None, ridge=0.0, **kwargs):
        _require_gpytorch()
        super().__init__(**kwargs)
        precision_t = torch.as_tensor(np.asarray(precision, dtype=np.float64))
        if precision_t.ndim != 2 or precision_t.shape[0] != precision_t.shape[1]:
            raise ValueError("precision must be a square 2D array")
        precision_t = 0.5 * (precision_t + precision_t.transpose(-1, -2))
        if ridge:
            precision_t = precision_t + float(ridge) * torch.eye(precision_t.shape[0], dtype=precision_t.dtype)
        self.register_buffer("precision", precision_t)
        self.has_center = center is not None
        if center is None:
            center_t = torch.zeros(precision_t.shape[0], dtype=precision_t.dtype)
        else:
            center_t = torch.as_tensor(np.asarray(center, dtype=np.float64))
            if center_t.shape != (precision_t.shape[0],):
                raise ValueError("center must have shape (n_features,)")
        self.register_buffer("center", center_t)

    @classmethod
    def from_metric(cls, metric, **kwargs):
        return cls(precision=metric.precision_, center=getattr(metric, "location_", None), **kwargs)

    def _distance(self, x1, x2, diag=False, **params):
        P = self.precision.to(dtype=x1.dtype, device=x1.device)
        if self.has_center:
            c = self.center.to(dtype=x1.dtype, device=x1.device)
            x1 = x1 - c
            x2 = x2 - c
        if diag:
            diff = x1 - x2
            return torch.sum((diff @ P) * diff, dim=-1).clamp_min(0)
        x1p = x1 @ P
        x2p = x2 @ P
        x1_norm = torch.sum(x1p * x1, dim=-1).unsqueeze(-1)
        x2_norm = torch.sum(x2p * x2, dim=-1).unsqueeze(-2)
        return (x1_norm + x2_norm - 2.0 * x1p @ x2.transpose(-1, -2)).clamp_min(0)


class RobustMahalanobisRBFKernel(_BaseRobustMahalanobisTorchKernel):
    """GPyTorch RBF kernel with a frozen robust Mahalanobis input metric."""

    def forward(self, x1, x2, diag=False, **params):
        d2 = self._distance(x1, x2, diag=diag, **params)
        ls2 = self.lengthscale.squeeze().pow(2)
        return torch.exp(-0.5 * d2 / ls2)


class RobustMahalanobisMaternKernel(_BaseRobustMahalanobisTorchKernel):
    """GPyTorch Matérn kernel with a frozen robust Mahalanobis input metric."""

    def __init__(self, precision, center=None, ridge=0.0, nu=1.5, **kwargs):
        super().__init__(precision=precision, center=center, ridge=ridge, **kwargs)
        self.nu = float(nu)
        if self.nu not in {0.5, 1.5, 2.5} and not math.isinf(self.nu):
            raise ValueError("GPyTorch adapter supports nu=0.5, 1.5, 2.5, or np.inf")

    def forward(self, x1, x2, diag=False, **params):
        d = torch.sqrt(self._distance(x1, x2, diag=diag, **params).clamp_min(0)) / self.lengthscale.squeeze()
        if math.isinf(self.nu):
            return torch.exp(-0.5 * d.pow(2))
        if self.nu == 0.5:
            return torch.exp(-d)
        if self.nu == 1.5:
            z = math.sqrt(3.0) * d
            return (1.0 + z) * torch.exp(-z)
        z = math.sqrt(5.0) * d
        return (1.0 + z + z.pow(2) / 3.0) * torch.exp(-z)


__all__ = ["RobustMahalanobisRBFKernel", "RobustMahalanobisMaternKernel"]
