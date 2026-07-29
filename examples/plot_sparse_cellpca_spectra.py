"""Sparse cellwise-robust PCA for interpretable spectra.

The example simulates a high-dimensional spectral panel in which each latent
component acts on a short wavelength band.  Isolated bad cells, abnormal full
spectra, and missing readings are added before fitting dense CellPCA and
SparseCellPCA.  The sparse model is evaluated both for subspace recovery and
for recovering the active wavelength bands.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

import robustcov as rc


OUT = Path("results/use_cases/sparse_cellpca_spectra")
OUT.mkdir(parents=True, exist_ok=True)


def make_data(seed: int = 41):
    rng = np.random.default_rng(seed)
    n, p, q = 120, 48, 3
    wavelengths = np.linspace(900.0, 1800.0, p)
    loadings = np.zeros((p, q))
    starts = (3, 19, 35)
    shape = np.array([0.28, 0.48, 0.70, 0.92, 1.0, 0.78, 0.45])
    for k, start in enumerate(starts):
        loadings[start : start + shape.size, k] = shape * (-1.0 if k == 1 else 1.0)
    loadings, _ = np.linalg.qr(loadings)
    scores = rng.normal(size=(n, q)) * np.array([3.0, 2.1, 1.4])
    clean = scores @ loadings.T + 0.10 * rng.normal(size=(n, p))

    X = clean.copy()
    cell_truth = np.zeros_like(X, dtype=bool)
    bad = rng.choice(X.size, size=int(round(0.045 * X.size)), replace=False)
    cell_truth.flat[bad] = True
    X.flat[bad] += rng.choice([-1.0, 1.0], bad.size) * rng.uniform(4.0, 7.0, bad.size)

    case_truth = np.zeros(n, dtype=bool)
    case_truth[:7] = True
    X[case_truth] += rng.normal(0.0, 3.5, size=(case_truth.sum(), p))

    missing = (rng.random(X.shape) < 0.03) & ~cell_truth
    X[missing] = np.nan
    return wavelengths, clean, X, loadings, cell_truth, case_truth, missing


def projection_error(components: np.ndarray, truth: np.ndarray) -> float:
    basis, _ = np.linalg.qr(np.asarray(components).T)
    return float(
        np.linalg.norm(basis @ basis.T - truth @ truth.T, ord="fro")
        / np.sqrt(2.0 * truth.shape[1])
    )


def aligned_loadings(components: np.ndarray, truth: np.ndarray) -> np.ndarray:
    estimate = np.asarray(components, dtype=float).T
    similarity = np.abs(truth.T @ estimate)
    rows, cols = linear_sum_assignment(-similarity)
    aligned = estimate[:, cols]
    aligned = aligned * np.sign(np.sum(aligned * truth[:, rows], axis=0))[None, :]
    return aligned


def support_metrics(components: np.ndarray, truth: np.ndarray):
    aligned = aligned_loadings(components, truth)
    expected = np.abs(truth) > 1e-12
    predicted = np.abs(aligned) > 1e-12
    tp = int(np.count_nonzero(expected & predicted))
    fp = int(np.count_nonzero(~expected & predicted))
    fn = int(np.count_nonzero(expected & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, aligned


wavelengths, clean, X, truth_loadings, cell_truth, case_truth, missing = make_data()
q = truth_loadings.shape[1]
medians = np.nanmedian(X, axis=0)
imputed = np.where(np.isnan(X), medians, X)

classical = PCA(n_components=q).fit(imputed)
dense = rc.CellPCA(n_components=q, max_iter=55, tol=5e-4).fit(X)
sparse = rc.SparseCellPCA(
    n_components=q,
    alpha=0.055,
    l1_ratio=1.0,
    sparsity_threshold=0.02,
    max_iter=45,
    loading_max_iter=45,
    tol=5e-4,
).fit(X)

valid = ~missing
clean_missing = missing & ~case_truth[:, None]
metrics = {
    "classical_subspace_error": projection_error(classical.components_, truth_loadings),
    "cellpca_subspace_error": projection_error(dense.components_, truth_loadings),
    "sparse_cellpca_subspace_error": projection_error(sparse.components_, truth_loadings),
    "cellpca_cell_auc": roc_auc_score(
        cell_truth[valid], np.abs(dense.standardized_residuals_[valid])
    ),
    "sparse_cellpca_cell_auc": roc_auc_score(
        cell_truth[valid], np.abs(sparse.standardized_residuals_[valid])
    ),
    "cellpca_missing_mae": np.mean(
        np.abs(dense.fitted_values_[clean_missing] - clean[clean_missing])
    ),
    "sparse_cellpca_missing_mae": np.mean(
        np.abs(sparse.fitted_values_[clean_missing] - clean[clean_missing])
    ),
}
precision, recall, f1, aligned_sparse = support_metrics(
    sparse.components_, truth_loadings
)
metrics.update(
    support_precision=precision,
    support_recall=recall,
    support_f1=f1,
    loading_sparsity=sparse.sparsity_,
    nonzero_loadings=int(np.count_nonzero(sparse.components_)),
    total_loadings=int(sparse.components_.size),
)

print(f"data shape: {X.shape}")
print(f"cell contamination: {cell_truth.mean():.1%}")
print(f"abnormal rows: {case_truth.sum()}")
print(f"missing cells: {missing.mean():.1%}")
print("subspace error, classical / CellPCA / SparseCellPCA:")
print(
    f"{metrics['classical_subspace_error']:.3f} / "
    f"{metrics['cellpca_subspace_error']:.3f} / "
    f"{metrics['sparse_cellpca_subspace_error']:.3f}"
)
print(
    "cell-outlier AUROC, CellPCA / SparseCellPCA: "
    f"{metrics['cellpca_cell_auc']:.3f} / {metrics['sparse_cellpca_cell_auc']:.3f}"
)
print(
    "clean missing-cell MAE, CellPCA / SparseCellPCA: "
    f"{metrics['cellpca_missing_mae']:.3f} / "
    f"{metrics['sparse_cellpca_missing_mae']:.3f}"
)
print(
    "loading support precision / recall / F1: "
    f"{precision:.3f} / {recall:.3f} / {f1:.3f}"
)
print(
    f"exact-zero loading fraction: {sparse.sparsity_:.1%} "
    f"({np.count_nonzero(sparse.components_)} / {sparse.components_.size} nonzero)"
)

# Loading comparison after component alignment.
aligned_dense = aligned_loadings(dense.components_, truth_loadings)
fig, axes = plt.subplots(q, 1, figsize=(10, 7), sharex=True)
for k, ax in enumerate(axes):
    ax.plot(wavelengths, truth_loadings[:, k], label="truth", linewidth=2)
    ax.plot(wavelengths, aligned_dense[:, k], label="CellPCA", alpha=0.75)
    ax.step(
        wavelengths,
        aligned_sparse[:, k],
        where="mid",
        label="SparseCellPCA",
        linewidth=1.6,
    )
    ax.axhline(0.0, linewidth=0.7)
    ax.set_ylabel(f"PC {k + 1}")
axes[-1].set_xlabel("wavelength")
axes[0].legend(ncol=3)
fig.suptitle("Dense and sparse robust loading recovery")
fig.tight_layout()
fig.savefig(OUT / "loading_comparison.png", dpi=160)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
axes[0].bar(
    ["Classical", "CellPCA", "Sparse"],
    [
        metrics["classical_subspace_error"],
        metrics["cellpca_subspace_error"],
        metrics["sparse_cellpca_subspace_error"],
    ],
)
axes[0].set_ylabel("normalized projection error")
axes[0].set_title("Subspace recovery")
axes[1].bar(["precision", "recall", "F1"], [precision, recall, f1])
axes[1].set_ylim(0.0, 1.05)
axes[1].set_title("Sparse loading support")
fig.tight_layout()
fig.savefig(OUT / "performance_comparison.png", dpi=160)
plt.close(fig)

rc.plot_sparse_cellpca_loadings(
    sparse,
    feature_names=[f"{value:.0f}" for value in wavelengths],
    title="Sparse robust loading matrix",
    output_path=OUT / "sparse_loadings.png",
    show=False,
)
rc.plot_cellpca_outlier_map(
    sparse,
    labels=case_truth,
    title="SparseCellPCA casewise and cellwise diagnostics",
    output_path=OUT / "outlier_map.png",
    show=False,
)

with (OUT / "metrics.csv").open("w", encoding="utf-8") as handle:
    handle.write("metric,value\n")
    for key, value in metrics.items():
        handle.write(f"{key},{value}\n")
