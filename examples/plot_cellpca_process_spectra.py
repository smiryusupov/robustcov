"""CellPCA for process spectra with bad wavelengths and abnormal batches.

Each row is a synthetic process spectrum.  The clean data follow a three-factor
subspace.  The observed table also contains isolated wavelength errors, a few
whole-batch deviations, and missing measurements.  The example compares
classical PCA after median imputation with CellPCA.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

import robustcov as rc


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "use_cases" / "cellpca_process_spectra"
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(92)
n_samples = 180
n_wavelengths = 28
rank = 3
wavelengths = np.linspace(900, 1700, n_wavelengths)

# Smooth loading curves make the synthetic matrix resemble low-resolution
# process or near-infrared spectra rather than an arbitrary Gaussian matrix.
raw_loadings = np.column_stack(
    [
        np.exp(-0.5 * ((wavelengths - 1080) / 150) ** 2),
        np.exp(-0.5 * ((wavelengths - 1390) / 180) ** 2)
        - 0.45 * np.exp(-0.5 * ((wavelengths - 1050) / 120) ** 2),
        np.sin((wavelengths - 900) / 800 * 2.4 * np.pi),
    ]
)
true_basis, _ = np.linalg.qr(raw_loadings)
latent = rng.normal(size=(n_samples, rank)) * np.array([3.0, 1.8, 1.0])
baseline = 0.8 + 0.00035 * (wavelengths - wavelengths.mean())
clean = baseline + latent @ true_basis.T + 0.10 * rng.normal(
    size=(n_samples, n_wavelengths)
)

X = clean.copy()
cell_truth = np.zeros_like(X, dtype=bool)
cell_indices = rng.choice(X.size, size=int(0.055 * X.size), replace=False)
cell_truth.flat[cell_indices] = True
X.flat[cell_indices] += rng.choice([-1.0, 1.0], size=cell_indices.size) * rng.uniform(
    3.5, 7.0, size=cell_indices.size
)

case_truth = np.zeros(n_samples, dtype=bool)
case_rows = rng.choice(n_samples, size=12, replace=False)
case_truth[case_rows] = True
# Abnormal batches depart smoothly from the clean subspace rather than through
# one extreme coordinate.
case_pattern = np.cos((wavelengths - 900) / 800 * 1.5 * np.pi)
case_pattern -= true_basis @ (true_basis.T @ case_pattern)
case_pattern /= np.linalg.norm(case_pattern)
X[case_rows] += rng.choice([-1.0, 1.0], size=(case_rows.size, 1)) * rng.uniform(
    2.5, 4.0, size=(case_rows.size, 1)
) * case_pattern

missing_truth = np.zeros_like(X, dtype=bool)
available = np.flatnonzero(~cell_truth.ravel())
missing_indices = rng.choice(available, size=int(0.03 * X.size), replace=False)
missing_truth.flat[missing_indices] = True
X[missing_truth] = np.nan

median = np.nanmedian(X, axis=0)
median_imputed = np.where(np.isnan(X), median, X)
classical = PCA(n_components=rank).fit(median_imputed)
cellpca = rc.CellPCA(
    n_components=rank,
    max_iter=80,
    tol=1e-6,
).fit(X)


def subspace_error(components):
    projection = components.T @ components
    truth = true_basis @ true_basis.T
    return np.linalg.norm(projection - truth, ord="fro")


classical_error = subspace_error(classical.components_)
cellpca_error = subspace_error(cellpca.components_)
valid_cells = ~missing_truth
cell_scores = np.nan_to_num(
    np.abs(cellpca.standardized_residuals_), nan=0.0, posinf=0.0, neginf=0.0
)
cell_auc = roc_auc_score(cell_truth[valid_cells], cell_scores[valid_cells])
case_auc = roc_auc_score(case_truth, cellpca.case_deviations_)
missing_mae = np.mean(
    np.abs(cellpca.imputed_data_[missing_truth] - clean[missing_truth])
)
median_mae = np.mean(np.abs(median_imputed[missing_truth] - clean[missing_truth]))

print(f"data shape: {X.shape}; retained rank: {rank}")
print(f"corrupted cells: {cell_truth.sum()} ({cell_truth.mean():.1%})")
print(f"abnormal batches: {case_truth.sum()} ({case_truth.mean():.1%})")
print(f"missing cells: {missing_truth.sum()} ({missing_truth.mean():.1%})")
print(
    "subspace error, classical PCA / CellPCA: "
    f"{classical_error:.3f} / {cellpca_error:.3f}"
)
print(f"cell-outlier AUROC: {cell_auc:.3f}")
print(f"case-outlier AUROC: {case_auc:.3f}")
print(
    "missing-cell MAE, column median / CellPCA: "
    f"{median_mae:.3f} / {missing_mae:.3f}"
)
print(f"CellPCA iterations: {cellpca.n_iter_}; converged: {cellpca.converged_}")

fig = plt.figure(figsize=(7.2, 4.6))
ax = fig.add_subplot(111)
labels = ["Classical PCA", "CellPCA"]
values = [classical_error, cellpca_error]
ax.bar(labels, values)
ax.set_ylabel("projection-matrix error")
ax.set_title("Subspace recovery under mixed contamination")
for index, value in enumerate(values):
    ax.text(index, value, f"{value:.3f}", ha="center", va="bottom")
fig.tight_layout()
fig.savefig(OUT / "subspace_recovery.png", dpi=160)
plt.close(fig)

row_score = np.nanmax(cell_scores, axis=1)
selected_rows = np.argsort(row_score)[-55:]
rc.plot_cellwise_residual_map(
    cellpca,
    X[selected_rows],
    row_labels=[f"batch {row + 1}" for row in selected_rows],
    column_labels=[f"{value:.0f}" for value in wavelengths],
    title="Residual cellmap for the most suspicious batches",
    output_path=OUT / "residual_cellmap.png",
    show=False,
)

combined_truth = case_truth | cell_truth.any(axis=1)
rc.plot_cellpca_outlier_map(
    cellpca,
    labels=combined_truth,
    title="Whole-batch deviation versus largest cell residual",
    output_path=OUT / "outlier_map.png",
    show=False,
)

fig = plt.figure(figsize=(8.2, 4.8))
ax = fig.add_subplot(111)
for component in range(rank):
    ax.plot(wavelengths, cellpca.components_[component], label=f"component {component + 1}")
ax.set_xlabel("wavelength")
ax.set_ylabel("loading")
ax.set_title("CellPCA loading curves")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "loading_curves.png", dpi=160)
plt.close(fig)

(OUT / "metrics.csv").write_text(
    "metric,value\n"
    f"classical_subspace_error,{classical_error:.8f}\n"
    f"cellpca_subspace_error,{cellpca_error:.8f}\n"
    f"cell_auc,{cell_auc:.8f}\n"
    f"case_auc,{case_auc:.8f}\n"
    f"median_imputation_mae,{median_mae:.8f}\n"
    f"cellpca_imputation_mae,{missing_mae:.8f}\n",
    encoding="utf-8",
)
