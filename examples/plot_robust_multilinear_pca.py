"""Robust multilinear PCA for contaminated sensor-by-time windows."""
from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

import robustcov as rc

OUT = Path("results/use_cases/robust_multilinear_pca")
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20260719)

n, n_sensors, n_times = 110, 8, 12
ranks = (2, 3)
U_true, _ = np.linalg.qr(rng.normal(size=(n_sensors, ranks[0])))
V_true, _ = np.linalg.qr(rng.normal(size=(n_times, ranks[1])))
cores = rng.normal(size=(n, *ranks)) * np.linspace(3.0, 1.0, 6).reshape(ranks)
center = 0.15 * rng.normal(size=(n_sensors, n_times))
clean = center + np.einsum("au,nuv,bv->nab", U_true, cores, V_true, optimize=True)
clean += 0.08 * rng.normal(size=clean.shape)
X = clean.copy()

cell_truth = np.zeros_like(X, dtype=bool)
bad_cells = rng.choice(X.size, size=int(0.04 * X.size), replace=False)
cell_truth.flat[bad_cells] = True
X.flat[bad_cells] += rng.choice([-1.0, 1.0], size=bad_cells.size) * rng.uniform(5.0, 8.0, size=bad_cells.size)
case_truth = np.zeros(n, dtype=bool)
case_truth[:10] = True
X[case_truth] += rng.normal(0.0, 4.0, size=X[case_truth].shape)
missing = rng.random(X.shape) < 0.03
X[missing] = np.nan

# Median-imputed multilinear baseline.
median = np.nanmedian(X, axis=0)
safe = np.where(np.isfinite(X), X, median)
center_baseline = safe.mean(axis=0)
centered = safe - center_baseline
row_cov = np.einsum("nac,nbc->ab", centered, centered) / (n * n_times)
col_cov = np.einsum("nra,nrb->ab", centered, centered) / (n * n_sensors)
_, row_vectors = np.linalg.eigh(row_cov)
_, col_vectors = np.linalg.eigh(col_cov)
U_base = row_vectors[:, -ranks[0]:]
V_base = col_vectors[:, -ranks[1]:]
base_cores = np.einsum("au,nab,bv->nuv", U_base, centered, V_base)
base_fit = center_baseline + np.einsum("au,nuv,bv->nab", U_base, base_cores, V_base)

model = rc.RobustMultilinearPCA(
    ranks=ranks,
    max_iter=100,
    backend="auto",
).fit(X)

row_error_base = np.linalg.norm(U_base @ U_base.T - U_true @ U_true.T, ord="fro")
row_error_robust = np.linalg.norm(model.row_components_ @ model.row_components_.T - U_true @ U_true.T, ord="fro")
col_error_base = np.linalg.norm(V_base @ V_base.T - V_true @ V_true.T, ord="fro")
col_error_robust = np.linalg.norm(model.column_components_ @ model.column_components_.T - V_true @ V_true.T, ord="fro")
regular = ~(cell_truth | case_truth[:, None, None] | missing)
mae_base = float(np.mean(np.abs(base_fit[regular] - clean[regular])))
mae_robust = float(np.mean(np.abs(model.fitted_values_[regular] - clean[regular])))
cell_auc = roc_auc_score(cell_truth[~missing], np.abs(model.standardized_residuals_[~missing]))
case_auc = roc_auc_score(case_truth, model.case_deviations_)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
labels = ["Median MPCA", "Robust multilinear PCA"]
axes[0].bar(labels, [row_error_base, row_error_robust])
axes[0].set_ylabel("row-mode projection error")
axes[0].tick_params(axis="x", rotation=15)
axes[1].bar(labels, [col_error_base, col_error_robust])
axes[1].set_ylabel("column-mode projection error")
axes[1].tick_params(axis="x", rotation=15)
fig.tight_layout()
fig.savefig(OUT / "mode_subspaces.png", dpi=150)
plt.close(fig)

rc.plot_multilinear_residual_map(
    model,
    index=int(np.argmax(model.max_cell_residuals_)),
    output_path=OUT / "residual_map.png",
    show=False,
)
rc.plot_multilinear_outlier_map(
    model,
    output_path=OUT / "outlier_map.png",
    show=False,
)

fig, ax = plt.subplots(figsize=(6.8, 4.2))
ax.bar(labels, [mae_base, mae_robust])
ax.set_ylabel("MAE on uncontaminated cells")
ax.tick_params(axis="x", rotation=15)
ax.set_title("Low-rank reconstruction")
fig.tight_layout()
fig.savefig(OUT / "reconstruction.png", dpi=150)
plt.close(fig)

metrics = {
    "row_error_baseline": row_error_base,
    "row_error_robust": row_error_robust,
    "column_error_baseline": col_error_base,
    "column_error_robust": col_error_robust,
    "clean_cell_mae_baseline": mae_base,
    "clean_cell_mae_robust": mae_robust,
    "cell_outlier_auc": cell_auc,
    "case_outlier_auc": case_auc,
    "iterations": model.n_iter_,
    "converged": model.converged_,
    "backend": model.backend_,
}
with (OUT / "metrics.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["metric", "value"])
    writer.writerows(metrics.items())

print(f"matrix sample shape: {X.shape}")
print(f"retained ranks: {ranks}")
print(f"backend: {model.backend_}")
print("row-mode projection error, baseline / robust:", f"{row_error_base:.3f} / {row_error_robust:.3f}")
print("column-mode projection error, baseline / robust:", f"{col_error_base:.3f} / {col_error_robust:.3f}")
print("clean-cell MAE, baseline / robust:", f"{mae_base:.3f} / {mae_robust:.3f}")
print("cell / case outlier AUROC:", f"{cell_auc:.3f} / {case_auc:.3f}")
