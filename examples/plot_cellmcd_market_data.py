"""CellMCD for isolated bad ticks and missing quotes.

The rows represent trading days and the columns represent asset returns.  A
small fraction of individual cells is corrupted, so more than half of the rows
contain at least one bad value.  The example compares empirical covariance,
rowwise MCD, and CellMCD.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score

import robustcov as rc


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "use_cases" / "cellmcd_market_data"
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(71)
n_days = 260
n_assets = 8
asset_names = [
    "Equity",
    "Credit",
    "Rates",
    "FX",
    "Energy",
    "Metals",
    "Real estate",
    "Volatility",
]

loadings = np.array(
    [
        [1.00, 0.20],
        [0.90, -0.15],
        [0.75, 0.35],
        [0.65, -0.40],
        [0.50, 0.55],
        [0.40, -0.60],
        [0.30, 0.75],
        [0.25, -0.80],
    ]
)
true_covariance = (
    loadings @ np.diag([0.00018, 0.00008]) @ loadings.T
    + np.diag(np.linspace(0.00004, 0.00009, n_assets))
)
clean = rng.multivariate_normal(np.zeros(n_assets), true_covariance, size=n_days)

X = clean.copy()
cell_truth = np.zeros_like(X, dtype=bool)
n_bad_cells = int(0.09 * X.size)
bad_flat = rng.choice(X.size, size=n_bad_cells, replace=False)
cell_truth.flat[bad_flat] = True
X.flat[bad_flat] += rng.choice([-1.0, 1.0], size=n_bad_cells) * rng.uniform(
    0.03, 0.07, size=n_bad_cells
)

missing_truth = np.zeros_like(X, dtype=bool)
available = np.flatnonzero(~cell_truth.ravel())
missing_flat = rng.choice(available, size=int(0.02 * X.size), replace=False)
missing_truth.flat[missing_flat] = True
X[missing_truth] = np.nan

median = np.nanmedian(X, axis=0)
median_imputed = np.where(np.isnan(X), median, X)
empirical_covariance = np.cov(median_imputed, rowvar=False)
rowwise = rc.FastMCD(
    contamination=0.40,
    quality="balanced",
    random_state=0,
    missing_values="median",
).fit(X)
cellwise = rc.CellMCD(
    alpha=0.75,
    quantile=0.99,
    max_iter=60,
    tol=1e-5,
).fit(X)

relative_error = {
    "Empirical": np.linalg.norm(empirical_covariance - true_covariance, ord="fro")
    / np.linalg.norm(true_covariance, ord="fro"),
    "FastMCD": np.linalg.norm(rowwise.covariance_ - true_covariance, ord="fro")
    / np.linalg.norm(true_covariance, ord="fro"),
    "CellMCD": np.linalg.norm(cellwise.covariance_ - true_covariance, ord="fro")
    / np.linalg.norm(true_covariance, ord="fro"),
}

valid = ~missing_truth
scores = np.nan_to_num(np.abs(cellwise.standardized_residuals_), nan=0.0)
cell_auc = roc_auc_score(cell_truth[valid], scores[valid])
cell_precision = precision_score(
    cell_truth[valid], cellwise.cell_outlier_mask_[valid], zero_division=0
)
cell_recall = recall_score(
    cell_truth[valid], cellwise.cell_outlier_mask_[valid], zero_division=0
)
rows_with_bad_cells = cell_truth.any(axis=1)

print(f"data shape: {X.shape}")
print(f"corrupted cells: {cell_truth.sum()} ({cell_truth.mean():.1%})")
print(f"rows containing at least one bad cell: {rows_with_bad_cells.sum()} ({rows_with_bad_cells.mean():.1%})")
print(f"missing cells: {missing_truth.sum()} ({missing_truth.mean():.1%})")
print(
    "relative covariance error, empirical / FastMCD / CellMCD: "
    f"{relative_error['Empirical']:.3f} / {relative_error['FastMCD']:.3f} / {relative_error['CellMCD']:.3f}"
)
print(
    "cell detection AUROC / precision / recall: "
    f"{cell_auc:.3f} / {cell_precision:.3f} / {cell_recall:.3f}"
)
print(f"CellMCD iterations: {cellwise.n_iter_}; converged: {cellwise.converged_}")

fig = plt.figure(figsize=(7.4, 4.8))
ax = fig.add_subplot(111)
labels = list(relative_error)
values = [relative_error[label] for label in labels]
ax.bar(labels, values)
ax.set_ylabel("relative Frobenius covariance error")
ax.set_title("Cellwise cleaning recovers the return covariance")
for i, value in enumerate(values):
    ax.text(i, value, f"{value:.3f}", ha="center", va="bottom")
fig.tight_layout()
fig.savefig(OUT / "covariance_error.png", dpi=160)
plt.close(fig)

row_score = np.nanmax(scores, axis=1)
selected_rows = np.argsort(row_score)[-60:]
rc.plot_cellwise_residual_map(
    cellwise,
    X[selected_rows],
    row_labels=[f"day {i + 1}" for i in selected_rows],
    column_labels=asset_names,
    title="Conditional residuals for the 60 most suspicious days",
    output_path=OUT / "cell_residual_map.png",
    show=False,
)


def correlation(covariance):
    scale = np.sqrt(np.diag(covariance))
    return covariance / np.outer(scale, scale)

fig = plt.figure(figsize=(11.2, 4.1))
ax1 = fig.add_subplot(121)
im1 = ax1.imshow(correlation(empirical_covariance), vmin=-1, vmax=1, aspect="equal")
ax1.set_title("Empirical correlation after median imputation")
ax1.set_xticks(np.arange(n_assets), labels=asset_names, rotation=45, ha="right")
ax1.set_yticks(np.arange(n_assets), labels=asset_names)
fig.colorbar(im1, ax=ax1, fraction=0.046)
ax2 = fig.add_subplot(122)
im2 = ax2.imshow(correlation(cellwise.covariance_), vmin=-1, vmax=1, aspect="equal")
ax2.set_title("CellMCD correlation")
ax2.set_xticks(np.arange(n_assets), labels=asset_names, rotation=45, ha="right")
ax2.set_yticks(np.arange(n_assets), labels=asset_names)
fig.colorbar(im2, ax=ax2, fraction=0.046)
fig.tight_layout()
fig.savefig(OUT / "correlation_comparison.png", dpi=160)
plt.close(fig)

(OUT / "metrics.csv").write_text(
    "metric,value\n"
    f"empirical_relative_error,{relative_error['Empirical']:.8f}\n"
    f"fastmcd_relative_error,{relative_error['FastMCD']:.8f}\n"
    f"cellmcd_relative_error,{relative_error['CellMCD']:.8f}\n"
    f"cell_auc,{cell_auc:.8f}\n"
    f"cell_precision,{cell_precision:.8f}\n"
    f"cell_recall,{cell_recall:.8f}\n"
    f"rows_with_bad_cells,{rows_with_bad_cells.sum()}\n"
    f"flagged_cells,{cellwise.cell_outlier_mask_.sum()}\n",
    encoding="utf-8",
)
