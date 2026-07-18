"""CellRCov for high-dimensional mixed contamination.

The example uses a low-rank covariance model with more variables than
observations.  Individual cells are corrupted, a minority of complete rows is
shifted outside the main subspace, and additional cells are missing.  CellRCov
is compared with regularized covariance estimators that operate after median
imputation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

import robustcov as rc


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "use_cases" / "cellrcov_high_dimensional"
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(2026)
n_samples = 90
n_features = 120
rank = 4

basis, _ = np.linalg.qr(rng.normal(size=(n_features, rank)))
factor_variance = np.array([8.0, 5.0, 3.0, 1.8])
residual_variance = np.linspace(0.15, 0.35, n_features)
true_covariance = (
    basis @ np.diag(factor_variance) @ basis.T
    + np.diag(residual_variance)
)
clean = rng.multivariate_normal(
    np.zeros(n_features), true_covariance, size=n_samples
)

X = clean.copy()
cell_truth = np.zeros_like(X, dtype=bool)
bad_flat = rng.choice(X.size, size=int(0.045 * X.size), replace=False)
bad_rows, bad_columns = np.unravel_index(bad_flat, X.shape)
X[bad_rows, bad_columns] += (
    rng.choice([-1.0, 1.0], size=bad_flat.size)
    * rng.uniform(6.0, 10.0, size=bad_flat.size)
)
cell_truth[bad_rows, bad_columns] = True

case_truth = np.zeros(n_samples, dtype=bool)
case_rows = rng.choice(n_samples, size=int(0.12 * n_samples), replace=False)
case_truth[case_rows] = True
orthogonal_direction = rng.normal(size=n_features)
orthogonal_direction -= basis @ (basis.T @ orthogonal_direction)
orthogonal_direction /= np.linalg.norm(orthogonal_direction)
X[case_rows] += 6.0 * orthogonal_direction

missing_truth = np.zeros_like(X, dtype=bool)
available = np.flatnonzero(~cell_truth.ravel())
missing_flat = rng.choice(available, size=int(0.05 * X.size), replace=False)
missing_rows, missing_columns = np.unravel_index(missing_flat, X.shape)
X[missing_rows, missing_columns] = np.nan
missing_truth[missing_rows, missing_columns] = True

median = np.nanmedian(X, axis=0)
median_imputed = np.where(np.isnan(X), median, X)

ledoit_wolf = LedoitWolf().fit(median_imputed)
cauchy = rc.RegularizedCauchy(
    alpha=0.15,
    max_iter=180,
    missing_values="median",
).fit(X)
mrcd = rc.MRCD(
    contamination=0.15,
    quality="fast",
    n_init=30,
    n_best=3,
    initial_c_steps=1,
    max_iter=45,
    random_state=0,
    missing_values="median",
).fit(X)
cellrcov = rc.CellRCov(
    n_components=rank,
    residual_shrinkage="auto",
    shrinkage_grid=[0.0, 0.25, 0.5, 0.75, 1.0],
    cv_splits=4,
    cell_pca=rc.CellPCA(n_components=rank, max_iter=55),
    score_estimator=rc.FastMCD(
        support_fraction=0.75,
        quality="fast",
        n_init=40,
        n_best=3,
        initial_c_steps=1,
        max_iter=50,
        random_state=0,
        scale_correction="none",
    ),
).fit(X)

models = {
    "Ledoit-Wolf": ledoit_wolf,
    "Regularized Cauchy": cauchy,
    "MRCD": mrcd,
    "CellRCov": cellrcov,
}

covariance_error = {
    name: np.linalg.norm(model.covariance_ - true_covariance, ord="fro")
    / np.linalg.norm(true_covariance, ord="fro")
    for name, model in models.items()
}

centered = median_imputed - ledoit_wolf.location_
ledoit_distances = np.einsum(
    "ij,jk,ik->i", centered, ledoit_wolf.precision_, centered
)
row_auc = {
    "Ledoit-Wolf": roc_auc_score(case_truth, ledoit_distances),
    "Regularized Cauchy": roc_auc_score(case_truth, cauchy.mahalanobis(X)),
    "MRCD": roc_auc_score(case_truth, mrcd.mahalanobis(X)),
    "CellRCov": roc_auc_score(case_truth, cellrcov.mahalanobis(X)),
}
observed = np.isfinite(X)
cell_auc = roc_auc_score(
    cell_truth[observed],
    np.abs(cellrcov.standardized_residuals_[observed]),
)

print(f"data shape: {X.shape}")
print(f"p / n ratio: {n_features / n_samples:.2f}")
print(f"corrupted cells: {cell_truth.sum()} ({cell_truth.mean():.1%})")
print(f"casewise outliers: {case_truth.sum()} ({case_truth.mean():.1%})")
print(f"missing cells: {missing_truth.sum()} ({missing_truth.mean():.1%})")
print(f"selected residual shrinkage: {cellrcov.residual_shrinkage_:.2f}")
print("relative covariance error:")
for name, value in covariance_error.items():
    print(f"  {name}: {value:.3f}")
print("case-outlier AUROC:")
for name, value in row_auc.items():
    print(f"  {name}: {value:.3f}")
print(f"CellRCov cell-outlier AUROC: {cell_auc:.3f}")

fig = plt.figure(figsize=(8.6, 4.8))
ax = fig.add_subplot(111)
labels = list(covariance_error)
values = [covariance_error[label] for label in labels]
ax.bar(np.arange(len(labels)), values)
ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=20, ha="right")
ax.set_ylabel("relative Frobenius error")
ax.set_title("Covariance recovery under mixed contamination")
fig.tight_layout()
fig.savefig(OUT / "covariance_error.png", dpi=160)
plt.close(fig)

fig = plt.figure(figsize=(8.6, 4.8))
ax = fig.add_subplot(111)
truth_values = np.linalg.eigvalsh(true_covariance)[::-1]
ax.plot(truth_values, label="truth", linewidth=2.0)
for name, model in models.items():
    ax.plot(np.linalg.eigvalsh(model.covariance_)[::-1], label=name, alpha=0.8)
ax.set_yscale("log")
ax.set_xlabel("ordered eigenvalue")
ax.set_ylabel("eigenvalue")
ax.set_title("Estimated covariance spectra")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "covariance_spectrum.png", dpi=160)
plt.close(fig)

mapping = cellrcov.outlier_map(X)
fig = plt.figure(figsize=(7.2, 5.4))
ax = fig.add_subplot(111)
ax.scatter(mapping[:, 0], mapping[:, 1], s=30, alpha=0.7)
ax.scatter(
    mapping[case_truth, 0],
    mapping[case_truth, 1],
    s=78,
    facecolors="none",
    edgecolors="black",
    label="injected case outlier",
)
ax.set_xlabel("squared distance in fitted subspace")
ax.set_ylabel("squared residual distance")
ax.set_title("CellRCov distance decomposition")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "distance_decomposition.png", dpi=160)
plt.close(fig)

suspicious_rows = np.argsort(cellrcov.max_cell_residuals_)[-25:]
rc.plot_cellwise_residual_map(
    cellrcov,
    X=X[suspicious_rows],
    row_labels=[str(index) for index in suspicious_rows],
    title="Cellwise residuals for the most suspicious rows",
    output_path=OUT / "cell_residual_map.png",
    show=False,
)

(OUT / "metrics.csv").write_text(
    "metric,value\n"
    + "\n".join(
        [
            *(f"covariance_error_{name.lower().replace(' ', '_')},{value:.8f}" for name, value in covariance_error.items()),
            *(f"case_auc_{name.lower().replace(' ', '_')},{value:.8f}" for name, value in row_auc.items()),
            f"cell_auc_cellrcov,{cell_auc:.8f}",
            f"residual_shrinkage,{cellrcov.residual_shrinkage_:.8f}",
        ]
    )
    + "\n",
    encoding="utf-8",
)
