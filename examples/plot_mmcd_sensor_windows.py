"""Matrix MCD for multichannel sensor windows.

Each observation is a sensor-by-time matrix.  The example compares an all-row
matrix-normal covariance fit with MMCD when a minority of windows contains
localized faults.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

import robustcov as rc


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "use_cases" / "mmcd_sensor_windows"
OUT.mkdir(parents=True, exist_ok=True)


def matrix_normal(rng, n, mean, row_covariance, column_covariance):
    row_root = np.linalg.cholesky(row_covariance)
    column_root = np.linalg.cholesky(column_covariance)
    noise = rng.normal(size=(n, *mean.shape))
    return np.asarray(
        [mean + row_root @ sample @ column_root.T for sample in noise]
    )


rng = np.random.default_rng(24)
n_clean = 110
n_outliers = 28
n_sensors = 5
n_times = 10

row_covariance = 0.35 * np.ones((n_sensors, n_sensors)) + 0.65 * np.eye(n_sensors)
column_covariance = 0.65 ** np.abs(
    np.subtract.outer(np.arange(n_times), np.arange(n_times))
)
mean = np.zeros((n_sensors, n_times))

clean = matrix_normal(
    rng, n_clean, mean, row_covariance, column_covariance
)
outliers = matrix_normal(
    rng, n_outliers, mean, row_covariance, column_covariance
)
# Two fault patterns: a sustained sensor bias and a short cross-sensor event.
outliers[:14, 1, 3:8] += 3.0
outliers[14:, :, 7:9] += np.linspace(1.8, 3.0, n_sensors)[:, None]

X = np.concatenate([clean, outliers])
y = np.concatenate([
    np.zeros(n_clean, dtype=int),
    np.ones(n_outliers, dtype=int),
])
order = rng.permutation(X.shape[0])
X = X[order]
y = y[order]

classical = rc.MMCD(
    support_fraction=1.0,
    n_init=1,
    n_best=1,
    initial_c_steps=0,
    max_iter=1,
    reweight=False,
    random_state=0,
).fit(X)

robust = rc.MMCD(
    contamination=0.22,
    quality="fast",
    n_init=50,
    n_best=7,
    random_state=0,
).fit(X)

classical_distance = classical.mahalanobis(X)
robust_distance = robust.mahalanobis(X)
classical_auc = roc_auc_score(y, classical_distance)
robust_auc = roc_auc_score(y, robust_distance)

print(f"matrix sample shape: {X.shape}")
print(f"raw MMCD support: {robust.raw_support_.sum()} / {X.shape[0]}")
print(f"fault windows retained in raw support: {np.count_nonzero(robust.raw_support_ & (y == 1))}")
print(f"outlier AUROC, all-row matrix MLE / MMCD: {classical_auc:.3f} / {robust_auc:.3f}")
print(f"median robust distance, regular / fault: {np.median(robust_distance[y == 0]):.2f} / {np.median(robust_distance[y == 1]):.2f}")

fig = plt.figure(figsize=(7.2, 5.4))
ax = fig.add_subplot(111)
ax.scatter(classical_distance[y == 0], robust_distance[y == 0], s=24, alpha=0.72, label="regular")
ax.scatter(classical_distance[y == 1], robust_distance[y == 1], s=42, marker="x", label="fault")
ax.set_xlabel("all-row matrix MLE squared distance")
ax.set_ylabel("MMCD squared distance")
ax.set_title("Localized faults are partly masked by the all-row fit")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "distance_comparison.png", dpi=160)
plt.close(fig)

selected = int(np.argmax(robust_distance))
rc.plot_matrix_outlier_contributions(
    robust,
    X,
    index=selected,
    row_labels=[f"sensor {i + 1}" for i in range(n_sensors)],
    column_labels=[f"t{i + 1}" for i in range(n_times)],
    title="Cell contributions for the highest-distance window",
    output_path=OUT / "contribution_heatmap.png",
    show=False,
)
plt.close("all")

row_scale = np.sqrt(np.diag(robust.row_covariance_))
column_scale = np.sqrt(np.diag(robust.column_covariance_))
row_correlation = robust.row_covariance_ / np.outer(row_scale, row_scale)
column_correlation = robust.column_covariance_ / np.outer(column_scale, column_scale)
fig = plt.figure(figsize=(10.5, 4.2))
ax1 = fig.add_subplot(121)
im1 = ax1.imshow(row_correlation, vmin=-1, vmax=1, aspect="equal")
ax1.set_title("Robust sensor correlation")
ax1.set_xlabel("sensor")
ax1.set_ylabel("sensor")
fig.colorbar(im1, ax=ax1, fraction=0.046)
ax2 = fig.add_subplot(122)
im2 = ax2.imshow(column_correlation, vmin=-1, vmax=1, aspect="auto")
ax2.set_title("Robust within-window correlation")
ax2.set_xlabel("time index")
ax2.set_ylabel("time index")
fig.colorbar(im2, ax=ax2, fraction=0.046)
fig.tight_layout()
fig.savefig(OUT / "covariance_factors.png", dpi=160)
plt.close(fig)

(OUT / "metrics.csv").write_text(
    "metric,value\n"
    f"classical_auc,{classical_auc:.6f}\n"
    f"mmcd_auc,{robust_auc:.6f}\n"
    f"raw_support,{robust.raw_support_.sum()}\n"
    f"outliers_in_raw_support,{np.count_nonzero(robust.raw_support_ & (y == 1))}\n",
    encoding="utf-8",
)
