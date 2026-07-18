"""DetS and DetMM: robustness versus efficiency.

A deterministic S-estimator supplies high-breakdown location and scatter.  The
MM refinement keeps its robust scale but uses a less aggressive bisquare weight
to improve Gaussian efficiency.  The example shows both the contaminated fit
and the clean-data tradeoff.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.stats import chi2, rankdata

import robustcov as rc


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    ranks = rankdata(np.asarray(scores, dtype=float), method="average")
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def covariance_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(estimate - truth, ord="fro") / np.linalg.norm(truth, ord="fro"))


def add_ellipse(ax, location, covariance, *, label, linestyle="-"):
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    radius = np.sqrt(chi2.ppf(0.975, 2))
    ellipse = Ellipse(
        location,
        width=2 * radius * np.sqrt(values[0]),
        height=2 * radius * np.sqrt(values[1]),
        angle=angle,
        fill=False,
        linewidth=1.6,
        linestyle=linestyle,
        label=label,
    )
    ax.add_patch(ellipse)


def main() -> None:
    rng = np.random.default_rng(5)
    covariance = np.array([[1.0, 0.65], [0.65, 1.2]])
    n = 260
    X = rng.multivariate_normal(np.zeros(2), covariance, size=n)
    labels = np.zeros(n, dtype=bool)
    labels[:26] = True
    directions = rng.normal(size=(26, 2))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    X[:26] += 3.0 * directions

    estimators = {
        "FastMCD": rc.FastMCD(
            quality="fast",
            n_init=80,
            n_best=5,
            initial_c_steps=2,
            max_iter=60,
            random_state=0,
        ),
        "DetS": rc.DetS(max_iter=80),
        "DetMM": rc.DetMM(efficiency=0.95, max_iter=80),
        "Student-t": rc.StudentTScatter(df=3, alpha=0.02, max_iter=180),
    }
    fitted = {name: estimator.fit(X) for name, estimator in estimators.items()}

    result_dir = Path("results/use_cases/dets_detmm_tradeoff")
    result_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.scatter(X[~labels, 0], X[~labels, 1], s=16, alpha=0.65, label="regular")
    ax.scatter(X[labels, 0], X[labels, 1], s=28, marker="x", label="radial outlier")
    add_ellipse(ax, np.zeros(2), covariance, label="clean covariance", linestyle="--")
    add_ellipse(ax, fitted["DetS"].location_, fitted["DetS"].covariance_, label="DetS")
    add_ellipse(ax, fitted["DetMM"].location_, fitted["DetMM"].covariance_, label="DetMM")
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.set_title("Robust scatter under radial contamination")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(result_dir / "robust_ellipses.png", dpi=150)
    plt.close(fig)

    errors = [covariance_error(model.covariance_, covariance) for model in fitted.values()]
    names = list(fitted)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    positions = np.arange(len(names))
    ax.bar(positions, errors)
    ax.set_xticks(positions, names, rotation=15)
    ax.set_ylabel("Relative covariance error")
    ax.set_title("Contaminated-sample covariance recovery")
    for index, value in enumerate(errors):
        ax.text(index, value + 0.008, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(result_dir / "covariance_error.png", dpi=150)
    plt.close(fig)

    u = np.linspace(0.0, 8.0, 300)
    s_c = fitted["DetS"].tuning_constant_
    mm_c = fitted["DetMM"].tuning_constant_
    s_weight = np.where(u < s_c, (1.0 - (u / s_c) ** 2) ** 2, 0.0)
    mm_weight = np.where(u < mm_c, (1.0 - (u / mm_c) ** 2) ** 2, 0.0)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(u, s_weight, label=f"DetS, c={s_c:.2f}")
    ax.plot(u, mm_weight, label=f"DetMM, c={mm_c:.2f}")
    ax.set_xlabel("Standardized radial distance")
    ax.set_ylabel("Bisquare weight")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("MM refinement downweights less aggressively")
    ax.legend()
    fig.tight_layout()
    fig.savefig(result_dir / "weight_functions.png", dpi=150)
    plt.close(fig)

    clean_rng = np.random.default_rng(100)
    clean_sample = clean_rng.multivariate_normal(np.zeros(2), covariance, size=n)
    clean_s = rc.DetS(max_iter=80).fit(clean_sample)
    clean_mm = rc.DetMM(efficiency=0.95, max_iter=80).fit(clean_sample)
    clean_errors = [
        covariance_error(clean_s.covariance_, covariance),
        covariance_error(clean_mm.covariance_, covariance),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.bar([0, 1], clean_errors)
    ax.set_xticks([0, 1], ["DetS", "DetMM"])
    ax.set_ylabel("Relative covariance error")
    ax.set_title("Clean Gaussian sample")
    for index, value in enumerate(clean_errors):
        ax.text(index, value + 0.005, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(result_dir / "clean_efficiency.png", dpi=150)
    plt.close(fig)

    metrics = {
        "n_samples": n,
        "outlier_fraction": float(labels.mean()),
        "dets_breakdown": fitted["DetS"].breakdown,
        "detmm_nominal_location_efficiency": fitted["DetMM"].nominal_location_efficiency_,
        "dets_covariance_error": covariance_error(fitted["DetS"].covariance_, covariance),
        "detmm_covariance_error": covariance_error(fitted["DetMM"].covariance_, covariance),
        "dets_outlier_auc": auc(labels, fitted["DetS"].distances_),
        "detmm_outlier_auc": auc(labels, fitted["DetMM"].distances_),
        "clean_dets_covariance_error": clean_errors[0],
        "clean_detmm_covariance_error": clean_errors[1],
    }
    with (result_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics.items())

    print(f"observations: {n}")
    print(f"injected radial outliers: {labels.sum()}")
    print(f"DetS tuning constant: {s_c:.3f}")
    print(f"DetMM tuning constant: {mm_c:.3f}")
    print(f"DetS covariance error: {metrics['dets_covariance_error']:.3f}")
    print(f"DetMM covariance error: {metrics['detmm_covariance_error']:.3f}")
    print(f"DetS / DetMM outlier AUROC: {metrics['dets_outlier_auc']:.3f} / {metrics['detmm_outlier_auc']:.3f}")
    print(f"clean-sample DetS / DetMM covariance error: {clean_errors[0]:.3f} / {clean_errors[1]:.3f}")


if __name__ == "__main__":
    main()
