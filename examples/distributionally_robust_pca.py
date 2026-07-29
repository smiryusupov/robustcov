"""Distributionally robust PCA under structured covariance shift.

This example contrasts ordinary PCA, contamination-robust PCA, an identity-
geometry Wasserstein control, and an anisotropic weighted-Wasserstein estimator.
The evaluation distribution is held out and differs from the training law.

Run from the repository root::

    python examples/distributionally_robust_pca.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import robustcov as rc
from robustcov.experimental import DistributionallyRobustPCA


def leading_basis(X: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    location = np.mean(X, axis=0)
    centered = X - location
    covariance = centered.T @ centered / len(X)
    values, vectors = np.linalg.eigh(covariance)
    return location, vectors[:, np.argsort(values)[::-1][:rank]]


def reconstruction_risk(X: np.ndarray, location: np.ndarray, basis: np.ndarray) -> float:
    centered = X - location
    residual = centered - (centered @ basis) @ basis.T
    return float(np.mean(np.einsum("ij,ij->i", residual, residual)))


def build_data(seed: int = 20260719) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_variances = np.array([6.0, 5.0, 2.5, 2.2, 1.5, 1.3, 1.1, 1.0, 0.9, 0.8])
    target_variances = np.array([4.5, 4.0, 9.0, 8.0, 1.5, 1.3, 1.1, 1.0, 0.9, 0.8])
    X_train = rng.normal(size=(260, 10)) * np.sqrt(train_variances)
    X_target = rng.normal(size=(12000, 10)) * np.sqrt(target_variances)
    return X_train, X_target, target_variances


def fit_methods(X_train: np.ndarray) -> dict[str, dict[str, object]]:
    location, basis = leading_basis(X_train, 2)
    robust = rc.RobustPCA(
        n_components=2,
        estimator=rc.RegularizedCauchy(alpha=0.10, max_iter=200, tol=1e-8),
    ).fit(X_train)
    identity = DistributionallyRobustPCA(
        n_components=2,
        radius=2.5,
        transport_geometry="identity",
        formulation="exact",
    ).fit(X_train)
    dro = DistributionallyRobustPCA(
        n_components=2,
        radius=2.5,
        transport_geometry="residual",
        formulation="exact",
    ).fit(X_train)
    return {
        "Empirical PCA": {"location": location, "basis": basis, "estimator": None},
        "RobustPCA (Cauchy)": {
            "location": robust.location_,
            "basis": robust.components_.T,
            "estimator": robust,
        },
        "DRO-PCA identity control": {
            "location": identity.location_,
            "basis": identity.components_.T,
            "estimator": identity,
        },
        "DRO-PCA residual geometry": {
            "location": dro.location_,
            "basis": dro.components_.T,
            "estimator": dro,
        },
    }


def save_plots(
    outdir: Path,
    methods: dict[str, dict[str, object]],
    X_target: np.ndarray,
    target_variances: np.ndarray,
) -> list[dict[str, object]]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional plotting dependency
        raise RuntimeError("this example requires robustcov[plot]") from exc

    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for name, fitted in methods.items():
        basis = np.asarray(fitted["basis"])
        location = np.asarray(fitted["location"])
        estimator = fitted["estimator"]
        rows.append(
            {
                "method": name,
                "target_risk": reconstruction_risk(X_target, location, basis),
                "selected_gamma": getattr(estimator, "selected_gamma_", ""),
                "exact_worst_case_risk": getattr(estimator, "exact_worst_case_risk_", ""),
            }
        )

    order = np.argsort([float(row["target_risk"]) for row in rows])
    ordered = [rows[index] for index in order]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    positions = np.arange(len(ordered))
    ax.barh(positions, [float(row["target_risk"]) for row in ordered])
    ax.set_yticks(positions, [str(row["method"]) for row in ordered])
    ax.invert_yaxis()
    ax.set_xlabel("Held-out target reconstruction risk (lower is better)")
    ax.set_title("Weighted-Wasserstein PCA under structured covariance shift")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "target_risk.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    feature_index = np.arange(1, len(target_variances) + 1)
    width = 0.18
    for offset, (name, fitted) in enumerate(methods.items()):
        basis = np.asarray(fitted["basis"])
        projector_mass = np.diag(basis @ basis.T)
        ax.bar(feature_index + (offset - 1.5) * width, projector_mass, width=width, label=name)
    ax.plot(feature_index, target_variances / np.max(target_variances), "o--", label="Target variance (scaled)")
    ax.set_xticks(feature_index)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Projector diagonal / scaled variance")
    ax.set_title("Which feature directions the retained subspace protects")
    ax.legend(fontsize=8, ncols=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "subspace_allocation.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    dro = methods["DRO-PCA residual geometry"]["estimator"]
    records = list(dro.candidate_results_)
    path_records = [record for record in records if record["source"] == "path"]
    gammas = np.array([float(record["gamma"]) for record in path_records])
    exact = np.array([float(record["exact_risk"]) for record in path_records])
    surrogate = np.array([float(record["surrogate_risk_bound"]) for record in path_records])
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(gammas, exact, marker="o", label="Exact worst-case risk")
    ax.plot(gammas, surrogate, marker="s", label="Squared surrogate bound")
    ax.axvline(float(dro.selected_gamma_), linestyle="--", label=f"Selected gamma = {dro.selected_gamma_:g}")
    ax.set_xscale("symlog", linthresh=0.03)
    ax.set_xlabel("Candidate-path multiplier gamma")
    ax.set_ylabel("Risk")
    ax.set_title("Exact ambiguity-set risk selects the candidate subspace")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "ambiguity_path.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    with (outdir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/use_cases/distributionally_robust_pca"),
    )
    args = parser.parse_args()
    X_train, X_target, target_variances = build_data()
    methods = fit_methods(X_train)
    rows = save_plots(args.outdir, methods, X_target, target_variances)
    print("method,target_risk,selected_gamma")
    for row in rows:
        print(f"{row['method']},{float(row['target_risk']):.6f},{row['selected_gamma']}")
    dro = methods["DRO-PCA residual geometry"]["estimator"]
    print(f"ambiguity_radius,{dro.radius_:.6f}")
    print(f"selected_path_gamma,{dro.selected_gamma_}")
    print(f"exact_worst_case_risk,{dro.exact_worst_case_risk_:.6f}")
    print(f"surrogate_risk_bound,{dro.surrogate_risk_bound_:.6f}")
    print(f"saved,{args.outdir}")


if __name__ == "__main__":
    main()
