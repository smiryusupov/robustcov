"""Track a gradually rotating subspace with sparse and rowwise corruption."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import robustcov as rc


def _basis(angle: float, p: int = 8) -> np.ndarray:
    basis = np.zeros((p, 2))
    basis[0, 0] = 1.0
    basis[1, 1] = np.cos(angle)
    basis[2, 1] = np.sin(angle)
    return basis


def _sample(rng: np.random.Generator, n: int, angle: float) -> np.ndarray:
    basis = _basis(angle)
    latent = rng.normal(size=(n, 2))
    return latent @ basis.T + rng.normal(scale=0.04, size=(n, basis.shape[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results/use_cases/online_subspace_tracking")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    initial = _sample(rng, 400, 0.0)
    tracker = rc.OnlineRobustSubspaceTracker(
        n_components=2,
        update_interval=50,
        buffer_size=200,
        adaptation_rate=0.7,
        max_update_angle=30.0,
    ).fit(initial)

    rows = []
    for step, angle in enumerate(np.linspace(0.05, 0.65, 16), start=1):
        batch = _sample(rng, 50, float(angle))
        sparse_rows = rng.choice(batch.shape[0], size=4, replace=False)
        sparse_cols = rng.integers(0, batch.shape[1], size=4)
        batch[sparse_rows, sparse_cols] += rng.choice([-20.0, 20.0], size=4)
        batch[0] += 12.0
        result = tracker.update(batch)
        target = _basis(float(angle))
        target_projector = target @ target.T
        estimated_projector = tracker.components_.T @ tracker.components_
        rows.append(
            {
                "step": step,
                "angle_radians": float(angle),
                "projector_error": float(
                    np.linalg.norm(estimated_projector - target_projector, ord="fro")
                ),
                "accepted": result.n_accepted,
                "rejected": result.n_rejected,
                "cell_corrections": result.n_cell_corrections,
                "candidate_max_angle": result.candidate_max_angle,
                "updated": result.update_performed,
            }
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "tracking_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"saved,{outdir / 'tracking_summary.csv'}")
        return

    steps = [row["step"] for row in rows]
    errors = [row["projector_error"] for row in rows]
    rejected = [row["rejected"] for row in rows]

    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(steps, errors, marker="o")
    axis.set_xlabel("streaming update")
    axis.set_ylabel("projector error")
    axis.set_title("Online robust subspace tracking")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "tracking_error.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(steps, rejected)
    axis.set_xlabel("streaming update")
    axis.set_ylabel("rejected rows")
    axis.set_title("Rows excluded from adaptation")
    fig.tight_layout()
    fig.savefig(outdir / "rejected_rows.png", dpi=160)
    plt.close(fig)
    print(f"saved,{outdir}")


if __name__ == "__main__":
    main()
