"""Benchmark optional C++ kernels against exact NumPy fallbacks."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

import robustcov as rc
from robustcov._native import matrix_mahalanobis2_batch, native_available, weighted_tucker_scores_2d


def timed(call, repeats):
    values = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = call()
        values.append(time.perf_counter() - start)
    return result, float(np.median(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    if not native_available():
        raise RuntimeError("native extension is required for this benchmark")
    rng = np.random.default_rng(20260719)
    rows = []

    for n, r, c in [(64, 5, 10), (256, 8, 16), (512, 12, 20)]:
        X = rng.normal(size=(n, r, c))
        location = rng.normal(size=(r, c))
        A = rng.normal(size=(r, r)); B = rng.normal(size=(c, c))
        RP = A @ A.T + np.eye(r); CP = B @ B.T + np.eye(c)
        py, py_time = timed(lambda: matrix_mahalanobis2_batch(X, location, RP, CP, backend="python"), args.repeats)
        cpp, cpp_time = timed(lambda: matrix_mahalanobis2_batch(X, location, RP, CP, backend="cpp"), args.repeats)
        rows.append({"kernel": "matrix_mahalanobis2", "shape": f"{n}x{r}x{c}", "python_seconds": py_time, "cpp_seconds": cpp_time, "speedup": py_time / cpp_time, "max_abs_difference": float(np.max(np.abs(py-cpp)))})

    for n, r, c, q1, q2 in [(64, 8, 10, 2, 3), (256, 12, 16, 3, 4), (512, 16, 20, 4, 4)]:
        X = rng.normal(size=(n, r, c)); W = rng.uniform(size=X.shape)
        W[rng.random(X.shape) < 0.08] = 0.0
        center = rng.normal(size=(r, c)); U, _ = np.linalg.qr(rng.normal(size=(r, q1))); V, _ = np.linalg.qr(rng.normal(size=(c, q2)))
        py, py_time = timed(lambda: weighted_tucker_scores_2d(X, W, center, U, V, backend="python"), args.repeats)
        cpp, cpp_time = timed(lambda: weighted_tucker_scores_2d(X, W, center, U, V, backend="cpp"), args.repeats)
        rows.append({"kernel": "weighted_tucker_scores", "shape": f"{n}x{r}x{c};{q1}x{q2}", "python_seconds": py_time, "cpp_seconds": cpp_time, "speedup": py_time / cpp_time, "max_abs_difference": float(np.max(np.abs(py-cpp)))})

    # End-to-end estimator comparison on a modest problem.
    n, r, c = 90, 10, 14
    U, _ = np.linalg.qr(rng.normal(size=(r, 2))); V, _ = np.linalg.qr(rng.normal(size=(c, 3)))
    cores = rng.normal(size=(n, 2, 3)); X = np.einsum("au,nuv,bv->nab", U, cores, V) + 0.1*rng.normal(size=(n,r,c))
    X.flat[rng.choice(X.size, 400, replace=False)] += 5.0
    py_model, py_time = timed(lambda: rc.RobustMultilinearPCA(ranks=(2,3), max_iter=15, backend="python").fit(X), 2)
    cpp_model, cpp_time = timed(lambda: rc.RobustMultilinearPCA(ranks=(2,3), max_iter=15, backend="cpp").fit(X), 2)
    rows.append({"kernel": "RobustMultilinearPCA.fit", "shape": f"{n}x{r}x{c}", "python_seconds": py_time, "cpp_seconds": cpp_time, "speedup": py_time / cpp_time, "max_abs_difference": float(np.max(np.abs(py_model.fitted_values_ - cpp_model.fitted_values_)))})

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    for row in rows:
        print(f"{row['kernel']:28s} {row['shape']:20s} speedup={row['speedup']:.2f}x diff={row['max_abs_difference']:.2e}")


if __name__ == "__main__":
    main()
