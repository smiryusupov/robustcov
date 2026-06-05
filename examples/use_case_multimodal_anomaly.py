"""Multimodal anomaly detection with cluster-aware robust distances."""

from pathlib import Path

import numpy as np
import csv

import robustcov as rc


def make_data(random_state=0):
    rng = np.random.default_rng(random_state)
    centers = np.array([[-4.0, 0.0], [0.5, 3.5], [4.0, -0.5]])
    covs = [
        np.array([[0.7, 0.25], [0.25, 0.5]]),
        np.array([[0.5, -0.15], [-0.15, 0.7]]),
        np.array([[0.9, 0.10], [0.10, 0.4]]),
    ]
    clouds = [rng.multivariate_normal(c, S, size=180) for c, S in zip(centers, covs)]
    X_in = np.vstack(clouds)
    X_out = rng.uniform(low=[-8, -5], high=[8, 7], size=(35, 2))
    X = np.vstack([X_in, X_out])
    y = np.r_[np.zeros(X_in.shape[0], dtype=int), np.ones(X_out.shape[0], dtype=int)]
    return X, y


def main():
    outdir = Path("results/use_cases/multimodal_anomaly")
    outdir.mkdir(parents=True, exist_ok=True)
    X, y = make_data()

    global_det = rc.RobustOutlierDetector(
        estimator=rc.RegularizedCauchy(alpha=0.05, warn_on_nonconvergence=False),
        threshold="empirical",
        alpha=1 - y.mean(),
    ).fit(X)
    cluster_det = rc.ClusterRobustOutlierDetector(
        n_clusters=3,
        contamination=float(y.mean()),
        random_state=0,
    ).fit(X)

    def metrics(pred):
        found = pred == -1
        tp = int(np.sum(found & (y == 1)))
        fp = int(np.sum(found & (y == 0)))
        fn = int(np.sum((~found) & (y == 1)))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        return precision, recall, f1, int(np.sum(found))

    g_prec, g_rec, g_f1, g_found = metrics(global_det.labels_)
    c_prec, c_rec, c_f1, c_found = metrics(cluster_det.labels_)

    metrics_path = outdir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "precision", "recall", "f1", "detected"])
        writer.writerow(["global RegularizedCauchy", f"{g_prec:.6f}", f"{g_rec:.6f}", f"{g_f1:.6f}", g_found])
        writer.writerow(["cluster robust detector", f"{c_prec:.6f}", f"{c_rec:.6f}", f"{c_f1:.6f}", c_found])

    cluster_fig = outdir / "cluster_distance_panel.png"
    global_fig = outdir / "global_distance_profile.png"

    rc.plot_cluster_robust_distances(
        cluster_det,
        X,
        labels=y,
        output_path=cluster_fig,
        show=False,
    )
    rc.plot_robust_distance_profile(
        global_det.estimator_,
        labels=y,
        title="Global robust distance profile on multimodal data",
        output_path=global_fig,
        show=False,
    )

    print("multimodal anomaly detection demo")
    print("method,precision,recall,f1,detected")
    print(f"global RegularizedCauchy,{g_prec:.3f},{g_rec:.3f},{g_f1:.3f},{g_found}")
    print(f"cluster robust detector,{c_prec:.3f},{c_rec:.3f},{c_f1:.3f},{c_found}")
    print(cluster_det.summary())
    print(f"saved metrics to {metrics_path}")
    print(f"saved figure: {cluster_fig}")
    print(f"saved figure: {global_fig}")
    print(f"saved diagnostics to {outdir}")


if __name__ == "__main__":
    main()
