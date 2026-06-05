import numpy as np

import robustcov as rc


def test_cluster_robust_detector_runs():
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(loc=(-3, 0), scale=0.5, size=(40, 2)),
        rng.normal(loc=(3, 0), scale=0.5, size=(40, 2)),
        rng.normal(loc=(0, 5), scale=0.5, size=(8, 2)),
    ])
    det = rc.ClusterRobustOutlierDetector(n_clusters=2, contamination=0.1, random_state=0).fit(X)
    assert det.distances_.shape == (X.shape[0],)
    assert det.cluster_centers_.shape == (2, 2)
    assert det.predict(X).shape == (X.shape[0],)
    D = det.distances_to_clusters(X[:5])
    assert D.shape == (5, 2)
    assert hasattr(det, "threshold_")
