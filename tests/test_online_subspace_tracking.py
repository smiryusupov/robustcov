import numpy as np
import pytest

import robustcov as rc


class EmpiricalScatter:
    def __init__(self, ridge=1e-6):
        self.ridge = ridge

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.location_ = np.mean(X, axis=0)
        centered = X - self.location_
        self.covariance_ = centered.T @ centered / X.shape[0]
        self.covariance_ += self.ridge * np.eye(X.shape[1])
        return self


def make_subspace_data(seed=0, n=300, p=6, angle=0.0):
    rng = np.random.default_rng(seed)
    basis = np.zeros((p, 2))
    basis[0, 0] = 1.0
    basis[1, 1] = np.cos(angle)
    basis[2, 1] = np.sin(angle)
    latent = rng.normal(size=(n, 2))
    X = latent @ basis.T + rng.normal(scale=0.03, size=(n, p))
    return X, basis


def projector_distance(components, basis):
    estimated = components.T @ components
    target = basis @ basis.T
    return np.linalg.norm(estimated - target, ord="fro")


def make_tracker(**kwargs):
    params = dict(
        n_components=2,
        estimator=EmpiricalScatter(),
        update_interval=40,
        buffer_size=160,
        adaptation_rate=0.8,
        residual_quantile=0.98,
        threshold_scale=2.0,
        cell_threshold=10.0,
        max_cell_fraction=0.30,
        change_detection_angle=2.0,
        max_update_angle=35.0,
        history_size=4,
    )
    params.update(kwargs)
    return rc.OnlineRobustSubspaceTracker(**params)


def test_fit_transform_scores_and_estimator_protocol():
    X, _ = make_subspace_data()
    tracker = make_tracker().fit(X)

    assert tracker.components_.shape == (2, X.shape[1])
    assert tracker.transform(X[:7]).shape == (7, 2)
    assert tracker.reconstruct(X[:7]).shape == (7, X.shape[1])
    assert tracker.residuals(X[:7]).shape == (7, X.shape[1])
    assert tracker.anomaly_scores(X[:7]).shape == (7,)
    assert np.allclose(tracker.score_samples(X[:7]), -tracker.anomaly_scores(X[:7]))
    assert set(np.unique(tracker.predict(X[:7]))).issubset({-1, 1})
    assert tracker.get_params(deep=False)["update_interval"] == 40
    assert tracker.set_params(update_interval=42) is tracker


def test_sparse_cell_corruption_is_repaired_for_updates():
    X, _ = make_subspace_data(seed=1)
    tracker = make_tracker(update_interval=20).fit(X)
    batch = X[:20].copy()
    batch[0, 5] += 100.0

    result = tracker.update(batch)

    assert result.n_cell_corrections >= 1
    assert not result.sample_outlier_mask[0]
    assert result.n_accepted == 20
    assert result.update_attempted
    assert result.update_performed
    assert result.cell_outlier_mask.shape == batch.shape


def test_dense_row_outlier_is_rejected():
    X, _ = make_subspace_data(seed=2)
    tracker = make_tracker(update_interval=20).fit(X)
    batch = X[:20].copy()
    batch[0] += 100.0

    result = tracker.update(batch)

    assert result.sample_outlier_mask[0]
    assert result.n_rejected >= 1
    assert result.n_accepted <= 19
    assert tracker.n_rejected_ >= 1


def test_tracker_adapts_to_gradual_rotation_better_than_frozen_subspace():
    initial, initial_basis = make_subspace_data(seed=3, n=320, angle=0.0)
    tracker = make_tracker(
        update_interval=40,
        buffer_size=120,
        adaptation_rate=1.0,
        max_update_angle=30.0,
    ).fit(initial)
    frozen_components = tracker.components_.copy()

    final_basis = initial_basis
    for index, angle in enumerate(np.linspace(0.08, 0.55, 8)):
        batch, final_basis = make_subspace_data(
            seed=30 + index,
            n=40,
            angle=float(angle),
        )
        rng = np.random.default_rng(100 + index)
        rows = rng.choice(batch.shape[0], size=3, replace=False)
        cols = rng.integers(0, batch.shape[1], size=3)
        batch[rows, cols] += 25.0
        tracker.partial_fit(batch)

    adaptive_error = projector_distance(tracker.components_, final_basis)
    frozen_error = projector_distance(frozen_components, final_basis)
    assert tracker.n_updates_ >= 6
    assert adaptive_error < 0.45 * frozen_error
    assert tracker.subspace_version_ == tracker.n_updates_


def test_slow_change_safeguard_rejects_abrupt_candidate():
    X, _ = make_subspace_data(seed=4)
    tracker = make_tracker(
        update_interval=40,
        buffer_size=40,
        adaptation_rate=1.0,
        max_update_angle=8.0,
        change_detection_angle=2.0,
        threshold_scale=100.0,
        cell_threshold=100.0,
    ).fit(X)
    components_before = tracker.components_.copy()

    abrupt, _ = make_subspace_data(seed=5, n=40, angle=1.2)
    result = tracker.update(abrupt)

    assert result.update_attempted
    assert not result.update_performed
    assert result.change_detected
    assert result.candidate_max_angle > 8.0
    assert np.allclose(tracker.components_, components_before)


def test_history_and_update_payload():
    X, _ = make_subspace_data(seed=6)
    tracker = make_tracker(update_interval=20, history_size=2).fit(X)
    first = tracker.update(X[:20])
    tracker.update(X[20:40])
    tracker.update(X[40:60])

    assert len(tracker.history_) == 2
    payload = first.as_dict(include_arrays=True)
    assert len(payload["anomaly_scores"]) == 20
    assert len(payload["sample_outlier_mask"]) == 20
    assert len(tracker.history_records(include_arrays=False)) == 2


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_components": 0}, "n_components"),
        ({"update_interval": 1}, "update_interval"),
        ({"buffer_size": 20, "update_interval": 40}, "buffer_size"),
        ({"adaptation_rate": 0.0}, "adaptation_rate"),
        ({"residual_quantile": 1.0}, "residual_quantile"),
        ({"cell_threshold": 0.0}, "cell_threshold"),
        ({"max_cell_fraction": 1.0}, "max_cell_fraction"),
        ({"change_detection_angle": 20.0, "max_update_angle": 10.0}, "cannot exceed"),
        ({"history_size": -1}, "history_size"),
    ],
)
def test_parameter_validation(kwargs, message):
    X, _ = make_subspace_data(seed=7)
    with pytest.raises((TypeError, ValueError), match=message):
        make_tracker(**kwargs).fit(X)


def test_rejects_unfitted_bad_shapes_and_full_rank_request():
    X, _ = make_subspace_data(seed=8)
    tracker = make_tracker()
    with pytest.raises(AttributeError, match="not fitted"):
        tracker.update(X[:10])
    with pytest.raises(ValueError, match="smaller than n_features"):
        make_tracker(n_components=X.shape[1]).fit(X)

    tracker.fit(X)
    with pytest.raises(ValueError, match="features"):
        tracker.update(np.ones((10, X.shape[1] + 1)))
    bad = X[:10].copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        tracker.update(bad)
