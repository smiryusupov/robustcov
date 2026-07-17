import numpy as np
import pytest

import robustcov as rc


class EmpiricalScatter:
    def __init__(self, ridge=1e-8):
        self.ridge = ridge

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.location_ = X.mean(axis=0)
        Xc = X - self.location_
        self.covariance_ = Xc.T @ Xc / X.shape[0]
        self.covariance_ += self.ridge * np.eye(X.shape[1])
        return self


def make_reference(seed=0, n=500, p=5):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 2))
    basis = np.zeros((p, 2))
    basis[0, 0] = 2.5
    basis[1, 1] = 1.5
    if p > 2:
        basis[2, :] = [0.6, -0.3]
    X = latent @ basis.T + rng.normal(scale=0.08, size=(n, p))
    return X, basis


def make_monitor(**kwargs):
    defaults = dict(
        n_components=2,
        estimator=EmpiricalScatter(),
        window_size=80,
        calibration_windows=10,
        threshold_quantile=0.99,
        sample_quantile=0.98,
        random_state=0,
    )
    defaults.update(kwargs)
    return rc.RobustSubspaceMonitor(**defaults)


def test_monitor_fit_warmup_evaluate_and_history():
    X, _ = make_reference()
    monitor = make_monitor(history_size=2).fit(X)

    assert monitor.reference_model_.n_components_ == 2
    assert set(monitor.thresholds_) >= {
        "location_shift",
        "shape_shift",
        "max_subspace_angle",
        "combined_outlier_fraction",
    }

    preview = monitor.evaluate(X[:40])
    assert not preview.ready
    assert monitor.window_.shape == (0, X.shape[1])
    assert monitor.history_ == []

    first = monitor.update(X[:40])
    second = monitor.update(X[40:80])
    assert not first.ready
    assert second.ready
    assert monitor.window_.shape == (80, X.shape[1])
    assert monitor.current_model_ is not None
    assert len(monitor.history_) == 2
    assert second.score_distances.shape == (40,)
    assert second.sample_outlier_mask.dtype == bool

    monitor.update(X[80:120])
    assert len(monitor.history_) == 2
    assert len(monitor.history_records()) == 2


def test_monitor_detects_location_shift_and_keeps_reference_frozen():
    X, _ = make_reference(seed=1)
    monitor = make_monitor(
        alarm_metrics=("location_shift",),
        threshold_quantile=0.95,
    ).fit(X)
    reference_location = monitor.reference_location_.copy()

    shifted = X[:80].copy()
    shifted[:, 0] += 5.0
    result = monitor.update(shifted)

    assert result.ready
    assert result.exceeded["location_shift"]
    assert result.raw_alarm
    assert result.alarm
    assert np.allclose(monitor.reference_location_, reference_location)
    assert not np.allclose(monitor.current_model_.location_, reference_location)


def test_monitor_detects_out_of_subspace_drift():
    X, _ = make_reference(seed=2)
    monitor = make_monitor(
        alarm_metrics=("orthogonal_distance_shift",),
        threshold_quantile=0.95,
    ).fit(X)

    drifted = X[:80].copy()
    drifted[:, 4] += 3.0
    result = monitor.update(drifted)

    assert result.orthogonal_distance_shift > result.thresholds[
        "orthogonal_distance_shift"
    ]
    assert result.exceeded["orthogonal_distance_shift"]
    assert result.batch_orthogonal_outlier_fraction > 0.9


def test_monitor_detects_subspace_rotation():
    rng = np.random.default_rng(3)
    latent = rng.normal(size=(600, 2))
    reference = np.column_stack(
        [2.0 * latent[:, 0], latent[:, 1], np.zeros((600, 2))]
    )
    reference += rng.normal(scale=0.03, size=reference.shape)

    monitor = make_monitor(
        window_size=100,
        alarm_metrics=("max_subspace_angle",),
        threshold_quantile=0.95,
    ).fit(reference)

    current_latent = rng.normal(size=(100, 2))
    rotated = np.column_stack(
        [np.zeros((100, 2)), 2.0 * current_latent[:, 0], current_latent[:, 1]]
    )
    rotated += rng.normal(scale=0.03, size=rotated.shape)
    result = monitor.update(rotated)

    assert result.max_subspace_angle > 70.0
    assert result.exceeded["max_subspace_angle"]
    assert result.principal_angles.shape == (2,)


def test_monitor_outlier_fraction_and_batch_arrays():
    X, _ = make_reference(seed=4)
    monitor = make_monitor(
        alarm_metrics=("combined_outlier_fraction",),
        threshold_quantile=0.95,
    ).fit(X)

    rng = np.random.default_rng(40)
    batch = X[:80].copy()
    batch[:20] += rng.normal(loc=0.0, scale=12.0, size=(20, X.shape[1]))
    result = monitor.update(batch)

    assert result.exceeded["combined_outlier_fraction"]
    assert result.combined_outlier_fraction >= 0.20
    assert result.batch_combined_outlier_fraction >= 0.20
    assert result.sample_outlier_mask.sum() >= 16
    payload = result.as_dict(include_arrays=True)
    assert len(payload["score_distances"]) == 80
    assert "ALARM" in result.summary()


def test_monitor_alarm_patience_requires_persistence():
    X, _ = make_reference(seed=5)
    monitor = make_monitor(
        alarm_metrics=("location_shift",),
        threshold_quantile=0.95,
        alarm_patience=2,
    ).fit(X)

    shifted_a = X[:80].copy()
    shifted_b = X[80:160].copy()
    shifted_a[:, 0] += 5.0
    shifted_b[:, 0] += 5.0

    first = monitor.update(shifted_a)
    second = monitor.update(shifted_b)

    assert first.raw_alarm and not first.alarm
    assert first.consecutive_alarms == 1
    assert second.raw_alarm and second.alarm
    assert second.consecutive_alarms == 2


def test_partial_fit_returns_self_and_reset_preserves_reference():
    X, _ = make_reference(seed=6)
    monitor = make_monitor().fit(X)
    assert monitor.partial_fit(X[:40]) is monitor
    assert monitor.last_result_ is not None

    reference_covariance = monitor.reference_covariance_.copy()
    monitor.reset()
    assert monitor.window_.shape == (0, X.shape[1])
    assert monitor.history_ == []
    assert np.allclose(monitor.reference_covariance_, reference_covariance)


def test_full_component_monitor_disables_subspace_only_metrics():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(300, 3))
    monitor = rc.RobustSubspaceMonitor(
        n_components=None,
        estimator=EmpiricalScatter(),
        window_size=60,
        calibration_windows=6,
    ).fit(X)

    assert not monitor.subspace_rotation_available_
    assert not monitor.orthogonal_distance_available_
    assert np.isinf(monitor.thresholds_["max_subspace_angle"])
    assert np.isinf(monitor.thresholds_["orthogonal_distance_shift"])

    result = monitor.update(X[:60])
    assert result.ready
    assert result.max_subspace_angle == 0.0
    assert np.isnan(result.orthogonal_distance_shift)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"window_size": 1}, "window_size"),
        ({"calibration_windows": 2}, "calibration_windows"),
        ({"threshold_quantile": 0.4}, "threshold_quantile"),
        ({"sample_quantile": 1.0}, "sample_quantile"),
        ({"alarm_patience": 0}, "alarm_patience"),
        ({"history_size": -1}, "history_size"),
        ({"alarm_metrics": ("unknown",)}, "unknown alarm_metrics"),
    ],
)
def test_monitor_parameter_validation(kwargs, message):
    X, _ = make_reference(seed=8, n=150)
    params = {"window_size": 60}
    params.update(kwargs)
    with pytest.raises((ValueError, TypeError), match=message):
        make_monitor(**params).fit(X)


def test_monitor_rejects_bad_batches_and_unfitted_calls():
    X, _ = make_reference(seed=9)
    monitor = make_monitor()
    with pytest.raises(AttributeError, match="not fitted"):
        monitor.update(X[:20])

    monitor.fit(X)
    with pytest.raises(ValueError, match="features"):
        monitor.update(np.ones((20, X.shape[1] + 1)))
    with pytest.raises(ValueError, match="finite"):
        bad = X[:20].copy()
        bad[0, 0] = np.nan
        monitor.update(bad)


def test_plot_subspace_monitor_history(tmp_path):
    X, _ = make_reference(seed=10)
    monitor = make_monitor(history_size=10).fit(X)
    monitor.update(X[:80])
    shifted = X[80:160].copy()
    shifted[:, 0] += 4.0
    monitor.update(shifted)

    output = tmp_path / "history.png"
    fig = rc.plot_subspace_monitor_history(
        monitor,
        output_path=output,
        show=False,
    )
    assert output.exists()
    assert len(fig.axes) == 2


def test_monitor_requires_reference_longer_than_window():
    X, _ = make_reference(seed=11, n=80)
    with pytest.raises(ValueError, match="smaller than"):
        make_monitor(window_size=80).fit(X)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"window_size": 80.0}, "window_size"),
        ({"calibration_windows": 6.0}, "calibration_windows"),
        ({"alarm_patience": 2.0}, "alarm_patience"),
        ({"history_size": 10.0}, "history_size"),
        ({"alarm_metrics": "location_shift"}, "not a string"),
        ({"threshold_scale": 0.0}, "threshold_scale"),
    ],
)
def test_monitor_rejects_ambiguous_parameter_types(kwargs, message):
    X, _ = make_reference(seed=12, n=200)
    params = {"window_size": 60}
    params.update(kwargs)
    with pytest.raises((ValueError, TypeError), match=message):
        make_monitor(**params).fit(X)
