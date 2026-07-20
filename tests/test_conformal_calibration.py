from __future__ import annotations

import numpy as np
import pytest

import robustcov as rc


def test_upper_tail_p_values_use_conservative_ties():
    calibrator = rc.ConformalAlertCalibrator(alpha=0.25).fit([1.0, 2.0, 2.0, 4.0])

    p = calibrator.p_values([0.0, 2.0, 3.0, 5.0])
    assert np.allclose(p, [1.0, 0.8, 0.4, 0.2])
    assert calibrator.p_values(5.0) == pytest.approx(0.2)
    assert calibrator.predict_alerts(5.0)
    assert calibrator.predict(5.0) == -1
    assert calibrator.predict(3.0) == 1
    assert calibrator.decision_function(5.0) == pytest.approx(-0.05)


def test_lower_tail_p_values_and_labels():
    calibrator = rc.ConformalAlertCalibrator(alpha=0.25, tail="lower").fit(
        [1.0, 2.0, 2.0, 4.0]
    )

    p = calibrator.p_values([0.0, 2.0, 3.0, 5.0])
    assert np.allclose(p, [0.2, 0.8, 0.8, 1.0])
    assert np.array_equal(calibrator.predict([0.0, 2.0]), [-1, 1])


def test_threshold_and_resolution_diagnostics():
    coarse = rc.ConformalAlertCalibrator(alpha=0.05).fit(np.arange(9.0))
    assert coarse.min_p_value_ == pytest.approx(0.1)
    assert coarse.resolution_limited_
    assert np.isposinf(coarse.threshold_)
    assert not coarse.predict_alerts(100.0)

    resolved = rc.ConformalAlertCalibrator(alpha=0.05).fit(np.arange(19.0))
    assert resolved.min_p_value_ == pytest.approx(0.05)
    assert not resolved.resolution_limited_
    assert resolved.threshold_ == pytest.approx(18.0)
    assert resolved.predict_alerts(19.0)
    summary = resolved.calibration_summary()
    assert summary["tie_handling"] == "conservative"
    assert summary["n_calibration"] == 19


def test_calibrator_controls_marginal_false_alert_rate_under_exchangeability():
    rng = np.random.default_rng(1234)
    calibration = rng.normal(size=999)
    test_scores = rng.normal(size=20_000)
    calibrator = rc.ConformalAlertCalibrator(alpha=0.05).fit(calibration)

    alert_rate = np.mean(calibrator.predict_alerts(test_scores))
    assert alert_rate <= 0.065
    assert alert_rate >= 0.035


def test_large_upper_tail_contamination_is_conservative_for_fixed_queries():
    clean = np.linspace(-2.0, 2.0, 99)
    contaminated = np.concatenate([clean, np.full(10, 100.0)])
    query = np.linspace(1.5, 5.0, 20)

    clean_calibrator = rc.ConformalAlertCalibrator(alpha=0.05).fit(clean)
    contaminated_calibrator = rc.ConformalAlertCalibrator(alpha=0.05).fit(
        contaminated
    )

    assert np.all(
        contaminated_calibrator.p_values(query)
        >= clean_calibrator.p_values(query)
    )


def test_scalar_and_array_outputs_are_stable():
    calibrator = rc.ConformalAlertCalibrator().fit(np.arange(30.0))
    assert isinstance(calibrator.p_values(100.0), float)
    assert isinstance(calibrator.predict_alerts(100.0), bool)
    assert isinstance(calibrator.predict(100.0), int)

    p = calibrator.p_values(np.array([1.0, 2.0]))
    alerts = calibrator.predict_alerts(np.array([1.0, 100.0]))
    labels = calibrator.predict(np.array([1.0, 100.0]))
    assert p.shape == (2,)
    assert alerts.dtype == bool
    assert np.array_equal(labels, [1, -1])


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({"alpha": 0.0}, ValueError, "alpha"),
        ({"alpha": 1.0}, ValueError, "alpha"),
        ({"alpha": True}, TypeError, "alpha"),
        ({"tail": "both"}, ValueError, "tail"),
        ({"tail": 1}, TypeError, "tail"),
    ],
)
def test_parameter_validation(kwargs, error, message):
    with pytest.raises(error, match=message):
        rc.ConformalAlertCalibrator(**kwargs).fit([1.0, 2.0])


@pytest.mark.parametrize(
    "scores, message",
    [
        ([], "at least one"),
        ([[1.0, 2.0]], "one-dimensional"),
        ([1.0, np.nan], "finite"),
    ],
)
def test_fit_input_validation(scores, message):
    with pytest.raises(ValueError, match=message):
        rc.ConformalAlertCalibrator().fit(scores)


def test_query_validation_and_unfitted_calls():
    calibrator = rc.ConformalAlertCalibrator()
    with pytest.raises(AttributeError, match="not fitted"):
        calibrator.p_values([1.0])

    calibrator.fit([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="scalar or one-dimensional"):
        calibrator.p_values([[1.0]])
    with pytest.raises(ValueError, match="finite"):
        calibrator.p_values([np.inf])


def test_sklearn_parameter_protocol_when_available():
    sklearn = pytest.importorskip("sklearn")
    clone = sklearn.base.clone(rc.ConformalAlertCalibrator(alpha=0.1, tail="lower"))
    assert clone.alpha == 0.1
    assert clone.tail == "lower"
