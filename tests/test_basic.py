import numpy as np
import pytest
import robustcov as rc


def test_fast_mcd_basic():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    X[:10] += 10
    est = rc.FastMCD(n_init=20, random_state=0).fit(X)
    assert est.location_.shape == (4,)
    assert est.covariance_.shape == (4, 4)
    assert est.distances_.shape == (120,)
    assert est.support_.dtype == bool


def test_tyler_scale_invariance():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 5))
    a = rc.TylerShape(max_iter=200).fit(X).shape_
    b = rc.TylerShape(max_iter=200).fit(10 * X).shape_
    np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-5)


def test_regularized_tyler_high_dim_runs():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 30))
    est = rc.RegularizedTyler(alpha=0.2, max_iter=100).fit(X)
    assert est.covariance_.shape == (30, 30)


def test_detector():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(100, 3))
    labels = rc.RobustOutlierDetector(alpha=0.95).fit_predict(X)
    assert set(labels).issubset({-1, 1})


def test_missing_values_median_imputation_fast_mcd():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(150, 4))
    X[::7, 2] = np.nan
    est = rc.FastMCD(n_init=20, random_state=0, missing_values="median").fit(X)
    assert np.isfinite(est.covariance_).all()
    d = est.mahalanobis(X)
    assert np.isfinite(d).all()


def test_missing_values_median_imputation_tyler():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(160, 5))
    X[::8, 1] = np.nan
    est = rc.RegularizedTyler(alpha=0.1, missing_values="median", max_iter=100).fit(X)
    assert np.isfinite(est.covariance_).all()


def test_robust_median_imputer():
    X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    Xt = rc.RobustMedianImputer().fit_transform(X)
    assert np.isfinite(Xt).all()
    assert Xt[0, 1] == 5.0


def test_fast_mcd_quality_and_raw_attributes():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(180, 5))
    X[:18] += 8
    est = rc.FastMCD(quality="balanced", n_init=30, n_best=4, random_state=0).fit(X)
    assert est.quality == "balanced"
    assert est.raw_support_.dtype == bool
    assert est.raw_support_.sum() == est.h_
    assert np.isfinite(est.raw_det_)
    assert np.isfinite(est.det_)
    assert est.raw_objective_value_ == pytest.approx(
        np.linalg.slogdet(est.raw_covariance_)[1]
    )
    assert est.objective_value_ == pytest.approx(
        np.linalg.slogdet(est.covariance_)[1]
    )
    assert np.isfinite(est.c_step_objective_value_)

    limited = rc.FastMCD(
        n_init=12,
        n_best=4,
        initial_c_steps=1,
        max_iter=1,
        random_state=0,
    ).fit(X)
    assert limited.n_iter_ == 1
    assert limited.converged_ is False


def test_fast_mcd_contamination_sets_h():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 5))
    est = rc.FastMCD(contamination=0.25, n_init=20, random_state=0).fit(X)
    assert 145 <= est.h_ <= 150
    assert abs(est.effective_support_fraction_ - est.h_ / 200) < 1e-12


def test_fast_mcd_parameter_validation():
    with pytest.raises(ValueError, match="either support_fraction or contamination"):
        rc.FastMCD(contamination=0.1, support_fraction=0.8).fit(
            np.zeros((12, 2))
        )
    with pytest.raises(ValueError, match="initial_c_steps must be positive"):
        rc.FastMCD(initial_c_steps=0).fit(np.zeros((12, 2)))


def test_plotting_helpers(tmp_path):
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(8)
    X = rng.normal(size=(80, 3))
    X[:8] += 6
    est = rc.FastMCD(n_init=20, random_state=0).fit(X)
    rc.plot_mahalanobis_diagnostics(est, output_path=tmp_path / "diag.png", show=False)
    rc.plot_covariance_heatmap(est.covariance_, output_path=tmp_path / "cov.png", show=False)
    assert (tmp_path / "diag.png").exists()
    assert (tmp_path / "cov.png").exists()


def test_2d_plotting_helpers(tmp_path):
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(9)
    X = rng.normal(size=(90, 2))
    X[:9] += 5
    y = np.zeros(90, dtype=int)
    y[:9] = 1
    est = rc.FastMCD(n_init=20, random_state=0).fit(X)
    rc.plot_anomaly_scatter_2d(est, X, labels=y, output_path=tmp_path / "scatter.png", show=False)
    rc.plot_distance_scatter_2d(est, X, output_path=tmp_path / "dist.png", show=False)
    assert (tmp_path / "scatter.png").exists()
    assert (tmp_path / "dist.png").exists()


def test_diagnostic_report_summary():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(120, 4))
    X[:12] += 7
    est = rc.FastMCD(n_init=20, random_state=0).fit(X)
    report = rc.diagnostic_report(est)
    text = report.summary()
    assert "Robust diagnostic report" in text
    assert report.n_samples == 120
    assert report.n_features == 4
    assert isinstance(report.as_dict(), dict)


def test_auto_robust_anomaly_detector():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(120, 4))
    X[:10] += 6
    det = rc.AutoRobustAnomalyDetector(contamination=0.1).fit(X)
    assert det.labels_.shape == (120,)
    assert det.score_.shape == (120,)
    labels = det.predict(X[:5])
    assert labels.shape == (5,)



def test_student_t_scatter_runs_high_dim():
    rng = np.random.default_rng(9)
    X = rng.standard_t(df=2, size=(30, 40))
    est = rc.StudentTScatter(df=3, alpha=0.1, max_iter=80, warn_on_nonconvergence=False).fit(X)
    assert est.covariance_.shape == (40, 40)
    assert np.isfinite(est.covariance_).all()
    assert est.distances_.shape == (30,)


def test_regularized_cauchy_runs():
    rng = np.random.default_rng(10)
    X = rng.standard_t(df=1, size=(50, 12))
    est = rc.RegularizedCauchy(alpha=0.2, max_iter=50).fit(X)
    assert est.covariance_.shape == (12, 12)
    assert np.isfinite(est.radial_kurtosis_)


def test_named_regularized_tyler_aliases():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(80, 10))
    a = rc.KLRegularizedTyler(alpha=0.1, max_iter=80).fit(X)
    b = rc.WieselTyler(alpha=0.1, max_iter=80).fit(X)
    assert a.penalty == "kl"
    assert b.penalty == "wiesel"
    assert a.shape_.shape == (10, 10)
    assert b.shape_.shape == (10, 10)


def test_hellinger_regularized_tyler_experimental_runs():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(60, 8))
    est = rc.HellingerRegularizedTyler(alpha=0.1, max_iter=30).fit(X)
    assert est.shape_.shape == (8, 8)
    assert np.isfinite(est.shape_).all()


def test_auto_robust_scatter_selects_estimator():
    rng = np.random.default_rng(13)
    X = rng.standard_t(df=2, size=(35, 45))
    est = rc.AutoRobustScatter().fit(X)
    assert est.covariance_.shape == (45, 45)
    assert est.best_estimator_name_
    assert len(est.candidate_results_) >= 2
    assert "AutoRobustScatter selected" in est.summary()


def test_m_estimator_damping_parameter():
    rng = np.random.default_rng(14)
    X = rng.standard_t(df=1.5, size=(40, 20))
    est = rc.RegularizedCauchy(alpha=0.1, damping=0.5, max_iter=80, warn_on_nonconvergence=False).fit(X)
    assert est.damping == 0.5
    assert np.isfinite(est.covariance_).all()


def test_auto_robust_scatter_stability_fields():
    rng = np.random.default_rng(15)
    X = rng.standard_t(df=1.5, size=(30, 35))
    est = rc.AutoRobustScatter(selection="stability", n_splits=2, random_state=0).fit(X)
    assert est.selection == "stability"
    assert np.isfinite(est.best_result_.diagnostic_score)
    assert np.isfinite(est.best_result_.stability_score)
    assert "stability=" in est.summary()


def test_auto_robust_scatter_diagnostic_mode():
    rng = np.random.default_rng(16)
    X = rng.standard_t(df=2, size=(40, 20))
    est = rc.AutoRobustScatter(selection="diagnostic").fit(X)
    assert est.best_estimator_name_
    assert est.best_result_.stability_score == 0.0


def test_distance_profile_plot_helpers(tmp_path):
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(10)
    X = rng.normal(size=(90, 4))
    X[:9] += 5
    est = rc.FastMCD(n_init=20, random_state=0).fit(X)
    rc.plot_robust_distance_profile(est, output_path=tmp_path / "profile.png", show=False)
    rc.plot_robust_distance_panel(est, output_path=tmp_path / "panel.png", show=False)
    assert (tmp_path / "profile.png").exists()
    assert (tmp_path / "panel.png").exists()


def test_openmp_thread_helpers_roundtrip():
    assert isinstance(rc.has_openmp(), bool)
    old = rc.get_num_threads()
    try:
        rc.set_num_threads(1)
        assert rc.get_num_threads() >= 1
        rng = np.random.default_rng(17)
        X = rng.normal(size=(120, 5))
        est = rc.FastMCD(n_init=10, random_state=0, n_jobs=1).fit(X)
        assert est.covariance_.shape == (5, 5)
        ty = rc.RegularizedTyler(alpha=0.1, max_iter=20, n_jobs=1).fit(X)
        assert ty.covariance_.shape == (5, 5)
    finally:
        rc.set_num_threads(old)


def test_thread_limit_context_restores_threads():
    old = rc.get_num_threads()
    try:
        with rc.thread_limit(1):
            assert rc.get_num_threads() >= 1
        assert rc.get_num_threads() == old
    finally:
        rc.set_num_threads(old)


def test_detector_contamination_api_matches_requested_fraction():
    rng = np.random.default_rng(18)
    X = rng.normal(size=(200, 4))
    X[:20] += 7.0
    det = rc.RobustOutlierDetector(
        estimator=rc.FastMCD(n_init=20, random_state=0),
        contamination=0.10,
    ).fit(X)
    assert (det.labels_ == -1).sum() == 20
    assert np.all((det.decision_function(X) >= 0.0) == (det.predict(X) == 1))


def test_core_estimators_follow_sklearn_parameter_protocol():
    from sklearn.base import clone

    est = rc.FastMCD(quality="balanced", n_init=25, random_state=7)
    cloned = clone(est)
    assert cloned is not est
    assert cloned.get_params(deep=False) == est.get_params(deep=False)
    assert "quality='balanced'" in repr(est)

    detector = rc.RobustOutlierDetector(
        estimator=est,
        contamination=0.1,
    )
    cloned_detector = clone(detector)
    assert cloned_detector.contamination == 0.1
    assert cloned_detector.estimator is not detector.estimator
    assert cloned_detector.estimator.get_params(deep=False) == est.get_params(deep=False)


def test_nested_set_params_updates_detector_estimator():
    detector = rc.RobustOutlierDetector(estimator=rc.FastMCD())
    detector.set_params(estimator__quality="high", estimator__random_state=11)
    assert detector.estimator.quality == "high"
    assert detector.estimator.random_state == 11


def test_fast_mcd_quality_defaults_resolve_at_fit_time():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(80, 3))
    est = rc.FastMCD()
    assert est.n_init is None
    est.set_params(quality="balanced", n_init=12, n_best=3).fit(X)
    assert est.effective_n_init_ == 12
    assert est.effective_n_best_ == 3
    assert est.effective_max_iter_ == 100


def test_m_estimator_uses_optimized_mahalanobis_contraction(monkeypatch):
    import robustcov.m_estimators as module

    calls = []
    original = module._mahalanobis_from_precision

    def wrapped(centered, precision):
        calls.append(centered.shape)
        return original(centered, precision)

    monkeypatch.setattr(module, "_mahalanobis_from_precision", wrapped)
    X = np.random.default_rng(17).normal(size=(80, 12))
    rc.RegularizedCauchy(
        alpha=0.1,
        max_iter=5,
        warn_on_nonconvergence=False,
    ).fit(X)
    assert calls
    assert all(shape == X.shape for shape in calls)
