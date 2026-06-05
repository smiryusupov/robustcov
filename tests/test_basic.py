import numpy as np
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


def test_fast_mcd_contamination_sets_h():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 5))
    est = rc.FastMCD(contamination=0.25, n_init=20, random_state=0).fit(X)
    assert 145 <= est.h_ <= 150
    assert abs(est.effective_support_fraction_ - est.h_ / 200) < 1e-12


def test_fast_mcd_contamination_conflicts_with_support_fraction():
    try:
        rc.FastMCD(contamination=0.1, support_fraction=0.8)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_plotting_helpers(tmp_path):
    rng = np.random.default_rng(8)
    X = rng.normal(size=(80, 3))
    X[:8] += 6
    est = rc.FastMCD(n_init=20, random_state=0).fit(X)
    rc.plot_mahalanobis_diagnostics(est, output_path=tmp_path / "diag.png", show=False)
    rc.plot_covariance_heatmap(est.covariance_, output_path=tmp_path / "cov.png", show=False)
    assert (tmp_path / "diag.png").exists()
    assert (tmp_path / "cov.png").exists()


def test_2d_plotting_helpers(tmp_path):
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
