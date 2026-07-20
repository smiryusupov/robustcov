# Examples

The examples are organized by **method family**. The Sphinx use-case gallery also
provides a secondary browse-by-domain view for finance, fraud, sensors,
biomedical data, embeddings, and real ML datasets.

List or run the registered groups:

```bash
python examples/run_use_case_gallery.py --list
python examples/run_use_case_gallery.py --group ica
python examples/run_use_case_gallery.py --group pca
python examples/run_use_case_gallery.py --group robust
python examples/run_use_case_gallery.py --group monitoring
python examples/run_use_case_gallery.py --all
```

## ICA, SOBI, and source separation

- `ica_two_scatter.py` – robust two-scatter ICA on a contaminated linear mixture.
- `sobi_source_separation.py` – classical versus robust SOBI under impulsive contamination.
- `source_separation_and_factor_models.py` – compact combined ICA/SOBI/factor-model smoke example.

## PCA and factor models

- `distributionally_robust_pca.py` – weighted-Wasserstein PCA under structured deployment shift.
- `distributionally_robust_pca_drift_monitoring.py` – calibrated window monitoring that tolerates geometry-aligned shift and flags off-geometry drift.
- `robust_factor_model.py` – robust static factor model with automatic factor-count selection.
- `plot_robust_pca_yield_curve.py` – robust level, slope, and curvature factors.
- `plot_robust_pca_subspace_stability.py` – bootstrap stability of robust PCA factors.
- `plot_robust_pca_dependent_stability.py` – dependent-bootstrap factor stability.
- `plot_robust_pca_market_risk.py` – systemic versus idiosyncratic market shocks.
- `plot_robust_pca_embedding_monitoring.py` – production embedding drift and OOD monitoring.
- `plot_cellpca_process_spectra.py` – cellwise robust PCA for process spectra.
- `plot_sparse_cellpca_spectra.py` – sparse, interpretable CellPCA loadings.
- `plot_density_power_pca.py` – density-power PCA under mixed contamination.
- `plot_robust_multilinear_pca.py` – robust PCA for matrix-valued observations.
- `robust_pca_outlier_map.py` – compact score/orthogonal-distance introduction.

## Robust covariance, scatter, and precision

- `use_case_finance_risk.py` – heavy-tail covariance for return-like data.
- `use_case_portfolio_stress.py` – empirical versus robust portfolio covariance under stress.
- `plot_mrcd_high_dimensional_outliers.py` – MRCD when the feature count approaches or exceeds the sample size.
- `plot_kmrcd_nonlinear_manifold.py` – kernel MRCD on curved inlier structure.
- `plot_dets_detmm_tradeoff.py` – DetS/DetMM robustness-efficiency comparison.
- `plot_cellmcd_market_data.py` – isolated bad ticks, missing values, and cellwise diagnostics.
- `plot_cellrcov_high_dimensional.py` – high-dimensional covariance under rowwise and cellwise contamination.
- `plot_mmcd_sensor_windows.py` – robust row/column covariance for matrix-valued windows.
- `plot_robust_graphical_lasso_market_network.py` – robust sparse conditional-dependence network.
- `plot_spatial_sign_graphical_lasso.py` – scale-free precision under radial heavy tails.
- `small_sample_heavy_tail.py` – Tyler, Student-t, Cauchy, and related small-sample estimators.
- `auto_robust_scatter.py` and `auto_selection_stability.py` – automatic estimator selection.

## Anomaly detection and monitoring

- `anomaly_detection.py` – robust distances and outlier detection.
- `use_case_fraud_screening.py` – fraud-style tabular anomaly screening.
- `use_case_network_traffic.py` – network-traffic anomaly simulation.
- `use_case_sensor_anomaly.py` – correlated multivariate sensor bursts.
- `use_case_maintenance_monitoring.py` – predictive-maintenance monitoring.
- `use_case_quality_control.py` – robust process monitoring.
- `plot_robust_subspace_monitoring.py` – score-space and orthogonal drift monitoring.
- `distributionally_robust_pca_drift_monitoring.py` – distribution-shift-aware PCA monitoring with empirical window calibration.
- `feature_geometry_drift_detection.py` – robust feature-geometry drift detection.
- `feature_geometry_embedding_monitoring.py` – practical embedding monitoring.
- `use_case_breast_cancer_screening.py`, `use_case_digits_one_class_baselines.py`, and `use_case_wine_class_screening.py` – reproducible real-data screening.
- `use_case_ml_preprocessing.py` – robust filtering before classification.
- `use_case_biomedical_signal.py`, `use_case_image_feature_anomaly.py`, `use_case_text_embedding_outliers.py`, and `use_case_multimodal_anomaly.py` – feature-vector and multimodal anomaly workflows.

## Geometry, kernels, and diagnostics

- `spd_geometry_diagnostics.py` and `spd_geometry_ml_use_cases.py` – SPD distances, geodesics, drift, and whitening.
- `feature_geometry_synthetic_ood.py` and related `feature_geometry_*` scripts – robust learned-feature geometry.
- `gp_robust_input_metric.py` – robust Gaussian-process/kernel input metric.
- `embedding_reranking_robust_geometry.py` – retrieval and RAG filtering.
- `visual_diagnostics.py`, `visual_anomaly_2d.py`, and `diagnostic_report_demo.py` – plots and reports.

## Benchmarks and external data

Benchmark plotting and contamination studies remain in `examples/`, while
optional downloaded-data workflows live in `examples_external/`. Raw data is
never committed: explicit loaders cache UCI gas-sensor and NASA C-MAPSS archives
outside the repository. Core tests exercise the loaders and protocols with tiny
local fixtures, while real-data runs remain optional and network-dependent.
