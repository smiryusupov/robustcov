# Examples

Core examples:

- `quickstart.py` – small smoke test.
- `anomaly_detection.py` – robust distances and outlier detection.
- `covariance_contamination_study.py` – covariance error under contamination.
- `missing_values_stability.py` – contamination plus missing values.
- `adaptive_contamination_fastmcd.py` – contamination-aware `FastMCD`.
- `visual_diagnostics.py` – generates general diagnostic PNG files.
- `visual_anomaly_2d.py` – 2D anomaly support, rejection, ellipse, and distance plots.
- `diagnostic_report_demo.py` – text diagnostic report with recommendations.
- `real_ml_anomaly.py` – sklearn breast cancer anomaly smoke test.
- `real_ml_one_class_digits.py` – one-class digits anomaly smoke test.
- `plot_contamination_accuracy.py` – plots benchmark CSVs as contamination curves.
- `plot_speed_comparison.py` – plots benchmark speed CSVs.
- `plot_speed_accuracy_pareto.py` – plots speed/error Pareto points.
- `plot_spd_geometry_ml_use_cases.py` – plots SPD-geometry ML diagnostics for drift monitoring, estimator stability, and robust whitening.
- `spd_geometry_ml_use_cases.py` – ML use-case patterns for SPD geometry: drift monitoring, scatter comparison, and robust similarity.
- `spd_geometry_diagnostics.py` – SPD-geometry utilities for comparing robust scatter matrices and Tyler fixed-point residuals.
- `gp_robust_input_metric.py` – robust GP kernel / input-metric example showing how contaminated input covariance can distort kernel geometry.


Small-sample heavy-tail examples:

- `small_sample_heavy_tail.py` – compares regularized Tyler, Student-t scatter, Cauchy, and experimental Hellinger-style Tyler when `p` is close to or larger than `n`.
- `estimator_selection_guide.py` – prints practical estimator guidance.

- `auto_robust_scatter.py` – fits several small-sample robust scatter candidates and prints the selected estimator plus diagnostics.

- `auto_selection_stability.py` – compares diagnostic and stability auto scatter selection.


Additional v0.13 examples:

- `robust_distance_profile_demo.py` – ordered robust distance / proline-style profile plots.
- `plot_benchmark_summary.py` – bar plots for benchmark rank/win-rate summaries.

## v0.14 application use cases

- `use_case_finance_risk.py` – small-sample heavy-tailed covariance for return-like data.
- `use_case_sensor_anomaly.py` – robust distances for multivariate sensor bursts.
- `use_case_quality_control.py` – robust covariance monitoring for correlated process measurements.

These are designed to make the package accessible to ML users who care about anomaly detection, risk, and monitoring workflows.


## Expanded use-case gallery

Run all application templates:

```bash
python examples/run_use_case_gallery.py
```

The gallery includes finance/risk covariance, portfolio stress, sensor anomaly detection, predictive maintenance, quality control, fraud-style tabular screening, network traffic anomaly simulation, biomedical/signal-window features, image-feature anomalies, text/embedding outliers, and robust ML preprocessing.


### Matching Sphinx pages

Each gallery example has a corresponding Sphinx page under `docs/gallery/`. The pages explain the synthetic/public data, the practical ML problem, the recommended estimator, and how to interpret the printed metrics and saved robust-distance diagnostics.


### v0.24 real ML use cases

The gallery now includes additional reproducible sklearn-based ML examples: breast-cancer anomaly screening, digits one-class anomaly detection, and wine class screening.  These pages include baseline comparisons, captured output, and generated plots.  Optional Kaggle/external dataset examples are planned under `examples_external/` so core docs remain reproducible without downloads.

## Optional external / Kaggle examples

External examples live in `examples_external/` rather than `examples/` because they require downloaded datasets. They are designed for Kaggle notebooks and reproducible local runs but are intentionally not part of tests.
