# Benchmarks

All benchmark scripts now support optional CSV output so you can create plots later.

Suggested workflow:

```bash
python benchmarks/accuracy_vs_contamination.py --csv results/accuracy.csv
python benchmarks/speed_estimators.py --csv results/speed.csv
python benchmarks/fastmcd_quality_speed_tradeoff.py --csv results/tradeoff.csv
python benchmarks/support_diagnostics.py --csv results/support.csv
```

Then visualize with:

```bash
python examples/plot_contamination_accuracy.py --input results/accuracy.csv --output results/accuracy.png
python examples/plot_speed_comparison.py --input results/speed.csv --output results/speed.png
```


Harder scenarios:

```bash
python benchmarks/hard_contamination_scenarios.py --csv results/hard_scenarios.csv
```

This benchmark is intended to expose limits, not just confirm a happy path.


Small-sample heavy-tail benchmark:

```bash
python benchmarks/small_sample_heavy_tail.py --csv results/small_sample.csv
```

This benchmark stresses `p/n`, heavy-tailed multivariate t samples, estimator failures, condition numbers, and covariance error.


Auto scatter selector benchmark:

```bash
python benchmarks/auto_scatter_small_sample.py --csv results/auto_scatter.csv
```

This benchmark reports the estimator selected by `AutoRobustScatter`, its unsupervised score, condition number, radial kurtosis, convergence status, and ground-truth covariance error for synthetic data.

Auto scatter selector benchmark:

```bash
python benchmarks/auto_scatter_small_sample.py --selection stability --csv results/auto_scatter.csv
```


## v0.13 summary tables

After generating benchmark CSVs, aggregate them with:

```bash
python benchmarks/benchmark_summary.py --input results/small_sample.csv --csv results/small_sample_summary.csv
```

The summary reports appearances, failures, win rate, mean rank, median error, mean error, and median seconds per method.

## One-command benchmark report

Run:

```bash
python benchmarks/make_report.py --outdir results/report
```

This produces CSVs, plots, and `results/report/benchmark_report.md`. It is the easiest way to regenerate the evidence used in the README/docs.


## OpenMP scaling benchmark

```bash
python benchmarks/openmp_scaling.py --n 8000 --p 20 --threads 1 2 4 --csv results/openmp_scaling.csv
```

For clean measurements, avoid BLAS/OpenMP oversubscription:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/openmp_scaling.py
```
