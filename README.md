# robustcov

`robustcov` is an experimental Python/C++ library for robust covariance, heavy-tail scatter estimation, and interpretable anomaly diagnostics.

It is designed for workflows where classical covariance estimates are unstable: contamination, heavy-tailed data, small samples, high-dimensional scatter estimation, and robust-distance based anomaly screening.

> Status: **alpha / experimental**. APIs and benchmark pages may change before a stable release.

## Highlights

- Fast robust covariance via `FastMCD`
- Heavy-tail scatter estimators: `RegularizedCauchy`, `StudentTScatter`, `RegularizedTyler`
- Robust anomaly detection with Mahalanobis-style robust distances
- Cluster-aware robust diagnostics for multimodal data
- Visual diagnostics: distance profiles, QQ plots, covariance heatmaps, anomaly panels
- Optional OpenMP acceleration in the C++ backend
- Sphinx documentation with benchmark and use-case galleries
- Optional external/Kaggle examples for fraud, finance, maintenance, and medical screening

## Installation

From PyPI after a release is published:

```bash
python -m pip install -U pip
python -m pip install robustcov
```

Supported release wheels are built for CPython 3.12, 3.13, and 3.14 on Ubuntu, Windows, and macOS by GitHub Actions. The package uses a C++/pybind11 backend built with `scikit-build-core`.

Inside a conda environment, install the PyPI wheels with pip:

```bash
conda create -n robustcov python=3.12 pip
conda activate robustcov
python -m pip install robustcov
```

For local development:

```bash
git clone https://github.com/smiryusupov/robustcov.git
cd robustcov

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

python -m pip install -U pip
python -m pip install -e ".[dev,docs,examples]"
python -m compileall -q robustcov tests examples benchmarks docs
python -m pytest -q
```

## Quickstart

```python
import numpy as np
import robustcov as rc

rng = np.random.default_rng(0)

# Heavy-tailed data with injected outliers
X = rng.standard_t(df=3, size=(400, 5))
X[:30] += 8.0

est = rc.FastMCD(quality="balanced", random_state=42).fit(X)

print(est.location_)
print(est.covariance_)
print(est.radial_kurtosis_)

det = rc.RobustOutlierDetector(estimator=est, contamination=0.075).fit(X)
print(det.labels_)
```

For small-sample or high-dimensional heavy-tailed data:

```python
est = rc.RegularizedCauchy(alpha=0.10).fit(X)
print(est.covariance_)

student = rc.StudentTScatter(df=3, alpha=0.05).fit(X)
print(student.radial_kurtosis_)
```

For automatic exploratory selection:

```python
auto = rc.AutoRobustScatter(selection="diagnostic").fit(X)

print(auto.best_estimator_name_)
print(auto.summary())
```

## Main estimators

| Estimator | Best use case | Notes |
|---|---|---|
| `FastMCD` | Separable contamination, `n >> p` | Fast robust covariance and support diagnostics |
| `RegularizedCauchy` | Very heavy tails, small samples, `p` close to `n` | Strong radial downweighting plus shrinkage |
| `StudentTScatter` | Diffuse heavy tails | Smooth heavy-tail scatter estimator |
| `RegularizedTyler` | Heavy-tailed shape estimation | Scale-free shape unless scale correction is requested |
| `AutoRobustScatter` | Exploratory estimator selection | Diagnostic or stability-based selector |
| `ClusterRobustOutlierDetector` | Multimodal data | Cluster-then-local-robust-scatter diagnostic |

`KLRegularizedTyler` and `WieselTyler` are currently documented as aliases/prototype variants around the regularized Tyler implementation. `HellingerRegularizedTyler` is experimental.

## Visual diagnostics

```python
est = rc.FastMCD(quality="balanced", random_state=0).fit(X)

rc.plot_robust_distance_profile(
    est,
    output_path="distance_profile.png",
    show=False,
)

rc.plot_mahalanobis_qq(
    est,
    output_path="qq.png",
    show=False,
)

rc.plot_covariance_heatmap(
    est.covariance_,
    title="FastMCD covariance",
    output_path="covariance.png",
    show=False,
)
```

Diagnostic reports summarize robust-distance behavior:

```python
report = rc.diagnostic_report(est)
print(report.summary())
```

Reports include radial kurtosis, detected fraction, condition number, support fraction, QQ tail deviation, and heuristic recommendations.

## Multimodal data

A single global robust covariance model can fail when the data have several legitimate modes. Use cluster-aware diagnostics when modes correspond to meaningful groups, regimes, or segments.

```python
det = rc.ClusterRobustOutlierDetector(
    n_clusters=3,
    contamination=0.05,
    random_state=0,
).fit(X)

scores = det.decision_function(X)
labels = det.predict(X)

rc.plot_cluster_robust_distances(
    det,
    X,
    output_path="cluster_distances.png",
    show=False,
)
```

This is not a full robust mixture model. It is a practical cluster-then-robust-scatter diagnostic.

## OpenMP acceleration

If OpenMP is available at build time, the C++ backend can parallelize distance evaluation, covariance accumulation, Tyler scatter updates, and FastMCD candidate evaluation.

```python
import robustcov as rc

print(rc.has_openmp())
rc.set_num_threads(4)

est = rc.FastMCD(n_init=500, n_jobs=4, random_state=0).fit(X)
```

For reproducible scaling benchmarks, avoid BLAS/OpenMP oversubscription:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python benchmarks/openmp_scaling.py \
  --n 8000 \
  --p 20 \
  --threads 1 2 4 \
  --csv results/openmp_scaling.csv
```

## Documentation

Build the Sphinx docs locally:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -b html docs docs/_build/html
```

Main documentation entry points:

- **Use-case gallery**: practical application pages organized by topic
- **Benchmark gallery**: benchmark plots, tables, and interpretation
- **Algorithms**: mathematical descriptions and references
- **Robust statistics background**: influence functions, Gateaux derivatives, breakdown point, geodesic convexity, and small-sample issues
- **External and Kaggle gallery**: optional external-data results

Do not commit `docs/_build/`; it is generated by Sphinx.

## Benchmarks

Generate the benchmark report:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python benchmarks/make_report.py --outdir results/report
```

This writes CSV files, plots, a Markdown report, and a standalone HTML report:

```text
results/report/benchmark_report.html
results/report/benchmark_report.md
results/report/*.csv
results/report/*.png
```

The benchmark documentation is intentionally honest. `robustcov` is not expected to win every anomaly-detection task. The package is strongest when the signal is covariance-shaped, heavy-tailed, high-dimensional, or benefits from interpretable robust distances.

## Examples

Run the reproducible use-case gallery:

```bash
python examples/run_use_case_gallery.py --all
```

Selected examples:

```bash
python examples/use_case_finance_risk.py
python examples/use_case_multimodal_anomaly.py
python examples/use_case_sensor_anomaly.py
python examples/use_case_breast_cancer_screening.py
python examples/use_case_digits_one_class_baselines.py
python examples/use_case_ml_preprocessing.py
```

Refresh generated gallery assets after editing examples:

```bash
python docs/generate_gallery_assets.py
python -m sphinx -b html docs docs/_build/html
```

## External and Kaggle examples

External examples live under `examples_external/`. They are optional and are not part of the test suite because they require manual dataset downloads and may have separate licenses.

Example:

```bash
python examples_external/kaggle_credit_card_fraud.py \
  --data examples_external/data/creditcard.csv \
  --outdir results/external/credit_card_fraud
```

Collect external result summaries:

```bash
python examples_external/collect_external_results.py \
  --root results/external \
  --outdir results/external_registry
```

External result pages should be read as evidence, not as leaderboard claims. Some datasets are strong wins, some are competitive but slower, and some are included mainly to show limitations.

## Scope

`robustcov` currently focuses on:

1. efficient robust covariance for classical contamination;
2. heavy-tail scatter estimators for small-sample/high-dimensional regimes;
3. robust-distance anomaly diagnostics;
4. application and benchmark galleries with reproducible scripts.

Minimum-volume ellipsoid and full robust mixture modeling are not core priorities yet. They may be added later as experimental features if they strengthen the package without distracting from the current scope.

## Development

```bash
python -m pip install -e ".[dev,docs]"
python -m pytest -q
python -m sphinx -b html docs docs/_build/html
```

Build distribution artifacts:

```bash
python -m build
python -m twine check dist/*
```

Release wheels are built by `.github/workflows/wheels.yml` using `cibuildwheel`. Push a `v*` tag to publish to PyPI via Trusted Publishing after configuring the `pypi` environment on PyPI/GitHub. See `RELEASE.md` for the full checklist.

## Project status

This is a pre-1.0 alpha package. Public APIs may change. The goal of the early releases is to make the estimators, diagnostics, benchmarks, and documentation easy to inspect before stabilizing the interface.

## License

Apache-2.0. See `LICENSE`.

## Citation

If you use `robustcov` in research or applied work, please cite the package using the metadata in `CITATION.cff`.