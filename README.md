# RobustCov

[![PyPI](https://img.shields.io/pypi/v/robustcov.svg)](https://pypi.org/project/robustcov/)
[![Python](https://img.shields.io/pypi/pyversions/robustcov.svg)](https://pypi.org/project/robustcov/)
[![Docs](https://readthedocs.org/projects/robustcov/badge/?version=latest)](https://robustcov.readthedocs.io/en/latest/)
[![CI](https://github.com/smiryusupov/robustcov/actions/workflows/tests.yml/badge.svg)](https://github.com/smiryusupov/robustcov/actions/workflows/tests.yml)
[![Wheels](https://github.com/smiryusupov/robustcov/actions/workflows/wheels.yml/badge.svg)](https://github.com/smiryusupov/robustcov/actions/workflows/wheels.yml)
[![License](https://img.shields.io/pypi/l/robustcov.svg)](https://github.com/smiryusupov/robustcov/blob/main/LICENSE)

**Robust covariance, anomaly scoring, PCA, and monitoring for difficult multivariate data.**

`robustcov` is a Python/C++ library for robust multivariate geometry. It
estimates covariance, scatter, precision, principal subspaces, and related
latent structure when empirical covariance is unreliable because the data are
contaminated, heavy-tailed, high-dimensional, incomplete, structured, or
shifting.

Use it to:

- fit robust covariance or scatter and compute Mahalanobis anomaly scores;
- convert held-out anomaly or monitoring scores into conformal p-values and calibrated alert labels;
- perform robust PCA, reconstruction diagnostics, and frozen-reference subspace monitoring;
- build robust whitening transforms, kernels, and metrics for learned features or embeddings;
- handle bad cells, missing values, matrix-valued observations, and multilinear low-rank structure;
- estimate sparse precision graphs or recover robust independent sources and latent factors.

The package provides numerical estimators and diagnostics with sklearn-style
`fit` APIs. It does not train neural networks or replace a production monitoring
platform.

> Status: **alpha / experimental**. Core estimator interfaces are intended to
> remain recognizable, but some APIs may change before 1.0.

## Start from your problem

| Your data or goal | Start with |
|---|---|
| A minority of complete rows are outliers and `n` is comfortably larger than `p` | `FastMCD`, `DetS`, or `DetMM` |
| Broad heavy tails or an ill-conditioned/high-dimensional covariance | `RegularizedCauchy`, `StudentTScatter`, `RegularizedTyler`, or `MRCD` |
| Isolated bad cells or missing entries | `CellMCD`, `CellRCov`, `CellPCA`, or `SparseCellPCA` |
| Matrix-valued or multilinear observations | `MMCD` or `RobustMultilinearPCA` |
| Robust dimensionality reduction or changing subspaces | `RobustPCA`, `DistributionallyRobustPCA`, `SubspaceStability`, or `RobustSubspaceMonitor` |
| Turn a held-out anomaly or monitoring score into a finite-sample alert | `ConformalAlertCalibrator` |
| Sparse conditional-dependence structure | `RobustGraphicalLasso` or `SGLASSO` |
| Learned features, embeddings, whitening, or robust kernels | `FeatureGeometry` |
| Independent or temporally correlated latent sources | `TwoScatterICA`, `RobustSOBI`, or `RobustFactorModel` |

See the [documentation](https://robustcov.readthedocs.io/en/latest/) for the
task-oriented workflow map, estimator selection guide, examples, benchmarks,
and API reference.

## Method families

- **Covariance and scatter:** `FastMCD`, `DetS`, `DetMM`, `MRCD`, `KMRCD`, regularized Cauchy, Student-t, and Tyler estimators.
- **Cellwise and structured data:** `CellMCD`, `CellRCov`, `MMCD`, `RobustMultilinearPCA`, `CellPCA`, and `SparseCellPCA`.
- **PCA and monitoring:** `RobustPCA`, `DensityPowerRobustPCA`, experimental `DistributionallyRobustPCA`, `SubspaceStability`, `RobustSubspaceMonitor`, and `ConformalAlertCalibrator`.
- **Sparse precision:** `RobustGraphicalLasso` and `SGLASSO`.
- **Latent structure:** `TwoScatterICA`, `SOBI`, `RobustSOBI`, and `RobustFactorModel`.
- **Reusable geometry:** robust distances, anomaly diagnostics, whitening, `FeatureGeometry`, full-matrix kernels, SPD utilities, and optional OpenMP acceleration.

## Installation

From PyPI after a release is published:

```bash
python -m pip install -U pip
python -m pip install robustcov
```

Supported release wheels are built for CPython 3.12, 3.13, and 3.14 on Ubuntu, Windows, and macOS by GitHub Actions. The package uses a C++/pybind11 backend built with `scikit-build-core`.

Dependency lower bounds are selected per Python version so Python 3.12 users are
not forced onto the versions needed only for Python 3.14. The exact oldest tested
sets are recorded in `requirements/minimum.txt` and exercised by CI.

Plotting is optional and is not installed with the numerical core:

```bash
python -m pip install "robustcov[plot]"
```

The package can be imported without the compiled extension. NumPy-backed estimators continue to work, while native-only estimators such as `FastMCD` and `TylerShape` raise an actionable error when fitted. Check the active installation with `robustcov.native_available()`. A native-free development wheel can be built explicitly with:

```bash
python -m build --wheel -Ccmake.define.ROBUSTCOV_BUILD_NATIVE=OFF
```

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

det = rc.RobustOutlierDetector(
    estimator=rc.FastMCD(quality="balanced", random_state=42),
    contamination=0.075,
).fit(X)
print(det.labels_)
```

Calibrate a separate held-out set of anomaly scores instead of choosing an
operational alert threshold heuristically:

```python
calibrator = rc.ConformalAlertCalibrator(alpha=0.05).fit(
    -det.score_samples(X_calibration)
)
p_values = calibrator.p_values(-det.score_samples(X_new))
alerts = calibrator.predict_alerts(-det.score_samples(X_new))
```

The usual finite-sample marginal interpretation requires exchangeability between
the held-out calibration scores and future inlier scores.

For deterministic smooth high-breakdown scatter and an efficiency refinement:

```python
dets = rc.DetS(breakdown=0.50).fit(X)
detmm = rc.DetMM(breakdown=0.50, efficiency=0.95).fit(X)

print(dets.weights_)
print(detmm.covariance_)
```

These estimators require the central half-sample to be nonsingular. Use `MRCD`
when `p` is too large for that condition.

For small-sample or high-dimensional heavy-tailed data:

```python
est = rc.RegularizedCauchy(alpha=0.10).fit(X)
print(est.covariance_)

student = rc.StudentTScatter(df=3, alpha=0.05).fit(X)
print(student.radial_kurtosis_)
```

For high-dimensional data with a minority of contaminated rows:

```python
mrcd = rc.MRCD(
    contamination=0.20,
    max_condition_number=50,
    random_state=0,
).fit(X)

print(mrcd.regularization_)
print(mrcd.standardized_condition_number_)
print(mrcd.support_)
```


For non-elliptical inlier structure, fit MRCD in a kernel feature space:

```python
kmrcd = rc.KMRCD(
    kernel="rbf",
    gamma="median",
    contamination=0.15,
    random_state=0,
).fit(X)

print(kmrcd.support_)
print(kmrcd.distances_)
```

The RBF bandwidth strongly affects the geometry. The median heuristic is a useful
starting point, not an automatic guarantee of good separation.

For matrix-valued observations such as sensor-by-time windows:

```python
mmcd = rc.MMCD(
    contamination=0.20,
    random_state=0,
).fit(X_matrices)

print(mmcd.row_covariance_)
print(mmcd.column_covariance_)
print(mmcd.mahalanobis(X_matrices))
```

For tables with isolated bad cells and missing entries:

```python
cellmcd = rc.CellMCD(alpha=0.75, quantile=0.99).fit(X)

print(cellmcd.cell_outlier_mask_)
X_corrected = cellmcd.corrected_data_
```

For high-dimensional tables with bad cells, abnormal rows, and missing entries:

```python
cellrcov = rc.CellRCov(
    n_components=4,
    residual_shrinkage="auto",
).fit(X)

print(cellrcov.covariance_)
print(cellrcov.residual_shrinkage_)
print(cellrcov.cell_outlier_mask_)
```


For sparse interpretable loadings under the same contamination model:

```python
sparse_pca = rc.SparseCellPCA(
    n_components=3,
    alpha=0.05,
    sparsity_threshold=0.01,
).fit(X)

print(sparse_pca.n_nonzero_loadings_)
print(sparse_pca.loading_support_)
```

For a sparse conditional-dependence graph:

```python
graph = rc.RobustGraphicalLasso(
    alpha="ebic",
    scatter_estimator=rc.CellMCD(
        alpha=0.75,
        min_samples_per_feature=None,
    ),
).fit(X)

print(graph.partial_correlation_)
print(graph.edge_list(feature_names))
```

For a sparse graph when radial magnitudes are extremely heavy-tailed:

```python
shape_graph = rc.SGLASSO(
    alpha=0.12,
).fit(X)

print(shape_graph.partial_correlation_)
print(shape_graph.edge_list(feature_names))
```

`SGLASSO` estimates a shape precision matrix up to a common scale. It is not
cellwise robust; use a CellMCD-based `RobustGraphicalLasso` when individual
coordinates are corrupted.

For dimensionality reduction under cellwise and rowwise contamination:

```python
cellpca = rc.CellPCA(n_components=3).fit(X)

Z = cellpca.transform(X)
print(cellpca.cell_outlier_mask_)
print(cellpca.case_outlier_mask_)
X_corrected = cellpca.corrected_data_
```

For automatic exploratory selection:

```python
auto = rc.AutoRobustScatter(selection="diagnostic").fit(X)

print(auto.best_estimator_name_)
print(auto.summary())
```

## Robust PCA

`RobustPCA` computes principal components from any compatible robust scatter
estimator. The interface follows ordinary PCA, with additional distances for
diagnosing unusual observations.

```python
pca = rc.RobustPCA(
    n_components=0.95,
    estimator=rc.RegularizedCauchy(alpha=0.10),
).fit(X)

Z = pca.transform(X)
score_distance = pca.score_distances(X)
orthogonal_distance = pca.orthogonal_distances(X)

rc.plot_robust_pca_outlier_map(
    pca,
    output_path="robust_pca_outlier_map.png",
    show=False,
)
```

Score distance measures how far a row lies along the retained components.
Orthogonal distance measures the part that those components cannot reconstruct.
This implementation uses an eigendecomposition of a robust scatter matrix; it
is not the low-rank-plus-sparse method with the same common name.

For a direct low-rank fit with density-power residual weighting:

```python
dpd_pca = rc.DensityPowerRobustPCA(
    n_components=5,
    alpha=0.30,
).fit(X)

Z_dpd = dpd_pca.transform(X)
cell_weights = dpd_pca.cell_weights(X)
```

This estimator requires a fixed component count and complete finite input.

Bootstrap the fitted loadings and retained subspace with:

```python
stability = rc.SubspaceStability(
    pca=pca,
    n_resamples=200,
    resampling="stationary",
    block_length=20,
    random_state=0,
).fit(X)

print(stability.loading_interval_)
print(stability.max_principal_angle_degrees_)
```

Use ``resampling="iid"`` for independent rows, a block or stationary bootstrap
for ordered weakly dependent observations, and ``resampling="cluster"`` for
repeated measurements grouped by subject, site, or account.

## Experimental distributionally robust PCA

`DistributionallyRobustPCA` is available only from `robustcov.experimental`.
It evaluates a weighted-Wasserstein worst-case reconstruction risk over a
deterministic adaptive candidate path. Identity transport geometry is retained
as a required ordinary-PCA control; anisotropic geometry is what expresses the
assumed train-to-deployment shift.

```python
from robustcov.experimental import DistributionallyRobustPCA

dro_pca = DistributionallyRobustPCA(
    n_components=2,
    radius=2.5,
    transport_geometry="residual",
    formulation="exact",
).fit(X_train)

print(dro_pca.exact_worst_case_risk_)
print(dro_pca.selected_gamma_)
```

The current exact formulation ranks a finite deterministic path using the exact
scalar-dual ambiguity-set risk; it does not claim a global solution of the
non-convex Grassmann problem. See `docs/distributionally_robust_pca.rst` and the
held-out shift benchmark before using it in scientific comparisons.

## Rolling subspace monitoring

`RobustSubspaceMonitor` compares incoming batches with a fixed reference fit. A
separate robust model is fitted to the current rolling window, allowing the
monitor to distinguish movement of the center from changes in scale, covariance
shape, or principal directions.

```python
monitor = rc.RobustSubspaceMonitor(
    n_components=0.95,
    estimator=rc.RegularizedCauchy(alpha=0.10),
    window_size=256,
    threshold_scale=1.2,
    alarm_patience=2,
).fit(X_reference)

result = monitor.update(X_batch)
if result.ready:
    print(result.summary())
    print(result.exceeded)
```

New rows are scored against the reference before the rolling model is updated.
A persistent production problem therefore cannot redefine the baseline before
it is detected.

## Main estimators

| Estimator | Best use case | Notes |
|---|---|---|
| `FastMCD` | Separable contamination, `n >> p` | Fast robust covariance and support diagnostics |
| `DetS` | Rowwise contamination with smooth high-breakdown weighting | Deterministic Tukey-bisquare S-estimator; requires `ceil(n/2) > p` |
| `DetMM` | The same regime when higher Gaussian efficiency is desired | DetS start with fixed robust scale and a less aggressive MM refinement |
| `MRCD` | Rowwise contamination with `p` close to or greater than `n` | Regularized high-breakdown subset covariance with automatic condition control |
| `KMRCD` | Non-elliptical inlier structure or implicit kernel data | MRCD subset search in a positive-semidefinite kernel feature space |
| `MMCD` | Matrix-valued observations with contaminated rows/samples | Robust mean matrix and Kronecker row/column covariance factors |
| `RobustMultilinearPCA` | Matrix-valued low-rank data with bad cells, abnormal samples, and missing entries | Robust Tucker-2 fit with cellwise and casewise redescending weights |
| `CellMCD` | Tables with isolated corrupted or missing cells and `n > p` | Observed-likelihood covariance fit with cell-level flags and conditional predictions |
| `CellRCov` | High-dimensional tables with bad cells, abnormal rows, and missing entries | Robust low-rank covariance plus a diagonally regularized residual covariance |
| `CellPCA` | Low-rank tables with cell errors, abnormal rows, and missing entries | Cellwise and casewise redescending weights in a weighted low-rank fit |
| `SparseCellPCA` | Interpretable low-rank tables with the same contamination model | CellPCA weights plus exact-zero elastic-net loading updates |
| `RegularizedCauchy` | Very heavy tails, small samples, `p` close to `n` | Strong radial downweighting plus shrinkage |
| `StudentTScatter` | Diffuse heavy tails | Smooth heavy-tail scatter estimator |
| `RegularizedTyler` | Heavy-tailed shape estimation | Scale-free shape unless scale correction is requested |
| `AutoRobustScatter` | Exploratory estimator selection | Diagnostic or stability-based selector |
| `ClusterRobustOutlierDetector` | Multimodal data | Cluster-then-local-robust-scatter diagnostic |
| `RobustPCA` | Robust dimensionality reduction and subspace diagnostics | Eigendecomposition of a robust location and scatter estimate |
| `DensityPowerRobustPCA` | Direct robust low-rank fitting with cell residual weights | Gaussian density-power-divergence alternating regressions |
| experimental `DistributionallyRobustPCA` | Principal subspaces under stated train-to-target distribution shift | Exact weighted-Wasserstein risk over a deterministic adaptive candidate path |

`KLRegularizedTyler` and `WieselTyler` are currently documented as aliases/prototype variants around the regularized Tyler implementation. `HellingerRegularizedTyler` is experimental.

For a scenario-specific decision table, capability limits, and cross-method results, see `docs/method_comparison.rst`. The comparison separates covariance, PCA, matrix-valued, and sparse-graph tasks rather than declaring one global winner.


## Robust kernels for GP and kernel methods

A robust scatter estimate can be used as a fixed full-matrix input metric for
kernel methods. `robustcov` supplies the metric and kernel adapters; model
fitting remains in scikit-learn, GPyTorch, or another downstream library.

```python
import robustcov as rc

metric = rc.RobustInputMetric(
    estimator=rc.RegularizedCauchy(alpha=0.05, scale_correction="radial_median"),
).fit(X_train)

K = rc.robust_rbf_kernel(
    X_train,
    precision=metric.precision_,
    center=metric.location_,
    length_scale=1.0,
)
```

For scikit-learn's `GaussianProcessRegressor`, use the optional adapter:

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel
from robustcov.sklearn_kernels import RobustMahalanobisRBF

kernel = (
    ConstantKernel(1.0)
    * RobustMahalanobisRBF(precision=metric.precision_, center=metric.location_)
    + WhiteKernel(1e-2)
)

gp = GaussianProcessRegressor(kernel=kernel).fit(X_train, y_train)
```

For GPyTorch, `robustcov.gpytorch_kernels.RobustMahalanobisRBFKernel` and
`RobustMahalanobisMaternKernel` provide frozen robust metric kernels that can be
wrapped by `gpytorch.kernels.ScaleKernel`.

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

print(rc.native_available())
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

- **What RobustCov does**: package scope, boundaries, and the reusable geometry model
- **Workflows**: anomaly scoring, PCA and monitoring, feature geometry, structured data, sparse precision, and latent factors
- **Choose an estimator**: recommendations by contamination model and dimensional regime
- **Examples by task and domain**: runnable examples with source and generated figures
- **Benchmarks and validation**: task-specific comparisons, failure cases, performance, and reviewed C-MAPSS snapshots
- **Methods and API reference**: mathematical details, provenance, fitted attributes, and public interfaces

Do not commit `docs/_build/`; it is generated by Sphinx.

## Benchmarks

Run the task-specific cross-method comparison:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=2 \
python benchmarks/compare_methods.py \
  --profile quick \
  --csv results/method_comparison.csv \
  --rst results/method_comparison.rst
```

The script compares methods only where their fitted quantities and ground-truth
metrics are compatible. It now covers scatter, kernel outlier detection, robust
PCA, matrix/tensor methods, sparse precision, ICA, SOBI, and robust factor
models. Use `--profile full --families scatter --repeats 3` (and repeat for the
other families) for slower, more stable local timing runs.

Run the focused latent-structure benchmark and generate its plots:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=2 \
python benchmarks/latent_structure_benchmarks.py \
  --profile quick \
  --families ica sobi pca factor \
  --csv results/latent_structure.csv \
  --plot-dir results/latent_structure_plots
```

Audit benchmark ownership across the public estimator surface:

```bash
python benchmarks/benchmark_inventory.py --strict
```

Generate the older benchmark report:

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
results/report/latent_structure/*.png
```

The benchmark pages report both successful and weak cases. Covariance-based
methods are most appropriate when anomalies or changes are expressed through
location, scale, correlation, or a low-dimensional subspace.

## Examples

The example gallery is grouped by method family. List the available groups:

```bash
python examples/run_use_case_gallery.py --list
```

Run one family:

```bash
python examples/run_use_case_gallery.py --group ica
python examples/run_use_case_gallery.py --group pca
python examples/run_use_case_gallery.py --group robust
python examples/run_use_case_gallery.py --group monitoring
```

The new source-separation and factor-model examples are explicit scripts:

```bash
python examples/ica_two_scatter.py
python examples/sobi_source_separation.py
python examples/robust_factor_model.py
```

Run every registered gallery example with:

```bash
python examples/run_use_case_gallery.py --all
```

Refresh generated gallery assets after editing examples:

```bash
python docs/generate_gallery_assets.py
python -m sphinx -b html docs docs/_build/html
```

## External and Kaggle examples

External examples live under `examples_external/`. Raw datasets are never bundled with the package or committed to the repository. Optional loaders cache explicit downloads under `ROBUSTCOV_DATA_DIR`, `XDG_CACHE_HOME/robustcov`, or `~/.cache/robustcov`.

List supported cached datasets:

```bash
python -m robustcov.datasets list
python -m robustcov.datasets info gas_sensor_drift
python -m robustcov.datasets info cmapss
```

Run the distribution-shift examples without storing data in the repository:

```bash
python examples_external/gas_sensor_drift_dro_pca.py --download
python examples_external/cmapss_dro_pca_monitoring.py --download --subset FD002
```

Kaggle-style manual example:

```bash
python examples_external/kaggle_credit_card_fraud.py \
  --data /path/to/creditcard.csv \
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

## Methods, attribution, and citation

`robustcov` distinguishes published algorithms, literature-based adaptations,
package-specific compositions, and software utilities. Each canonical estimator
records its primary references, the package's implementation contribution, and
material differences from the cited method.

```python
import robustcov as rc

info = rc.get_method_provenance(rc.RobustSOBI)
print(info.status)
print(info.references)
print(info.robustcov_contribution)
```

The full registry is documented in
[`docs/methods_and_references.rst`](docs/methods_and_references.rst), and the
project's restrained claims policy is in
[`docs/project_contributions.rst`](docs/project_contributions.rst). No current
public estimator is claimed as an original statistical method.

When using `robustcov`, cite both:

1. the software release using [`CITATION.cff`](CITATION.cff); and
2. the primary methodological references for the estimators used.

The machine-readable method bibliography is available in
[`docs/references.bib`](docs/references.bib). A JOSS paper draft is maintained in
the `paper/` directory.

## Contributing

Contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for
development setup and checks before opening a pull request. New public
estimators must add both benchmark ownership and method-provenance metadata.
Release notes are tracked in [`CHANGELOG.md`](CHANGELOG.md).
