# Benchmarks

The benchmark suite is organized by statistical task.  Covariance/scatter
accuracy and speed use one shared method catalog, so new estimators cannot be
added to one headline benchmark while remaining absent from the others.

## Covariance/scatter coverage

Run the small-sample heavy-tail grid:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=2 \
python benchmarks/small_sample_heavy_tail.py \
  --profile quick \
  --repeat 2 \
  --csv results/small_sample.csv
```

The catalog includes FastMCD, MRCD, DetS/DetMM, Tyler variants, Student-t and
Cauchy M-scatter, the labelled Hellinger prototype, AutoRobustScatter, and
sklearn empirical/shrinkage/MCD baselines.  Structural restrictions are written
as `not_applicable`, not failures.  Use `--profile full` for the larger grid.

Aggregate the grid with eligibility-aware statistics:

```bash
python benchmarks/benchmark_summary.py \
  --input results/small_sample.csv \
  --csv results/small_sample_summary.csv
```

The summary reports eligible scenarios, successes, failures, inapplicable
scenarios, success rate, win rate, rank, error, and runtime.

## Workload-aware covariance speed

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=2 \
python benchmarks/speed_estimators.py \
  --profile quick \
  --repeat 2 \
  --csv results/speed.csv

python examples/plot_speed_comparison.py \
  --input results/speed.csv \
  --output results/speed.png
```

This times the same catalog on low-dimensional row contamination,
moderate-dimensional heavy tails, and high-dimensional `p > n` data.  It does
not mix covariance timing with ICA/PCA/factor-model timing.

## OpenMP scaling

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python benchmarks/openmp_scaling.py \
  --n 8000 --p 20 --threads 1 2 4 \
  --csv results/openmp_scaling.csv
```

The OpenMP benchmark includes every currently threaded native workload:
FastMCD, TylerShape, RegularizedTyler, vector and matrix Mahalanobis batches,
and weighted Tucker score solves.  It also checks numerical drift versus the
one-thread result.  Native but single-threaded joint diagonalization is measured
by `source_separation_gate.py`, not mislabeled as OpenMP scaling.

## One-command benchmark report

```bash
python benchmarks/make_report.py --outdir results/report
```

This produces workload-aware CSVs and plots, a Markdown report, and a standalone
HTML report.  Use `--speed-profile full`, larger OpenMP dimensions, and repeated
Monte Carlo jobs for publication-grade runs.

## Other focused benchmarks

```bash
python benchmarks/accuracy_vs_contamination.py --csv results/accuracy.csv
python benchmarks/fastmcd_quality_speed_tradeoff.py --csv results/tradeoff.csv
python benchmarks/support_diagnostics.py --csv results/support.csv
python benchmarks/hard_contamination_scenarios.py --csv results/hard_scenarios.csv
python benchmarks/auto_scatter_small_sample.py --selection stability --csv results/auto_scatter.csv
```

Hard scenarios are intended to expose limits rather than confirm only favorable
cases.

## Cross-method suitability benchmark

The cross-method benchmark compares methods only within compatible tasks. It
covers scatter estimation, nonlinear kernel outlier detection, principal
subspaces, matrix-valued covariance, and sparse precision recovery:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=2 \
python benchmarks/compare_methods.py \
  --profile quick \
  --csv results/method_comparison.csv \
  --rst results/method_comparison.rst
```

Use `--profile full --families scatter --repeats 3` and repeat for `kernel`, `pca`, `matrix`, and `graph` for slower local runs. The generated tables
are discussed in `docs/method_comparison.rst`. There is intentionally no global
winner: each scenario uses metrics appropriate to that estimator family.

## Repeated scenario summaries

`monte_carlo_methods.py` reruns the task-specific scenarios across independent
seeds and reports medians, interquartile ranges, failure rates, and runtime
tails. It intentionally keeps covariance, PCA, tensor, and graph tasks separate.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
python benchmarks/monte_carlo_methods.py \
  --profile quick \
  --families scatter pca tensor \
  --n-seeds 20 \
  --csv results/monte_carlo_summary.csv \
  --rst results/monte_carlo_summary.rst
```

## Native-kernel scaling

`native_scaling.py` compares optional C++ kernels with exact NumPy fallbacks and
records speedup together with the maximum absolute numerical difference.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
python benchmarks/native_scaling.py \
  --repeats 5 \
  --csv results/native_scaling.csv
```

## Native port acceptance gate

Use `native_port_gate.py` before replacing a NumPy implementation with C++.
A port is accepted only if it matches the reference numerically and reaches at
least a 1.5x median speedup on its representative workload:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python benchmarks/native_port_gate.py --kernel vector-mahalanobis --min-speedup 1.5
```

Run the benchmark on the same machine before and after a native change, keep the
raw JSON output with the pull request, and do not use microbenchmarks alone to
remove the Python fallback. Small workloads should stay on NumPy when extension
call overhead makes the native path slower.

## Complete-estimator profiling

Profile representative complete fits before choosing a native-port target:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/profile_estimators.py \
  --output-dir results/estimator_profiles
```

The command writes a `cProfile` file and readable cumulative/internal-time
report for FastMCD, Tyler, MRCD, Matrix MCD, and robust PCA. Prefer complete-fit
profiles over isolated arithmetic microbenchmarks when selecting work.

## Estimator optimization acceptance gate

Use `estimator_optimization_gate.py` for profiling-driven Python/NumPy changes.
It reconstructs the pre-optimization algorithm, checks fitted-result
equivalence, and requires at least a 1.5x median improvement on complete fits:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/estimator_optimization_gate.py \
  --case all \
  --min-speedup 1.5 \
  --json-output results/estimator_optimization_gate.json
```

Keep the raw JSON result with the change. If a proposed optimization does not
clear the complete-fit gate, revert it even when a smaller internal operation
looks faster.

## Advanced-estimator profiling

Profile CellMCD, cellwise PCA, multilinear PCA, sparse precision, and kernel
MRCD before changing their implementation:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/profile_advanced_estimators.py \
  --output-dir results/advanced_estimator_profiles
```

## Cellwise optimization acceptance gate

`advanced_estimator_optimization_gate.py` reconstructs the previous rowwise
CellMCD and cellwise-PCA algorithms, checks complete fitted-result equivalence,
and enforces the same 1.5x median speedup policy:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/advanced_estimator_optimization_gate.py \
  --case all \
  --min-speedup 1.5 \
  --json-output results/advanced_estimator_optimization_gate.json
```

## Workflow profiling

Profile CellRCov, sparse cellwise PCA, automatic scatter selection, and robust
subspace-monitoring fits before changing shared numerical helpers:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/profile_workflow_estimators.py \
  --output-dir results/workflow_profiles
```

## Workflow optimization acceptance gate

`workflow_optimization_gate.py` reconstructs the previous feature-by-feature
sparse coordinate descent and pseudoinverse/einsum M-scatter paths. It checks
complete fitted-result equivalence and requires at least a 1.5x median speedup:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/workflow_optimization_gate.py \
  --case all \
  --min-speedup 1.5 \
  --json-output results/workflow_optimization_gate.json
```

The M-scatter changes are accepted on complete automatic-selection and
monitoring workflows rather than on the inverse helper alone. CellRCov remains
in the profiler even when no change clears the end-to-end gate.

## Statistical validation gate

Run `statistical_validation.py` after changing estimator equations, ridge
handling, standardization, or native covariance kernels. The gate checks
measurement-unit equivariance, singular-input behavior, a contamination curve,
and agreement with scikit-learn's `MinCovDet` when scikit-learn is installed:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/statistical_validation.py \
  --json-output results/statistical_validation.json
```

Unlike the performance gates, this command has no speed threshold. Every
numerical and statistical check must pass before the change is accepted.

## Structured-estimator statistical validation

Run `structured_statistical_validation.py` after changing Matrix MCD, Kernel
MRCD, CellMCD, robust PCA, or multilinear PCA. It checks tiny-unit behavior,
structured transformation properties, PSD validation, and explicit handling of
singular inputs:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/structured_statistical_validation.py \
  --json-output results/structured_statistical_validation.json
```

Every section must pass. Coordinatewise cell-robust estimators are tested under
feature or mode permutations and scale changes, not arbitrary rotations that
change the cellwise contamination model.

## Latent-structure benchmarks: ICA, SOBI, PCA, and factors

The latent-structure suite compares estimators only within compatible tasks and
uses permutation/sign/scale-aware recovery metrics:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=2 \
python benchmarks/latent_structure_benchmarks.py \
  --profile quick \
  --families ica sobi pca factor \
  --csv results/latent_structure.csv \
  --plot-dir results/latent_structure_plots \
  --rst results/latent_structure.rst
```

The principal metrics are:

- ICA and SOBI: minimum-distance index, Amari index, and optimally matched source correlation.
- Robust PCA: projection/subspace error, row/cell AUROC, missing-value reconstruction error, and runtime.
- Robust factor models: loading-subspace error, matched factor-score correlation, common-component error, factor-count error, covariance error, and runtime.

Use `--profile full --repeats 3` for a more stable local comparison.  Timing
numbers are machine-dependent; recovery metrics are deterministic for a fixed
seed apart from small numerical differences.

The same families are available from the cross-method runner:

```bash
python benchmarks/compare_methods.py \
  --profile quick \
  --families ica sobi pca factor \
  --csv results/latent_method_comparison.csv
```

## Distributionally robust PCA under held-out shift

The experimental weighted-Wasserstein PCA benchmark is intentionally separate
from contamination-only PCA rankings. It evaluates complete fits on independent
target distributions under structured covariance shift, no shift, and row
contamination without target shift:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/distributionally_robust_pca.py \
  --profile quick \
  --repeats 3 \
  --csv results/distributionally_robust_pca.csv \
  --plot results/distributionally_robust_pca.png
```

The primary metric is held-out target reconstruction risk. The identity
transport geometry is a required ordinary-PCA control, while the anisotropic
geometries test a stated shift model. Exact worst-case risk and the squared
surrogate bound are retained in the CSV for every DRO fit.

## Benchmark coverage inventory

Every canonical public estimator must be assigned to a comparative benchmark,
a statistical/numerical validation gate, a performance gate, or a documented
end-to-end workflow.  Audit that mapping with:

```bash
python benchmarks/benchmark_inventory.py \
  --strict \
  --csv results/benchmark_inventory.csv \
  --rst results/benchmark_inventory.rst
```

Aliases are mapped to their canonical estimator rather than counted as separate
algorithms.  Add an inventory row whenever a new public estimator is added.

### Principal Component Pursuit validation

```bash
python benchmarks/principal_component_pursuit_validation.py
```

Checks low-rank recovery on a clean control and under sparse gross entrywise
corruption. The comparison is with rank-matched truncated SVD and is specific
to the PCP data model.
