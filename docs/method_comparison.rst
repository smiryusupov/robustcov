Compare methods and interpret results
=====================================

There is no single estimator that is robust to every kind of bad data.  The
important choice is the contamination model: complete rows, isolated cells,
diffuse heavy tails, matrix-valued observations, or a low-rank representation.
A method can perform well outside its intended setting, but that should be
treated as an empirical result rather than an assumption.

A practical decision order
--------------------------

Start with four questions.

#. **What is one observation?** A vector, a matrix, or a time-ordered batch?
#. **How is the data damaged?** Complete rows, individual cells, broad heavy
   tails, or missing entries?
#. **What is the dimensional regime?** Is :math:`n` comfortably larger than
   :math:`p`, or is :math:`p` close to or greater than :math:`n`?
#. **What output is needed?** A covariance matrix, a principal subspace,
   independent sources, temporally separated sources, factors, a sparse graph,
   an anomaly score, or a change alarm?

The following table gives a starting point.  It is deliberately more specific
than a ranking table.

.. list-table::
   :header-rows: 1
   :widths: 24 21 25 30

   * - Data problem
     - Start with
     - Useful alternatives
     - Avoid or qualify
   * - A minority of complete rows are far from an elliptical bulk; :math:`n \gg p`
     - ``FastMCD``
     - ``MRCD``; ``RegularizedCauchy``
     - Do not use FastMCD when a clean, nonsingular :math:`h`-subset cannot exist.
   * - Complete-row contamination with smooth weights and deterministic fitting
     - ``DetMM`` when efficiency matters; ``DetS`` when breakdown protection is primary
     - ``FastMCD`` for explicit subset support; ``StudentTScatter`` for diffuse tails
     - Requires :math:`\lceil n/2 \rceil > p` and does not address cellwise contamination.
   * - Complete-row contamination with :math:`p` close to or greater than :math:`n`
     - ``MRCD``
     - ``RegularizedCauchy``; ``RegularizedTyler``
     - Empirical covariance and unregularized subset covariance are singular.
   * - The regular observations follow a curved or kernel-defined structure
     - ``KMRCD``
     - Linear ``MRCD`` as a baseline; compare several defensible bandwidths
     - A poorly chosen kernel can manufacture separation or hide genuine outliers.
   * - Broad heavy tails rather than a small separated outlier group
     - ``StudentTScatter``
     - ``RegularizedCauchy``; ``RegularizedTyler``
     - A hard-subset method can discard legitimate tail observations.
   * - Very heavy tails and a small or ill-conditioned sample
     - ``RegularizedCauchy``
     - ``StudentTScatter``; ``RegularizedTyler``
     - The shrinkage target matters when the data contain strong anisotropy.
   * - Individual cells are corrupted or missing and :math:`n` is comfortably larger than :math:`p`
     - ``CellMCD``
     - ``CellPCA`` when only a low-rank representation is needed
     - Rowwise methods may fail once bad cells are spread over many rows.
   * - Bad cells, abnormal rows, and missing entries with :math:`p` close to or greater than :math:`n`
     - ``CellRCov``
     - ``CellPCA`` for the subspace only; ``MRCD`` when contamination is truly rowwise
     - The covariance decomposition depends on a useful low-rank component and a chosen rank.
   * - One matrix is low rank plus sparse, arbitrarily large cell corruption
     - ``PrincipalComponentPursuit``
     - ``CellPCA`` when missing values or mixed row/cell weighting are required
     - PCP assumes a sufficiently incoherent low-rank component and dispersed sparse support; it is not a dense-noise model.
   * - Low-rank data contain bad cells, abnormal rows, and missing entries
     - ``CellPCA``
     - ``CellMCD`` followed by an eigendecomposition
     - Median-imputed ordinary PCA can rotate toward corrupted coordinates.
   * - The low-rank factors should use only a small, interpretable variable subset
     - ``SparseCellPCA``
     - Dense ``CellPCA`` as the robustness baseline
     - A larger penalty can improve sparsity while damaging subspace recovery.
   * - Each observation is naturally a matrix
     - ``MMCD``
     - Vector methods only when Kronecker structure is not scientifically meaningful
     - Flattening can discard row/column structure and create a much larger covariance.
   * - Matrix observations have low-rank row/column structure plus bad cells, abnormal samples, or missing entries
     - ``RobustMultilinearPCA``
     - ``MMCD`` for covariance factors; flattened ``CellPCA`` as a structural baseline
     - Fixed mode ranks are required and the package initialization is not reference ROMPCA parity.
   * - Robust dimensionality reduction under rowwise contamination or heavy tails
     - ``RobustPCA`` with a matching scatter estimator
     - ``CellPCA`` for cellwise errors
     - ``RobustPCA`` is scatter PCA, not low-rank-plus-sparse decomposition.
   * - Latent signals are statistically independent but observations are instantaneous mixtures
     - ``TwoScatterICA``
     - Symmetrized two-scatter ICA when source skewness is problematic; FastICA as a baseline
     - Identifiability requires non-Gaussian sources and sufficiently distinct scatter signatures.
   * - Latent time series have distinct autocorrelation signatures
     - ``SOBI`` on clean data; ``RobustSOBI`` under impulsive contamination
     - FastICA as a non-temporal baseline
     - Similar source autocorrelations or poorly chosen lags can make separation ill-conditioned.
   * - A static low-rank factor model is required under heavy tails or row contamination
     - ``RobustFactorModel(method='kendall')``
     - Huber refinement when cellwise residual loss is appropriate; ``n_factors='auto'`` for exploratory selection
     - Factor-number selection needs separated eigenvalue ratios and is not guaranteed in weak-factor regimes.
   * - A sparse conditional-dependence graph is required
     - ``RobustGraphicalLasso``
     - Choose ``CellMCD`` or a heavy-tail scatter estimator underneath it
     - Graph edges are conditional associations, not causal links.
   * - A sparse graph is required for high-dimensional elliptical data with unreliable radial magnitudes
     - ``SGLASSO``
     - ``RobustGraphicalLasso`` with ``RegularizedCauchy``
     - Spatial signs do not identify absolute covariance scale and are not cellwise robust.
   * - Production batches must be compared with a fixed reference
     - ``RobustSubspaceMonitor``
     - ``FeatureGeometry`` for direct distance or kernel monitoring
     - Do not update the reference before scoring the incoming batch.
   * - Sampling uncertainty of a principal subspace is required
     - ``SubspaceStability``
     - Stationary/block bootstrap for time series; cluster bootstrap for grouped data
     - IID bootstrap understates uncertainty when observations are dependent.

Capability matrix
-----------------

``Yes`` means that the capability is part of the estimator's model. ``Limited``
means that it is available through imputation, regularization, or composition
with another estimator rather than handled directly.

.. list-table::
   :header-rows: 1
   :widths: 20 18 12 12 10 10 18

   * - Method
     - Primary output
     - Rowwise
     - Cellwise
     - Missing
     - :math:`p \ge n`
     - Structured input
   * - ``FastMCD``
     - covariance and support
     - Yes
     - No
     - Limited
     - No
     - vectors
   * - ``DetS`` / ``DetMM``
     - smooth high-breakdown covariance and radial weights
     - Yes
     - No
     - Limited
     - No
     - vectors
   * - ``MRCD``
     - regularized covariance and support
     - Yes
     - No
     - Limited
     - Yes
     - vectors
   * - ``KMRCD``
     - kernel-space support and anomaly scores
     - Yes
     - No
     - Limited
     - Yes, through the kernel matrix
     - vectors or PSD kernels
   * - ``StudentTScatter``
     - covariance/scatter
     - Smooth downweighting
     - No
     - Limited
     - With shrinkage
     - vectors
   * - ``RegularizedCauchy``
     - covariance/scatter
     - Smooth downweighting
     - No
     - Limited
     - Yes
     - vectors
   * - ``RegularizedTyler``
     - shape or scaled scatter
     - Smooth downweighting
     - No
     - Limited
     - Yes
     - vectors
   * - ``CellMCD``
     - covariance and cell mask
     - Yes
     - Yes
     - Yes
     - No
     - tabular vectors
   * - ``CellRCov``
     - regularized full covariance and cell/case diagnostics
     - Yes
     - Yes
     - Yes
     - Yes
     - tabular vectors
   * - ``MMCD``
     - row/column covariance factors
     - Yes
     - No
     - Limited
     - Structured
     - matrices
   * - ``RobustMultilinearPCA``
     - row/column low-rank subspaces and cell/case weights
     - Yes
     - Yes
     - Yes
     - Structured
     - matrices
   * - ``PrincipalComponentPursuit``
     - explicit low-rank and sparse matrices
     - No
     - Yes, sparse gross cells
     - No
     - Yes
     - complete matrices
   * - ``RobustPCA``
     - principal subspace
     - Depends on scatter
     - No
     - No
     - Depends on scatter
     - vectors
   * - experimental ``DistributionallyRobustPCA``
     - shift-protected principal subspace
     - Not its primary model
     - No
     - No
     - Yes, through regularized geometry
     - vectors plus an SPD transport geometry
   * - ``DensityPowerRobustPCA``
     - direct low-rank subspace and residual weights
     - DPD residual downweighting
     - Yes, through cell residuals
     - No
     - Yes
     - complete tables
   * - ``CellPCA``
     - low-rank subspace and cell/case weights
     - Yes
     - Yes
     - Yes
     - Yes
     - tables
   * - ``SparseCellPCA``
     - sparse low-rank loadings and cell/case weights
     - Yes
     - Yes
     - Yes
     - Yes
     - tables
   * - ``RobustGraphicalLasso``
     - sparse precision graph
     - Depends on scatter
     - Depends on scatter
     - Depends on scatter
     - Yes
     - vectors
   * - ``SGLASSO``
     - sparse shape-precision graph
     - Radial heavy-tail robustness
     - No
     - Limited through median imputation
     - Yes
     - complete elliptical vectors
   * - ``TwoScatterICA``
     - independent components and mixing/unmixing matrices
     - Robust whitening and bounded radial scatter
     - No
     - No
     - No
     - Full-rank vector mixtures
   * - ``SOBI`` / ``RobustSOBI``
     - temporally separated sources
     - RobustSOBI handles impulsive rows
     - No
     - No
     - No
     - Ordered multichannel time series
   * - ``RobustFactorModel``
     - loadings, factors, common/idiosyncratic components
     - Kendall or Huber robustness
     - Huber residual downweighting only
     - No
     - Limited
     - static vector panels

Methods that are not direct competitors
---------------------------------------

Some public classes consume an estimator rather than replace it:

* ``FeatureGeometry`` turns a fitted scatter estimate into distances, whitening,
  and kernels.
* ``RobustSubspaceMonitor`` compares a rolling window with a frozen reference.
* ``SubspaceStability`` resamples and refits a PCA estimator to measure sampling
  variability.
* ``AutoRobustScatter`` selects among candidate scatter estimators; its quality
  depends on the candidate set and the selection score.

These classes are covered by functional tests and application examples, but it
would be misleading to put them in the same numerical ranking as covariance
estimators.

Benchmark design
----------------

The comparison script uses synthetic data because the true covariance,
principal subspace, source mixing matrix, factor loading space,
matrix-normal covariance, and graph are then known.  It contains nine
benchmark families:

``scatter``
   Rowwise outliers, diffuse heavy tails, :math:`p>n`, low-dimensional cellwise
   corruption, and a high-dimensional low-rank mixture of bad cells, bad rows,
   and missing entries.  The main metric is relative Frobenius covariance
   error.  AUROC is shown only when an injected outlier label is meaningful.


``kernel outlier detection``
   A noisy curved manifold with off-manifold row outliers.  Linear MRCD and
   linear KMRCD are compared with RBF KMRCD using row-outlier AUROC and runtime.
   No covariance error is reported because a nonlinear kernel does not define
   an ordinary input-space covariance estimate.

``pca``
   Rowwise low-rank outliers, dense cellwise low-rank corruption, and a sparse
   loading scenario with missing entries.  The metrics are projection-subspace
   error, outlier AUROC, hidden-cell reconstruction error, and—where the true
   loading support is defined—support F1 and fitted sparsity.

``matrix covariance``
   Matrix-normal observations with localized faulty windows.  MMCD is compared
   with an all-sample matrix-normal maximum-likelihood fit using Kronecker
   covariance error and matrix-distance AUROC.

``multilinear pca``
   Matrix-valued observations follow a known low-rank row/column structure with
   contaminated cells and abnormal samples.  Robust multilinear PCA is compared
   with a non-robust multilinear baseline using row/column subspace error,
   clean-cell reconstruction error, and anomaly AUROC.

``sparse precision``
   Two known sparse graphs are used.  One contains radial heavy tails under an
   elliptical model and includes ``SGLASSO``.  The other contains bad cells and
   missing values and compares empirical, Cauchy, and CellMCD scatter inputs.
   Fixed penalties isolate the covariance/shape input; EBIC path selection is
   evaluated separately in the gallery examples.

``ica``
   Independent non-Gaussian sources are mixed through a known full-rank matrix.
   Clean and impulsively contaminated fits are evaluated with the
   minimum-distance index, Amari index, matched source correlation, and runtime.

``sobi``
   Autoregressive sources with distinct temporal signatures are mixed through a
   known matrix.  Classical and robust SOBI are compared on clean and impulsive
   sequences with permutation/scale-aware recovery metrics.

``factor model``
   Heavy-tailed factors generate a known loading subspace and common component.
   Classical PCA factors are compared with Kendall, Huber-refined, and automatic
   robust factor models using subspace, factor-score, reconstruction, covariance,
   factor-count, and runtime metrics.

The quick profile is small enough for documentation and continuous integration.
The full profile increases sample sizes, dimensions, and subset starts.  Run
separate families when collecting repeated timing measurements.  Neither profile is a substitute for a benchmark designed around a
specific application.

.. include:: _generated/method_comparison_results.rst

How to read this snapshot
-------------------------

The committed quick run supports several practical conclusions, but not a
universal ordering.

* In the nonlinear-manifold scenario, RBF KMRCD separates off-manifold points
  that lie inside the broad linear covariance envelope.  The result depends on
  the selected RBF bandwidth, so linear MRCD remains an important baseline.
* Under separated rowwise contamination, FastMCD gives the smallest covariance
  error in this run.  DetS and DetMM provide deterministic smooth weighting;
  DetMM generally moves toward the clean-sample efficiency of classical
  covariance without replacing the high-breakdown S-scale.  MRCD is somewhat
  more conservative but remains usable in regimes where a raw subset covariance
  would be singular.
* Under diffuse Student-t tails, StudentTScatter and RegularizedCauchy recover
  covariance more accurately than empirical covariance or a hard subset.
* In the :math:`p>n` scenario, the regularized M-estimators give the lowest
  covariance error, while MRCD gives the clearest separation of the injected
  rows.  The preferred method therefore depends on whether covariance recovery
  or high-breakdown support selection is the primary goal.
* When bad cells are spread across more than half of the rows, CellMCD recovers
  covariance far more accurately than methods that treat complete rows as the
  unit of contamination.
* In the high-dimensional mixed-contamination scenario, CellRCov is the only
  method in the table designed to use cellwise diagnostics and a regularized
  full covariance at the same time.  Its advantage depends on the supplied
  low-rank structure and should not be generalized to arbitrary dense covariance.
* For low-rank data with missing and corrupted cells, CellPCA combines accurate
  subspace recovery with substantially better reconstruction of the hidden
  entries.  CellMCD scatter PCA recovers the subspace well but is slower and is
  not itself a low-rank missing-data model.
* In the sparse-loading scenario, SparseCellPCA preserves the CellPCA subspace
  and hidden-cell reconstruction while recovering the simulated loading support.
  Dense PCA methods necessarily score poorly on exact support because they do
  not set coefficients to zero.
* MMCD improves Kronecker covariance recovery in the matrix-valued example.
* The ICA and SOBI rows must be interpreted with permutation/sign/scale-aware
  metrics.  In the committed source-separation scenarios, robust whitening and
  lag weighting protect recovery against impulsive rows, while the clean-data
  baseline can remain faster.
* The factor benchmark separates loading-subspace recovery from implied
  covariance recovery.  A method can recover factors well without matching the
  entire covariance if the idiosyncratic variance model is misspecified.
* In the radial heavy-tail graph, spatial-sign and Cauchy-scatter graphical
  lasso substantially reduce partial-correlation error relative to empirical
  covariance.  Spatial signs target shape rather than absolute covariance scale.
* In the bad-cell graph, CellMCD has higher recall but produces a denser graph.
  Spatial-sign graphical lasso is omitted because a single damaged coordinate
  can rotate the entire sign vector.  These are tuning and contamination-model
  tradeoffs, not a universal graph ranking.

Reproducing the tables
----------------------

From the repository root:

.. code-block:: bash

   OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=2 \
   python benchmarks/compare_methods.py \
       --profile quick \
       --csv docs/_static/benchmarks/method_comparison_quick.csv \
       --rst docs/_generated/method_comparison_results.rst

For a more stable local timing comparison:

.. code-block:: bash

   OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4 \
   python benchmarks/compare_methods.py \
       --profile full \
       --families scatter \
       --repeats 3 \
       --csv results/scatter_method_comparison.csv

Repeat the command with ``kernel``, ``pca``, ``matrix``, ``graph``, ``ica``,
``sobi``, or ``factor``.  Running the families
separately gives progress at natural checkpoints and avoids losing an entire
long run if one method is interrupted.

Use an otherwise idle machine and keep BLAS thread counts fixed.  The reported
runtime includes estimator fitting but not Python startup.  The optional
``--measure-python-memory`` flag records ``tracemalloc`` peak memory, which
excludes native C++ and BLAS allocations and slows Python-heavy methods.

Limits of the comparison
------------------------

* The committed table is one deterministic synthetic snapshot, not a claim of
  state-of-the-art performance on every dataset.
* Hyperparameters are intentionally modest so the quick profile remains useful
  during development.  A production analysis should tune support fractions,
  shrinkage, component count, and graph penalties.  The CellRCov benchmark is
  given the simulated low-rank dimension rather than estimating it.
* Runtime depends on compiler flags, BLAS, CPU, and thread settings.
* Shape estimators require a scale convention before covariance error is
  meaningful.
* A high anomaly AUROC does not imply that the covariance estimate is accurate;
  the high-dimensional MRCD example illustrates this distinction.
