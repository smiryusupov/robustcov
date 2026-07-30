Choose an estimator
===================

Start from the failure mode
---------------------------

Choose the estimator from the structure of one observation, the way the data
are contaminated, the ``n``-to-``p`` regime, and the fitted quantity you need.
The short table below is a starting point.  See :doc:`method_comparison` for
capability limits, methods that should not be compared directly, and
reproducible cross-method benchmarks.

.. list-table::
   :header-rows: 1
   :widths: 24 20 28 28

   * - Situation
     - Recommended estimator
     - Why
     - Main limitation
   * - ``n`` much larger than ``p`` and outliers are separable
     - ``FastMCD``
     - High-breakdown covariance with explicit support diagnostics.
     - Not suitable when a clean nonsingular subset cannot exist.
   * - Rowwise contamination with smooth weights and deterministic fitting
     - ``DetS`` or ``DetMM``
     - DetS emphasizes breakdown; DetMM keeps the robust scale and improves Gaussian efficiency.
     - Requires :math:`\lceil n/2 \rceil > p` and is not a cellwise or high-dimensional method.
   * - Rowwise contamination with ``p`` close to or larger than ``n``
     - ``MRCD``
     - High-breakdown subset estimation with target regularization.
     - The target and condition-number bound influence covariance recovery.
   * - The regular observations follow a curved or otherwise non-elliptical structure
     - ``KMRCD``
     - Runs the regularized subset search in a kernel feature space.
     - Kernel and bandwidth choices define the geometry and can dominate the result.
   * - Each observation is a matrix and contamination affects complete observations
     - ``MMCD``
     - Estimates separate row and column covariance factors.
     - Assumes a scientifically meaningful separable covariance structure.
   * - Individual cells are corrupted or missing but the rest of each row is useful
     - ``CellMCD``
     - Conditional prediction and cell-level flagging preserve clean cells.
     - Not intended for unrestricted ``p >= n`` covariance estimation.
   * - Bad cells, abnormal rows, and missing entries occur with ``p`` close to or above ``n``
     - ``CellRCov``
     - Combines a cellwise-robust low-rank covariance with a regularized residual covariance.
     - Requires a defensible rank and benefits from genuine low-dimensional structure.
   * - Matrix-valued low-rank data contain bad cells, abnormal samples, and missing entries
     - ``RobustMultilinearPCA``
     - Preserves row and column modes while applying cellwise and casewise robust weights.
     - Requires fixed mode ranks; the package initialization is not reference ROMPCA parity.
   * - One observed matrix is low rank plus sparse, arbitrarily large cell corruption
     - ``PrincipalComponentPursuit``
     - Recovers explicit low-rank and sparse matrices through the canonical convex PCP program.
     - Requires incoherence/sparsity assumptions; no missing values, dense-noise model, or out-of-sample sparse decomposition.
   * - Complete low-rank data contain large rowwise or cellwise reconstruction errors
     - ``DensityPowerRobustPCA``
     - Fits scores and loadings directly with a tunable density-power loss.
     - Requires a fixed rank and alpha; missing values need separate handling.
   * - Low-rank data contain bad cells, abnormal rows, and missing entries
     - ``CellPCA``
     - Fits the low-rank model with separate cellwise and casewise weights.
     - Requires a defensible component count and low-rank structure.
   * - The same low-rank setting, but component interpretation requires a small variable set
     - ``SparseCellPCA``
     - Adds exact-zero elastic-net loadings to the cellwise robust fit.
     - Requires a penalty choice and sparse components are not generally orthogonal.
   * - Conditional-dependence graph with heavy tails, outliers, or bad cells
     - ``RobustGraphicalLasso``
     - Sparse inverse covariance from a selectable robust scatter estimate.
     - Edge recovery is sensitive to the penalty and the scatter estimator.
   * - Sparse graph under high-dimensional elliptical data with unreliable radial magnitudes
     - ``SGLASSO``
     - Spatial signs remove observation-specific radius before graph estimation.
     - Estimates shape only and is not robust to isolated bad cells.
   * - Small sample, very heavy tails, or ``p`` close to or larger than ``n``
     - ``RegularizedCauchy``
     - Strong radial downweighting with shrinkage.
     - Does not identify a high-breakdown clean subset.
   * - Diffuse heavy tails rather than point anomalies
     - ``StudentTScatter``
     - Smooth heavy-tail weighting retains legitimate tail observations.
     - The fixed degrees of freedom encode a tail assumption.
   * - Shape estimation for elliptical data
     - ``RegularizedTyler``
     - Scale-free shape estimate with high-dimensional regularization.
     - Absolute covariance scale needs an explicit correction.
   * - Unsure which heavy-tail estimator to choose
     - ``AutoRobustScatter``
     - Fits candidates and selects with a diagnostic or stability score.
     - Selection is only as good as the candidate set and score.


Estimator status
----------------

The authoritative symbol-by-symbol classification is maintained in
``robustcov/_public_api.json`` and explained in :doc:`api_stability`.  The
manifest is checked against the installed top-level and experimental namespaces
for every release.

In broad terms, the mature covariance core is stable, newer structured and
workflow APIs are provisional, and research interfaces with changing algorithms
or defaults are experimental.  This guide intentionally does not duplicate the
full lists, because a second hand-maintained status table can drift from the
actual compatibility contract.
