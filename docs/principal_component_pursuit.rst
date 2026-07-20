Principal Component Pursuit
===========================

``PrincipalComponentPursuit`` implements the canonical low-rank-plus-sparse
matrix decomposition that is often called *robust PCA* in the optimization and
computer-vision literature. It solves

.. math::

   \min_{L,S}\; \lVert L\rVert_* + \lambda\lVert S\rVert_1
   \quad\text{subject to}\quad X=L+S,

where ``low_rank_`` stores :math:`L` and ``sparse_`` stores :math:`S`.
The nuclear norm promotes low rank and the entrywise L1 norm promotes sparse
gross corruption.

This is a different task from :class:`robustcov.RobustPCA`:

.. list-table::
   :header-rows: 1
   :widths: 24 36 40

   * - Method
     - Fitted object
     - Appropriate contamination model
   * - ``PrincipalComponentPursuit``
     - A decomposition of one matrix into ``low_rank_`` and ``sparse_``
     - Arbitrarily large errors in a sufficiently sparse, dispersed set of cells
   * - ``RobustPCA``
     - A robust location, scatter, and principal subspace with row diagnostics
     - Heavy tails or rowwise contamination handled through the chosen scatter estimator
   * - ``CellPCA``
     - A weighted low-rank model with cellwise/casewise weights and missing values
     - Mixed bad cells, abnormal rows, and missing entries

Quick example
-------------

.. code-block:: python

   import robustcov as rc

   pcp = rc.PrincipalComponentPursuit(tol=1e-7).fit(X)

   low_rank = pcp.low_rank_
   sparse = pcp.sparse_
   flagged_cells = pcp.sparse_support_

   print(pcp.decomposition_summary())
   print(pcp.history_records())

The alias ``rc.PCP`` is available for concise interactive use.

Algorithm
---------

The implementation uses the inexact augmented Lagrange multiplier method of
Lin, Chen, and Ma. Each iteration alternates:

#. singular-value thresholding for the low-rank component;
#. elementwise soft thresholding for the sparse component;
#. a dual update and increasing augmented-Lagrangian penalty.

The default

.. math::

   \lambda = \frac{1}{\sqrt{\max(n,p)}}

is the canonical Principal Component Pursuit choice. Convergence is measured by

.. math::

   \frac{\lVert X-L-S\rVert_F}{\lVert X\rVert_F}.

Fitted diagnostics
------------------

The estimator exposes:

``low_rank_``
   Recovered low-rank matrix.
``sparse_``
   Recovered entrywise-sparse corruption matrix.
``residual_``
   Numerical equality-constraint residual.
``rank_`` and ``singular_values_``
   Numerical rank and spectrum of the recovered low-rank component.
``sparse_support_`` and ``cell_outlier_scores_``
   Cell-level support and absolute sparse magnitudes.
``row_outlier_scores_`` and ``column_outlier_scores_``
   Aggregated sparse energy by row or column.
``history_records()``
   Rank, sparse fraction, objective, penalty, and reconstruction residual by iteration.

``transform`` projects new rows onto the fitted low-rank row space. It does
*not* estimate a new sparse component. There is no canonical out-of-sample PCP
rule without adding a new model or optimization problem.

Assumptions and limits
----------------------

The exact-recovery theory requires conditions such as incoherence of the
low-rank singular vectors, sufficiently low rank, and sufficiently sparse and
dispersed corruption. The implementation does not test those assumptions.

Use another method when:

* small dense noise is scientifically important: canonical equality-constrained
  PCP may assign some of it to ``sparse_``; stable PCP is not implemented;
* entire rows or columns are corrupted: use a row-robust scatter/PCA method or a
  column-sparse method such as Outlier Pursuit, which is not implemented here;
* entries are missing: this implementation is not robust matrix completion;
* observations arrive continuously: use :doc:`online_subspace_tracking` for an
  adaptive subspace workflow rather than rerunning PCP silently.

Validation
----------

Run the deterministic low-rank recovery validation with:

.. code-block:: bash

   python benchmarks/principal_component_pursuit_validation.py

The benchmark contains an exact low-rank control and an entrywise-sparse gross
corruption scenario. It supports a narrow recovery claim under that controlled
model, not universal superiority over robust scatter PCA.

References
----------

* Candès, Li, Ma, and Wright (2011), *Robust Principal Component Analysis?*
* Lin, Chen, and Ma (2010), *The Augmented Lagrange Multiplier Method for Exact Recovery of Corrupted Low-Rank Matrices*.
