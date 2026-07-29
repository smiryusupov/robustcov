Cellwise covariance estimation
==============================

Most robust covariance estimators treat an entire row as clean or contaminated.
That model is appropriate when a record comes from another population, but it is
wasteful when only one measurement in the row is wrong.  ``CellMCD`` keeps the
usable cells and estimates the covariance from the resulting incomplete data.

A typical example is a panel of asset returns with isolated bad ticks.  A single
bad quote can contaminate one asset on one day while the remaining returns from
that day still carry useful information.

Mathematical formulation
------------------------

Let :math:`X=(x_{ij})` be an :math:`n\times p` data matrix.  CellMCD introduces a
binary matrix :math:`W=(w_{ij})`, where :math:`w_{ij}=1` means that cell
:math:`x_{ij}` is retained and :math:`w_{ij}=0` means that it is treated as
missing or outlying.

For row :math:`i`, let :math:`x_i^{(w_i)}` contain the retained coordinates,
and let :math:`\mu^{(w_i)}` and :math:`\Sigma^{(w_i)}` denote the corresponding
parts of the location vector and covariance matrix.  The partial squared
Mahalanobis distance is

.. math::

   d_i^2(w_i)
   =
   \left(x_i^{(w_i)}-\mu^{(w_i)}\right)^T
   \left(\Sigma^{(w_i)}\right)^{-1}
   \left(x_i^{(w_i)}-\mu^{(w_i)}\right).

The fitted parameters and cell mask approximately minimize

.. math::

   \sum_{i=1}^{n}
   \left[
      \log\left|\Sigma^{(w_i)}\right|
      + |w_i|\log(2\pi)
      + d_i^2(w_i)
   \right]
   +
   \sum_{j=1}^{p} q_j\sum_{i=1}^{n}(1-w_{ij}),

subject to retaining at least :math:`h` cells in every column and keeping the
smallest covariance eigenvalue above a fixed lower bound.  The penalty
:math:`q_j` discourages unnecessary flags.

Concentration step
------------------

One iteration has two parts.

First, each column of :math:`W` is updated while the location and covariance are
held fixed.  For cell :math:`x_{ij}`, the model predicts its value from the
currently retained cells in the same row.  If :math:`\widehat{x}_{ij}` is that
conditional prediction and :math:`C_{ij}` its conditional variance, the cost of
retaining the cell is compared with the cost of flagging it through

.. math::

   \Delta_{ij}
   =
   \log(C_{ij}) + \log(2\pi)
   + \frac{(x_{ij}-\widehat{x}_{ij})^2}{C_{ij}}.

Cells with :math:`\Delta_{ij}\le q_j` are retained, subject to the minimum
column support :math:`h`.

Second, the new mask is treated as a missing-data pattern.  One Gaussian EM step
updates the location and covariance, after which small eigenvalues are truncated
at ``min_eigenvalue``.  The observed-likelihood objective is recorded in
``objective_history_``.

Using the estimator
-------------------

.. code-block:: python

   import robustcov as rc

   model = rc.CellMCD(
       alpha=0.75,
       quantile=0.99,
       min_eigenvalue=1e-4,
   ).fit(X)

   clean_cells = model.cell_support_
   flagged_cells = model.cell_outlier_mask_
   corrected = model.corrected_data_
   residuals = model.standardized_residuals_

``predicted_values_`` contains the conditional prediction for every training
cell.  ``corrected_data_`` replaces flagged and missing cells by those
predictions while leaving retained cells unchanged.

New rows can be inspected without refitting:

.. code-block:: python

   diagnostics = model.cellwise_diagnostics(X_new)
   mask = diagnostics["cell_outlier_mask"]
   X_corrected = model.transform(X_new)

The standardized residual map is often the quickest way to inspect the result:

.. code-block:: python

   rc.plot_cellwise_residual_map(
       model,
       column_labels=feature_names,
       output_path="cellwise_residuals.png",
       show=False,
   )

Interpretation and limits
-------------------------

``CellMCD`` is coordinate dependent.  That is intentional: a cell is tied to a
specific measured variable, so rotating the feature space changes the meaning
of cellwise contamination.

The method is intended for low- or moderate-dimensional tables.  The published
rule of thumb is about five observations per feature; the implementation checks
this by default.  Use ``MRCD`` or a regularized scatter estimator when
:math:`p` is close to or larger than :math:`n`.

The reference CellMCD software starts from DDCW.  This package uses a
deterministic median/MAD and clipped-correlation initialization.  The objective
and concentration step follow CellMCD, but numerical solutions need not match
the reference implementation exactly.

Worked example
--------------

:doc:`Cleaning isolated bad ticks in market data <gallery/cellmcd_market_data>`
compares empirical covariance, rowwise MCD, and CellMCD when individual cells
are corrupted and some quotes are missing.
