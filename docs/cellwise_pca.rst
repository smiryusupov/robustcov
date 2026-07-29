Cellwise and casewise robust PCA
================================

``CellPCA`` fits a low-rank subspace when a data table may contain three
problems at once: unusual complete rows, isolated bad cells, and missing
entries.  A row with one faulty measurement can still contribute through its
other observed cells, while a row that departs from the fitted subspace as a
whole can be downweighted separately.

This is a different contamination model from :doc:`robust_pca`.  ``RobustPCA``
eigendecomposes a robust scatter estimate and is primarily rowwise robust.
``CellPCA`` fits the low-rank approximation directly with cell and row weights.

Model and residuals
-------------------

Let :math:`X=(x_{ij})` be an :math:`n\times p` data matrix.  For a chosen rank
:math:`q`, the fitted values are

.. math::

   \widehat X = \mathbf{1}\mu^T + T V^T,
   \qquad V^T V = I_q,

where :math:`\mu` is the center, :math:`T` contains the scores, and the columns
of :math:`V` span the principal subspace.  The residual in an observed cell is

.. math::

   r_{ij}=x_{ij}-\widehat x_{ij}.

Each variable has a fixed robust residual scale :math:`s_j`.  The standardized
cell residual is :math:`u_{ij}=r_{ij}/s_j`.

Cell and row weights
--------------------

The implementation uses a bounded, redescending wrapping loss.  Its cellwise
IRLS weight is

.. math::

   w^{\mathrm{cell}}_{ij}
   =
   \frac{\psi_c(u_{ij})}{u_{ij}},

with the value one at zero.  Central residuals keep weight one, residuals in a
transition region are downweighted, and sufficiently large residuals receive
zero weight.

To prevent a complete abnormal row from defining the subspace, the cell losses
are also summarized into a casewise total deviation

.. math::

   t_i
   =
   \left[
      \frac{1}{|O_i|}
      \sum_{j\in O_i}
      2s_j^2\rho_c(u_{ij})
   \right]^{1/2},

where :math:`O_i` is the set of observed cells in row :math:`i`.  A second
redescending weight is then computed:

.. math::

   w^{\mathrm{case}}_i
   =
   \frac{\psi_r(t_i/s_r)}{t_i/s_r}.

The weight used by the least-squares update is

.. math::

   w_{ij}
   =
   \delta_{ij}
   w^{\mathrm{cell}}_{ij}
   w^{\mathrm{case}}_i,

where :math:`\delta_{ij}=0` for a missing cell and one otherwise.  The center,
scores, and loadings are updated by alternating weighted least squares until
the fitted matrix stabilizes.

Using the estimator
-------------------

.. code-block:: python

   import robustcov as rc

   pca = rc.CellPCA(
       n_components=3,
       max_iter=100,
       tol=1e-5,
   ).fit(X)

   scores = pca.transform(X)
   fitted = pca.fitted_values_
   corrected = pca.corrected_data_

The main diagnostics are kept separately:

.. code-block:: python

   pca.cell_weights_
   pca.case_weights_
   pca.standardized_residuals_
   pca.cell_outlier_mask_
   pca.case_outlier_mask_
   pca.case_deviations_

``imputed_data_`` replaces missing cells only.  ``corrected_data_`` also
replaces cells whose final weight falls below ``weight_threshold``.  In both
cases the replacements are predictions from the fitted subspace.

New rows may themselves be incomplete or contain bad cells:

.. code-block:: python

   Z_new = pca.transform(X_new)
   diagnostics = pca.cellwise_diagnostics(X_new)
   X_new_corrected = pca.correct(X_new)

Outlier displays
----------------

A residual cellmap shows which variables drive the unusual rows:

.. code-block:: python

   rc.plot_cellwise_residual_map(
       pca,
       column_labels=feature_names,
       output_path="cellpca_residuals.png",
       show=False,
   )

The CellPCA outlier map places the casewise total deviation on one axis and the
largest absolute cell residual on the other:

.. code-block:: python

   rc.plot_cellpca_outlier_map(
       pca,
       output_path="cellpca_outlier_map.png",
       show=False,
   )

A row high on the vertical axis may be driven by one or two bad measurements.
A row far to the right has a broader departure from the fitted subspace.

Implementation scope
--------------------

The published cellPCA method combines cellwise and casewise losses in one
objective and minimizes it with IRLS.  It starts from MacroPCA and uses fixed
M-scales.  This package follows the same two-level weighting model and weighted
low-rank updates, but starts from robust marginal clipping followed by SVD and
uses fixed MAD-type residual scales.  Numerical equality with the reference
implementation is therefore not claimed.

Cellwise robustness is coordinate dependent.  Rotating the variables changes
what constitutes a cell, so orthogonal equivariance is neither expected nor
desirable for this contamination model.

Worked example
--------------

:doc:`Process spectra with bad wavelengths and abnormal batches
<gallery/cellpca_process_spectra>` compares CellPCA with classical PCA when
cellwise errors, rowwise deviations, and missing measurements occur together.

When the loading vectors should select a small set of variables, see
:doc:`sparse_cellwise_pca`.
