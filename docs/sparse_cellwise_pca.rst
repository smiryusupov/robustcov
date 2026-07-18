Sparse cellwise robust PCA
==========================

``SparseCellPCA`` is intended for a low-rank table in which only a small subset
of variables should define each component.  It keeps the cellwise and casewise
weights used by :doc:`cellwise_pca`, then adds an elastic-net penalty to the
loading matrix.  The result is a component model that can tolerate isolated
bad cells and missing entries while setting many loadings exactly to zero.

This is useful when a dense robust subspace is statistically adequate but hard
to interpret.  Typical examples include spectra, gene panels, sensor arrays,
and engineered feature sets where each latent mechanism is expected to involve
only a limited group of variables.

Penalized reconstruction model
------------------------------

Let :math:`X=(x_{ij})` be an :math:`n\times p` data matrix, with missing cells
omitted from the loss.  For rank :math:`q`, write

.. math::

   \widehat x_{ij}
   = \mu_j + t_i^T b_j,

where :math:`t_i\in\mathbb{R}^q` is the score vector for row :math:`i` and
:math:`b_j\in\mathbb{R}^q` is the loading row for variable :math:`j`.

The package-specific objective is

.. math::

   \sum_{(i,j)\in\mathcal O}
   \rho\!\left(
      \frac{x_{ij}-\mu_j-t_i^T b_j}{s_j}
   \right)
   +
   \sum_{k=1}^{q}\alpha_k
   \left[
      \eta\lVert b_{\cdot k}\rVert_1
      + \frac{1-\eta}{2}\lVert b_{\cdot k}\rVert_2^2
   \right].

Here :math:`\mathcal O` is the set of observed cells, :math:`s_j` is a robust
residual scale, :math:`\rho` is the redescending cell loss, :math:`\alpha_k`
controls the penalty on component :math:`k`, and :math:`\eta` is
``l1_ratio``.

The fit alternates between:

#. robust cell and row weights;
#. weighted score and center updates;
#. weighted elastic-net regressions for the loading rows.

Coordinate descent produces exact zeros.  A final ``sparsity_threshold`` can
remove coefficients that are numerically small but not exactly zero.

Using the estimator
-------------------

.. code-block:: python

   import robustcov as rc

   pca = rc.SparseCellPCA(
       n_components=3,
       alpha=0.05,
       l1_ratio=1.0,
       sparsity_threshold=0.01,
   ).fit(X)

   scores = pca.transform(X)
   reconstructed = pca.fitted_values_

The main sparsity diagnostics are:

.. code-block:: python

   pca.loading_support_
   pca.n_nonzero_loadings_
   pca.component_sparsity_
   pca.sparsity_
   pca.feature_importances_

The usual CellPCA diagnostics remain available:

.. code-block:: python

   pca.cell_outlier_mask_
   pca.case_outlier_mask_
   pca.standardized_residuals_
   pca.corrected_data_

Plot the loading matrix with:

.. code-block:: python

   rc.plot_sparse_cellpca_loadings(
       pca,
       feature_names=feature_names,
       output_path="sparse_loadings.png",
       show=False,
   )

Choosing the penalty
--------------------

``alpha`` may be a scalar or one value per component.  Larger values produce
fewer selected variables but can also distort the subspace.  ``l1_ratio=1`` is
a pure lasso penalty; smaller values retain more correlated variables through
the ridge part of the elastic net.

There is no automatic penalty selector in this first implementation.  Choose
``alpha`` using a held-out reconstruction task, stability across resamples, or
a scientifically defensible target range for the number of selected variables.
The example reports both subspace error and support recovery because sparsity
alone is not evidence of a good component model.

Geometry of the sparse components
---------------------------------

Unlike ordinary PCA and ``CellPCA``, the sparse loading vectors are not forced
to remain mutually orthogonal after penalization.  ``component_gram_`` records

.. math::

   B^T B,

and should be inspected when strongly overlapping components would be hard to
interpret.  Scores are calculated by weighted least squares rather than by a
simple orthogonal projection.

Relationship to SCRAMBLE
------------------------

SCRAMBLE introduced a cellwise robust sparse PCA objective with an elastic-net
penalty and optimized a smooth approximation on the Stiefel manifold using
Riemannian stochastic gradient descent.  ``SparseCellPCA`` shares the modeling
idea but uses the package's CellPCA weights and alternating weighted
coordinate-descent updates.  It should therefore be treated as a separate
experimental estimator, not as a numerical reproduction of SCRAMBLE.

Worked example
--------------

:doc:`Sparse CellPCA for interpretable spectra <gallery/sparse_cellpca_spectra>`
compares dense CellPCA and SparseCellPCA when the true loading vectors are
localized wavelength bands.
