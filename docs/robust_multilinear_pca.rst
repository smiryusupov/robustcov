Robust multilinear PCA
======================

``RobustMultilinearPCA`` reduces matrix-valued observations without flattening
them.  Each sample keeps its row and column modes, which is useful for data such
as sensor-by-time windows, image patches, asset-by-horizon panels, and
excitation-by-emission spectra.

The estimator is intended for data containing a mixture of abnormal complete
matrices, isolated bad cells, and missing entries.

Model
-----

For matrices :math:`X_i\in\mathbb{R}^{r\times c}`, the fitted Tucker-2 model is

.. math::

   X_i \approx M + U G_i V^\mathsf{T},

where :math:`M` is a center matrix, :math:`U\in\mathbb{R}^{r\times q_r}` and
:math:`V\in\mathbb{R}^{c\times q_c}` have orthonormal columns, and
:math:`G_i\in\mathbb{R}^{q_r\times q_c}` is the core score matrix for sample
:math:`i`.

The fit combines a weight for each observed cell with a second weight for the
complete matrix:

.. math::

   w_{iab}=\delta_{iab}\,w^{\mathrm{cell}}_{iab}\,w^{\mathrm{case}}_i.

Missing cells have :math:`\delta_{iab}=0`. Large standardized reconstruction
residuals receive small cell weights, while matrices with broad reconstruction
departures receive an additional casewise downweighting.

Usage
-----

.. code-block:: python

   import robustcov as rc

   model = rc.RobustMultilinearPCA(
       ranks=(2, 3),
       backend="auto",
   ).fit(X)

   core_scores = model.transform(X_new)
   reconstructed = model.reconstruct(X_new)
   corrected = model.correct(X_new)

``transform`` returns an array with shape
``(n_samples, row_rank, column_rank)``.  The fitted mode loadings are available
as ``row_components_`` and ``column_components_``.

Diagnostics
-----------

The fitted object reports:

* ``cell_weights_`` and ``cell_outlier_mask_``;
* ``case_weights_`` and ``case_outlier_mask_``;
* ``standardized_residuals_``;
* ``imputed_data_`` and ``corrected_data_``;
* a two-axis outlier map combining case deviation and the largest cell residual.

.. code-block:: python

   rc.plot_multilinear_residual_map(model, index=7, show=False)
   rc.plot_multilinear_outlier_map(model, show=False)

Native backend
--------------

The repeated weighted core-score solve can use the package's C++ extension:

.. code-block:: python

   model = rc.RobustMultilinearPCA(ranks=(2, 3), backend="cpp")

``backend="auto"`` selects C++ when the native extension is available and uses
the NumPy implementation otherwise.  Backend-equivalence tests compare fitted
values and weights to floating-point precision.

Scope
-----

The class follows the casewise/cellwise robust multilinear PCA structure of
Hirari, Centofanti, Hubert, and Van Aelst.  It uses a package-specific robust
HOSVD initialization and fixed MAD residual scales.  It does not reproduce the
reference ROMPCA initialization, recentering, or automatic rank-selection
procedure exactly, so numerical parity with the reference software is not
claimed.

Use ``MMCD`` instead when the main target is a separable row/column covariance
rather than a low-rank reconstruction.  Use ``CellPCA`` when flattening the
observations is scientifically acceptable.

Worked example
--------------

See :doc:`gallery/robust_multilinear_pca` for a sensor-by-time simulation with
casewise faults, damaged cells, and missing measurements.
