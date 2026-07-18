CellRCov in a high-dimensional contaminated table
==================================================

This synthetic example has 90 observations and 120 variables.  The clean
covariance contains four dominant factors plus heteroscedastic residual
variation.  The observed table contains isolated bad cells, a minority of rows
shifted outside the factor subspace, and missing entries.

Run the example
---------------

.. code-block:: bash

   python examples/plot_cellrcov_high_dimensional.py

.. literalinclude:: ../_static/gallery/cellrcov_high_dimensional/output.txt
   :language: text
   :caption: Console output

Covariance recovery
-------------------

Ledoit-Wolf, regularized Cauchy scatter, and MRCD are fitted after or with
median imputation.  CellRCov uses the individual observed cells, a robust
low-rank fit, and a regularized residual covariance.

.. image:: ../_static/gallery/cellrcov_high_dimensional/covariance_error.png
   :alt: Relative covariance error for four estimators
   :width: 760px

Covariance spectrum
-------------------

The spectrum plot shows whether the dominant factors and residual eigenvalues
are recovered at the same time.  The vertical scale is logarithmic.

.. image:: ../_static/gallery/cellrcov_high_dimensional/covariance_spectrum.png
   :alt: True and estimated covariance eigenvalue spectra
   :width: 820px

Subspace and residual distances
-------------------------------

CellRCov keeps the distance inside the robust fitted subspace separate from the
residual distance outside it.  The outlined points are the injected complete-row
outliers.

.. image:: ../_static/gallery/cellrcov_high_dimensional/distance_decomposition.png
   :alt: CellRCov subspace and residual distance map
   :width: 720px

Cell residuals
--------------

The cellmap shows the rows with the largest standardized cell residuals.  A
single isolated mark is interpreted differently from a broad departure across
many variables.

.. image:: ../_static/gallery/cellrcov_high_dimensional/cell_residual_map.png
   :alt: CellRCov residual cellmap for suspicious observations
   :width: 980px

Scope
-----

The simulation is deliberately favorable to a low-rank-plus-residual covariance
decomposition, and the true rank is supplied to every CellRCov fit.  The result
does not establish that the same rank or estimator is optimal for arbitrary
high-dimensional tables.

Source
------

.. literalinclude:: ../../examples/plot_cellrcov_high_dimensional.py
   :language: python
   :caption: examples/plot_cellrcov_high_dimensional.py
