Robust multilinear PCA for sensor windows
=========================================

This example treats each observation as a sensor-by-time matrix.  The regular
windows follow a low-rank multilinear model, while some complete windows are
abnormal and other windows contain only a few damaged measurements.

A median-imputed multilinear PCA baseline is compared with
``RobustMultilinearPCA``.  The robust fit keeps the two modes separate, estimates
one loading matrix for sensors and one for time, and returns cell- and
case-level diagnostics.

.. literalinclude:: ../../examples/plot_robust_multilinear_pca.py
   :language: python
   :linenos:

Mode-subspace recovery
----------------------

.. image:: ../_static/gallery/robust_multilinear_pca/mode_subspaces.png
   :alt: Row-mode and column-mode subspace errors

Residual diagnostics
--------------------

.. image:: ../_static/gallery/robust_multilinear_pca/residual_map.png
   :alt: Standardized residual map for an unusual matrix observation

.. image:: ../_static/gallery/robust_multilinear_pca/outlier_map.png
   :alt: Casewise versus cellwise multilinear PCA diagnostics

Reconstruction
--------------

.. image:: ../_static/gallery/robust_multilinear_pca/reconstruction.png
   :alt: Reconstruction error comparison

Captured output
---------------

.. literalinclude:: ../_static/gallery/robust_multilinear_pca/output.txt
   :language: text
