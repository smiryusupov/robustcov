API reference
=============

The supported package-root exports and their stability tiers are defined in
:doc:`api_stability`.  Names not listed in ``robustcov.__all__`` are internal
implementation details unless a documented submodule explicitly exposes them.

Method provenance
-----------------

.. automodule:: robustcov.provenance
   :members:
   :undoc-members:

Covariance estimators
---------------------

.. automodule:: robustcov.covariance
   :members:
   :undoc-members:

Minimum regularized covariance determinant
------------------------------------------

.. automodule:: robustcov.mrcd
   :members:
   :undoc-members:
   :show-inheritance:

Kernel minimum regularized covariance determinant
-------------------------------------------------

.. automodule:: robustcov.kernel_mrcd
   :members:
   :undoc-members:
   :show-inheritance:

Deterministic S and MM estimators
---------------------------------

.. automodule:: robustcov.s_estimators
   :members:
   :undoc-members:
   :show-inheritance:

Matrix minimum covariance determinant
-------------------------------------

.. automodule:: robustcov.mmcd
   :members:
   :undoc-members:
   :show-inheritance:

Cellwise minimum covariance determinant
---------------------------------------

.. automodule:: robustcov.cellmcd
   :members:
   :undoc-members:
   :show-inheritance:

Cellwise regularized covariance
-------------------------------

.. automodule:: robustcov.cellrcov
   :members:
   :undoc-members:
   :show-inheritance:

Sparse cellwise robust PCA
---------------------------

.. automodule:: robustcov.sparse_cellpca
   :members:
   :undoc-members:
   :show-inheritance:

Robust sparse precision matrices
--------------------------------

.. automodule:: robustcov.sparse_precision
   :members:
   :undoc-members:
   :show-inheritance:

M-estimators
------------

.. automodule:: robustcov.m_estimators
   :members:
   :undoc-members:

Auto selection
--------------

.. automodule:: robustcov.auto
   :members:
   :undoc-members:


Robust input metrics and kernels
---------------------------------

.. automodule:: robustcov.metrics
   :members:
   :undoc-members:

.. automodule:: robustcov.kernels
   :members:
   :undoc-members:

Scikit-learn kernel adapters
----------------------------

.. automodule:: robustcov.sklearn_kernels
   :members:
   :undoc-members:

GPyTorch kernel adapters
------------------------

.. automodule:: robustcov.gpytorch_kernels
   :members:
   :undoc-members:

Outlier detection
-----------------

.. automodule:: robustcov.outliers
   :members:
   :undoc-members:

Multimodal diagnostics
----------------------

.. automodule:: robustcov.multimodal
   :members:
   :undoc-members:

Diagnostics
-----------

.. automodule:: robustcov.diagnostics
   :members:
   :undoc-members:

Plotting
--------

.. automodule:: robustcov.plotting
   :members:
   :undoc-members:

Geometry utilities
------------------

.. automodule:: robustcov.geometry
   :members:
   :undoc-members:
   :show-inheritance:

Blind source separation
-----------------------

.. automodule:: robustcov.ica
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: robustcov.sobi
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: robustcov.joint_diagonalization
   :members:
   :undoc-members:

Robust factor models
--------------------

.. automodule:: robustcov.factor_models
   :members:
   :undoc-members:
   :show-inheritance:

Low-rank plus sparse decomposition
----------------------------------

.. automodule:: robustcov.decomposition
   :members:
   :undoc-members:
   :show-inheritance:

Robust principal component analysis
-----------------------------------

.. autoclass:: robustcov.RobustPCA
   :members:
   :undoc-members:
   :show-inheritance:

Experimental adversarial covariance filtering
---------------------------------------------

The filtering estimator is available only from ``robustcov.experimental``.

.. autoclass:: robustcov.experimental.SpectralFilteringCovariance
   :members:
   :undoc-members:
   :show-inheritance:

Experimental distributionally robust PCA
-----------------------------------------

The distributionally robust estimator is intentionally available only from
``robustcov.experimental`` while its geometry and radius defaults are validated.

.. autoclass:: robustcov.experimental.DistributionallyRobustPCA
   :members:
   :undoc-members:
   :show-inheritance:

Density-power robust PCA
------------------------

.. automodule:: robustcov.dpd_pca
   :members:
   :undoc-members:
   :show-inheritance:

Bootstrap PCA stability
-----------------------

.. automodule:: robustcov.stability
   :members:
   :undoc-members:
   :show-inheritance:

Robust multilinear PCA
----------------------

.. automodule:: robustcov.multilinear_pca
   :members:
   :undoc-members:
   :show-inheritance:

Cellwise and casewise robust PCA
--------------------------------

.. automodule:: robustcov.cellpca
   :members:
   :undoc-members:
   :show-inheritance:

Robust rolling monitoring
-------------------------

.. autoclass:: robustcov.RobustSubspaceMonitor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: robustcov.SubspaceDriftResult
   :members:
   :undoc-members:
   :show-inheritance:

Online robust subspace tracking
-------------------------------

.. autoclass:: robustcov.OnlineRobustSubspaceTracker
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: robustcov.OnlineSubspaceUpdate
   :members:
   :undoc-members:
   :show-inheritance:

Conformal alert calibration
----------------------------

.. autoclass:: robustcov.ConformalAlertCalibrator
   :members:
   :undoc-members:
   :show-inheritance:

Feature geometry
----------------

.. autoclass:: robustcov.FeatureGeometry
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: robustcov.ClassConditionalFeatureGeometry
   :members:
   :undoc-members:
   :show-inheritance:

Optional external dataset loaders
---------------------------------

These loaders never download during import.  Raw data is stored in a user cache
outside the repository; see :doc:`external_data`.

.. autofunction:: robustcov.datasets.fetch_gas_sensor_drift

.. autoclass:: robustcov.datasets.GasSensorDriftDataset
   :members:

.. autofunction:: robustcov.datasets.fetch_cmapss

.. autoclass:: robustcov.datasets.CMapssDataset
   :members:

.. autofunction:: robustcov.datasets.get_data_home
