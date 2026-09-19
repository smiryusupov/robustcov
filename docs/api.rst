API reference
=============

Use the API reference when you already know which object you need and want its
constructor parameters, fitted attributes, methods, or return types.  If you
are still deciding what to use, start with :doc:`estimator_guide` or
:doc:`user_guide` instead.

The supported package-root exports and their stability tiers are defined in
:doc:`api_stability`.  Names not listed in ``robustcov.__all__`` are internal
implementation details unless a documented submodule explicitly exposes them.

Common entry points
-------------------

.. list-table:: API map
   :header-rows: 1
   :widths: 28 37 35

   * - Task
     - Common objects
     - Reference page
   * - covariance, scatter, and robust distances
     - ``FastMCD``, ``MRCD``, ``DetS``, ``DetMM``, ``RegularizedCauchy``
     - :doc:`api_reference/covariance_scatter`
   * - cellwise, matrix, and structured observations
     - ``CellMCD``, ``CellRCov``, ``MatrixMCD``
     - :doc:`api_reference/structured`
   * - PCA, decomposition, sources, and factors
     - ``RobustScatterPCA``, ``CellPCA``, ``PrincipalComponentPursuit``
     - :doc:`api_reference/latent_structure`
   * - anomaly detection, calibration, and monitoring
     - ``RobustOutlierDetector``, ``ConformalAlertCalibrator``, ``RobustSubspaceMonitor``
     - :doc:`api_reference/detection_monitoring`
   * - sparse precision, distances, and kernels
     - ``RobustGraphicalLasso``, ``SGLASSO``, SPD geometry utilities
     - :doc:`api_reference/geometry_precision`
   * - explanation helpers and external data loaders
     - ``RobustExplanationReference`` and dataset loaders
     - :doc:`api_reference/integrations_data`
   * - research-stage interfaces
     - adversarial filtering, distributionally robust PCA, online tracking
     - :doc:`api_reference/experimental`
   * - method metadata and citations
     - provenance registry and canonical method names
     - :doc:`api_reference/provenance`

Reference groups
----------------

.. toctree::
   :maxdepth: 1

   api_reference/covariance_scatter
   api_reference/structured
   api_reference/latent_structure
   api_reference/detection_monitoring
   api_reference/geometry_precision
   api_reference/integrations_data
   api_reference/experimental
   api_reference/provenance
