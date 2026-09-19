Experimental API
================

These interfaces are intentionally research-stage.  Check :doc:`../api_stability`
before depending on their exact defaults or signatures.

Adversarial covariance filtering
--------------------------------

The filtering estimator is available only from ``robustcov.experimental``.

.. autoclass:: robustcov.experimental.SpectralFilteringCovariance
   :members:
   :show-inheritance:

Distributionally robust PCA
---------------------------

The distributionally robust estimator is intentionally available only from
``robustcov.experimental`` while its geometry and radius defaults are validated.

.. autoclass:: robustcov.experimental.DistributionallyRobustPCA
   :members:
   :show-inheritance:

Online robust subspace tracking
-------------------------------

.. autoclass:: robustcov.OnlineRobustSubspaceTracker
   :members:
   :show-inheritance:

.. autoclass:: robustcov.OnlineSubspaceUpdate
   :members:
   :show-inheritance:
