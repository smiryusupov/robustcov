Integrations and data API
=========================

SHAP and LIME adapters
----------------------

.. automodule:: robustcov.explain
   :members:
   :undoc-members:

Optional external dataset loaders
---------------------------------

These loaders never download during import.  Raw data is stored in a user cache
outside the repository; see :doc:`../external_data`.

.. autofunction:: robustcov.datasets.fetch_gas_sensor_drift

.. autoclass:: robustcov.datasets.GasSensorDriftDataset
   :members:

.. autofunction:: robustcov.datasets.fetch_cmapss

.. autoclass:: robustcov.datasets.CMapssDataset
   :members:

.. autofunction:: robustcov.datasets.get_data_home
