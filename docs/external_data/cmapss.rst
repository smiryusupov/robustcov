NASA C-MAPSS degradation monitoring
====================================

Dataset
-------

NASA C-MAPSS contains multiple multivariate run-to-failure engine trajectories.
The four standard subsets vary in operating conditions and fault modes.  FD002
and FD004 contain six operating conditions, making them useful for testing
whether a shift-aware subspace can tolerate operating-regime changes while
remaining sensitive to degradation.

Source and citation
-------------------

* NASA Open Data page: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
* NASA PCoE repository: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
* Citation: A. Saxena and K. Goebel (2008), *Turbofan Engine Degradation Simulation Data Set*, NASA Ames Prognostics Data Repository.

The NASA Open Data record does not specify a license.  Review the current source
terms before use.  ``robustcov`` does not redistribute the archive.

Fetch or use a manual archive
-----------------------------

.. code-block:: bash

   python -m robustcov.datasets fetch cmapss --subset FD002

or:

.. code-block:: python

   from robustcov.datasets import fetch_cmapss

   data = fetch_cmapss("FD002", download=True)
   print(data.train.sensors.shape)
   print(data.train.settings.shape)

The loader accepts both the current NASA PCoE archive and the legacy
``CMAPSSData.zip`` structure, including an outer archive that contains a nested
ZIP.

DRO-PCA monitoring workflow
----------------------------

Run:

.. code-block:: bash

   python examples_external/cmapss_dro_pca_monitoring.py \
     --download \
     --subset FD002

The script:

1. uses the first 20% of each training trajectory as the reference period;
2. uses the 20--35% life interval for independent alert calibration;
3. estimates transport geometry from healthy operating-regime mean shifts;
4. scores rolling windows over normalized engine life;
5. reports alert rates by life interval and late-life sensor contributions.

The fault-onset time is not supplied in the training data, so normalized-life
bins are a transparent evaluation proxy rather than a claim of exact onset
labels. Results are written under
``results/external/cmapss_dro_pca_monitoring``; raw data remains in the cache.

FD002 and FD004 are the approved reviewed external snapshot profiles. After
local review, publish their aggregate figures and summary table with
``scripts/publish_external_snapshot.py``. Read the Docs renders only those
committed snapshots and never downloads or executes C-MAPSS.
