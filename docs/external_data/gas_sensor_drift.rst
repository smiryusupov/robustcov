UCI gas-sensor temporal drift
=============================

Dataset
-------

The UCI Gas Sensor Array Drift at Different Concentrations dataset contains
13,910 measurements, 128 derived features, six gases, concentration labels, and
ten temporal batches collected across 36 months.  The dataset is not included in
``robustcov``.

Source and citation
-------------------

* Dataset page: https://archive.ics.uci.edu/dataset/270/gas%2Bsensor%2Barray%2Bdrift%2Bdataset%2Bat%2Bdifferent%2Bconcentrations
* DOI: https://doi.org/10.24432/C5MK6M
* Citation: Vergara, A. (2012), *Gas Sensor Array Drift at Different Concentrations*, UCI Machine Learning Repository.

The current UCI metadata displays CC BY 4.0, while older descriptive text on the
record contains a research-only statement.  Review the current source terms and
your intended use.  The loader does not redistribute the archive.

Fetch or use a manual archive
-----------------------------

.. code-block:: bash

   python -m robustcov.datasets fetch gas_sensor_drift

or:

.. code-block:: python

   from robustcov.datasets import fetch_gas_sensor_drift

   data = fetch_gas_sensor_drift(download=True)
   print(data.X.shape)
   print(data.batch)

A manually downloaded ZIP can be used without network access:

.. code-block:: python

   data = fetch_gas_sensor_drift(
       archive_path="/path/to/gas_sensor_drift.zip",
       download=False,
   )

Exploratory batch analysis
--------------------------

Run:

.. code-block:: bash

   python examples_external/gas_sensor_drift_dro_pca.py --download

The script residualizes known gas and concentration effects, compares early and
later batches, and writes local diagnostic figures. It is retained as an
**exploratory batch-level analysis**, not as a reviewed external benchmark. In
particular, within-batch row ordering and the small number of calibration windows
make the current alert-rate output unsuitable as an operational monitoring claim.

Outputs are written under
``results/external/gas_sensor_drift_dro_pca`` and remain untracked. The committed
repository contains only the loader, exploratory protocol, tests, and
documentation—not the raw measurements or generated result snapshots.
