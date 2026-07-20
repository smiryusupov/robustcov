External benchmark snapshot policy
==================================

External benchmarks are executed locally or by the manually triggered
``external-data`` workflow. Read the Docs never downloads datasets and never
runs these protocols. Instead, reviewed aggregate outputs are copied into the
documentation as immutable reference snapshots.

Repository boundaries
---------------------

Tracked source and documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* dataset loaders and protocol scripts;
* small aggregate figures and summary tables under
  ``docs/_static/external_results``;
* one ``snapshot.json`` provenance record per published result;
* generated result pages and gallery cards.

Never tracked
~~~~~~~~~~~~~

* raw archives or extracted datasets;
* cache directories or temporary downloads;
* embeddings or model checkpoints;
* row-level and window-level scores;
* complete ``results/external`` workspaces.

Publishing a reviewed result
----------------------------

First run the protocol locally. For example:

.. code-block:: bash

   python examples_external/gas_sensor_drift_dro_pca.py \
     --download \
     --outdir results/external/gas_sensor_drift_dro_pca

Then publish only its approved outputs:

.. code-block:: bash

   python scripts/publish_external_snapshot.py publish gas_sensor_drift \
     --results results/external/gas_sensor_drift_dro_pca \
     --command "python examples_external/gas_sensor_drift_dro_pca.py --download --outdir results/external/gas_sensor_drift_dro_pca"

Review the generated Git diff before committing. The publisher removes local
cache paths, copies only allowlisted files, records SHA-256 digests, and creates
the result page and gallery registry.

Validate all committed snapshots with:

.. code-block:: bash

   python scripts/publish_external_snapshot.py check

Selected benchmark roadmap
--------------------------

.. list-table:: External validation roadmap
   :header-rows: 1
   :widths: 22 20 28 30

   * - Dataset
     - Repository interface
     - Scientific role
     - Delivery phase
   * - UCI Gas Sensor Drift
     - Managed public archive loader
     - Direct temporal sensor drift
     - Available now; publish the first reviewed snapshot.
   * - NASA C-MAPSS FD002 and FD004
     - Managed public archive loader
     - Operating-regime tolerance versus degradation
     - Available now; publish separate FD002 and FD004 snapshots.
   * - UCI Air Quality
     - Managed public archive loader
     - Sensor response drift against certified reference measurements
     - Next lightweight loader and protocol.
   * - Electricity Load Diagrams 2011--2014
     - Managed public archive loader with streaming parser
     - Long-horizon unsupervised monitoring stress test
     - Next time-series loader; summaries only.
   * - Office-Home embeddings
     - Local embedding artifact, not a core image downloader
     - Accessible domain-shift visualization
     - Add a standard embedding schema and optional preparation script.
   * - Camelyon17 embeddings
     - Local WILDS embedding artifact
     - Publication-grade hospital domain shift
     - Preferred first heavyweight benchmark because its WILDS release is CC0.
   * - RxRx1 embeddings
     - Local WILDS embedding artifact
     - Experimental batch-effect domain shift
     - Optional later benchmark; preserve the non-commercial share-alike terms.

Embedding-based datasets
------------------------

Image datasets do not belong in the lightweight core dataset downloader. Their
preparation scripts may use optional packages such as ``torch``, ``torchvision``,
or ``wilds``, but the RobustCov protocol should consume a small, documented local
artifact such as ``.npz`` with arrays named ``X``, ``domain``, ``label``, and
``sample_id``. Embeddings themselves remain untracked; only aggregate plots and
summary tables are published.
