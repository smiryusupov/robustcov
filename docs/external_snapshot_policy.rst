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

   python examples_external/cmapss_dro_pca_monitoring.py \
     --download \
     --subset FD002 \
     --outdir results/external/cmapss_fd002

Then publish only its approved outputs:

.. code-block:: bash

   python scripts/publish_external_snapshot.py publish cmapss_fd002 \
     --results results/external/cmapss_fd002 \
     --command "python examples_external/cmapss_dro_pca_monitoring.py --download --subset FD002 --outdir results/external/cmapss_fd002"

Review the generated Git diff before committing. The publisher removes local
cache paths, copies only allowlisted files, records SHA-256 digests, and creates
the result page and gallery registry.

Validate all committed snapshots with:

.. code-block:: bash

   python scripts/publish_external_snapshot.py check

Remove a previously published snapshot with:

.. code-block:: bash

   python scripts/publish_external_snapshot.py remove <slug>

Only the approved C-MAPSS FD002 and FD004 profiles may be published. Dataset
loaders and exploratory protocols may remain available without appearing in the
public snapshot gallery.

Selected benchmark roadmap
--------------------------

.. list-table:: External validation roadmap
   :header-rows: 1
   :widths: 22 20 28 30

   * - Dataset
     - Repository interface
     - Scientific role
     - Delivery phase
   * - NASA C-MAPSS FD002 and FD004
     - Managed public archive loader
     - Operating-regime tolerance versus degradation
     - Current reviewed public benchmark family; publish separate FD002 and FD004 snapshots.
   * - UCI Gas Sensor Drift
     - Managed public archive loader
     - Exploratory batch-level covariance drift
     - Loader and local protocol retained; not an approved public snapshot profile.
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

Clean provenance
----------------

Publish reviewed snapshots from a committed implementation with a clean Git
working tree. The publisher records both the commit and dirty-state flag, and
registry validation rejects snapshots produced from uncommitted code.

For release-candidate validation, require both reviewed C-MAPSS profiles:

.. code-block:: bash

   python scripts/publish_external_snapshot.py check \
     --require cmapss_fd002 \
     --require cmapss_fd004

``--allow-dirty`` exists only for local preview generation; such snapshots cannot
pass the registry check and must not be committed.
