External datasets and local caches
==================================

``robustcov`` does not bundle or commit external datasets.  Optional loaders keep
raw archives, extracted files, and processed arrays in a user cache outside the
repository.  Downloads occur only after an explicit request.

Cache location
--------------

The cache root is resolved in this order:

1. an explicit ``cache_dir=...`` argument;
2. ``ROBUSTCOV_DATA_DIR``;
3. ``XDG_CACHE_HOME/robustcov``;
4. ``~/.cache/robustcov``.

For example:

.. code-block:: bash

   export ROBUSTCOV_DATA_DIR="$HOME/data/robustcov"

List the supported datasets and inspect their source metadata:

.. code-block:: bash

   python -m robustcov.datasets list
   python -m robustcov.datasets info gas_sensor_drift
   python -m robustcov.datasets info cmapss

Safety and reproducibility
--------------------------

The loaders:

* never access the network during ``import robustcov``;
* use atomic ``.partial`` downloads;
* validate published checksums when available;
* record and revalidate a local SHA-256 fingerprint when an upstream archive has
  no published SHA-256 value;
* reject path traversal, absolute paths, symbolic links, and oversized ZIP
  archives;
* record the source URL, archive fingerprint, citation, and terms metadata in
  the cache;
* support manually downloaded archives through ``archive_path=...``.

Normal tests use tiny locally generated archives and never require internet
access.  External examples write only result summaries and figures under
``results/external``.  Raw data remains in the user cache.

.. toctree::
   :maxdepth: 1

   external_data/gas_sensor_drift
   external_data/cmapss
   external_snapshot_policy
