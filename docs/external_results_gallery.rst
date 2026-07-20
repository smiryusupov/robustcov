External benchmark case studies
=================================

This page contains reviewed aggregate snapshots from public datasets. The
underlying datasets and full benchmark outputs are not stored in the repository.
Protocols are run locally or through a manually triggered workflow; Read the
Docs only renders committed plots, compact summaries, and provenance metadata.

The current public external benchmark family is NASA C-MAPSS FD002/FD004. These
subsets provide run-to-failure engine trajectories with multiple operating
conditions, allowing the documentation to distinguish tolerated operating
regimes from progressive degradation.

How to interpret these results
------------------------------

The C-MAPSS case studies validate the package's monitoring workflow:

* fit a reference subspace on early-life observations;
* calibrate upper-tail conformal p-values on a separate healthy-life interval;
* keep the fitted reference frozen;
* score rolling windows over later engine life;
* report risk and alert frequency by normalized-life interval;
* inspect sensors contributing to late-life residual risk.

They do not imply that every robust estimator must outperform empirical PCA on
every subset. When empirical PCA and DRO-PCA produce overlapping curves, the
result is reported as evidence for the calibrated monitoring pipeline, not as a
superiority claim.

Reviewed snapshots
------------------

The cards below are generated from the committed snapshot manifest. Every
published snapshot records file hashes, the source commit, the reproduction
command, dataset fingerprints, and aggregate outputs. See
:doc:`external_snapshot_policy`.

Refresh the generated cards and page tree after changing the publisher or
manifest:

.. code-block:: bash

   python scripts/publish_external_snapshot.py check --rewrite-generated


.. include:: _generated/external_snapshot_cards.rst

.. include:: _generated/external_snapshot_toctree.rst

Reproduce locally
-----------------

.. code-block:: bash

   python examples_external/cmapss_dro_pca_monitoring.py \
     --download \
     --subset FD002 \
     --outdir results/external/cmapss_fd002

   python scripts/publish_external_snapshot.py publish cmapss_fd002 \
     --results results/external/cmapss_fd002 \
     --command "python examples_external/cmapss_dro_pca_monitoring.py --download --subset FD002 --outdir results/external/cmapss_fd002"

Run FD004 with the same protocol after the archive is present locally:

.. code-block:: bash

   python examples_external/cmapss_dro_pca_monitoring.py \
     --subset FD004 \
     --outdir results/external/cmapss_fd004

The UCI Gas Sensor Drift loader and exploratory analysis remain available under
:doc:`external_data`, but they are not part of the reviewed public benchmark
gallery.
