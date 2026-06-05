External demo workflow
======================

The external examples are optional: they are not part of tests and they do not
run during the default documentation build.  They are meant for Kaggle notebooks,
external benchmark pages, and public examples where a dataset has to be
manually downloaded or generated.

A reproducible finance demo
---------------------------

Use this command when you want an external-style demo without downloading any
Kaggle or market data:

.. code-block:: bash

   python examples_external/run_external_demo_suite.py --synthetic

The command runs the following steps:

1. generate a synthetic multi-asset price file with correlated heavy-tailed
   returns and injected stress windows;
2. run the market-stress day detector;
3. run the rolling-window market-regime detector;
4. collect all external-result folders into a compact registry.

Outputs
-------

.. code-block:: text

   examples_external/data/prices.csv
   results/external/finance_market_stress/
   results/external/finance_rolling_window/
   results/external_registry/external_results.csv
   results/external_registry/external_results.md
   results/external_registry/external_results.html

Manual finance commands
-----------------------

The demo suite is just a convenience wrapper.  You can also run each step
manually:

.. code-block:: bash

   python examples_external/make_synthetic_prices.py \
     --out examples_external/data/prices.csv

   python examples_external/finance_market_stress.py \
     --prices examples_external/data/prices.csv \
     --outdir results/external/finance_market_stress

   python examples_external/finance_rolling_window_anomaly.py \
     --prices examples_external/data/prices.csv \
     --window 20 \
     --step 5 \
     --outdir results/external/finance_rolling_window

   python examples_external/collect_external_results.py \
     --root results/external \
     --outdir results/external_registry

Kaggle examples
---------------

Kaggle examples should remain manual because datasets require downloads,
account terms, and sometimes competition-specific schemas.  The recommended
workflow is:

1. download the dataset manually;
2. run the matching script under ``examples_external/``;
3. inspect ``metrics.csv`` and plots;
4. add good results to the external-results gallery.
