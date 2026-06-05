Notebooks
=========

The notebooks are optional companion material.  They are not part of the main
documentation navigation because the public docs focus on galleries, algorithms,
benchmarks, and API references.

Available notebooks
-------------------

* ``01_quickstart_robust_distances.ipynb`` — first robust-distance workflow.
* ``02_small_sample_heavy_tail.ipynb`` — Cauchy, Student-t, and Tyler comparison.
* ``03_credit_card_fraud_external_template.ipynb`` — optional Kaggle fraud walkthrough.
* ``04_finance_market_stress.ipynb`` — market stress and rolling-window diagnostics.

Run them from the repository root:

.. code-block:: bash

   python -m pip install notebook ipykernel
   jupyter notebook notebooks

External-data notebooks assume that the relevant CSV files have already been
downloaded under ``examples_external/data``.
