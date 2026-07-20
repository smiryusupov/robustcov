Benchmark coverage inventory
============================

The inventory prevents new public estimators from being added without an
explicit evidence owner.  Coverage is classified as:

``comparative``
   The estimator appears in a task-specific accuracy and timing comparison.

``validation``
   Numerical/statistical properties are checked, but the estimator is not
   ranked against alternatives in that file.

``performance``
   A complete-estimator or native-kernel acceleration gate exists.

``workflow``
   The estimator is exercised through a documented end-to-end application.

Regenerate and validate the inventory with:

.. code-block:: bash

   python benchmarks/benchmark_inventory.py \
       --strict \
       --csv results/benchmark_inventory.csv \
       --rst docs/_generated/benchmark_inventory.rst

Aliases are mapped to their canonical implementation rather than counted as
new algorithms.

.. include:: _generated/benchmark_inventory.rst
   :start-after: .. benchmark-inventory-body-start
