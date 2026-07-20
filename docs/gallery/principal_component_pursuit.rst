:orphan:

Principal Component Pursuit
===========================

This example separates an incoherent low-rank matrix from sparse, arbitrarily
large cell corruption and compares the recovered low-rank component with a
rank-matched truncated SVD.

.. code-block:: bash

   python examples/principal_component_pursuit.py \
     --outdir results/use_cases/principal_component_pursuit

The output contains a decomposition panel, convergence history, and aggregate
recovery metrics. The controlled example matches the PCP contamination model;
it is not a benchmark for heavy tails, rowwise outliers, or missing data.

See :doc:`../principal_component_pursuit` for assumptions and API details.
