:orphan:

Adversarial covariance filtering
================================

This example compares empirical covariance with the experimental quadratic
spectral filter under a structured whole-row attack.

.. code-block:: bash

   python examples/adversarial_covariance_filtering.py \
     --outdir results/use_cases/adversarial_covariance_filtering

The output contains aggregate covariance error, attack-row recall, and the
lifted-operator diagnostic across filtering iterations. The scenario is a
controlled validation, not a claim of universal superiority.

See :doc:`../adversarial_covariance_filtering` for assumptions and limitations.
