:orphan:

Online robust subspace tracking
===============================

This synthetic streaming example rotates a two-dimensional latent subspace
slowly, injects isolated corrupted cells, and adds dense row anomalies. The
tracker repairs limited projected-residual cells, excludes dense rows from
adaptation, and updates from a bounded recent-sample buffer.

.. code-block:: bash

   python examples/online_robust_subspace_tracking.py

The example writes ``tracking_summary.csv``, ``tracking_error.png``, and
``rejected_rows.png`` under
``results/use_cases/online_subspace_tracking``.

Interpretation
--------------

The example validates a practical software behavior: the adaptive projector
follows gradual rotation better than a frozen initial projector while corrupted
observations are prevented from dominating updates. It does not validate NORST
theory and should not be cited as a NORST reproduction.

See :doc:`../online_subspace_tracking` for assumptions and API details.
