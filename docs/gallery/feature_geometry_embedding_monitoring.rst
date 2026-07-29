:orphan:

Practical embedding monitoring
==============================

This example shows how robust feature geometry can be used as a practical
monitoring layer for embedding or feature matrices.

The workflow mimics a production setting:

* a reference window of embeddings is available;
* a new batch of embeddings arrives;
* the reference window may itself be contaminated;
* empirical and robust feature geometries are fitted;
* a central reference anchor is selected;
* an MMD-style drift signal is calibrated from reference splits.

The important point is that the upstream model does not change. ``robustcov``
only receives feature matrices.

In this example, empirical covariance geometry keeps contaminated reference
points inside the central anchor and largely absorbs the drift direction.
Robust FastMCD geometry removes the contaminated reference points from the
central anchor and preserves the MMD-style drift signal.

Example output
--------------

.. literalinclude:: ../_static/gallery/feature_geometry_embedding_monitoring_output.txt
   :language: text

Source
------

.. literalinclude:: ../../examples/feature_geometry_embedding_monitoring.py
   :language: python
   :caption: examples/feature_geometry_embedding_monitoring.py
