Methods, provenance, and references
===================================

``robustcov`` distinguishes the origin of a statistical method from the work
performed in this package.  This distinction is important: independently
implementing, accelerating, validating, and integrating a published method can
be a substantial software contribution without making the underlying method a
new invention.

How to cite a result
--------------------

When a result uses ``robustcov``:

#. cite the software release using ``CITATION.cff`` or the release DOI;
#. cite the primary methodological references listed for each estimator used;
#. describe material implementation choices such as the scatter estimator,
   regularization target, robust weighting, initialization, and backend.

The complete machine-readable bibliography is stored in
``docs/references.bib``.  The runtime registry is available through
``robustcov.get_method_provenance``:

.. code-block:: python

   import robustcov as rc

   info = rc.get_method_provenance(rc.RobustSOBI)
   print(info.status)
   print(info.references)
   print(info.robustcov_contribution)
   print(info.implementation_notes)

Provenance labels
-----------------

``Literature implementation``
   The estimator implements a named published method.  Independent numerical
   choices and engineering work are still documented.

``Literature-based adaptation``
   The estimator follows a published method family but deliberately changes or
   combines parts of the algorithm.  The implementation notes state the main
   differences.

``robustcov composite/workflow``
   The package combines established components into a practical estimator or
   workflow.  This label does not claim that the component methods originated in
   ``robustcov``.

``robustcov utility/infrastructure``
   The object primarily provides preprocessing, numerical, metric, or API
   infrastructure around established ideas.

``Original methodological contribution``
   Reserved for a genuinely new statistical method with a clear technical
   description and validation.  **No current public estimator uses this label.**

.. include:: _generated/method_provenance.inc
   :start-after: .. method-provenance-body-start
