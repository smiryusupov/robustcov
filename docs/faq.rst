FAQ
===

Why not prioritize MVE?
-----------------------

Minimum-volume ellipsoid estimators are historically important, but they are not the current differentiator for this package.  The benchmark evidence is stronger for efficient FastMCD and regularized heavy-tail scatter estimators such as ``RegularizedCauchy`` and ``StudentTScatter``.

When should I use FastMCD?
--------------------------

Use ``FastMCD`` when the data are mostly clean, the outliers are separable, and ``n`` is comfortably larger than ``p``.  It is the most interpretable choice for classical robust distances and support diagnostics.

When should I use RegularizedCauchy?
------------------------------------

Use ``RegularizedCauchy`` for small-sample, high-dimensional, heavy-tailed covariance problems.  It is currently the strongest method in the package's small-sample heavy-tail benchmark gallery.

When should I use StudentTScatter?
----------------------------------

Use ``StudentTScatter`` when you want a smooth heavy-tail covariance-like estimator with a fixed degrees-of-freedom parameter.  It is often competitive with Cauchy and can be easier to tune conceptually.

When should I use AutoRobustScatter?
------------------------------------

Use ``AutoRobustScatter`` for exploratory workflows when you want the package to compare several robust scatter candidates.  It is a diagnostic selector, not an oracle.  For production, inspect the chosen estimator and the robust-distance diagnostics.

Are OpenMP results deterministic?
---------------------------------

For fixed random seeds, estimator choices are intended to be reproducible.  Parallel floating-point reductions can still introduce tiny numerical differences because summation order changes.  This is normal for parallel numerical code.

Why are there both a Benchmark Gallery and a Use-case Gallery?
----------------------------------------------------------------

The benchmark gallery answers evidence questions: speed, ranking, scaling, and failure modes.  The use-case gallery answers application questions: what to do for finance, fraud, sensors, embeddings, and ML preprocessing.
