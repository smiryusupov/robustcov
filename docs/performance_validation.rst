Performance and validation
==========================

The package keeps estimator logic in Python unless profiling identifies a
repeated numerical kernel whose native implementation gives a measurable gain.
Every C++ kernel has a NumPy fallback and a backend-equivalence test.

Current native kernels
----------------------

``MMCD`` uses a native batched matrix-Mahalanobis kernel when available.
``RobustMultilinearPCA`` uses a native weighted Tucker core-score solver.  Both
classes accept ``backend="auto"``, ``"python"``, or ``"cpp"``.

Run the local scaling benchmark with:

.. code-block:: bash

   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
   python benchmarks/native_scaling.py \
       --repeats 5 \
       --csv results/native_scaling.csv

The benchmark records Python and C++ runtimes, speedup, and maximum absolute
numerical difference.  Results depend on compiler, BLAS implementation, CPU,
and thread settings; they should not be treated as portable constants.

Monte Carlo comparison
----------------------

The deterministic documentation benchmark is useful for reproducibility but is
not enough to characterize estimator variability.  Repeated scenario summaries
are generated with:

.. code-block:: bash

   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
   python benchmarks/monte_carlo_methods.py \
       --profile quick \
       --families scatter pca tensor \
       --n-seeds 20 \
       --csv results/monte_carlo_summary.csv \
       --rst results/monte_carlo_summary.rst

The output reports medians, interquartile ranges, failure rates, median runtime,
and 95th-percentile runtime.  Comparisons remain task-specific; there is no
single ranking across covariance, PCA, matrix, and graph estimators.

What remains in Python
----------------------

Eigendecompositions and small dense linear solves already use optimized
NumPy/LAPACK routines. Moving their orchestration into C++ would add complexity
without necessarily improving performance. Candidate kernels should be moved
only after profiling shows that Python-level loops dominate end-to-end runtime.

Local validation snapshot
-------------------------

The repository includes one small validation snapshot generated on the
development environment.  It is useful for checking the reporting pipeline,
not for portable performance claims.

.. literalinclude:: _static/benchmarks/native_scaling.csv
   :language: text
   :lines: 1-8

A five-seed matrix/tensor Monte Carlo smoke run is stored in
``docs/_static/benchmarks/monte_carlo_summary.csv``.  Larger conclusions should
be based on more seeds and the target deployment hardware.

Release-candidate statistical validation
----------------------------------------

The focused release evidence command also runs
``benchmarks/statistical_validation.py`` after changes to covariance equations
or native kernels.  In the committed snapshot all checks pass.  FastMCD agrees
with scikit-learn ``MinCovDet`` on the fixed contaminated Gaussian reference to
relative covariance error ``0.00535`` with distance correlation effectively
``1.0``.  At 10--30 percent injected row contamination, FastMCD covariance
error remains between ``0.126`` and ``0.135`` while empirical covariance error
ranges from ``9.90`` to ``23.34``.

.. literalinclude:: _static/benchmarks/statistical_validation.json
   :language: json

Regenerate and verify the focused snapshot with:

.. code-block:: bash

   python scripts/generate_release_evidence.py --source-revision "$(git rev-parse HEAD)"
   python scripts/generate_release_evidence.py --check
