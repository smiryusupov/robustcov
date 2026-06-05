Optional OpenMP acceleration
============================

``robustcov`` can use OpenMP in its C++ extension when the compiler and build
environment provide it. OpenMP is optional: builds without OpenMP still work and
fall back to the serial kernels.

What is parallelized?
---------------------

The current OpenMP path targets the loops that are shared by several estimators:

* robust / Mahalanobis distance evaluation over observations;
* column means and covariance accumulation over selected subsets;
* Tyler and regularized Tyler scatter accumulation;
* FastMCD random-start evaluation and final candidate polishing.

These are useful first targets because they are repeated many times inside
FastMCD C-steps, reweighting, Tyler iterations, diagnostics, and benchmark runs.

Thread-control API
------------------

Use the process-wide helpers when you want global control:

.. code-block:: python

   import robustcov as rc

   print(rc.has_openmp())
   print(rc.get_num_threads())
   rc.set_num_threads(4)

Estimators that call the C++ backend also accept ``n_jobs``:

.. code-block:: python

   est = rc.FastMCD(n_init=500, n_jobs=4, random_state=0).fit(X)
   ty = rc.RegularizedTyler(alpha=0.1, n_jobs=4).fit(X)

For temporary changes, use ``thread_limit``:

.. code-block:: python

   with rc.thread_limit(2):
       est = rc.FastMCD(n_init=1000, random_state=0).fit(X)

Benchmarking scaling
--------------------

Run:

.. code-block:: bash

   python benchmarks/openmp_scaling.py --n 8000 --p 20 --threads 1 2 4 --csv results/openmp_scaling.csv

Interpret scaling carefully. Small data can be slower with multiple threads
because thread launch and reduction overhead dominate. The gains should be most
visible for larger ``n``, larger ``p``, and many FastMCD starts.

BLAS and OpenMP interaction
---------------------------

Some NumPy/SciPy builds also use threaded BLAS. If BLAS and OpenMP both use many
threads, oversubscription can hurt performance. For clean OpenMP benchmarks, set
BLAS thread counts explicitly, for example:

.. code-block:: bash

   OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/openmp_scaling.py

Determinism
-----------

Random starts are generated serially so ``random_state`` remains reproducible.
Parallel reductions may still cause tiny floating-point differences because
summation order changes. These should be numerically negligible.


Benchmark integration
---------------------

The benchmark report includes OpenMP scaling by default::

   OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
   python benchmarks/make_report.py --outdir results/report

The generated report includes ``openmp_scaling.csv`` and ``openmp_scaling.png``. For larger
``n``, higher ``p``, and more FastMCD starts, the parallel benefit should become more visible.

A common shell mistake is to paste Python snippets directly into bash. Use ``python - <<'PY'``
or an interactive Python session when checking OpenMP helpers::

   python - <<'PY'
   import robustcov as rc
   print(rc.has_openmp())
   print(rc.get_num_threads())
   PY
