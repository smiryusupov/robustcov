Release readiness
=================

This page records the mechanical checks required before a public release. It is
an engineering checklist, not a claim that every experimental estimator has a
stable API.

Supported environments
----------------------

Release wheels target CPython 3.12, 3.13, and 3.14 on Linux x86-64, Windows
AMD64, macOS x86-64, and macOS arm64. The normal CI matrix runs the tests on all
three operating systems and Python versions. A separate Linux job tests the
oldest supported NumPy, SciPy, and scikit-learn versions from
``requirements/minimum.txt``.

Source checks
-------------

Run the source metadata and public-export audit:

.. code-block:: bash

   python scripts/release_check.py \
     --json-output results/release-check-source.json

Run compilation, tests, and documentation with warnings treated as errors:

.. code-block:: bash

   python -m compileall -q robustcov tests examples benchmarks docs scripts
   OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=2 \
     python -m pytest -q
   python -m sphinx -W --keep-going -b html docs docs/_build/html

Minimum dependencies
--------------------

For the active Python interpreter:

.. code-block:: bash

   python -m pip install -U pip scikit-build-core pybind11 pytest
   python -m pip install --only-binary=:all: -r requirements/minimum.txt
   python -m pip install --no-build-isolation --no-deps -e .
   python -m pytest -q

Distribution checks
-------------------

Build fresh artifacts and validate both metadata and archive contents:

.. code-block:: bash

   rm -rf build dist
   python -m build
   python -m twine check dist/*
   python scripts/release_check.py \
     --json-output results/release-check-dist.json \
     dist/*
   python scripts/package_smoke_test.py --expect-native yes dist/*

A native-free wheel is an additional fallback test, not the normal release
artifact:

.. code-block:: bash

   rm -rf native-free-wheel
   python -m build --wheel \
     -Ccmake.define.ROBUSTCOV_BUILD_NATIVE=OFF \
     --outdir native-free-wheel
   python scripts/release_check.py native-free-wheel/*.whl
   python scripts/package_smoke_test.py \
     --expect-native no native-free-wheel/*.whl

Release sign-off
----------------

Before tagging, confirm that:

* ``pyproject.toml``, ``robustcov.__version__``, and ``CITATION.cff`` agree;
* ``CHANGELOG.md`` describes the release and no generated benchmark claims are stale;
* license metadata uses the SPDX expression and both ``LICENSE`` and ``NOTICE``
  are present in the source distribution and wheels;
* all GitHub Actions jobs are green;
* artifacts were built from the intended commit and their checksums were saved;
* installation was smoke-tested outside the repository source tree.

Release-candidate evidence gate
-------------------------------

After reviewed FD002 and FD004 snapshots are committed, run:

.. code-block:: bash

   python scripts/release_check.py --release-candidate

This adds the public API manifest checks and requires clean, commit-pinned
C-MAPSS evidence. Ordinary source checks remain available without this flag
during development.
