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

Version and source checks
-------------------------

Verify that the package metadata, runtime version, citation metadata, and public
API manifest agree:

.. code-block:: bash

   python scripts/check_release_version.py --require-prerelease

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

Artifact checksums
------------------

Create a deterministic checksum manifest after building all release artifacts:

.. code-block:: bash

   python scripts/write_artifact_checksums.py \
     dist/* native-free-wheel/*.whl \
     --output release-metadata/SHA256SUMS

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


TestPyPI rehearsal
------------------

Run the ``release distributions`` GitHub Actions workflow manually with
``publish_target=testpypi``. The workflow uses the ``testpypi`` GitHub
environment and Trusted Publishing, then installs the published wheel outside
the checkout and runs the installed-package smoke test.

The TestPyPI and production publishing jobs are separate from the build jobs and
are the only jobs granted ``id-token: write``. Configure the production
``pypi`` environment with a required reviewer and release-tag restrictions.

A package-index version cannot be overwritten. Increment the alpha serial before
repeating a published rehearsal.

Production publication
----------------------

The tag must match the declared package version exactly:

.. code-block:: bash

   python scripts/check_release_version.py --tag v0.1.0a3

Pushing that signed tag triggers the same distribution build and evidence gate,
then publishes through the protected ``pypi`` environment. The PyPI publishing
action creates PEP 740-compatible attestations by default when Trusted
Publishing is used.
