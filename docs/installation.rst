Installation
============

PyPI install
------------

After a release is published, install ``robustcov`` from PyPI:

.. code-block:: bash

   python -m pip install -U pip
   python -m pip install robustcov

Release wheels are built for CPython 3.12, 3.13, and 3.14 on Ubuntu, Windows, and macOS. The project contains a C++/pybind11 extension built with ``scikit-build-core``.

Optional plotting dependency
----------------------------

The numerical core does not require Matplotlib. Install plotting helpers with:

.. code-block:: bash

   python -m pip install "robustcov[plot]"

Importing ``robustcov`` without Matplotlib remains supported. Calling a plotting helper without the extra raises an error that points to the command above.

Native extension availability
-----------------------------

``robustcov`` can still be imported when the compiled extension is unavailable.
NumPy-backed estimators continue to work, ``robustcov.native_available()``
returns ``False``, and thread helpers report a serial fallback. Native-only
estimators such as ``FastMCD`` and ``TylerShape`` fail at ``fit`` time with an
actionable message rather than breaking the package import.

Source-build and native-free packaging instructions are maintained with the
contributor workflow rather than the normal user installation path.

Using a conda environment
-------------------------

A PyPI wheel can be installed inside a conda environment:

.. code-block:: bash

   conda create -n robustcov python=3.12 pip
   conda activate robustcov
   python -m pip install robustcov

Development and documentation builds
------------------------------------

Contributor setup, editable installs, documentation builds, and release
packaging commands are maintained in the repository's
`CONTRIBUTING.md <https://github.com/smiryusupov/robustcov/blob/main/CONTRIBUTING.md>`_.
Keeping those commands out of the user installation path avoids mixing package
installation with maintainer workflow.
