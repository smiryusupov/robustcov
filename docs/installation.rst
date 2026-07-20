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

Native-free installation behavior
---------------------------------

``robustcov`` can be imported when the compiled extension is unavailable. NumPy-backed estimators continue to work, ``robustcov.native_available()`` returns ``False``, and thread helpers report a serial fallback. Native-only estimators such as ``FastMCD`` and ``TylerShape`` fail at ``fit`` time with an actionable message rather than breaking the package import.

For packaging tests or constrained environments, build a native-free wheel explicitly:

.. code-block:: bash

   python -m pip install build
   python -m build --wheel -Ccmake.define.ROBUSTCOV_BUILD_NATIVE=OFF

Release wheels should continue to include the native extension. The native-free option is a fallback and packaging-validation path, not a replacement for the normal release build.

Conda environment install
-------------------------

PyPI wheels can be installed inside a conda environment immediately:

.. code-block:: bash

   conda create -n robustcov python=3.12 pip
   conda activate robustcov
   python -m pip install robustcov

For native ``conda install -c conda-forge robustcov`` support, publish to PyPI first, then generate a conda-forge recipe from the PyPI metadata with Grayskull:

.. code-block:: bash

   conda create -n grayskull -c conda-forge grayskull conda-build
   conda activate grayskull
   grayskull pypi robustcov

A starter recipe is included under ``conda/recipe/meta.yaml``. Replace the PyPI sdist SHA256 after the first release and submit the recipe to ``conda-forge/staged-recipes``.

Local editable install
----------------------

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -U pip
   python -m pip install -e ".[dev,docs,examples]"
   python -m compileall -q robustcov tests examples benchmarks docs
   pytest -q

Build the documentation
-----------------------

.. code-block:: bash

   python -m pip install -e ".[docs,examples]"
   python -m compileall -q robustcov examples benchmarks docs
   sphinx-build -b html docs docs/_build/html

The built documentation will be available at ``docs/_build/html/index.html``. Read the Docs uses the root ``.readthedocs.yaml`` and also compiles all Python examples before running Sphinx.
