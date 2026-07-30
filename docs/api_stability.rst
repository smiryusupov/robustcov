API stability
=============

Plain-language summary
----------------------

``robustcov`` is under active development. The package is usable, tested, and
documented, but some APIs may still change before a stable 1.0 release.

In practice:

* estimator names and core fitted attributes such as ``location_``,
  ``covariance_``, and ``precision_`` are intended to remain stable;
* documented examples and user-guide workflows are maintained;
* newer utility layers, such as SPD geometry, robust PCA, rolling monitoring,
  and optional integration helpers, may evolve based on user feedback;
* breaking changes and deprecations are recorded in ``CHANGELOG.md``.

The stability tier for each public symbol is recorded explicitly so users can
distinguish mature interfaces from provisional and experimental ones.

What counts as public
---------------------

At the package root, the supported namespace is exactly the set of names listed
in ``robustcov.__all__``.  The same rule applies to
``robustcov.experimental.__all__``.  A helper that is reachable only as an
implementation detail is not part of the compatibility contract, even if Python
introspection can discover it through an internal module.

Public classes and functions documented from their defining submodules remain
usable from those documented paths.  The top-level stability manifest does not
turn every imported helper or module attribute into a supported API.

Stability tiers
---------------

The machine-readable contract is stored in ``robustcov/_public_api.json``. Every
name exported from ``robustcov`` is classified in exactly one tier, and release
checks fail when the manifest and the actual namespaces diverge.

Stable
~~~~~~

Stable names are intended to remain available with recognizable constructor and
fitted-attribute contracts. Changes follow the deprecation policy below. This
tier currently contains the mature core covariance interfaces,
``PrincipalComponentPursuit``, ``RobustOutlierDetector``, and native-thread
control helpers.

Provisional
~~~~~~~~~~~

Provisional names are supported public APIs, but details may evolve before 1.0.
They still require changelog entries and a compatibility path when practical.
Most structured estimators, workflow composites, plotting helpers, and newer PCA
or precision APIs are currently provisional.

Experimental
~~~~~~~~~~~~

Experimental names may change substantially and are normally exposed from
``robustcov.experimental``. A small number remain available at the top level for
compatibility; the manifest records those exceptions explicitly. Experimental
methods must state which published guarantees do and do not carry over to the
implementation.

Inspect the installed contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import importlib.resources
   import json

   manifest = json.loads(
       importlib.resources.files("robustcov")
       .joinpath("_public_api.json")
       .read_text(encoding="utf-8")
   )
   print(manifest["stable_top_level"])

Deprecation policy
------------------

The project uses an explicit deprecation path whenever a compatibility shim is
practical.

Before 1.0
~~~~~~~~~~

* A renamed or replaced public API should emit ``DeprecationWarning`` with the
  replacement and planned removal release.
* Deprecated aliases should normally remain available for at least one public
  feature release and at least 90 days.
* A direct breaking change is reserved for cases where retaining compatibility
  would produce incorrect numerical results, unsafe behavior, or unreasonable
  maintenance cost.
* Every deprecation or direct break must appear under ``Unreleased`` in
  ``CHANGELOG.md`` and in the corresponding release notes.

After 1.0
~~~~~~~~~

* Deprecated public APIs should normally remain available for at least two
  minor releases and at least six months.
* Removal warnings must name the replacement when one exists.
* Constructor parameters should not silently change meaning. A changed default
  must be called out in release notes and, when practical, warned about during
  the transition period.

Warnings are intentionally based on Python's ``DeprecationWarning`` category so
library users can opt in during tests with ``python -Wd`` or an equivalent
warning filter without producing noise in ordinary interactive use.
