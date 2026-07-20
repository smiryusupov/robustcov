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

The goal is to be honest about project maturity without suggesting that the
implemented algorithms are experimental toys.

Stability tiers
---------------

Stable prototype
~~~~~~~~~~~~~~~~

These APIs are intended to remain recognizable:

* ``FastMCD``
* ``RegularizedCauchy``
* ``StudentTScatter``
* ``RobustOutlierDetector``
* ``diagnostic_report``
* robust distance plotting helpers

Experimental
~~~~~~~~~~~~

These may change significantly:

* ``AutoRobustScatter`` scoring internals
* ``HellingerRegularizedTyler``
* exact KL/Wiesel variants beyond current aliases/prototypes
* ``MRCD`` search presets, initialization strategy, and diagnostic attribute names
* ``KMRCD`` kernel defaults, initial-support search, and out-of-sample diagnostics
* ``SpatialSignGraphicalLasso`` penalty selection, diagonal-penalty defaults, and shape-score diagnostics
* ``MMCD`` initialization, numerical regularization, and contribution APIs
* ``RobustPCA`` and ``RobustSubspaceMonitor`` calibration and monitoring APIs
* benchmark script schemas

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
