API stability
=============


Plain-language summary
----------------------

``robustcov`` is under active development. This means the package is usable,
tested, and documented, but some APIs may still change as the project matures.

In practice:

* estimator names and core fitted attributes such as ``location_``,
  ``covariance_``, and ``precision_`` are intended to remain stable;
* documented examples and user-guide workflows are maintained;
* newer utility layers, such as SPD geometry and optional integration helpers,
  may evolve based on user feedback;
* breaking changes should be documented in release notes.

The goal is to be honest about project maturity without suggesting that the
implemented algorithms are experimental toys.


The project is still pre-1.0. APIs are divided into practical stability tiers.

Stable prototype
----------------

These APIs are intended to remain recognizable:

* ``FastMCD``
* ``RegularizedCauchy``
* ``StudentTScatter``
* ``RobustOutlierDetector``
* ``diagnostic_report``
* robust distance plotting helpers

Experimental
------------

These may change significantly:

* ``AutoRobustScatter`` scoring internals
* ``HellingerRegularizedTyler``
* exact KL/Wiesel variants beyond current aliases/prototypes
* benchmark script schemas
