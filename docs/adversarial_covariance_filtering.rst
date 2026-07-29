Spectral filtering for adversarial row contamination
====================================================

``SpectralFilteringCovariance`` is an experimental estimator for approximately
Gaussian data when an upper bound on the fraction of arbitrarily corrupted
*rows* is available. It is exposed from ``robustcov.experimental``.

The method fills a different role from the package's classical robust scatter
estimators:

* use FastMCD or DetS/DetMM for classical high-breakdown elliptical scatter;
* use Tyler, Student-t, Cauchy, or spatial signs for clean heavy-tailed data;
* use CellMCD or CellRCov for isolated bad cells;
* use spectral filtering when a near-Gaussian reference may contain a bounded
  fraction of adversarial whole-row replacements.

Algorithmic idea
----------------

At each iteration the estimator:

#. computes a geometric-median location and regularized provisional covariance;
#. whitens the retained observations;
#. applies matrix-free power iteration to the covariance operator of the lifted
   quadratic features :math:`zz^T-I`;
#. combines the dominant directional quadratic score with a radial score;
#. removes only the most extreme rows, subject to the declared contamination
   budget;
#. stops when the lifted operator is within a conservative finite-sample
   tolerance or no defensible removal remains.

For a clean standard Gaussian and symmetric matrix :math:`V`, the centered
quadratic operator satisfies

.. math::

   \mathbb{E}\left[\langle zz^T-I,V\rangle(zz^T-I)\right] = 2V.

The implementation uses this identity as a diagnostic target. Its automatic
finite-sample tolerance is a transparent heuristic, not a theorem for the
implemented composite.

Quick example
-------------

.. code-block:: python

   from robustcov.experimental import SpectralFilteringCovariance

   estimator = SpectralFilteringCovariance(
       contamination=0.10,
       filter_strength=8.0,
       random_state=0,
   ).fit(X)

   print(estimator.covariance_)
   print(estimator.n_removed_)
   print(estimator.stopping_reason_)
   print(estimator.history_records())

The fitted ``support_`` identifies rows retained for the final estimate.
``filter_scores_`` stores the largest robust standardized filtering score seen
for each training row. Standard covariance attributes and methods are available:
``location_``, ``covariance_``, ``precision_``, ``mahalanobis``,
``score_samples``, and ``predict``.

Assumptions and limits
----------------------

This implementation is a robustcov experimental composite inspired by the
filtering literature of Diakonikolas et al. and Cheng et al. It is **not** their
optimal Gaussian covariance algorithm: it omits the full recursive machinery
and carries none of their finite-sample error or runtime guarantees.

The clean distribution should be close to Gaussian after an affine transform.
Heavy tails can create the same fourth-moment signal as adversarial corruption
and may cause unnecessary filtering. The declared ``contamination`` is a hard
row-removal budget, not an estimated probability. The matrix-free lifted
operator costs roughly :math:`O(np^2)` per power step, so the estimator is not
intended for extremely large feature counts.

Validation
----------

Run the deterministic validation with:

.. code-block:: bash

   python benchmarks/spectral_filter_covariance_validation.py

The validation includes a clean-Gaussian control and a rank-one adversarial-row
attack. It supports only the narrow claim that the implemented filter improves
covariance recovery in that controlled attack while leaving the clean control
essentially unchanged.

References
----------

* Diakonikolas et al. (2017), *Being Robust (in High Dimensions) Can Be Practical*.
* Cheng et al. (2019), *Faster Algorithms for High-Dimensional Robust Covariance Estimation*.
* Novikov (2025), *Robust Scatter Matrix Estimation for Elliptical Distributions in Polynomial Time*.
