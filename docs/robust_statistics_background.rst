Robust statistics background
============================

This page gives the short mathematical vocabulary needed to read the
algorithm pages.  It is intentionally practical: the goal is to explain why
``robustcov`` uses robust scatter estimators and robust-distance diagnostics,
not to replace a robust-statistics textbook.

Why robustness matters
----------------------

Classical covariance is sensitive to a small number of extreme observations.
For data points ``x_i \in R^p``, the empirical covariance is

.. math::

   \widehat\Sigma = \frac{1}{n}\sum_{i=1}^n (x_i - \bar x)(x_i - \bar x)^T.

The quadratic term means that a few high-leverage points can dominate the
estimate.  Robust covariance estimators replace this direct average with
subset selection, reweighting, bounded radial weights, shrinkage, or shape
normalization.

Influence function
------------------

The **influence function** describes the first-order effect of putting an
infinitesimal amount of probability mass at a point ``z``.  If ``T(F)`` is a
statistical functional, the influence function is

.. math::

   \operatorname{IF}(z; T, F)
   = \lim_{\varepsilon \downarrow 0}
     \frac{T((1-\varepsilon)F + \varepsilon \Delta_z) - T(F)}{\varepsilon},

where ``\Delta_z`` is a point mass at ``z``.

A bounded influence function means that a single very extreme observation has
limited first-order effect.  Classical mean and covariance have unbounded
influence.  Many robust M-estimators use radial weights that downweight large
Mahalanobis distances, reducing this sensitivity.

Gateaux derivative
------------------

The influence function is a special case of a **Gateaux derivative**.  The
Gateaux derivative measures the directional derivative of a functional ``T``
at distribution ``F`` in the direction of another signed measure ``H``:

.. math::

   D T(F)[H] = \lim_{\varepsilon \to 0}
   \frac{T(F + \varepsilon H) - T(F)}{\varepsilon}.

For influence functions, the direction is ``H = \Delta_z - F``.  This is the
formal way to ask: “What happens to the estimator if the data-generating
distribution is perturbed slightly toward a contaminating point?”

Gross-error contamination model
-------------------------------

A common robustness model writes the observed distribution as

.. math::

   F_\varepsilon = (1-\varepsilon)F_0 + \varepsilon G,

where ``F_0`` is the ideal distribution and ``G`` is arbitrary contamination.
Robust covariance estimators are designed to remain useful when
``\varepsilon`` is nonzero and ``G`` can generate outliers or heavy tails.

Breakdown point
---------------

The **breakdown point** is the smallest contamination fraction that can make an
estimator arbitrarily bad.  Classical covariance has a very low breakdown
point: one sufficiently extreme point can destroy it.

Subset estimators such as MCD are designed for high-breakdown behavior under
separable contamination.  Heavy-tail M-estimators such as Tyler, Student-t, and
Cauchy scatter are instead designed to reduce radial sensitivity and stabilize
shape/covariance estimation under non-Gaussian tails.

Robust distances
----------------

Given a robust location ``\hat\mu`` and scatter/covariance estimate
``\hat\Sigma``, robust squared Mahalanobis distances are

.. math::

   d_i^2 = (x_i - \hat\mu)^T \hat\Sigma^{-1} (x_i - \hat\mu).

These distances are the basis for many diagnostics in ``robustcov``:

* ranked distance profiles;
* QQ plots against a Gaussian reference;
* empirical or chi-square thresholds;
* top-anomaly tables;
* radial kurtosis and tail diagnostics.

Why different estimators behave differently
-------------------------------------------

``FastMCD`` is most appropriate when the data contain a mostly clean central
subset plus separable outliers.  It searches for a subset with small covariance
determinant and then reweights observations.

``TylerShape`` estimates shape, not covariance scale.  It is useful for
elliptical heavy-tailed data but can be unstable without regularization in
small-sample or high-dimensional regimes.

``StudentTScatter`` and ``RegularizedCauchy`` use radial weights of the form

.. math::

   \widehat\Sigma \leftarrow
   \frac{1}{n}\sum_{i=1}^n w(d_i^2)(x_i-\hat\mu)(x_i-\hat\mu)^T
   + \text{shrinkage},

where large distances receive smaller weights.  Cauchy-type weights are more
aggressive in very heavy tails; Student-t weights are smoother.


Geodesic convexity on covariance matrices
-----------------------------------------

Covariance and scatter matrices live in the cone of symmetric positive-definite
matrices, not in ordinary Euclidean space.  This space has useful curved
geometries.  A common affine-invariant geodesic between two positive-definite
matrices ``A`` and ``B`` is

.. math::

   \gamma(t) = A^{1/2}
   \left(A^{-1/2} B A^{-1/2}\right)^t
   A^{1/2},
   \qquad 0 \le t \le 1.

A function ``f`` on positive-definite matrices is **geodesically convex** if

.. math::

   f(\gamma(t)) \le (1-t) f(A) + t f(B).

This matters because several robust scatter objectives are not ordinary convex
in the matrix entries but become well behaved under the right positive-definite
geometry.  Tyler-type and Wiesel/KL-regularized scatter estimators are often
studied through this lens: geodesic convexity helps explain uniqueness, stable
fixed-point algorithms, and why the optimization should be viewed as occurring
on the SPD cone rather than in flat Euclidean space.

For users, the practical message is simple: regularized Tyler/Cauchy-style
scatter estimators are not just ad-hoc reweighting rules.  They are connected to
a geometry where covariance matrices remain positive definite and where
regularization can improve existence, uniqueness, and conditioning in
small-sample regimes.


Small samples, large samples, and asymptotic comfort
----------------------------------------------------

Classical covariance theory is often introduced through asymptotics: as the
number of observations ``n`` grows with a fixed number of variables ``p``, the
sample covariance converges to the population covariance under suitable moment
conditions.  That theory is useful, but it can be a poor guide for the regimes
that motivate ``robustcov``.

Two common failure modes are easy to miss.

First, in small-sample or high-dimensional data, ``p`` may be close to ``n`` or
even larger than ``n``.  The empirical covariance can then become ill-conditioned
or singular, and inverse-covariance distances can be unstable.  A method that is
asymptotically consistent may still be unusable at the sample size available in
a real biomedical, finance, sensor, or embedding problem.

Second, asymptotic normal approximations usually rely on finite moments and on a
model that is not too contaminated.  Heavy tails, leverage points, clustered
outliers, and regime shifts can dominate the empirical covariance long before
``n`` is large enough for the asymptotic story to help.  In these settings the
wrong estimator can look precise while actually estimating the tail behavior or
outlier geometry rather than the central scatter.

Regularization and robust radial weights are not cosmetic additions.  They are
finite-sample safeguards: shrinkage controls conditioning, Tyler/Cauchy/Student-t
weights limit radial influence, and robust-distance diagnostics reveal when a
Gaussian chi-square threshold is not credible.  The right question is therefore
not only “is the estimator consistent as ``n`` goes to infinity?” but also “is it
stable, invertible, and interpretable at the ``n`` and ``p`` I actually have?”

Practical reading guide
-----------------------

When reading benchmark pages, ask:

* Is the task truly rare-anomaly detection, or is it closer to classification?
* Is the anomaly signal covariance-shaped or mostly categorical/nonlinear?
* Does the method win quality, runtime, interpretability, or only one of them?
* Is the robust covariance score best used alone, or as an additional feature?

A method that appears to win every benchmark should be treated skeptically.
``robustcov`` reports strong wins, competitive results, and losses separately
because different anomaly problems reward different geometry.
