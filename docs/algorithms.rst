
Algorithms
==========

This page gives the mathematical and practical description of the estimators used by
``robustcov``. The package focuses on robust covariance/scatter estimation and robust-distance
diagnostics, not on fitting a full probability model with density, sampler, AIC, or BIC.

Notation
--------

Let :math:`X = \{x_i\}_{i=1}^n`, with :math:`x_i \in \mathbb{R}^p`. A location estimate is
:math:`\hat\mu`, a covariance or scatter estimate is :math:`\hat\Sigma`, and robust squared
Mahalanobis distances are

.. math::

   d_i^2 = (x_i - \hat\mu)^T \hat\Sigma^{-1} (x_i - \hat\mu).

For shape-only estimators such as Tyler's estimator, the scale of :math:`\hat\Sigma` is not
identified by the estimating equation. ``robustcov`` normalizes shape matrices and optionally
applies a radial scale correction for diagnostics.

FastMCD / MinCovDet
-------------------

``FastMCD`` is the package's estimator for the classical contamination model: most observations
come from a compact elliptical bulk and a minority are outliers. It approximates the minimum
covariance determinant problem: find a subset :math:`H` of size :math:`h` whose empirical covariance
has small determinant.

.. math::

   H^* \approx \arg\min_{|H|=h} \det\left(
       \frac{1}{h-1} \sum_{i \in H} (x_i-\bar x_H)(x_i-\bar x_H)^T
   \right).

The raw subset location and covariance are

.. math::

   \hat\mu_H = \frac{1}{h}\sum_{i\in H} x_i,
   \qquad
   \hat\Sigma_H = \frac{1}{h-1}\sum_{i\in H}(x_i-\hat\mu_H)(x_i-\hat\mu_H)^T.

The FastMCD idea is the **C-step**. Starting from a candidate subset, compute Mahalanobis distances
with the current subset covariance and keep the :math:`h` observations with smallest distances. The
C-step has the monotonicity property that it does not increase the determinant under regularity
conditions.

.. code-block:: text

   random elemental starts
       ↓
   short C-steps
       ↓
   retain best determinant candidates
       ↓
   full C-step polishing
       ↓
   raw robust location/covariance
       ↓
   reweighting by robust distances
       ↓
   final location/covariance and support diagnostics

In ``robustcov``, the final covariance is computed on the selected/reweighted support. This is
important under contamination: rescaling the final covariance using all observations can reintroduce
outlier inflation. ``FastMCD`` is best when :math:`n \gg p` and outliers are separable. It is not the
right tool for :math:`p > n` covariance recovery or diffuse heavy tails.

Tyler shape estimator
---------------------

Tyler's estimator is a distribution-free shape estimator for elliptical data. It estimates the
shape matrix up to scale by solving the fixed-point equation

.. math::

   \hat S = \frac{p}{n}\sum_{i=1}^n
       \frac{z_i z_i^T}{z_i^T \hat S^{-1} z_i},
   \qquad z_i = x_i - \hat\mu,

with a normalization such as

.. math::

   \operatorname{tr}(\hat S) = p.

The radial weight is

.. math::

   w_i(d_i^2) = \frac{p}{d_i^2}.

This makes Tyler's estimator highly robust to radial outliers because observations with large
robust distances receive small weights. Since the estimator is shape-only, it is often paired with a
separate scale correction or used primarily for robust distances and shape diagnostics.

Regularized Tyler / KL Tyler / Wiesel Tyler
-------------------------------------------

When :math:`p` is close to :math:`n` or :math:`p > n`, unregularized scatter estimates can become
singular or unstable. ``RegularizedTyler`` shrinks the Tyler update toward a target matrix
:math:`T`, typically the identity or a diagonal target:

.. math::

   S_{\text{Tyler}} = \frac{p}{n}\sum_{i=1}^n
       \frac{z_i z_i^T}{z_i^T S^{-1} z_i},

.. math::

   S_{\text{new}} = (1-\alpha) S_{\text{Tyler}} + \alpha T,
   \qquad 0 \leq \alpha \leq 1.

The result is normalized after each update. Shrinkage improves conditioning and makes the estimator
usable in high-dimensional small-sample regimes. In the current MVP, ``KLRegularizedTyler`` and
``WieselTyler`` are documented aliases around this regularized Tyler prototype. They keep the API
space open for a future exact objective-specific implementation.

Geometry note.  Regularized Tyler and Wiesel-style estimators are often
understood through the geometry of the symmetric positive-definite cone.  Their
objectives can be geodesically convex under appropriate formulations, even when
they are not ordinary Euclidean-convex functions of the matrix entries.  This is
why the package documentation separates the fixed-point update used in the MVP
from stronger mathematical claims about an exact KL/Wiesel objective.  The
current implementation is pragmatic; future versions may expose objective-level
solvers once the exact formulation is stabilized.

Student-t scatter
-----------------

``StudentTScatter`` is an iteratively reweighted covariance estimator motivated by the multivariate
Student-t model with fixed degrees of freedom :math:`\nu`. Given squared robust distances
:math:`d_i^2`, it uses the radial weight

.. math::

   w_i(d_i^2) = \frac{\nu + p}{\nu + d_i^2}.

The weighted update is

.. math::

   S_{\text{M}} = \frac{1}{\sum_i w_i}\sum_{i=1}^n
       w_i z_i z_i^T,

followed by optional shrinkage

.. math::

   S_{\text{new}} = (1-\alpha)S_{\text{M}} + \alpha T.

Smaller :math:`\nu` means heavier tails and more aggressive downweighting. Unlike MCD, Student-t
scatter does not try to identify a hard subset. It is therefore useful when the whole data set is
heavy-tailed rather than clean data plus a clearly separated outlier cloud.

Regularized Cauchy
------------------

``RegularizedCauchy`` is the very-heavy-tail member of the same M-estimator family. It corresponds
to a Cauchy-like radial downweighting rule and shrinkage toward a stable target. In practice this is
the current flagship estimator for small-sample heavy-tail covariance recovery.

A simplified view is

.. math::

   w_i(d_i^2) \propto \frac{1 + p}{1 + d_i^2},
   \qquad
   S_{\text{new}} = (1-\alpha)S_{\text{Cauchy}} + \alpha T.

The benchmark gallery shows that this combination of aggressive radial downweighting and shrinkage
can strongly outperform empirical covariance, Ledoit-Wolf, OAS, and MCD when the data are very
heavy-tailed and :math:`p` is close to or larger than :math:`n`.

HellingerRegularizedTyler, experimental
---------------------------------------

``HellingerRegularizedTyler`` is intentionally marked experimental. It applies Tyler-like radial
weights with square-root-space shrinkage. It is useful for exploratory comparisons, but it should
not yet be cited as the exact optimizer of a specific Hellinger objective. The API label is
experimental until the objective and fixed-point update are finalized.

AutoRobustScatter
-----------------

``AutoRobustScatter`` is a practical selector. It fits a small candidate set and chooses an
estimator using a diagnostic or stability score.

.. code-block:: text

   candidate estimators
       ↓
   fit each candidate
       ↓
   compute convergence, condition, tail, and distance diagnostics
       ↓
   optionally compute split-sample stability
       ↓
   choose the lowest score

The diagnostic score combines convergence, finite covariance checks, condition-number penalties,
and tail diagnostics. The stability score adds split-sample scatter stability. This is not an oracle:
it is a pragmatic default for users who do not yet know whether Cauchy, Student-t, or Tyler is the
best fit.



Multimodal robust diagnostics
-----------------------------

A single robust covariance estimator is designed for a setting that is approximately one central
elliptical cloud plus contamination.  In a genuinely multimodal distribution there may be several
valid clouds:

.. math::

   X \sim \sum_{k=1}^K \pi_k F_k + \epsilon G,

where each :math:`F_k` is a legitimate local population and :math:`G` is contamination.  If a
single global covariance is fitted to this mixture, smaller valid modes may be assigned very large
robust distances and incorrectly flagged as outliers.

``ClusterRobustOutlierDetector`` is a pragmatic diagnostic for this case.  It is not a full robust
mixture model.  It uses a two-stage procedure:

.. code-block:: text

   cluster observations into K modes
       ↓
   fit a robust scatter estimator inside each cluster
       ↓
   score each point by distance to its assigned local cluster
       ↓
   flag points with large local robust distances

For an observation assigned to cluster :math:`c(i)`, the local score is

.. math::

   d_i^2 = (x_i - \hat\mu_{c(i)})^T
           \hat\Sigma_{c(i)}^{-1}
           (x_i - \hat\mu_{c(i)}).

This is useful when multiple clusters are valid but each cluster is locally elliptical.  It should
not be sold as a replacement for robust mixture modeling: there is no likelihood, no EM algorithm,
no automatic number-of-components selection, and no claim that the clustering step is itself robust.
Its purpose is to prevent a global robust covariance model from treating legitimate modes as
outliers.

A future experimental layer could add trimmed Gaussian mixtures or robust Student-t mixtures, but
that would move the package toward robust clustering.  The current feature stays within the package
scope: robust scatter plus interpretable diagnostics.

Robust-distance diagnostics
---------------------------

All estimators can be inspected through robust distances. ``robustcov`` reports radial kurtosis,
QQ-tail deviation, condition number, detected fraction, and distance-profile plots.

A useful normalized radial kurtosis diagnostic is

.. math::

   \kappa_r = \frac{\mathbb{E}[d^4]}{p(p+2)},

which is close to one for an ideal Gaussian elliptical model and larger for heavy tails or
outlier-contaminated data. In practice, radial kurtosis should be interpreted together with QQ
plots and the distance profile: high radial kurtosis can be a valid property of heavy-tailed data,
not necessarily estimator failure.

Estimator selection summary
---------------------------

.. list-table:: Practical estimator guidance
   :header-rows: 1

   * - Situation
     - Recommended estimator
     - Reason
   * - Separable outliers, :math:`n \gg p`
     - ``FastMCD``
     - robust subset/support estimation and classical outlier diagnostics
   * - Small sample, heavy tails, :math:`p \approx n` or :math:`p > n`
     - ``RegularizedCauchy``
     - aggressive radial downweighting plus shrinkage
   * - Smooth heavy-tailed covariance-like estimate
     - ``StudentTScatter``
     - softer radial weights than Cauchy
   * - Shape estimation under elliptical heavy tails
     - ``RegularizedTyler``
     - scale-free robust shape with shrinkage
   * - Unsure which heavy-tail estimator to use
     - ``AutoRobustScatter``
     - diagnostic or stability-based selection

References
----------

See :doc:`references` for the full bibliography. Key background includes Rousseeuw and Van Driessen
for FastMCD, Tyler for shape estimation, Wiesel for regularized robust covariance, and standard
Student-t/Cauchy M-estimation literature.
