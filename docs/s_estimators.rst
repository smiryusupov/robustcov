Deterministic S and MM scatter estimators
=========================================

``DetS`` and ``DetMM`` estimate multivariate location and scatter without
random subset sampling.  They are intended for low- and moderate-dimensional
problems where entire observations may be contaminated and a smooth
high-breakdown estimator is preferable to a hard retained subset.

Use ``DetS`` when breakdown protection is the main concern.  Use ``DetMM`` when
the same robust starting point is desired with better efficiency near a
Gaussian model.

Basic use
---------

.. code-block:: python

   import robustcov as rc

   dets = rc.DetS(
       breakdown=0.50,
       max_iter=100,
   ).fit(X)

   detmm = rc.DetMM(
       breakdown=0.50,
       efficiency=0.95,
       max_iter=100,
   ).fit(X)

   print(dets.location_)
   print(dets.covariance_)
   print(detmm.weights_)

Both classes expose ``location_``, ``covariance_``, ``precision_``, squared
Mahalanobis ``distances_``, continuous ``weights_``, and a Boolean ``support_``
indicating which observations receive nonzero bisquare weight.

S-estimator
-----------

Write the scatter matrix as

.. math::

   \Sigma = \sigma^2 \Gamma,
   \qquad |\Gamma|=1.

The multivariate S-estimator minimizes :math:`\sigma` subject to

.. math::

   \frac{1}{n}\sum_{i=1}^{n}
   \rho_0\!\left(
      \frac{
      \sqrt{(x_i-\mu)^\mathsf{T}\Gamma^{-1}(x_i-\mu)}}
      {\sigma}
   \right)
   = b.

The implementation uses Tukey's bisquare loss,

.. math::

   \rho_c(u)=
   \begin{cases}
   \frac{u^2}{2}-\frac{u^4}{2c^2}+\frac{u^6}{6c^4},
      & |u|\le c,\\
   \frac{c^2}{6}, & |u|>c,
   \end{cases}

with radial weight

.. math::

   w_c(u)=
   \begin{cases}
   \left(1-u^2/c^2\right)^2, & |u|<c,\\
   0, & |u|\ge c.
   \end{cases}

For dimension :math:`p`, ``breakdown`` determines the tuning constant
:math:`c_0`.  The normal-model expectation :math:`b=E[\rho_{c_0}(R)]`, with
:math:`R\sim\chi_p`, makes the S-scale consistent under a multivariate normal
model.  A breakdown value of 0.5 gives the most aggressive default weighting.

The I-step alternates:

#. an S-scale update;
#. bisquare radial weights;
#. a weighted location and covariance update;
#. determinant normalization of the shape matrix.

MM refinement
-------------

``DetMM`` first fits ``DetS`` and fixes its robust scale
:math:`\widetilde\sigma`.  It then minimizes

.. math::

   \frac{1}{n}\sum_{i=1}^{n}
   \rho_1\!\left(
      \frac{
      \sqrt{(x_i-\mu)^\mathsf{T}\Gamma^{-1}(x_i-\mu)}}
      {\widetilde\sigma}
   \right),
   \qquad |\Gamma|=1.

The second bisquare tuning constant is larger than the S tuning constant, so
moderately distant observations receive more weight.  The ``efficiency``
parameter calibrates nominal asymptotic **location efficiency** under the
multivariate normal model.  The default is 0.95.

The final MM covariance is

.. math::

   \widehat\Sigma_{MM}
   = \widetilde\sigma^2\widehat\Gamma_{MM}.

Deterministic starts
--------------------

The fit begins with robust marginal standardization followed by six
deterministic correlation and projection starts.  Each start is converted to a
central half-sample estimate, briefly refined, and the best candidates are
polished to convergence.

This construction follows the DetS idea but does not reproduce the exact six
DetMCD starts of the reference implementation.  The S-scale equation, bisquare
I-steps, and fixed-scale MM refinement follow the published formulation;
numerical identity with ``rrcov`` or FSDA is not claimed.

When to use these estimators
----------------------------

``DetS`` and ``DetMM`` are useful when:

* contamination affects complete rows;
* :math:`n` is comfortably larger than :math:`p`;
* deterministic, permutation-invariant fitting is desirable;
* smooth weighting is preferred to a binary MCD support;
* a robustness--efficiency continuum is scientifically useful.

They are not the right first choice when:

* :math:`\lceil n/2\rceil\le p`; use ``MRCD`` or a regularized M-estimator;
* individual cells are corrupted; use ``CellMCD`` or ``CellRCov``;
* the clean distribution is broadly heavy-tailed rather than contaminated by a
  minority of observations; compare ``StudentTScatter`` and
  ``RegularizedCauchy``;
* the inlier structure is nonlinear; consider ``KMRCD``.

Worked example
--------------

See :doc:`DetS and DetMM: robustness versus efficiency
<gallery/dets_detmm_tradeoff>`.
