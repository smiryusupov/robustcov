Distributionally robust PCA
============================

``DistributionallyRobustPCA`` is an experimental estimator for principal
subspaces that must remain useful after a structured change in the data
distribution.  It is not a renamed outlier-resistant PCA method.  The fitted
criterion is a worst-case expected squared reconstruction loss over a weighted
type-2 Wasserstein ambiguity set around the centered empirical distribution.

The implementation follows the weighted-geometry formulation of Xu, Wood, and
Yang and the Wasserstein DRO-PCA formulation studied by Wang, Liu, and Chen.
See the entries for ``DistributionallyRobustPCA`` in
:doc:`methods_and_references` for the exact citations and implementation notes.

Experimental status
-------------------

Import the estimator from the experimental namespace:

.. code-block:: python

   from robustcov.experimental import DistributionallyRobustPCA

The API, default geometry, and radius calibration may change while the method
is validated on additional real distribution-shift problems.  It is not
exported from the top-level ``robustcov`` namespace.

Mathematical contract
---------------------

For an orthogonal residual projector :math:`Q`, a transport matrix
:math:`G \succ 0`, empirical distribution :math:`\widehat P_n`, and ambiguity
radius :math:`\delta`, the estimator evaluates

.. math::

   \sup_{P:\; W_{2,G}(P,\widehat P_n) \leq \delta}
   \mathbb{E}_{P}\left[\lVert Q(X-\mu)\rVert_2^2\right].

With ``formulation="exact"``, candidate subspaces are ranked using the exact
scalar dual of this worst-case risk.  The optimizer searches a deterministic,
data-adaptive path of candidate subspaces.  Therefore, **exact** describes the
ambiguity-set risk evaluated for each candidate; it does not claim a global
solution of the non-convex Grassmann optimization problem.

With ``formulation="surrogate"``, candidates are ranked by the spectral
surrogate

.. math::

   \sqrt{\operatorname{tr}(\widehat\Sigma Q)}
   + \delta\sqrt{\lambda_{\max}(G^{-1/2}QG^{-1/2})}.

The fitted object exposes both the exact risk and the squared surrogate bound.

Identity geometry is a required control
---------------------------------------

For homogeneous geometry, ``transport_geometry="identity"``, the ambiguity
penalty is the same for every rank-fixed residual subspace.  The selected
subspace must therefore equal ordinary PCA.  ``robustcov`` treats that reduction
as a regression invariant and exposes the identity case as a control rather
than presenting it as a new PCA estimator.

A useful DRO fit requires an anisotropic geometry that states which
perturbations are plausible or inexpensive.  Available choices are:

``"residual"``
   Construct an inverse-variance geometry from directions outside the initial
   PCA block.  This is the default structured-shift adaptation.

``"pca_block"``
   Construct the inverse-variance geometry from directions inside the initial
   PCA block.

``"identity"``
   Homogeneous control that recovers ordinary PCA.

``transport_matrix=G``
   Use a supplied symmetric positive-definite geometry.  Positive scalar
   multiples are normalized to the same geometry convention.

Radius calibration
------------------

A numeric ``radius`` is used directly in the data's measurement units.
``radius="sqrt_n"`` uses

.. math::

   \delta_n = c\sqrt{\overline{\operatorname{var}}(X)/n}.

This is a transparent, scale-equivariant heuristic.  It is **not** labelled as
the robust Wasserstein profile-inference calibration from the cited paper.
Users making scientific claims should report the radius, geometry, candidate
path, and a sensitivity analysis.

Example
-------

.. code-block:: python

   from robustcov.experimental import DistributionallyRobustPCA

   model = DistributionallyRobustPCA(
       n_components=2,
       radius=2.5,
       transport_geometry="residual",
       formulation="exact",
   ).fit(X_train)

   scores = model.transform(X_target)
   target_errors = model.reconstruction_error(X_target)

   print(model.exact_worst_case_risk_)
   print(model.surrogate_risk_bound_)
   print(model.selected_gamma_)
   print(model.ambiguity_set_)

Important fitted diagnostics include:

``transport_matrix_``
   Normalized SPD transport geometry used by the ambiguity set.

``radius_``
   Effective Wasserstein radius.

``exact_worst_case_risk_``
   Exact scalar-dual risk for the selected candidate.

``surrogate_risk_bound_``
   Squared spectral surrogate for the selected candidate.

``candidate_results_``
   Exact and surrogate values for every distinct path candidate.

``selected_gamma_``
   Path multiplier selected by the requested formulation.

When to use it
--------------

Use distributionally robust PCA when the primary concern is a plausible change
between the training and deployment distributions, such as feature-specific
measurement degradation or structured covariance shift.  Use other package
families when the data problem is different:

.. list-table::
   :header-rows: 1

   * - Problem
     - Prefer
   * - Rowwise outliers or heavy tails
     - ``RobustPCA`` or ``DensityPowerRobustPCA``
   * - Cellwise corruption and missing entries
     - ``CellPCA`` or ``SparseCellPCA``
   * - Structured train-to-target distribution shift
     - ``DistributionallyRobustPCA``

Data-drift monitoring
---------------------

DRO-PCA can serve as the reference subspace in a drift-monitoring workflow, but
the fitted worst-case expected risk is not itself a per-window alarm threshold.
Calibrate a window statistic on independent reference batches, then monitor
mean reconstruction risk and residual feature contributions.  The runnable
example distinguishes an anticipated geometry-aligned covariance change from an
off-geometry drift:

:doc:`DRO-PCA data-drift monitoring <gallery/distributionally_robust_pca_drift_monitoring>`

Validation should use held-out target-distribution reconstruction or downstream
performance.  Training reconstruction alone cannot establish distributional
robustness.
