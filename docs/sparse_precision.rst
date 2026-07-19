Robust sparse precision matrices
================================

A covariance matrix describes marginal variation and association.  Its inverse,
the precision matrix, describes conditional association: for a Gaussian model,
features :math:`j` and :math:`k` are conditionally independent given the other
features when :math:`\Theta_{jk}=0`.

``RobustGraphicalLasso`` estimates a sparse precision matrix from a robust
scatter estimate.  This separates two decisions:

* the scatter estimator determines how observations or cells are downweighted;
* the graphical-lasso penalty determines which conditional associations are
  retained.

Basic use
---------

.. code-block:: python

   import robustcov as rc

   graph = rc.RobustGraphicalLasso(
       alpha="ebic",
       scatter_estimator=rc.CellMCD(
           alpha=0.75,
           min_samples_per_feature=None,
       ),
   ).fit(X)

   print(graph.alpha_)
   print(graph.n_edges_)
   print(graph.partial_correlation_)
   print(graph.edge_list(feature_names))

The default scatter estimator is ``RegularizedCauchy(alpha=0.10)``.  Pass
``CellMCD`` when isolated cells or missing values are the main concern, ``MRCD``
for rowwise contamination in a high-dimensional sample, or another fitted
``robustcov`` covariance estimator.

Penalized objective
-------------------

Let :math:`S` be the fitted robust scatter matrix.  The estimator solves

.. math::

   \widehat\Theta
   =
   \arg\min_{\Theta \succ 0}
   \left\{
     \operatorname{tr}(S\Theta)
     - \log\det(\Theta)
     + \alpha\sum_{j\ne k}|\Theta_{jk}|
   \right\}.

Only off-diagonal entries are penalized.  A zero off-diagonal entry removes the
corresponding edge from the fitted conditional-dependence graph.

By default, the optimization is performed on the robust correlation matrix.
If :math:`D` contains the robust marginal standard deviations and
:math:`R=D^{-1}SD^{-1}`, the standardized precision :math:`\Omega` is estimated
from :math:`R` and mapped back through

.. math::

   \widehat\Theta = D^{-1}\widehat\Omega D^{-1}.

This makes a single penalty more comparable across features with different
units.  Setting ``standardize=False`` applies the penalty directly in the
original units.

Partial correlations
--------------------

The fitted partial correlation between features :math:`j` and :math:`k` is

.. math::

   \rho_{jk\,\cdot\,-\{j,k\}}
   =
   -\frac{\widehat\Theta_{jk}}
          {\sqrt{\widehat\Theta_{jj}\widehat\Theta_{kk}}}.

``partial_correlation_`` stores this matrix and ``adjacency_`` stores the
non-zero precision pattern.  ``conditional_coefficients_`` contains the linear
coefficients implied by the precision matrix:

.. math::

   \mathbb{E}[X_j\mid X_{-j}]
   =
   \mu_j
   - \frac{1}{\Theta_{jj}}
     \Theta_{j,-j}(X_{-j}-\mu_{-j}).

Penalty selection with EBIC
---------------------------

With ``alpha="ebic"``, the estimator evaluates a geometric penalty path.  For
an estimated graph with :math:`|E|` undirected edges, it minimizes

.. math::

   \operatorname{EBIC}(\Theta)
   =
   n\left[\operatorname{tr}(S\Theta)-\log\det(\Theta)\right]
   + |E|\log n
   + 4\gamma |E|\log p.

``ebic_gamma`` controls the extra high-dimensional sparsity penalty.  The
selected path and scores are available as ``alphas_``, ``ebic_scores_``, and
``path_n_edges_``.  EBIC is a model-selection rule, not a guarantee that every
selected edge is scientifically meaningful.  Stability analysis and domain
knowledge remain important when the graph will be interpreted.

Numerical method
----------------

The package solves the graphical-lasso objective with an alternating direction
method of multipliers.  The positive-definite update is obtained from an
eigendecomposition, and the sparse update applies soft thresholding only to
off-diagonal entries.  If the final thresholded iterate has a small negative
eigenvalue from numerical error, its off-diagonal entries are shrunk toward
zero until positive definiteness is restored; this preserves the estimated zero
pattern.

The fitted object reports ``converged_``, ``n_iter_``, the objective path, and
primal and dual residual paths.  Increase ``max_iter`` or tighten the tolerances
when a high-accuracy numerical solution is required.

Relationship to other robust graph estimators
----------------------------------------------

This class is a **robust-scatter graphical lasso**.  Its robustness is inherited
from the estimator that produces :math:`S`.  The package now exposes the
spatial-sign SGLASSO objective separately through
:doc:`spatial_sign_precision`.  Robust CLIME, trimmed graphical lasso, and the
multivariate-t graphical lasso still use different objectives and are not
implemented here.

Example
-------

:doc:`Sparse market network with bad ticks <gallery/robust_graphical_lasso_market_network>`
compares empirical and cellwise-robust scatter inputs when a small proportion of
individual returns are corrupted.
