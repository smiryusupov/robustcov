Spatial-sign sparse precision matrices
======================================

``SpatialSignGraphicalLasso`` estimates a sparse conditional-association graph
from multivariate spatial signs.  It is intended for high-dimensional
elliptical data whose main difficulty is radial heavy tails rather than a small
number of corrupted coordinates.

The method discards the length of each centered observation and keeps only its
direction.  This makes the fitted graph insensitive to observation-specific
radial rescaling, but it also means that absolute covariance scale is not
identified.

Basic use
---------

.. code-block:: python

   import robustcov as rc

   graph = rc.SGLASSO(
       alpha="ebic",
       n_alphas=20,
       ebic_gamma=0.5,
   ).fit(X)

   print(graph.alpha_)
   print(graph.n_edges_)
   print(graph.partial_correlation_)
   print(graph.edge_list(feature_names))

``SGLASSO`` and ``SpatialSignSparsePrecision`` are aliases for
``SpatialSignGraphicalLasso``.

Spatial median and sign covariance
----------------------------------

Let :math:`\widehat\mu` be the sample spatial median,

.. math::

   \widehat\mu
   = \arg\min_{\mu\in\mathbb{R}^p}
     \sum_{i=1}^{n}\lVert x_i-\mu\rVert_2.

For a nonzero vector :math:`z`, define its spatial sign as

.. math::

   U(z)=\frac{z}{\lVert z\rVert_2}.

The sample spatial-sign covariance matrix is

.. math::

   \widehat S
   = \frac{1}{n}\sum_{i=1}^{n}
     U(x_i-\widehat\mu)U(x_i-\widehat\mu)^T.

Observations equal to the spatial median contribute the zero vector.  The
fitted object reports their count through ``zero_sign_count_``.

Under the high-dimensional elliptical assumptions studied by Lu and Feng,
:math:`p\widehat S` approximates a trace-normalized covariance shape.  The
resulting inverse is therefore a shape precision matrix, defined only up to a
common scale.  Graph support and partial correlations do not depend on that
common scale.

Sparse objective
----------------

The default objective follows the spatial-sign graphical-lasso proposal:

.. math::

   \widehat V
   = \arg\min_{V\succ0}
     \left\{
       \operatorname{tr}(p\widehat S V)
       - \log\det V
       + \alpha\lVert V\rVert_1
     \right\}.

``penalize_diagonal=True`` matches this full elementwise penalty.  Set it to
``False`` to use the more common graphical-lasso convention in which only
off-diagonal entries are penalized.

The optimization uses the same ADMM solver as ``RobustGraphicalLasso``.  The
fitted ``precision_`` is the sparse shape precision, and ``covariance_`` is its
normalized inverse shape.  ``partial_correlation_``, ``adjacency_``, and
``conditional_coefficients_`` have the same interpretation as in the existing
robust graphical-lasso API.

Penalty selection
-----------------

A fixed numerical penalty gives the clearest comparison across estimators:

.. code-block:: python

   graph = rc.SGLASSO(alpha=0.12).fit(X)

With ``alpha="ebic"``, the package evaluates a geometric path using

.. math::

   n\left[
      \operatorname{tr}(p\widehat S V)-\log\det V
   \right]
   + |E|\log n
   + 4\gamma |E|\log p.

This EBIC rule is a package addition.  The SGLASSO paper selects the penalty
using a separate validation sample and spatial-sign likelihood loss.  Users
with enough data can reproduce that strategy by fitting a numerical penalty
grid and evaluating each candidate on a held-out sign covariance matrix.

When the method is suitable
---------------------------

Use spatial-sign graphical lasso when:

* the observations are reasonably modeled by a centered elliptical family;
* radial magnitudes are extremely heavy-tailed or contain scale shocks;
* the graph of interest is sparse;
* :math:`p` is comparable to, or larger than, :math:`n`;
* graph support and partial correlations matter more than absolute covariance
  scale.

Prefer ``RobustGraphicalLasso`` with another scatter estimator when:

* bad values occur in individual cells—use ``CellMCD`` or ``CellRCov``;
* the contamination is a separated set of complete rows—consider ``FastMCD``
  or ``MRCD`` underneath the graph solver;
* the covariance scale is needed for likelihoods, simulation, or portfolio
  variance;
* the clean distribution is strongly non-elliptical.

Missing values
--------------

The published spatial-sign construction assumes complete rows.  The default
``missing_values="raise"`` enforces that assumption.  A practical
``missing_values="median"`` mode is available, but it is ordinary coordinate
median imputation and does not provide cellwise robustness or inherit the
published theory.

Example
-------

:doc:`Spatial-sign graph under radial heavy tails <gallery/spatial_sign_graphical_lasso>`
compares empirical, Cauchy-scatter, and spatial-sign graph estimates.  It also
shows the exact radial-rescaling invariance on a symmetric sample.
