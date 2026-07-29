Robust principal component analysis
===================================

``RobustPCA`` computes principal components from a robust location and scatter
estimate.  Use it when a few unusual rows, heavy tails, or leverage points would
otherwise pull ordinary PCA toward the wrong directions.


.. note::

   This page describes scatter-based robust PCA. If your data model is one
   matrix equal to a low-rank component plus sparse gross cell corruption, use
   :doc:`principal_component_pursuit` instead.

The class follows the familiar PCA interface: fit a model, project new data,
and reconstruct observations from the retained components.  It also reports two
distances that are useful when PCA is used for diagnostics.


## Mathematical formulation

Let :math:`x_1,\ldots,x_n \in \mathbb{R}^p` denote the observations.
`RobustPCA` first fits a robust location estimate
:math:`\widehat{\mu}` and a robust scatter estimate
:math:`\widehat{\Sigma}`.

The scatter matrix is decomposed as

.. math::

# \widehat{\Sigma}

V \Lambda V^\mathsf{T},

where

.. math::

# \Lambda

\operatorname{diag}(\lambda_1,\ldots,\lambda_p),
\qquad
\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_p,

and the columns of :math:`V` are orthonormal eigenvectors.

If :math:`q` components are retained, let

.. math::

V_q = [v_1,\ldots,v_q].

The robust principal-component scores of an observation :math:`x` are

.. math::

z = V_q^\mathsf{T}(x-\widehat{\mu}).

The corresponding reconstruction is

.. math::

# \widehat{x}

\widehat{\mu} + V_qz.

The explained-variance ratio of component :math:`j` is

.. math::

# r_j

\frac{\lambda_j}
{\sum_{k=1}^{p}\lambda_k}.

The difference from ordinary PCA is the source of
:math:`\widehat{\mu}` and :math:`\widehat{\Sigma}`. Ordinary PCA uses the
sample mean and empirical covariance. `RobustPCA` obtains them from the
selected robust scatter estimator, which reduces the influence of contaminated
or heavy-tailed observations.


Fit and transform
-----------------

.. code-block:: python

   import robustcov as rc

   pca = rc.RobustPCA(
       n_components=5,
       estimator=rc.RegularizedCauchy(alpha=0.10),
   ).fit(X_train)

   Z_train = pca.transform(X_train)
   Z_test = pca.transform(X_test)
   X_reconstructed = pca.inverse_transform(Z_test)

``n_components`` accepts an integer, ``None`` to keep every direction, or a
fraction such as ``0.95`` to retain enough components to explain that share of
the robust variance.

Whitening
---------

With ``whiten=True``, each retained score is divided by the square root of its
robust eigenvalue:

.. code-block:: python

   pca = rc.RobustPCA(
       n_components=0.95,
       estimator=rc.FastMCD(quality="balanced", random_state=0),
       whiten=True,
   ).fit(X_train)

   Z_white = pca.transform(X_test)

Whitening depends on the scale of the fitted scatter estimate.  For shape-only
estimators such as Tyler's estimator, choose a scale correction when the
absolute scale matters.

Score distance and orthogonal distance
--------------------------------------

PCA outliers do not all look the same.  An observation can be extreme while
still following the fitted subspace, or it can depart from the subspace
altogether.

``score_distances(X)`` measures distance from the robust center inside the
retained subspace.  Large values often correspond to leverage points or large
factor moves.

For an observation :math:`x`, the score distance is

.. math::

   \operatorname{SD}(x)
   =
   \left(
   \sum_{j=1}^{q}
   \frac{
      \left[v_j^\mathsf{T}(x-\widehat{\mu})\right]^2
   }{\lambda_j}
   \right)^{1/2}.

``orthogonal_distances(X)`` measures reconstruction error outside the retained
subspace.  Large values indicate a direction that the fitted components do not
explain well.

.. code-block:: python

   score_distance = pca.score_distances(X_test)
   orthogonal_distance = pca.orthogonal_distances(X_test)
   outlier_map = pca.outlier_map(X_test)

The orthogonal distance is

.. math::

   \operatorname{OD}(x)
   =
   \left\|
   (I-V_qV_q^\mathsf{T})(x-\widehat{\mu})
   \right\|_2.

``outlier_map`` returns these two quantities as separate columns.  Keeping them
separate usually gives a more useful diagnosis than combining them into one
number.

.. code-block:: python

   rc.plot_robust_pca_outlier_map(
       pca,
       X_test,
       output_path="robust_pca_outlier_map.png",
       show=False,
   )

Choosing a scatter estimator
----------------------------

``RobustPCA`` clones and fits the estimator passed through ``estimator``.  A
compatible estimator must implement ``fit(X)`` and expose a finite square
``covariance_`` matrix.  If it also exposes ``location_``, that location is used
for centering; otherwise the arithmetic mean is used.

A few common choices are:

.. code-block:: python

   candidates = [
       rc.FastMCD(quality="balanced", random_state=0),
       rc.RegularizedCauchy(alpha=0.10),
       rc.StudentTScatter(df=3, alpha=0.05),
       rc.RegularizedTyler(alpha=0.10, scale_correction="radial_median"),
   ]

   models = [
       rc.RobustPCA(n_components=10, estimator=est).fit(X_train)
       for est in candidates
   ]

``FastMCD`` is a good starting point when the sample is comfortably larger than
the feature dimension and contamination is sparse.  A regularized estimator is
usually safer when ``p`` is close to, or larger than, ``n``.

Bootstrap stability
-------------------

A single fit does not show how much the loadings or retained subspace would
change under resampling.  :doc:`subspace_stability` bootstraps the PCA fit,
aligns the loading matrices, and reports loading intervals and principal-angle
distributions.

Worked examples
---------------

The gallery includes three examples built around different data structures:

* :doc:`Production embedding monitoring <gallery/robust_pca_embedding_monitoring>`
  tracks movement within an embedding subspace and detects vectors that leave
  it.
* :doc:`Yield-curve factor extraction <gallery/robust_pca_yield_curve>`
  estimates level-, slope-, and curvature-like directions in the presence of
  stress days and quote errors.
* :doc:`Cross-asset market-risk decomposition <gallery/robust_pca_market_risk>`
  separates broad factor moves from instrument-specific dislocations.

The examples use deterministic synthetic data, so they can run in CI without
network access.

.. code-block:: bash

   python docs/generate_gallery_assets.py --only \
       robust_pca_embedding_monitoring \
       robust_pca_yield_curve \
       robust_pca_market_risk

What this class implements
--------------------------

This is scatter-based robust PCA: the class eigendecomposes a robust scatter
matrix.  It is not the low-rank-plus-sparse decomposition sometimes called
"robust PCA," and it is not a full projection-pursuit ROBPCA implementation.
Those methods solve different problems and should be exposed under separate
APIs.
