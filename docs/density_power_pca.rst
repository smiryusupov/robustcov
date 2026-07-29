Density-power robust PCA
========================

``DensityPowerRobustPCA`` fits a low-rank model directly with a
Gaussian density-power-divergence loss.  It is useful when ordinary PCA is
pulled toward a relatively small number of large reconstruction errors and a
fixed component count is available.

Unlike :class:`~robustcov.RobustPCA`, it does not first estimate a full scatter
matrix.  Scores and loadings are updated together by robust alternating
regressions.  This can be attractive when :math:`p` is large or when corruption
is concentrated in individual entries of the data matrix.

Basic use
---------

.. code-block:: python

   import robustcov as rc

   pca = rc.DensityPowerRobustPCA(
       n_components=5,
       alpha=0.30,
       max_iter=100,
   ).fit(X)

   scores = pca.transform(X)
   reconstructed = pca.reconstruct(X)
   cell_weights = pca.cell_weights(X)

The rank is explicit.  The current implementation does not select
``n_components`` automatically.

Working model and loss
----------------------

After estimating a robust location :math:`\widehat\mu`, the centered data are
represented as

.. math::

   x_{ij}-\widehat\mu_j
   \approx
   a_i^\mathsf{T}b_j + \varepsilon_{ij},

where :math:`a_i\in\mathbb{R}^q` is the score vector for row :math:`i` and
:math:`b_j\in\mathbb{R}^q` contains the loadings for feature :math:`j`.

For a residual :math:`r=y-c^\mathsf{T}d`, the Gaussian DPD contribution is

.. math::

   V_\alpha(r;\sigma^2)
   =
   (2\pi)^{-\alpha/2}\sigma^{-\alpha}
   \left[
      \frac{1}{\sqrt{1+\alpha}}
      -\frac{1+\alpha}{\alpha}
       \exp\left(-\frac{\alpha r^2}{2\sigma^2}\right)
   \right],

for :math:`\alpha>0`.  Its fixed-point weight is

.. math::

   w_{ij}
   =
   \exp\left(-\frac{\alpha r_{ij}^2}{2\sigma^2}\right).

At :math:`\alpha=0`, the limiting fit is ordinary Gaussian least squares.
Increasing :math:`\alpha` downweights large residuals more rapidly, with a
corresponding loss of clean-sample efficiency.

Alternating updates
-------------------

With scores fixed, every feature loading is updated by weighted least squares:

.. math::

   b_j
   =
   \left(A^\mathsf{T}W_jA+\lambda I\right)^{-1}
   A^\mathsf{T}W_jx_{\cdot j}.

With loadings fixed, each row score is updated similarly:

.. math::

   a_i
   =
   \left(B^\mathsf{T}W_iB+\lambda I\right)^{-1}
   B^\mathsf{T}W_ix_i.

The residual variance uses the Gaussian DPD fixed-point equation

.. math::

   \widehat\sigma^2
   =
   \frac{\operatorname{mean}(w_{ij}r_{ij}^2)}
        {\operatorname{mean}(w_{ij})
          -\alpha(1+\alpha)^{-3/2}}.

The loading matrix is re-orthogonalized between updates, and the final fitted
low-rank matrix is put into a canonical SVD form.  The reported component
eigenvalues are :math:`s_k^2/n`, where :math:`s_k` is a fitted singular value.

Location and initialization
---------------------------

The default center is the geometric median, which is orthogonally equivariant.
The default initialization clips each marginal coordinate at four robust scales
before computing an SVD.  These choices reduce the chance that the first
iteration is already dominated by extreme cells.

The following alternatives are available:

.. code-block:: python

   rc.DensityPowerRobustPCA(
       n_components=5,
       location="coordinate_median",
       init="winsorized_svd",
   )

   rc.DensityPowerRobustPCA(
       n_components=5,
       alpha=0.0,
       location="mean",
       init="svd",
   )

The second configuration is useful as a numerical check against ordinary PCA.

Diagnostics
-----------

The fitted cell weights and residuals are available as

.. code-block:: python

   pca.weights_
   pca.cell_outlier_scores_       # 1 - weights
   pca.row_outlier_scores_        # row average of 1 - weights
   pca.residuals_
   pca.residual_scale_

The usual robust PCA distances are also provided:

.. code-block:: python

   score_distance = pca.score_distances(X)
   orthogonal_distance = pca.orthogonal_distances(X)
   outlier_map = pca.outlier_map(X)

``plot_robust_pca_outlier_map`` accepts the fitted estimator directly.

Choosing alpha
--------------

There is no universally optimal value.  A practical starting grid is

.. code-block:: python

   [0.05, 0.10, 0.20, 0.30, 0.50]

Small values remain close to ordinary PCA.  Values around 0.2--0.4 often give a
useful compromise in synthetic contamination studies, but the result depends
on signal strength, noise scale, and the type of corruption.  Compare subspace
stability, reconstruction on held-out clean data, and the sensitivity of the
result across a modest grid.

Limitations and implementation scope
------------------------------------

The estimator assumes a fixed low rank and currently requires complete finite
input.  Median imputation can be used before fitting, but ``CellPCA`` is the
better starting point when missing values are central to the problem.

The algorithm follows the Gaussian DPD alternating-regression formulation of
Roy, Basu, and Ghosh.  Their reference rSVDdpd implementation uses a different
normalization sequence and additional tuning utilities.  ``robustcov`` uses
block weighted-least-squares updates, QR reparameterization, and final SVD
canonicalization, so numerical identity with that software is not claimed.

See also
--------

* :doc:`Robust PCA from a scatter estimator <robust_pca>`
* :doc:`Cellwise and casewise robust PCA <cellwise_pca>`
* :doc:`Sparse cellwise robust PCA <sparse_cellwise_pca>`
* :doc:`Density-power PCA example <gallery/density_power_pca>`
