Cellwise regularized covariance
===============================

``CellRCov`` estimates a full covariance matrix when the data may contain
isolated bad cells, abnormal complete rows, missing entries, and more variables
than observations.  It combines the low-rank fit from :doc:`cellwise_pca` with
a regularized covariance estimate for the remaining variation.

Use ``CellMCD`` when the dimension is modest and a direct cellwise likelihood
fit is practical.  Use ``CellRCov`` when :math:`p` is close to or greater than
:math:`n`, provided that a defensible low-rank approximation is available.

Covariance decomposition
------------------------

Let :math:`Z` be the data after robust columnwise centering and scaling.  For a
chosen rank :math:`q`, ``CellPCA`` gives fitted points and residuals

.. math::

   Z_i = \widehat Z_i + E_i,
   \qquad
   \widehat Z_i = \widehat\mu + V t_i.

This is a computational decomposition, not a claim that the data were generated
by a uniquely identifiable signal-plus-noise model.  The covariance is estimated
as the sum of two terms:

.. math::

   \widehat\Sigma_Z
   =
   \widehat\Sigma_{\mathrm{fit}}
   +
   \widehat\Sigma_{\mathrm{res}}^{(\delta)}.

Fitted-subspace covariance
--------------------------

The CellPCA scores can still contain casewise leverage points.  ``CellRCov``
therefore fits a robust covariance estimator to the score matrix.  If
:math:`\widehat\Sigma_T` is the robust score covariance and the columns of
:math:`V` are the loading vectors, then

.. math::

   \widehat\Sigma_{\mathrm{fit}}
   =
   V\widehat\Sigma_TV^T.

The default score estimator is ``FastMCD`` with a 75% support fraction.  A
configured alternative can be passed through ``score_estimator``.

Weighted residual covariance
----------------------------

CellPCA supplies a cell weight :math:`w_{ij}^{\mathrm{cell}}`, a case weight
:math:`w_i^{\mathrm{case}}`, and a missingness indicator :math:`d_{ij}`.  The
combined cell weight is

.. math::

   a_{ij}
   =
   d_{ij}w_{ij}^{\mathrm{cell}}.

Large or missing residual cells are therefore replaced by their fitted value,
which is equivalent to setting the corresponding residual to zero.  In matrix
form, the residual contribution is based on weighted outer products

.. math::

   \widehat\Sigma_{\mathrm{res}}
   =
   \frac{1}{c n}
   \sum_{i=1}^{n}
   w_i^{\mathrm{case}}
   D_i E_iE_i^T D_i,

where :math:`D_i=\operatorname{diag}(a_{i1},\ldots,a_{ip})` and :math:`c`
corrects for the effective fraction of observed and downweighted cells.

The implementation uses a scalar effective-weight correction that preserves a
positive-semidefinite finite-sample residual matrix.  This plays the same role
as the effective-pair normalization in the reference method, but numerical
identity with the reference software is not claimed.

Residual regularization
-----------------------

The residual covariance is shrunk toward its diagonal:

.. math::

   \widehat\Sigma_{\mathrm{res}}^{(\delta)}
   =
   (1-\delta)\widehat\Sigma_{\mathrm{res}}
   +
   \delta\operatorname{diag}
   (\widehat\Sigma_{\mathrm{res}}),
   \qquad 0\leq\delta\leq1.

This retains the residual variances while stabilizing the off-diagonal
covariances.  A fixed value can be supplied directly.  With
``residual_shrinkage="auto"``, the low-rank fit and weights are held fixed and
the shrinkage value is chosen by row-split cross-validation.

Finally, if :math:`D` contains the original robust marginal scales, the
covariance in the original units is

.. math::

   \widehat\Sigma_X
   =
   D\widehat\Sigma_ZD.

Using the estimator
-------------------

.. code-block:: python

   import robustcov as rc

   model = rc.CellRCov(
       n_components=4,
       residual_shrinkage="auto",
       shrinkage_grid=[0.0, 0.25, 0.5, 0.75, 1.0],
   ).fit(X)

   print(model.covariance_)
   print(model.residual_shrinkage_)
   print(model.cell_outlier_mask_)

The fitted decomposition is available through
``fitted_covariance_``, ``residual_covariance_``, and
``residual_covariance_regularized_``.  ``corrected_data_`` replaces missing and
flagged cells with low-rank predictions.

Two distance components
-----------------------

The total corrected-data Mahalanobis distance is returned by
``mahalanobis(X)``.  The two parts of the decomposition can also be inspected
separately:

.. code-block:: python

   along_subspace = model.subspace_distances(X_new)
   outside_subspace = model.residual_distances(X_new)
   map_values = model.outlier_map(X_new)

A large subspace distance indicates an unusual position within the retained
factor structure.  A large residual distance indicates variation that is not
well represented by that structure.

Choosing the rank
-----------------

The reference cellRCov procedure uses robust parallel analysis to select the
rank.  This implementation requires an explicit integer ``n_components``.
Choose it from domain knowledge, held-out reconstruction, or a separate
stability analysis.  Trying several nearby ranks is advisable when the
covariance estimate is sensitive to this choice.

Scope and limitations
---------------------

``CellRCov`` is coordinate dependent because the definition of a contaminated
cell is tied to the measured variables.  It assumes that the missingness
mechanism is non-informative after conditioning on the observed data.

The method is most useful when dominant low-dimensional structure exists.  If
there is no credible low-rank component, a pairwise robust covariance method or
a regularized heavy-tail estimator may be easier to justify.

The reference implementation uses robust M-scales, reference cellPCA, DetMCD,
robust parallel analysis, and a specific cross-validation construction.  This
package uses its own CellPCA and FastMCD implementations and a fixed-weight
cross-validation approximation.  It should therefore be treated as a
package-native implementation of the covariance decomposition, not as a claim
of exact reference parity.

Worked example
--------------

:doc:`High-dimensional covariance with mixed cellwise and casewise contamination
<gallery/cellrcov_high_dimensional>` compares CellRCov with Ledoit-Wolf,
regularized Cauchy scatter, and MRCD when :math:`p>n` and entries are also
missing.
