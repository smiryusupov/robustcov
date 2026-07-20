Robust source separation and factor models
==========================================

``robustcov`` includes a shared blind-source-separation layer for independent
components, temporally correlated sources, and static latent factors.  These
estimators reuse the package's robust scatter geometry rather than presenting a
thin wrapper around a conventional ICA implementation.


Runnable examples
-----------------

The three workflows have separate scripts and detailed gallery pages:

* :doc:`Robust two-scatter ICA <gallery/ica_two_scatter>` — ``python examples/ica_two_scatter.py``
* :doc:`Robust SOBI <gallery/sobi_source_separation>` — ``python examples/sobi_source_separation.py``
* :doc:`Robust static factor model <gallery/robust_factor_model>` — ``python examples/robust_factor_model.py``

Run the complete source-separation group with:

.. code-block:: bash

   python examples/run_use_case_gallery.py --group ica

Two-scatter robust ICA
----------------------

``TwoScatterICA`` whitens observations with a robust scatter estimator and then
diagonalizes a second scatter matrix in the whitened coordinates.  Its default
second scatter is a winsorized radial fourth-moment construction whose
per-observation matrix contribution is bounded.  The method is useful when
sources have different marginal tail shapes.

.. code-block:: python

   import robustcov as rc

   ica = rc.TwoScatterICA(
       radial_clip_quantile=0.90,
       random_state=0,
   ).fit(X)

   sources = ica.sources_
   reconstructed = ica.inverse_transform(sources)

Set ``symmetrize=True`` to estimate the second scatter from pairwise
differences.  Symmetrization is helpful for asymmetric source distributions,
but it spreads an isolated bad row across many pairs; the default therefore
prioritizes direct row-contamination resistance.

SOBI and robust SOBI
--------------------

``SOBI`` separates temporally correlated sources by whitening a multivariate
time series and jointly diagonalizing lagged covariance matrices.  Gaussian
sources can be separated when their autocorrelation signatures differ.

``RobustSOBI`` replaces empirical whitening with Student-t scatter and uses
Huber-weighted lagged cross-scatter matrices.  This targets heavy-tailed series
and isolated temporal impulses.

.. code-block:: python

   sobi = rc.RobustSOBI(lags=20).fit(multichannel_series)
   latent_series = sobi.transform(multichannel_series)

   print(sobi.temporal_signatures_)
   print(sobi.off_diagonal_energy_)

The Jacobi joint diagonalizer supports ``backend='python'``, ``'cpp'``, and
``'auto'``.  The automatic backend uses the compiled implementation only after
the complete-SOBI benchmark clears the package's 1.5x acceleration gate.

Robust static factor models
---------------------------

``RobustFactorModel`` exposes loadings, factor scores, common components,
idiosyncratic residuals, and a covariance decomposition.

``method='kendall'`` estimates the loading subspace from the multivariate
spatial Kendall matrix.  Pair directions are bounded and the eigenspace remains
well defined without requiring ordinary covariance moments under an elliptical
factor model.

``method='huber'`` starts from the Kendall subspace and alternates batched
Huber-weighted regressions for factor scores and loadings.

.. code-block:: python

   factor_model = rc.RobustFactorModel(
       n_factors='auto',
       method='kendall',
       max_factors=8,
   ).fit(X)

   factors = factor_model.factor_scores_
   loadings = factor_model.loadings_
   common = factor_model.common_component_
   residual = factor_model.idiosyncratic_

Source-separation metrics
-------------------------

ICA and SOBI are identifiable only up to source permutation, sign, and scale.
Use ``minimum_distance_index`` as the primary recovery metric and
``amari_index`` as a familiar secondary metric.

.. code-block:: python

   mdi = rc.minimum_distance_index(model.unmixing_, true_mixing)
   amari = rc.amari_index(model.unmixing_, true_mixing)

Both are zero for exact recovery up to the unavoidable indeterminacies.

Assumptions and limitations
---------------------------

* ``TwoScatterICA`` requires source distributions that make the two scatter
  functionals distinguishable.  Sources with equal scatter signatures are not
  identifiable by this method alone.
* SOBI requires distinct temporal autocorrelation signatures across sources.
* Automatic factor-number selection uses a spatial-Kendall eigenvalue-ratio
  rule and should be checked with domain knowledge for weak factors.
* Complex-valued mixtures, convolutive mixtures, sparse ICA, distance-
  correlation ICA, and dynamic/tensor factor models remain future experimental
  extensions.
