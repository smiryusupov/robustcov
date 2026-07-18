Bootstrap stability for robust PCA
==================================

A fitted PCA model gives one set of loadings and eigenvalues.  It does not show
how much those quantities would change if the sample were collected again.
``SubspaceStability`` approximates that variation by resampling observations and
refitting the same PCA model.

The resampling design matters.  Independent rows can be sampled one at a time,
but financial returns, sensor streams, and repeated measurements usually carry
dependence that should be preserved.  The class therefore supports IID, block,
stationary, and cluster bootstrap designs.

The analysis answers two related questions:

* are the individual loading vectors repeatable;
* is the retained subspace repeatable, even when nearby components rotate or
  exchange order?

Basic use
---------

For independent observations, the default IID bootstrap is sufficient:

.. code-block:: python

   import robustcov as rc

   analysis = rc.SubspaceStability(
       pca=rc.RobustPCA(
           n_components=3,
           estimator=rc.FastMCD(random_state=0),
       ),
       n_resamples=200,
       confidence_level=0.95,
       random_state=0,
   ).fit(X)

   print(analysis.summary())
   print(analysis.loading_interval_lower_)
   print(analysis.loading_interval_upper_)
   print(analysis.max_principal_angle_degrees_)

Plot one component and the subspace-angle distribution:

.. code-block:: python

   rc.plot_subspace_stability(
       analysis,
       component=0,
       feature_names=feature_names,
       output_path="subspace_stability.png",
       show=False,
   )

Choosing a resampling design
----------------------------

``resampling="iid"``
   Sample individual rows with replacement.  Use this when observations can
   reasonably be treated as independent.

``resampling="moving_block"``
   Sample fixed-length contiguous blocks without wrapping around the end of the
   series.  Row order is treated as time order.

``resampling="circular_block"``
   Sample fixed-length contiguous blocks on a circle, so a block can continue
   from the last row back to the first.

``resampling="stationary"``
   Use the stationary bootstrap.  Consecutive rows remain in the same block
   with probability :math:`1-1/L`, where :math:`L` is the expected block
   length.  Otherwise a new starting row is drawn uniformly.  Block lengths are
   therefore geometrically distributed.

``resampling="cluster"``
   Sample complete groups with replacement.  This is appropriate for repeated
   measurements grouped by subject, site, account, machine, or experiment.

For an ordered multivariate series:

.. code-block:: python

   analysis = rc.SubspaceStability(
       pca=pca,
       n_resamples=300,
       resampling="stationary",
       block_length=20,
       random_state=0,
   ).fit(X_time_ordered)

For grouped observations:

.. code-block:: python

   analysis = rc.SubspaceStability(
       pca=pca,
       n_resamples=300,
       resampling="cluster",
       random_state=0,
   ).fit(X, groups=subject_id)

``sample_fraction`` applies to rows for IID and block methods.  Under cluster
resampling it applies to the number of unique clusters, and every row belonging
to a selected cluster is retained.

Block length
------------

The block length controls the amount of serial dependence retained in each
bootstrap sample.  Short blocks approach IID resampling and may understate
uncertainty.  Very long blocks preserve more dependence but provide fewer
independent block starts and can make the bootstrap distribution noisy.

When ``block_length=None``, the implementation uses
``ceil(n_samples**(1/3))``.  This is a convenient default, not an automatic
optimality result.  For serious inference, compare several plausible block
lengths or choose one with a method designed for the statistic and dependence
structure at hand.

Bootstrap construction
----------------------

Let :math:`V_q` contain the :math:`q` loading vectors fitted to the full data.
For bootstrap replicate :math:`b`, the selected resampling design produces an
index sequence :math:`I_b`.  The same PCA estimator is refitted to
:math:`X[I_b]`, giving :math:`V_q^{(b)}` and eigenvalues
:math:`\lambda_j^{(b)}`.

All features are resampled together.  Block sampling therefore preserves both
contemporaneous dependence among variables and local serial dependence among
observations inside each block.

The number of retained components is fixed to the value selected by the
full-data fit.  This keeps every replicate comparable when the original
``n_components`` was a variance threshold such as ``0.95``.

Principal angles
----------------

The singular values of

.. math::

   V_q {V_q^{(b)}}^T

are the cosines of the principal angles between the reference and bootstrap
subspaces.  If :math:`\sigma_j^{(b)}` is one of those singular values, then

.. math::

   \theta_j^{(b)} = \arccos\left(\sigma_j^{(b)}\right).

Small angles mean that the retained subspace changes little under resampling.
The largest angle in each replicate is available through
``max_principal_angle_degrees_``.  Projection-matrix distances are stored in
``projection_distance_samples_`` as a second rotation-invariant summary.

Loading alignment
-----------------

Eigenvectors have arbitrary signs.  They may also swap order or rotate when
several eigenvalues are close.  Loading intervals are therefore calculated only
after aligning each bootstrap basis with the full-data basis.

The default ``alignment="procrustes"`` chooses an orthogonal matrix
:math:`Q_b` that minimizes

.. math::

   \left\|Q_b V_q^{(b)} - V_q\right\|_F.

Two alternatives are available:

``alignment="sign_permutation"``
   Match components by absolute cosine similarity, then orient their signs.

``alignment="sign"``
   Preserve component order and correct signs only.

Procrustes alignment is usually the safest choice for subspace summaries.
Sign-and-permutation alignment can be easier to interpret when eigenvalues are
well separated and component labels have a fixed meaning.

Reported intervals
------------------

The fitted object stores central percentile intervals for:

* aligned loadings;
* retained eigenvalues;
* explained-variance ratios;
* principal angles.

For example, with ``confidence_level=0.95`` the endpoints are the 2.5th and
97.5th percentiles of the successful bootstrap fits.  ``stable_loading_mask_``
marks loading intervals that do not cross zero.

The fitted object also records the effective design:

.. code-block:: python

   analysis.resampling_
   analysis.block_length_
   analysis.bootstrap_sample_sizes_
   analysis.n_clusters_in_
   analysis.bootstrap_cluster_count_

These are bootstrap stability intervals, not universal guarantees.  Their
interpretation depends on the fitted model, the dependence assumptions, and the
chosen block or cluster design.

What dependence is and is not addressed
----------------------------------------

Block and stationary bootstrap sampling address dependence **between rows over
time**.  Because every row is resampled as one multivariate vector, they also
retain the cross-sectional dependence among observed variables inside each
selected time point.

PCA itself does not assume that latent factors are statistically independent.
Its fitted component scores are orthogonal, and therefore uncorrelated in the
fitted covariance geometry, but orthogonality is weaker than independence.  A
model seeking independent latent sources is an ICA or dynamic latent-factor
model rather than PCA.

Cluster resampling assumes that complete clusters are the natural exchangeable
units.  It does not model dependence between clusters, and a small number of
clusters can still produce unreliable intervals.

Other cautions
--------------

Individual loadings are weakly identified when adjacent eigenvalues are nearly
equal.  In that situation the same subspace can be represented by many rotated
component bases.  Use principal angles and projection distances rather than
interpreting narrow differences between component loadings.  The fitted
``relative_eigenvalue_gaps_`` attribute helps identify this case.

A bootstrap fit can fail if the selected scatter estimator becomes singular on
a resample.  Failed fits are recorded in ``failure_messages_`` and excluded.
The analysis stops if fewer than ``min_successful_resamples`` fits succeed.  A
regularized scatter estimator is often preferable for small or
high-dimensional samples.

Worked examples
---------------

:doc:`Bootstrap stability of yield-curve factors
<gallery/robust_pca_subspace_stability>` compares empirical and robust PCA on a
contaminated factor model.

:doc:`Dependent bootstrap stability for robust factors
<gallery/robust_pca_dependent_stability>` keeps the PCA estimator fixed and
compares IID with stationary resampling for a serially dependent multivariate
series.  The IID bootstrap produces visibly narrower loading and eigenvalue
uncertainty in that example.
