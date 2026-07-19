
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

Deterministic S and MM estimators
---------------------------------

A multivariate S-estimator writes the scatter matrix as

.. math::

   \Sigma=\sigma^2\Gamma,\qquad |\Gamma|=1,

and minimizes :math:`\sigma` subject to

.. math::

   \frac{1}{n}\sum_{i=1}^{n}
   \rho_0\!\left(
      \frac{\sqrt{(x_i-\mu)^\mathsf{T}\Gamma^{-1}(x_i-\mu)}}{\sigma}
   \right)=b.

``DetS`` uses Tukey's bisquare loss

.. math::

   \rho_c(u)=
   \begin{cases}
   u^2/2-u^4/(2c^2)+u^6/(6c^4), & |u|\le c,\\
   c^2/6, & |u|>c,
   \end{cases}

whose radial weight is

.. math::

   w_c(u)=\left(1-u^2/c^2\right)^2\mathbf{1}(|u|<c).

For :math:`R\sim\chi_p`, the normal-consistency constant is
:math:`b=E[\rho_c(R)]`.  The requested breakdown value determines the S tuning
constant through :math:`b/\rho_c(\infty)`.  I-steps alternate an S-scale update,
bisquare weighting, a weighted location/covariance update, and determinant
normalization of :math:`\Gamma`.

``DetMM`` begins from ``DetS`` and fixes its robust scale
:math:`\widetilde\sigma`.  It then minimizes

.. math::

   \frac{1}{n}\sum_{i=1}^{n}
   \rho_1\!\left(
      \frac{\sqrt{(x_i-\mu)^\mathsf{T}\Gamma^{-1}(x_i-\mu)}}
           {\widetilde\sigma}
   \right),
   \qquad |\Gamma|=1.

The second bisquare cutoff is calibrated to a requested nominal location
efficiency under a Gaussian model.  Its larger cutoff gives moderately distant
observations more weight while the high-breakdown S-scale remains fixed.

The package uses six deterministic robust correlation/projection starts inspired
by DetS rather than the exact six DetMCD starts of the reference software.
Consequently, the estimating equations follow DetS/DetMM but numerical identity
with ``rrcov`` or FSDA is not claimed.  These estimators require
:math:`\lceil n/2\rceil>p`; use MRCD or a regularized M-estimator when the
central half-sample cannot have full-rank covariance.

Minimum Regularized Covariance Determinant
------------------------------------------

``MRCD`` extends the MCD subset idea to settings where the subset covariance is
singular or poorly conditioned, including :math:`p \geq h`.  The observations
are first standardized with coordinatewise medians and robust marginal scales:

.. math::

   u_i = D_X^{-1}(x_i-\nu_X),

where :math:`\nu_X` contains the marginal medians and :math:`D_X` is diagonal.
The default scale estimate is :math:`Q_n`.

For an :math:`h`-subset :math:`H`, let :math:`S_U(H)` be its covariance in the
standardized coordinates.  MRCD replaces that covariance by

.. math::

   K(H) = \rho T + (1-\rho)c_\alpha S_U(H),

where :math:`T` is a fixed positive-definite target, :math:`c_\alpha` is the
Gaussian consistency correction for the retained fraction, and
:math:`0 \leq \rho \leq 1` controls regularization.  The selected subset is

.. math::

   H_{\mathrm{MRCD}}
   \approx
   \arg\min_{|H|=h} \det(K(H))^{1/p}.

The default target is the identity matrix in standardized coordinates.  In the
original units this corresponds to a diagonal covariance target based on the
robust marginal scales.  An equicorrelation target or a custom SPD target can
also be supplied.

With automatic regularization, ``MRCD`` chooses the smallest target weight that
keeps the target-whitened covariance below a requested condition number.  If
:math:`\lambda_{\min}` and :math:`\lambda_{\max}` are the extreme eigenvalues
of :math:`c_\alpha S_W(H)`, the calibrated matrix has condition number

.. math::

   \frac{\rho + (1-\rho)\lambda_{\max}}
        {\rho + (1-\rho)\lambda_{\min}}.

The package default bounds this quantity by 50.  This is a condition number
relative to the target after robust standardization; rescaling the variables can
change the ordinary condition number of the covariance in the original units.

The subset search uses regularized C-steps.  At each step, robust distances are
computed from the current subset mean and regularized covariance, and the next
subset contains the :math:`h` smallest distances.  The regularized determinant
does not increase under the MRCD C-step.

``robustcov`` uses deterministic central starts together with randomized
projection and elemental starts, short C-step screening, and full polishing of
the best candidates.  This differs from the six DetMCD initial estimates used
in the reference R implementation, but optimizes the same MRCD subset
criterion.  ``MRCD`` is intended for rowwise contamination.  It does not model
individual corrupted cells and it is not a sparse precision-matrix estimator.

Kernel Minimum Regularized Covariance Determinant
-------------------------------------------------

``KMRCD`` applies the MRCD subset criterion in a reproducing-kernel Hilbert
space.  Let :math:`k(x,y)=\langle\phi(x),\phi(y)\rangle` be a positive-semidefinite
kernel and let :math:`H` contain :math:`h` observations.  Their feature-space
center is

.. math::

   c_{\mathcal F}^H = \frac{1}{h}\sum_{i\in H}\phi(x_i).

The centered subset covariance is regularized toward the identity in feature
space,

.. math::

   \widehat\Sigma_{\mathrm{reg}}^H
   = \frac{1-\rho}{h-1}
     \sum_{i\in H}(\phi(x_i)-c_{\mathcal F}^H)
                    (\phi(x_i)-c_{\mathcal F}^H)^T
     + \rho I.

KMRCD approximately minimizes its determinant over all :math:`h`-subsets.  The
same optimization can be carried out with the centered subset kernel matrix
:math:`\widetilde K^H` because

.. math::

   \widetilde K_{\mathrm{reg}}^H
   = (1-\rho)\widetilde K^H + (h-1)\rho I_h

has a determinant proportional to the feature-space regularized covariance
determinant.

For a new point :math:`x`, define the kernel vector and self-kernel after
centering relative to the selected subset.  Its squared robust kernel distance
is

.. math::

   d_{\mathcal F}^2(x)
   = \frac{1}{\rho}\left[
       \widetilde k(x,x)
       -(1-\rho)\widetilde k(H,x)^T
       (\widetilde K_{\mathrm{reg}}^H)^{-1}
       \widetilde k(H,x)
     \right].

A kernel C-step keeps the :math:`h` observations with the smallest current
kernel distances.  As in MRCD, the regularized determinant does not increase
under this update.  ``regularization='auto'`` chooses a positive target weight
that keeps the regularized subset kernel below the requested condition-number
bound.

The RBF default uses the paper's median squared-distance bandwidth heuristic.
That heuristic is only a starting value: a bandwidth that is too large makes
the model nearly linear, whereas a bandwidth that is too small can fragment the
inlier cloud.

``robustcov`` implements the KMRCD objective, kernel distance, and C-steps.  It
uses kernel-central and randomized initial subsets instead of the four refined
initial estimators in the reference MATLAB code, so numerical identity with
that implementation is not claimed.  Use KMRCD for rowwise outliers around a
non-elliptical majority structure.  It is not a cellwise method, and it does not
produce an ordinary input-space covariance matrix for nonlinear kernels.

Cellwise Minimum Covariance Determinant
---------------------------------------

``CellMCD`` handles contamination at the level of individual entries rather
than complete rows.  It introduces a binary mask :math:`W=(w_{ij})` and fits
location, covariance, and the mask through an observed Gaussian likelihood with
a penalty for flagged cells.  At least :math:`h` cells must remain in every
column.

For a retained pattern :math:`w_i`, the row contributes

.. math::

   \log|\Sigma^{(w_i)}| + |w_i|\log(2\pi)
   + (x_i^{(w_i)}-\mu^{(w_i)})^T
     (\Sigma^{(w_i)})^{-1}
     (x_i^{(w_i)}-\mu^{(w_i)}).

A concentration step first updates each column of the cell mask using
conditional standardized residuals, then applies one missing-data EM update to
:math:`\mu` and :math:`\Sigma`.  See :doc:`cellwise_covariance` for the full
objective, diagnostics, and implementation limits.

Cellwise Regularized Covariance
---------------------------------

``CellRCov`` targets high-dimensional tables containing cellwise contamination,
casewise contamination, and missing entries.  After robust marginal
standardization, a rank-:math:`q` CellPCA fit gives

.. math::

   Z_i = \widehat Z_i + E_i,
   \qquad \widehat Z_i = \widehat\mu + Vt_i.

A robust score covariance :math:`\widehat\Sigma_T` is mapped back to feature
space,

.. math::

   \widehat\Sigma_{\mathrm{fit}}
   = V\widehat\Sigma_TV^T.

Cell and case weights from CellPCA are then used to form a positive-semidefinite
residual covariance :math:`\widehat\Sigma_{\mathrm{res}}`.  The residual term
is stabilized by diagonal shrinkage,

.. math::

   \widehat\Sigma_{\mathrm{res}}^{(\delta)}
   = (1-\delta)\widehat\Sigma_{\mathrm{res}}
   + \delta\operatorname{diag}(\widehat\Sigma_{\mathrm{res}}),

and the standardized covariance estimate is

.. math::

   \widehat\Sigma_Z
   = \widehat\Sigma_{\mathrm{fit}}
   + \widehat\Sigma_{\mathrm{res}}^{(\delta)}.

See :doc:`cellwise_regularized_covariance` for the effective-weight
normalization, shrinkage selection, rank choice, and differences from the
reference cellRCov software.

Density-power robust PCA
----------------------------

``DensityPowerRobustPCA`` estimates a low-rank model directly rather than
first estimating a full scatter matrix.  For centered data it uses

.. math::

   x_{ij}-\mu_j \approx a_i^T b_j + \varepsilon_{ij},

where :math:`a_i` is the score vector of row :math:`i` and :math:`b_j` is the
loading vector of feature :math:`j`.  With density-power parameter
:math:`\alpha>0`, each residual contributes a bounded Gaussian
density-power-divergence loss.  The corresponding weight is

.. math::

   w_{ij}
   = \exp\left(
       -\frac{\alpha r_{ij}^2}{2\sigma^2}
     \right),
   \qquad
   r_{ij}=x_{ij}-\mu_j-a_i^Tb_j.

Scores and loadings are updated by alternating weighted least squares.  The
residual scale is updated from the density-power fixed-point equation

.. math::

   \sigma^2
   =
   \frac{\frac{1}{np}\sum_{i,j} w_{ij}r_{ij}^2}
        {\frac{1}{np}\sum_{i,j}w_{ij}
         -\alpha(1+\alpha)^{-3/2}}.

At :math:`\alpha=0`, all weights equal one and the fit reduces to ordinary
least-squares PCA.  Larger values bound the influence of large reconstruction
errors more strongly.  A final singular-value decomposition puts the fitted
rank-:math:`q` model into canonical PCA form, with eigenvalues
:math:`s_k^2/n`.

The package uses a geometric-median center, winsorized SVD initialization,
block weighted least-squares updates, QR stabilization, and a final SVD.  It
therefore implements a package-native density-power PCA variant and does not
claim numerical identity with the reference rPCAdpd software.  See
:doc:`density_power_pca` for diagnostics, tuning guidance, and limitations.

Cellwise and Casewise Robust PCA
-----------------------------------

``CellPCA`` estimates a rank-:math:`q` approximation

.. math::

   \widehat X = \mathbf{1}\mu^T + T V^T,
   \qquad V^T V = I_q,

while assigning a weight to every observed cell and a second weight to every
row.  For residual :math:`r_{ij}` and fixed robust residual scale :math:`s_j`,
the cell weight is

.. math::

   w^{\mathrm{cell}}_{ij}
   = \frac{\psi_c(r_{ij}/s_j)}{r_{ij}/s_j}.

The bounded cell losses are summarized into a casewise total deviation
:math:`t_i`, which yields a row weight

.. math::

   w^{\mathrm{case}}_i
   = \frac{\psi_r(t_i/s_r)}{t_i/s_r}.

Missing cells have weight zero, and the combined weights are used in
alternating weighted least-squares updates of the center, scores, and loadings.
See :doc:`cellwise_pca` for the residual scales, diagnostics, prediction on
incomplete rows, and the implementation's relationship to the reference
cellPCA algorithm.

Sparse cellwise robust PCA
---------------------------

``SparseCellPCA`` replaces the dense loading update with a weighted elastic-net
regression.  For loading matrix :math:`B=(b_{jk})`, the package-specific
criterion adds

.. math::

   \sum_{k=1}^{q}\alpha_k
   \left[
      \eta\lVert b_{\cdot k}\rVert_1
      + \frac{1-\eta}{2}\lVert b_{\cdot k}\rVert_2^2
   \right]


to the robust cellwise reconstruction loss.  The same cell and row weights as
``CellPCA`` protect the low-rank fit, while coordinate descent gives exact-zero
loadings.  Sparse loading vectors are normalized but need not be mutually
orthogonal.  See :doc:`sparse_cellwise_pca` for tuning, diagnostics, and the
implementation's relationship to SCRAMBLE.

Bootstrap PCA subspace stability
--------------------------------

``SubspaceStability`` repeatedly resamples observations and refits a PCA-style
estimator.  The resampling design can be IID, moving-block, circular-block,
stationary, or cluster based.  Block methods preserve consecutive multivariate
rows, while cluster sampling preserves every row belonging to a selected
subject, site, or account.  If :math:`V_q` is the full-data basis and
:math:`V_q^{(b)}` is a bootstrap basis, the singular values of

.. math::

   V_q {V_q^{(b)}}^T

are the cosines of the principal angles between the two retained subspaces.
Small angles indicate that the fitted subspace is insensitive to the sampled
rows.

Loading intervals require an additional alignment because eigenvector signs are
arbitrary and nearby components can rotate.  The default orthogonal Procrustes
alignment solves

.. math::

   Q_b = \arg\min_{Q^TQ=I}
   \left\|Q V_q^{(b)} - V_q\right\|_F.

Percentile intervals are then calculated from the aligned bootstrap loadings,
eigenvalues, and explained-variance ratios.  These are sampling-stability
diagnostics rather than finite-sample guarantees.  See
:doc:`subspace_stability` for block-length choice, grouped resampling, and
interpretation when eigenvalues are nearly tied.

Robust graphical lasso
----------------------

``RobustGraphicalLasso`` estimates a sparse precision matrix from a robust
scatter estimate.  If :math:`S` denotes that scatter matrix, it solves

.. math::

   \widehat\Theta
   =
   \arg\min_{\Theta \succ 0}
   \left\{
      \operatorname{tr}(S\Theta)-\log\det(\Theta)
      + \alpha\sum_{j\ne k}|\Theta_{jk}|
   \right\}.

Zeros in the off-diagonal precision matrix define the estimated conditional-
dependence graph.  The default implementation works on the robust correlation
matrix before mapping the result back to the original feature scales.

With ``alpha="ebic"``, a geometric penalty path is scored by the extended
Bayesian information criterion.  The optimizer uses ADMM with an eigenvalue
update for the positive-definite variable and off-diagonal soft thresholding for
the sparse variable.

Robustness comes from the supplied scatter estimator rather than from a new
graphical-lasso likelihood.  ``RegularizedCauchy`` is the default; ``CellMCD``
can be used for cellwise contamination and missing values.  See
:doc:`sparse_precision` for partial correlations, EBIC selection, and the
difference from robust CLIME and spatial-sign graphical-lasso methods.

Spatial-sign graphical lasso
--------------------------------

``SpatialSignGraphicalLasso`` estimates a sparse precision for the
trace-normalized elliptical shape rather than starting from an ordinary
covariance estimate.  With spatial median :math:`\widehat\mu`, define

.. math::

   \widehat S
   = \frac{1}{n}\sum_{i=1}^{n}
     U(x_i-\widehat\mu)U(x_i-\widehat\mu)^T,
   \qquad U(z)=\frac{z}{\lVert z\rVert_2}.

The published SGLASSO objective is

.. math::

   \widehat V
   = \arg\min_{V\succ0}
     \left\{
       \operatorname{tr}(p\widehat S V)-\log\det V
       + \alpha\lVert V\rVert_1
     \right\}.

The factor :math:`p` arises because, under the high-dimensional elliptical
assumptions of Lu and Feng, :math:`pS` approximates a trace-normalized covariance
shape.  Consequently, ``precision_`` is defined up to a common scale; its zero
pattern and partial correlations remain meaningful.  See
:doc:`spatial_sign_precision` for diagonal penalization, EBIC selection, missing
values, and the distinction from cellwise-robust graph estimation.

Matrix Minimum Covariance Determinant
-------------------------------------

``MMCD`` extends the MCD subset principle to matrix-valued observations.  For
:math:`X_i \in \mathbb{R}^{r\times c}`, it estimates a mean matrix :math:`M`, a
row covariance :math:`R`, and a column covariance :math:`C` under

.. math::

   \operatorname{Cov}(\operatorname{vec}(X_i)) = C \otimes R.

The squared matrix Mahalanobis distance is

.. math::

   d_i^2 = \operatorname{tr}\left[
      C^{-1}(X_i-M)^T R^{-1}(X_i-M)
   \right].

For each candidate :math:`h`-subset, the two covariance factors are fitted by
alternating matrix-normal maximum-likelihood updates.  The selected subset
approximately minimizes

.. math::

   \det(C_H \otimes R_H)
   = \det(R_H)^c \det(C_H)^r.

A matrix C-step retains the :math:`h` observations with the smallest current
matrix Mahalanobis distances.  This avoids estimating an unrestricted
:math:`rc \times rc` covariance after vectorization and preserves separate row
and column dependence.  See :doc:`matrix_covariance` for the fitted equations,
scale normalization, contribution decomposition, and implementation limits.

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
