Matrix-valued covariance estimation
===================================

``MMCD`` estimates a mean matrix together with separate row and column
covariance factors.  It is intended for data in which each observation is
naturally a matrix, such as a sensor-by-time window, an image patch, or a
variable-by-condition panel.

The input shape is ``(n_samples, n_rows, n_columns)``:

.. code-block:: python

   import robustcov as rc

   model = rc.MMCD(
       contamination=0.20,
       n_init=200,
       random_state=0,
   ).fit(X)

   distances = model.mahalanobis(X)
   labels = model.predict(X)

The fitted factors are available as ``row_covariance_`` and
``column_covariance_``.  Their Kronecker product is the covariance of a
column-major vectorization of each matrix:

.. code-block:: python

   covariance = model.kronecker_covariance()

The full matrix is not stored automatically.  For an observation with ``r``
rows and ``c`` columns it would have shape ``(r*c, r*c)``, while the two factors
require only ``r*r + c*c`` entries.

Statistical model
-----------------

Let :math:`X_i \in \mathbb{R}^{r\times c}`.  A separable matrix covariance model
assumes

.. math::

   \operatorname{Cov}(\operatorname{vec}(X_i))
   = C \otimes R,

where :math:`R` is the row covariance and :math:`C` is the column covariance.
For a mean matrix :math:`M`, the squared matrix Mahalanobis distance is

.. math::

   d_i^2
   =
   \operatorname{tr}\left[
      C^{-1}(X_i-M)^\mathsf{T}R^{-1}(X_i-M)
   \right].

For a candidate subset :math:`H` of size :math:`h`, MMCD computes matrix-normal
maximum-likelihood estimates.  The row and column factors satisfy the
flip-flop equations

.. math::

   R_H
   =
   \frac{1}{hc}
   \sum_{i\in H}
   (X_i-M_H)C_H^{-1}(X_i-M_H)^\mathsf{T},

.. math::

   C_H
   =
   \frac{1}{hr}
   \sum_{i\in H}
   (X_i-M_H)^\mathsf{T}R_H^{-1}(X_i-M_H).

The subset objective is the determinant of the Kronecker covariance:

.. math::

   H^*
   \approx
   \arg\min_{|H|=h}
   \det(C_H\otimes R_H)
   =
   \arg\min_{|H|=h}
   \det(R_H)^c\det(C_H)^r.

A concentration step fits the two factors on the current subset, calculates
all matrix Mahalanobis distances, and keeps the :math:`h` smallest distances.
The package screens several central and randomized elemental starts, then
polishes the best candidates.

The Kronecker factors have a scale ambiguity: multiplying :math:`R` by a
positive constant and dividing :math:`C` by the same constant leaves
:math:`C\otimes R` unchanged.  ``MMCD`` fixes this representation by normalizing
:math:`\det(R)=1`.

Distance contributions
----------------------

The squared distance can be decomposed over matrix cells.  For
:math:`E=X-M`, define

.. math::

   G
   =
   E \odot \left(R^{-1}EC^{-1}\right),

where :math:`\odot` denotes elementwise multiplication.  Then

.. math::

   \sum_{a=1}^{r}\sum_{b=1}^{c}G_{ab}=d^2(X).

``cell_contributions``, ``row_contributions``, and ``column_contributions``
return this decomposition.  Individual terms can be negative when variables
are correlated.  These are signed quadratic-form contributions, not Shapley
values.

.. code-block:: python

   cell = model.cell_contributions(X_test)
   row = model.row_contributions(X_test)
   column = model.column_contributions(X_test)

   rc.plot_matrix_outlier_contributions(
       model,
       X_test,
       index=0,
       output_path="matrix_contributions.png",
       show=False,
   )

Numerical and modeling limits
-----------------------------

``MMCD`` assumes rowwise contamination: a minority of complete matrix
observations may come from another process.  It does not identify isolated bad
cells during fitting.  The row/column covariance assumption is also a modeling
choice; it can be too restrictive when the vectorized covariance is not close
to a Kronecker product.

The implementation follows the MMCD objective and concentration-step structure,
but its initialization is a Python-oriented approximation to FastMMCD rather
than a reproduction of every subsampling optimization in the reference R
implementation.  ``ridge`` adds a small trace-relative diagonal term for
numerical stability.  Set ``ridge=0`` only when every subset update is known to
remain positive definite.

Example
-------

The sensor-window example keeps each window as a sensor-by-time matrix and
shows how localized faults appear in the cell contribution map:

* :doc:`Matrix MCD for multichannel sensor windows <gallery/mmcd_sensor_windows>`
