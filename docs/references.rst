
References
==========

The MVP documentation cites the following robust-statistics and covariance-estimation background.
The implementation is intentionally pragmatic and experimental; these references provide the
mathematical context rather than a claim that every estimator is a line-by-line reproduction of a
specific paper.

Robust covariance and MCD
-------------------------

* P. J. Rousseeuw. 1984. Least median of squares regression. *Journal of the American Statistical Association*.
* P. J. Rousseeuw. 1985. Multivariate estimation with high breakdown point. In *Mathematical Statistics and Applications*.
* P. J. Rousseeuw and K. Van Driessen. 1999. A fast algorithm for the minimum covariance determinant estimator. *Technometrics*.
* M. Hubert, M. Debruyne, and P. J. Rousseeuw. 2018. Minimum covariance determinant and extensions. *WIREs Computational Statistics*.
* M. Hubert, P. J. Rousseeuw, D. Vanpaemel, and T. Verdonck. 2015. The DetS and DetMM estimators for multivariate location and scatter. *Computational Statistics & Data Analysis*.
* K. Boudt, P. J. Rousseeuw, S. Vanduffel, and T. Verdonck. 2020. The minimum regularized covariance determinant estimator. *Statistics and Computing*.
* J. Schreurs, I. Vranckx, M. Hubert, J. A. K. Suykens, and P. J. Rousseeuw. 2021. Outlier detection in non-elliptical data by kernel MRCD. *Statistics and Computing*, 31:66.
* M. Mayrhofer, U. Radojičić, and P. Filzmoser. 2025. Robust covariance estimation and explainable outlier detection for matrix-valued data. *Technometrics*.
* J. Raymaekers and P. J. Rousseeuw. 2024. The cellwise minimum covariance determinant estimator. *Journal of the American Statistical Association*, 119(548), 2610–2621.
* F. Centofanti, M. Hubert, and P. J. Rousseeuw. 2026. Cellwise and casewise robust covariance in high dimensions. arXiv:2505.19925.
* P. Dutilleul. 1999. The MLE algorithm for the matrix normal distribution. *Journal of Statistical Computation and Simulation*.

Tyler and robust scatter M-estimation
-------------------------------------

* D. E. Tyler. 1987. A distribution-free M-estimator of multivariate scatter. *The Annals of Statistics*.
* R. A. Maronna. 1976. Robust M-estimators of multivariate location and scatter. *The Annals of Statistics*.
* E. Ollila, D. E. Tyler, V. Koivunen, and H. V. Poor. 2012. Complex elliptically symmetric distributions: survey, new results and applications. *IEEE Transactions on Signal Processing*.

Regularization and shrinkage
----------------------------

* O. Ledoit and M. Wolf. 2004. A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*.
* Y. Chen, A. Wiesel, and A. O. Hero. 2011. Robust shrinkage estimation of high-dimensional covariance matrices. *IEEE Transactions on Signal Processing*.
* A. Wiesel. 2012. Unified framework to regularized covariance estimation in scaled Gaussian models. *IEEE Transactions on Signal Processing*.
* Y. Sun, P. Babu, and D. P. Palomar. 2014. Regularized Tyler's scatter estimator: existence, uniqueness, and algorithms. *IEEE Transactions on Signal Processing*.


Matrix geometry and geodesic convexity
--------------------------------------

* R. Bhatia. 2007. *Positive Definite Matrices*. Princeton University Press.
* S. Sra and R. Hosseini. 2015. Conic geometric optimization on the manifold of positive definite matrices. *SIAM Journal on Optimization*.
* M. Moakher. 2005. A differential geometric approach to the geometric mean of symmetric positive-definite matrices. *SIAM Journal on Matrix Analysis and Applications*.

Heavy-tailed covariance and Student-t models
--------------------------------------------

* K. L. Lange, R. J. A. Little, and J. M. G. Taylor. 1989. Robust statistical modeling using the t distribution. *Journal of the American Statistical Association*.
* G. McLachlan and T. Krishnan. 2008. *The EM Algorithm and Extensions*. Wiley.
* R. A. Maronna, R. D. Martin, V. J. Yohai, and M. Salibián-Barrera. 2019. *Robust Statistics: Theory and Methods*. Wiley.

Robust anomaly diagnostics
--------------------------

* P. J. Rousseeuw and A. M. Leroy. 1987. *Robust Regression and Outlier Detection*. Wiley.
* M. Hubert, P. J. Rousseeuw, and K. Vanden Branden. 2005. ROBPCA: a new approach to robust principal component analysis. *Technometrics*.
* F. Centofanti, M. Hubert, and P. J. Rousseeuw. 2026. Robust principal components by casewise and cellwise weighting. *Technometrics*.
* P. Pfeiffer, L. Vana-Gür, and P. Filzmoser. 2025. Cellwise robust and sparse principal component analysis. *Advances in Data Analysis and Classification*.
* M. E. Timmerman, E. Kiers, and A. C. Smilde. 2007. Estimating confidence intervals for principal component loadings: a comparison between the bootstrap and asymptotic results. *British Journal of Mathematical and Statistical Psychology*.
* R. H. Abul Naga and G. Antille. 1990. Stability of robust and non-robust principal components analysis. *Computational Statistics & Data Analysis*.
* H. R. Künsch. 1989. The jackknife and the bootstrap for general stationary observations. *The Annals of Statistics*.
* D. N. Politis and J. P. Romano. 1994. The stationary bootstrap. *Journal of the American Statistical Association*.
* D. N. Politis and H. White. 2004. Automatic block-length selection for the dependent bootstrap. *Econometric Reviews*.


Robust clustering and mixtures
------------------------------

* A. C. Atkinson and M. Riani. 2000. *Robust Diagnostic Regression Analysis*. Springer.
* A. García-Escudero, A. Gordaliza, C. Matrán, and A. Mayo-Iscar. 2008. A general trimming approach to robust cluster analysis. *The Annals of Statistics*.
* G. J. McLachlan and D. Peel. 2000. *Finite Mixture Models*. Wiley.
* G. J. McLachlan and D. Peel. 1998/2000. Robust cluster analysis via mixtures of multivariate t-distributions. Related robust mixture-model literature.
