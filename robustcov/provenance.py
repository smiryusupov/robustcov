# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Method provenance, references, and package-contribution metadata.

The registry deliberately separates the origin of a statistical method from the
work performed in :mod:`robustcov`.  A package implementation can be a
substantial software contribution without implying that the underlying
statistical methodology originated in this project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class Reference:
    """A primary methodological or numerical reference."""

    key: str
    short: str
    citation: str
    url: str


@dataclass(frozen=True)
class MethodProvenance:
    """Provenance metadata for one canonical public estimator or algorithm."""

    name: str
    family: str
    status: str
    summary: str
    references: tuple[str, ...]
    robustcov_contribution: str
    implementation_notes: str
    aliases: tuple[str, ...] = ()


STATUS_LABELS: Mapping[str, str] = {
    "literature_implementation": "Literature implementation",
    "literature_adaptation": "Literature-based adaptation",
    "robustcov_composite": "robustcov composite/workflow",
    "robustcov_utility": "robustcov utility/infrastructure",
    "original_method": "Original methodological contribution",
}


REFERENCE_CATALOG: Mapping[str, Reference] = {
    "amari_1995": Reference(
        "amari_1995",
        "Amari (1995)",
        "S.-I. Amari. A new learning algorithm for blind signal separation. Advances in Neural Information Processing Systems 8, 1995.",
        "https://proceedings.neurips.cc/paper/1995/hash/e19347e1c3ca0c0b97de5fb3b690855a-Abstract.html",
    ),
    "belouchrani_etal_1997": Reference(
        "belouchrani_etal_1997",
        "Belouchrani et al. (1997)",
        "A. Belouchrani, K. Abed-Meraim, J.-F. Cardoso, and E. Moulines. A blind source separation technique using second-order statistics. IEEE Transactions on Signal Processing 45(2), 434–444, 1997.",
        "https://ieeexplore.ieee.org/document/554307/",
    ),
    "bashari_etal_2025": Reference(
        "bashari_etal_2025",
        "Bashari et al. (2025)",
        "M. Bashari, M. Sesia, and Y. Romano. Robust conformal outlier detection under contaminated reference data. Proceedings of the 42nd International Conference on Machine Learning, PMLR 267, 3091–3141, 2025.",
        "https://proceedings.mlr.press/v267/bashari25a.html",
    ),
    "bhatia_2007": Reference(
        "bhatia_2007",
        "Bhatia (2007)",
        "R. Bhatia. Positive Definite Matrices. Princeton University Press, 2007.",
        "https://press.princeton.edu/books/hardcover/9780691129181/positive-definite-matrices",
    ),
    "boudt_etal_2020": Reference(
        "boudt_etal_2020",
        "Boudt et al. (2020)",
        "K. Boudt, P. J. Rousseeuw, S. Vanduffel, and T. Verdonck. The minimum regularized covariance determinant estimator. Statistics and Computing 30, 113–128, 2020.",
        "https://doi.org/10.1007/s11222-019-09869-x",
    ),
    "cardoso_souloumiac_1996": Reference(
        "cardoso_souloumiac_1996",
        "Cardoso & Souloumiac (1996)",
        "J.-F. Cardoso and A. Souloumiac. Jacobi angles for simultaneous diagonalization. SIAM Journal on Matrix Analysis and Applications 17(1), 161–164, 1996.",
        "https://doi.org/10.1137/S0895479893259546",
    ),
    "centofanti_etal_2025_cellrcov": Reference(
        "centofanti_etal_2025_cellrcov",
        "Centofanti et al. (2025)",
        "F. Centofanti, M. Hubert, and P. J. Rousseeuw. Cellwise and casewise robust covariance in high dimensions. arXiv:2505.19925, 2025.",
        "https://arxiv.org/abs/2505.19925",
    ),
    "centofanti_etal_2026_cellpca": Reference(
        "centofanti_etal_2026_cellpca",
        "Centofanti et al. (2026)",
        "F. Centofanti, M. Hubert, and P. J. Rousseeuw. Robust principal components by casewise and cellwise weighting. Technometrics, 2026.",
        "https://doi.org/10.1080/00401706.2026.2643216",
    ),
    "chen_etal_2011": Reference(
        "chen_etal_2011",
        "Chen et al. (2011)",
        "Y. Chen, A. Wiesel, and A. O. Hero. Robust shrinkage estimation of high-dimensional covariance matrices. IEEE Transactions on Signal Processing 59(9), 4097–4107, 2011.",
        "https://doi.org/10.1109/TSP.2011.2138698",
    ),
    "dutilleul_1999": Reference(
        "dutilleul_1999",
        "Dutilleul (1999)",
        "P. Dutilleul. The MLE algorithm for the matrix normal distribution. Journal of Statistical Computation and Simulation 64(2), 105–123, 1999.",
        "https://doi.org/10.1080/00949659908811970",
    ),
    "fan_etal_2018": Reference(
        "fan_etal_2018",
        "Fan et al. (2018)",
        "J. Fan, H. Liu, and W. Wang. Large covariance estimation through elliptical factor models. The Annals of Statistics 46(4), 1383–1414, 2018.",
        "https://doi.org/10.1214/17-AOS1588",
    ),
    "friedman_etal_2008": Reference(
        "friedman_etal_2008",
        "Friedman et al. (2008)",
        "J. Friedman, T. Hastie, and R. Tibshirani. Sparse inverse covariance estimation with the graphical lasso. Biostatistics 9(3), 432–441, 2008.",
        "https://doi.org/10.1093/biostatistics/kxm045",
    ),
    "hirari_etal_2026": Reference(
        "hirari_etal_2026",
        "Hirari et al. (2026)",
        "M. Hirari, F. Centofanti, M. Hubert, and S. Van Aelst. Casewise and cellwise robust multilinear principal component analysis. Journal of Computational and Graphical Statistics, 2026.",
        "https://doi.org/10.1080/10618600.2026.2637632",
    ),
    "huber_1964": Reference(
        "huber_1964",
        "Huber (1964)",
        "P. J. Huber. Robust estimation of a location parameter. The Annals of Mathematical Statistics 35(1), 73–101, 1964.",
        "https://doi.org/10.1214/aoms/1177703732",
    ),
    "hubert_etal_2005": Reference(
        "hubert_etal_2005",
        "Hubert et al. (2005)",
        "M. Hubert, P. J. Rousseeuw, and K. Vanden Branden. ROBPCA: a new approach to robust principal component analysis. Technometrics 47(1), 64–79, 2005.",
        "https://doi.org/10.1198/004017004000000563",
    ),
    "hubert_etal_2015": Reference(
        "hubert_etal_2015",
        "Hubert et al. (2015)",
        "M. Hubert, P. J. Rousseeuw, D. Vanpaemel, and T. Verdonck. The DetS and DetMM estimators for multivariate location and scatter. Computational Statistics & Data Analysis 81, 64–75, 2015.",
        "https://doi.org/10.1016/j.csda.2014.07.013",
    ),
    "ilmonen_etal_2010": Reference(
        "ilmonen_etal_2010",
        "Ilmonen et al. (2010)",
        "P. Ilmonen, K. Nordhausen, H. Oja, and E. Ollila. A new performance index for ICA: properties, computation and asymptotic analysis. Latent Variable Analysis and Signal Separation, 229–236, 2010.",
        "https://doi.org/10.1007/978-3-642-15995-4_29",
    ),
    "kunsch_1989": Reference(
        "kunsch_1989",
        "Künsch (1989)",
        "H. R. Künsch. The jackknife and the bootstrap for general stationary observations. The Annals of Statistics 17(3), 1217–1241, 1989.",
        "https://doi.org/10.1214/aos/1176347265",
    ),
    "lange_etal_1989": Reference(
        "lange_etal_1989",
        "Lange et al. (1989)",
        "K. L. Lange, R. J. A. Little, and J. M. G. Taylor. Robust statistical modeling using the t distribution. Journal of the American Statistical Association 84(408), 881–896, 1989.",
        "https://doi.org/10.1080/01621459.1989.10478852",
    ),
    "lu_feng_2025": Reference(
        "lu_feng_2025",
        "Lu & Feng (2025)",
        "Z. Lu and L. Feng. Robust sparse precision matrix estimation and its applications. arXiv:2503.03575, 2025.",
        "https://arxiv.org/abs/2503.03575",
    ),
    "maronna_1976": Reference(
        "maronna_1976",
        "Maronna (1976)",
        "R. A. Maronna. Robust M-estimators of multivariate location and scatter. The Annals of Statistics 4(1), 51–67, 1976.",
        "https://doi.org/10.1214/aos/1176343347",
    ),
    "maronna_etal_2019": Reference(
        "maronna_etal_2019",
        "Maronna et al. (2019)",
        "R. A. Maronna, R. D. Martin, V. J. Yohai, and M. Salibián-Barrera. Robust Statistics: Theory and Methods (with R), 2nd ed. Wiley, 2019.",
        "https://doi.org/10.1002/9781119214656",
    ),
    "mayrhofer_etal_2025": Reference(
        "mayrhofer_etal_2025",
        "Mayrhofer et al. (2025)",
        "M. Mayrhofer, U. Radojičić, and P. Filzmoser. Robust covariance estimation and explainable outlier detection for matrix-valued data. Technometrics 67(3), 516–530, 2025.",
        "https://doi.org/10.1080/00401706.2025.2475781",
    ),
    "narayanamurthy_vaswani_2018": Reference(
        "narayanamurthy_vaswani_2018",
        "Narayanamurthy & Vaswani (2018)",
        "P. Narayanamurthy and N. Vaswani. Nearly Optimal Robust Subspace Tracking. Proceedings of the 35th International Conference on Machine Learning, PMLR 80, 3701–3709, 2018.",
        "https://proceedings.mlr.press/v80/narayanamurthy18a.html",
    ),
    "zhu_shen_2022": Reference(
        "zhu_shen_2022",
        "Zhu & Shen (2022)",
        "T. Zhu and J. Shen. Residual-Based Sampling for Online Outlier-Robust PCA. Proceedings of the 39th International Conference on Machine Learning, PMLR 162, 27591–27611, 2022.",
        "https://proceedings.mlr.press/v162/zhu22i.html",
    ),
    "nordhausen_2014": Reference(
        "nordhausen_2014",
        "Nordhausen (2014)",
        "K. Nordhausen. On robustifying some second order blind source separation methods for nonstationary time series. Statistical Papers 55, 141–156, 2014.",
        "https://doi.org/10.1007/s00362-012-0487-5",
    ),
    "oja_etal_2006": Reference(
        "oja_etal_2006",
        "Oja et al. (2006)",
        "H. Oja, S. Sirkiä, and J. Eriksson. Scatter matrices and independent component analysis. Austrian Journal of Statistics 35(2&3), 175–189, 2006.",
        "https://doi.org/10.17713/ajs.v35i2&3.377",
    ),
    "ollila_tyler_2014": Reference(
        "ollila_tyler_2014",
        "Ollila & Tyler (2014)",
        "E. Ollila and D. E. Tyler. Regularized M-estimators of scatter matrix. IEEE Transactions on Signal Processing 62(22), 6059–6070, 2014.",
        "https://doi.org/10.1109/TSP.2014.2360826",
    ),
    "pennec_etal_2006": Reference(
        "pennec_etal_2006",
        "Pennec et al. (2006)",
        "X. Pennec, P. Fillard, and N. Ayache. A Riemannian framework for tensor computing. International Journal of Computer Vision 66, 41–66, 2006.",
        "https://doi.org/10.1007/s11263-005-3222-z",
    ),
    "pfeiffer_etal_2025": Reference(
        "pfeiffer_etal_2025",
        "Pfeiffer et al. (2025)",
        "P. Pfeiffer, L. Vana-Gür, and P. Filzmoser. Cellwise robust and sparse principal component analysis. Advances in Data Analysis and Classification, 2025.",
        "https://doi.org/10.1007/s11634-025-00656-3",
    ),
    "politis_romano_1994": Reference(
        "politis_romano_1994",
        "Politis & Romano (1994)",
        "D. N. Politis and J. P. Romano. The stationary bootstrap. Journal of the American Statistical Association 89(428), 1303–1313, 1994.",
        "https://doi.org/10.1080/01621459.1994.10476870",
    ),
    "raymaekers_rousseeuw_2024": Reference(
        "raymaekers_rousseeuw_2024",
        "Raymaekers & Rousseeuw (2024)",
        "J. Raymaekers and P. J. Rousseeuw. The cellwise minimum covariance determinant estimator. Journal of the American Statistical Association 119(548), 2610–2621, 2024.",
        "https://doi.org/10.1080/01621459.2023.2267777",
    ),
    "rousseeuw_vandriessen_1999": Reference(
        "rousseeuw_vandriessen_1999",
        "Rousseeuw & Van Driessen (1999)",
        "P. J. Rousseeuw and K. Van Driessen. A fast algorithm for the minimum covariance determinant estimator. Technometrics 41(3), 212–223, 1999.",
        "https://doi.org/10.1080/00401706.1999.10485670",
    ),
    "roy_etal_2024": Reference(
        "roy_etal_2024",
        "Roy et al. (2024)",
        "S. Roy, A. Basu, and A. Ghosh. Robust principal component analysis using density power divergence. Journal of Machine Learning Research 25(324), 1–40, 2024.",
        "https://jmlr.org/papers/v25/22-1380.html",
    ),
    "schreurs_etal_2021": Reference(
        "schreurs_etal_2021",
        "Schreurs et al. (2021)",
        "J. Schreurs, I. Vranckx, M. Hubert, J. A. K. Suykens, and P. J. Rousseeuw. Outlier detection in non-elliptical data by kernel MRCD. Statistics and Computing 31, 66, 2021.",
        "https://doi.org/10.1007/s11222-021-10041-7",
    ),
    "sun_etal_2014": Reference(
        "sun_etal_2014",
        "Sun et al. (2014)",
        "Y. Sun, P. Babu, and D. P. Palomar. Regularized Tyler's scatter estimator: existence, uniqueness, and algorithms. IEEE Transactions on Signal Processing 62(19), 5143–5156, 2014.",
        "https://doi.org/10.1109/TSP.2014.2345351",
    ),
    "taskinen_etal_2007": Reference(
        "taskinen_etal_2007",
        "Taskinen et al. (2007)",
        "S. Taskinen, A. Kankainen, and H. Oja. Independent component analysis based on symmetrised scatter matrices. Computational Statistics & Data Analysis 51(10), 5103–5111, 2007.",
        "https://doi.org/10.1016/j.csda.2006.08.031",
    ),
    "timmerman_etal_2007": Reference(
        "timmerman_etal_2007",
        "Timmerman et al. (2007)",
        "M. E. Timmerman, H. A. L. Kiers, and A. K. Smilde. Estimating confidence intervals for principal component loadings: a comparison between the bootstrap and asymptotic results. British Journal of Mathematical and Statistical Psychology 60(2), 295–314, 2007.",
        "https://doi.org/10.1348/000711006X109636",
    ),
    "tyler_1987": Reference(
        "tyler_1987",
        "Tyler (1987)",
        "D. E. Tyler. A distribution-free M-estimator of multivariate scatter. The Annals of Statistics 15(1), 234–251, 1987.",
        "https://doi.org/10.1214/aos/1176350263",
    ),
    "wang_liu_chen_2025_drpca": Reference(
        "wang_liu_chen_2025_drpca",
        "Wang, Liu & Chen (2025)",
        "L. Wang, X. Liu, and X. Chen. Enhancing distributional robustness in principal component analysis by Wasserstein distances. arXiv:2503.02494, 2025.",
        "https://arxiv.org/abs/2503.02494",
    ),
    "xu_wood_yang_2026": Reference(
        "xu_wood_yang_2026",
        "Xu, Wood & Yang (2026)",
        "C. Xu, A. T. A. Wood, and Y. Yang. Distributionally robust PCA with data-adaptive Wasserstein geometry. arXiv:2606.10463, 2026.",
        "https://arxiv.org/abs/2606.10463",
    ),
    "wiesel_2012": Reference(
        "wiesel_2012",
        "Wiesel (2012)",
        "A. Wiesel. Geodesic convexity and covariance estimation. IEEE Transactions on Signal Processing 60(12), 6182–6189, 2012.",
        "https://doi.org/10.1109/TSP.2012.2218241",
    ),
    "vovk_etal_2005": Reference(
        "vovk_etal_2005",
        "Vovk et al. (2005)",
        "V. Vovk, A. Gammerman, and G. Shafer. Algorithmic Learning in a Random World. Springer, 2005.",
        "https://doi.org/10.1007/b106715",
    ),
    "yu_etal_2019": Reference(
        "yu_etal_2019",
        "Yu et al. (2019)",
        "L. Yu, Y. He, and X. Zhang. Robust factor number specification for large-dimensional elliptical factor models. Journal of Multivariate Analysis 174, 104543, 2019.",
        "https://doi.org/10.1016/j.jmva.2019.104543",
    ),
}


def _entry(
    name: str,
    family: str,
    status: str,
    summary: str,
    references: Iterable[str],
    contribution: str,
    notes: str,
    aliases: Iterable[str] = (),
) -> MethodProvenance:
    return MethodProvenance(
        name=name,
        family=family,
        status=status,
        summary=summary,
        references=tuple(references),
        robustcov_contribution=contribution,
        implementation_notes=notes,
        aliases=tuple(aliases),
    )


METHOD_PROVENANCE: Mapping[str, MethodProvenance] = {
    "FastMCD": _entry(
        "FastMCD", "Rowwise covariance", "literature_implementation",
        "Minimum covariance determinant estimation with FAST-MCD-style concentration steps.",
        ("rousseeuw_vandriessen_1999",),
        "C++/OpenMP implementation, workload-aware native dispatch, sklearn-compatible API, diagnostics, and scale-relative numerical safeguards.",
        "The start-generation and polishing schedule is robustcov-specific; numerical identity with FAST-MCD implementations in other packages is not claimed.",
        ("MinCovDet",),
    ),
    "MRCD": _entry(
        "MRCD", "Rowwise covariance", "literature_adaptation",
        "Minimum regularized covariance determinant estimation for high-dimensional data.",
        ("boudt_etal_2020",),
        "Python implementation, deterministic/randomized starts, target handling, automatic regularization, diagnostics, and benchmark coverage.",
        "The package optimizes the MRCD criterion but uses a different initialization strategy from the reference R implementation.",
        ("MinimumRegularizedCovarianceDeterminant", "MinRegularizedCovDet"),
    ),
    "KMRCD": _entry(
        "KMRCD", "Kernel covariance", "literature_adaptation",
        "Kernel minimum regularized covariance determinant estimation in a reproducing-kernel feature space.",
        ("schreurs_etal_2021", "boudt_etal_2020"),
        "Kernel API, precomputed-kernel validation, automatic bandwidth/regularization support, and robust distance diagnostics.",
        "The objective and kernel C-steps follow KMRCD, while initialization differs from the reference MATLAB implementation.",
        ("KernelMRCD", "KernelMinimumRegularizedCovarianceDeterminant"),
    ),
    "DetS": _entry(
        "DetS", "Rowwise covariance", "literature_adaptation",
        "Deterministic high-breakdown S-estimation of multivariate location and scatter.",
        ("hubert_etal_2015",),
        "Pure-Python deterministic starts, stable S-scale updates, common covariance API, and edge-case hardening.",
        "The implementation follows the DetS construction but is independently implemented and may differ in initialization details.",
        ("DeterministicSEstimator",),
    ),
    "DetMM": _entry(
        "DetMM", "Rowwise covariance", "literature_adaptation",
        "Deterministic MM-estimation combining high breakdown with improved efficiency.",
        ("hubert_etal_2015",),
        "DetS initialization, fixed-scale MM refinement, diagnostics, and sklearn-compatible estimator behavior.",
        "The implementation is literature-based and independently coded; exact reproduction of external software is not claimed.",
        ("DeterministicMMEstimator",),
    ),
    "TylerShape": _entry(
        "TylerShape", "Elliptical shape", "literature_implementation",
        "Tyler's scale-free M-estimator of multivariate shape.",
        ("tyler_1987",),
        "C++/OpenMP fixed-point solver, convergence diagnostics, explicit rank checks, and extreme-scale hardening.",
        "The estimator returns trace-normalized shape rather than an identified covariance scale.",
    ),
    "RegularizedTyler": _entry(
        "RegularizedTyler", "Elliptical shape", "literature_implementation",
        "Regularized Tyler shape estimation for singular or high-dimensional regimes.",
        ("chen_etal_2011", "sun_etal_2014", "ollila_tyler_2014"),
        "C++/OpenMP solver, scale-relative regularization, convergence diagnostics, and a unified covariance/shape API.",
        "The target and normalization conventions are documented package choices and should be considered when comparing implementations.",
    ),
    "IterativeMScatter": _entry(
        "IterativeMScatter", "M-scatter infrastructure", "literature_adaptation",
        "General fixed-point infrastructure for multivariate M-estimators of scatter.",
        ("maronna_1976", "maronna_etal_2019", "ollila_tyler_2014"),
        "Reusable weight-function interface, stable regularization, optimized contractions, diagnostics, and fitted-estimator protocol.",
        "This is package infrastructure rather than a single named statistical estimator.",
    ),
    "StudentTScatter": _entry(
        "StudentTScatter", "Heavy-tail covariance", "literature_implementation",
        "Student-t maximum-likelihood/M-estimation of location and scatter.",
        ("lange_etal_1989", "maronna_1976"),
        "Regularized complete-fit implementation, convergence diagnostics, stable small-scale behavior, and common estimator API.",
        "The optional ridge term is a robustcov extension to the unregularized Student-t likelihood updates.",
    ),
    "RegularizedCauchy": _entry(
        "RegularizedCauchy", "Heavy-tail covariance", "literature_adaptation",
        "Cauchy/Student-t scatter estimation with explicit ridge regularization.",
        ("lange_etal_1989", "ollila_tyler_2014"),
        "A practical Cauchy specialization, optimized Mahalanobis contractions, automatic precision output, and numerical safeguards.",
        "This is a package adaptation of Student-t M-scatter rather than a separately claimed original estimator.",
    ),
    "KLRegularizedTyler": _entry(
        "KLRegularizedTyler", "Elliptical shape", "literature_adaptation",
        "Regularized Tyler estimation exposed with KL/geodesic regularization terminology.",
        ("wiesel_2012", "sun_etal_2014"),
        "Unified estimator API and diagnostics over the package's regularized Tyler solver.",
        "This class currently shares the regularized Tyler computational engine; it is not presented as a distinct exact reproduction of a separate algorithm.",
    ),
    "WieselTyler": _entry(
        "WieselTyler", "Elliptical shape", "literature_adaptation",
        "Wiesel-style regularized Tyler estimator interface.",
        ("wiesel_2012", "sun_etal_2014"),
        "Compatibility-oriented API over the scale-hardened regularized Tyler engine.",
        "The class is an implementation alias/adaptation, not a claim of new methodology.",
    ),
    "HellingerRegularizedTyler": _entry(
        "HellingerRegularizedTyler", "Elliptical shape", "literature_adaptation",
        "Experimental Hellinger-inspired robust scatter approximation.",
        ("maronna_1976", "tyler_1987", "ollila_tyler_2014"),
        "An experimental weight-function implementation in the common M-scatter framework, with validation and benchmark visibility.",
        "No claim is made that this class exactly reproduces a named published Hellinger-Tyler algorithm; it remains experimental.",
    ),
    "CellMCD": _entry(
        "CellMCD", "Cellwise covariance", "literature_adaptation",
        "Cellwise minimum covariance determinant estimation with mask updates and missing-data EM steps.",
        ("raymaekers_rousseeuw_2024",),
        "Independent implementation, grouped support-pattern solves, diagnostics, corrected-data transforms, and performance gates.",
        "The package follows the CellMCD objective while using implementation-specific initialization and numerical safeguards.",
        ("CellwiseMinimumCovarianceDeterminant", "CellwiseMCD"),
    ),
    "CellRCov": _entry(
        "CellRCov", "Cellwise covariance", "literature_adaptation",
        "High-dimensional covariance estimation under mixed casewise/cellwise contamination and missingness.",
        ("centofanti_etal_2025_cellrcov", "centofanti_etal_2026_cellpca"),
        "Integration with CellPCA, score-space robust scatter, shrinkage selection, complete covariance output, and diagnostics.",
        "The implementation is based on the public CellRCov description and is not claimed to be numerically identical to reference research code.",
        ("CellwiseRegularizedCovariance", "CellwiseRobustCovariance"),
    ),
    "MatrixMCD": _entry(
        "MatrixMCD", "Matrix-valued covariance", "literature_adaptation",
        "Matrix minimum covariance determinant estimation of mean, row covariance, and column covariance.",
        ("mayrhofer_etal_2025", "dutilleul_1999"),
        "Independent implementation, vectorized flip-flop updates, auto backend routing, contribution diagnostics, and transformation tests.",
        "The MMCD criterion is literature-derived; robustcov uses its own start and quality-preset implementation.",
        ("MatrixMinimumCovarianceDeterminant", "MMCD"),
    ),
    "RobustPCA": _entry(
        "RobustPCA", "Principal components", "robustcov_composite",
        "Principal-component analysis driven by a user-selectable robust scatter estimator.",
        ("hubert_etal_2005", "maronna_etal_2019"),
        "A unified estimator-driven PCA workflow with projection, reconstruction, whitening, robust score/orthogonal distances, plots, and monitoring integration.",
        "This is not an implementation of the complete ROBPCA algorithm; it diagonalizes the fitted scatter supplied to the estimator.",
    ),
    "DensityPowerRobustPCA": _entry(
        "DensityPowerRobustPCA", "Principal components", "literature_adaptation",
        "Direct robust low-rank estimation using density-power-divergence weighting.",
        ("roy_etal_2024",),
        "Independent alternating-regression implementation, diagnostics, missing-data handling, and common PCA API.",
        "Numerical defaults and stopping rules are package choices; exact identity with authors' code is not claimed.",
        ("DPDRobustPCA",),
    ),
    "CellPCA": _entry(
        "CellPCA", "Principal components", "literature_adaptation",
        "PCA with simultaneous casewise weights, cellwise weights, and missing-value handling.",
        ("centofanti_etal_2026_cellpca",),
        "Batched row/loading solves, diagnostics, corrected values, plots, and common estimator protocol.",
        "The package independently implements the published objective and uses its own numerical initialization and convergence safeguards.",
        ("CellwiseRobustPCA", "CasewiseCellwisePCA"),
    ),
    "SparseCellPCA": _entry(
        "SparseCellPCA", "Principal components", "literature_adaptation",
        "Cellwise/casewise robust PCA with exact-zero elastic-net loadings.",
        ("pfeiffer_etal_2025", "centofanti_etal_2026_cellpca"),
        "Batched sparse coordinate updates, support diagnostics, plotting, and benchmark coverage.",
        "The implementation is literature-based and may differ in tuning-path and initialization details.",
        ("SparseCellwiseRobustPCA", "SparseCasewiseCellwisePCA"),
    ),
    "RobustMultilinearPCA": _entry(
        "RobustMultilinearPCA", "Tensor/matrix principal components", "literature_adaptation",
        "Casewise and cellwise robust multilinear PCA for matrix-valued observations.",
        ("hirari_etal_2026",),
        "Independent implementation with missing values, native weighted Tucker kernels, diagnostics, and transformation hardening.",
        "The loss structure is literature-derived; initialization, backend routing, and numerical floors are package choices.",
        ("CasewiseCellwiseMultilinearPCA", "CellwiseRobustMultilinearPCA"),
    ),
    "DistributionallyRobustPCA": _entry(
        "DistributionallyRobustPCA", "Distributionally robust principal components", "literature_adaptation",
        "Weighted-Wasserstein PCA with anisotropic transport geometry and exact or surrogate worst-case reconstruction criteria.",
        ("xu_wood_yang_2026", "wang_liu_chen_2025_drpca"),
        "Experimental exact-dual candidate evaluation, deterministic adaptive-geometry path search, scale-equivariant geometry normalization, sklearn-compatible API, diagnostics, and shift-focused benchmarks.",
        "The default exact formulation minimizes the genuine weighted-Wasserstein worst-case risk over a finite deterministic candidate path, not over the entire Grassmann manifold. The sqrt-n radius is a transparent heuristic rather than the full RWPI calibration from Xu, Wood, and Yang.",
        ("WassersteinRobustPCA",),
    ),
    "RobustGraphicalLasso": _entry(
        "RobustGraphicalLasso", "Sparse precision", "robustcov_composite",
        "Graphical lasso applied to a configurable robust scatter estimate.",
        ("friedman_etal_2008", "maronna_etal_2019"),
        "Scatter-estimator composition, standalone ADMM solver, EBIC path selection, graph diagnostics, and scale/constant-feature safeguards.",
        "The graphical-lasso objective is established; the general robust-scatter composition and package workflow are robustcov contributions.",
        ("SparseRobustPrecision",),
    ),
    "SGLASSO": _entry(
        "SGLASSO", "Sparse precision", "literature_adaptation",
        "Sparse precision/shape estimation from spatial signs under elliptical heavy tails.",
        ("lu_feng_2025", "friedman_etal_2008"),
        "Spatial-median and sign-covariance implementation, EBIC selection, numerical hardening, and common graph diagnostics.",
        "This experimental implementation follows the published objective but is independently coded.",
        ("SpatialSignGraphicalLasso", "SpatialSignSparsePrecision"),
    ),
    "TwoScatterICA": _entry(
        "TwoScatterICA", "Independent component analysis", "literature_adaptation",
        "ICA by whitening with one scatter and diagonalizing a second scatter with the independence property.",
        ("oja_etal_2006", "taskinen_etal_2007", "cardoso_souloumiac_1996"),
        "Configurable robust whitening, bounded radial second scatter, optional symmetrization, canonicalization, diagnostics, and native joint diagonalization.",
        "The two-scatter ICA principle is established. The bounded radial scatter and exact composition used here are a robustcov adaptation, not a claim of inventing ICA.",
    ),
    "SOBI": _entry(
        "SOBI", "Second-order source separation", "literature_implementation",
        "Second-order blind identification by joint diagonalization of lagged covariance matrices.",
        ("belouchrani_etal_1997", "cardoso_souloumiac_1996"),
        "Python/C++ symmetric joint diagonalization, canonicalized outputs, diagnostics, backend selection, and sklearn-style estimator behavior.",
        "Lag defaults, canonicalization, and numerical safeguards are package choices.",
    ),
    "RobustSOBI": _entry(
        "RobustSOBI", "Second-order source separation", "robustcov_composite",
        "SOBI composed with robust whitening and weighted lagged cross-scatter matrices.",
        ("belouchrani_etal_1997", "nordhausen_2014", "huber_1964"),
        "A package-specific composition of Student-t whitening, Huber/Tukey lag weighting, native joint diagonalization, and source-recovery diagnostics.",
        "robustcov does not claim the SOBI principle or robust second-order BSS as original; the exact default composition is a package workflow.",
    ),
    "RobustFactorModel": _entry(
        "RobustFactorModel", "Static factor models", "literature_adaptation",
        "Static factor estimation from spatial Kendall eigenspaces with optional Huber alternating refinement.",
        ("fan_etal_2018", "yu_etal_2019", "huber_1964"),
        "Unified loadings/scores/common-component API, factor-number selection, Huber refinement, diagnostics, and benchmark integration.",
        "The Kendall and robust-factor ideas are literature-derived; the combined estimator API and refinement workflow are robustcov adaptations.",
    ),
    "RobustOutlierDetector": _entry(
        "RobustOutlierDetector", "Anomaly detection", "robustcov_composite",
        "Thresholded robust-distance detector built around a fitted location/scatter estimator.",
        ("rousseeuw_vandriessen_1999", "maronna_etal_2019"),
        "Estimator cloning, contamination/quantile thresholding, score API, diagnostics, and sklearn-compatible behavior.",
        "This is a package workflow over established robust Mahalanobis-distance ideas.",
    ),
    "AutoRobustAnomalyDetector": _entry(
        "AutoRobustAnomalyDetector", "Anomaly detection", "robustcov_composite",
        "Automatic robust scatter selection followed by robust-distance anomaly scoring.",
        ("maronna_etal_2019", "rousseeuw_vandriessen_1999", "tyler_1987"),
        "Automatic candidate fitting, contamination thresholding, diagnostics, and end-to-end anomaly API.",
        "The selection workflow is package-specific and should not be interpreted as a universally optimal statistical selector.",
    ),
    "ClusterRobustOutlierDetector": _entry(
        "ClusterRobustOutlierDetector", "Anomaly detection", "robustcov_composite",
        "Cluster-conditioned robust-distance detection for multimodal data.",
        ("maronna_etal_2019",),
        "Composition of clustering and within-cluster robust scatter, with cluster-aware distances and diagnostics.",
        "This is an applied robustcov workflow rather than a claim of a new robust clustering theory.",
    ),
    "AutoRobustScatter": _entry(
        "AutoRobustScatter", "Estimator selection", "robustcov_composite",
        "Automatic selection among robust scatter candidates using stability and fit diagnostics.",
        ("maronna_etal_2019", "rousseeuw_vandriessen_1999", "tyler_1987"),
        "Candidate registry, reproducible stability scoring, selection diagnostics, and fallback behavior.",
        "The selection criterion and defaults are package-specific engineering choices, not a published universal decision rule.",
    ),
    "OnlineRobustSubspaceTracker": _entry(
        "OnlineRobustSubspaceTracker", "Online subspace tracking", "robustcov_composite",
        "Bounded-memory robust mini-batch tracking of a slowly changing principal subspace with projected-residual cell repair and dense-row rejection.",
        ("narayanamurthy_vaswani_2018", "zhu_shen_2022", "hubert_etal_2005"),
        "A sklearn-style streaming workflow combining robust PCA initialization, projected-residual screening, bounded recent-sample updates, projector interpolation, diagnostics, and slow-change safeguards.",
        "This experimental estimator is inspired by robust subspace-tracking research but is not NORST: it does not solve projected l1 recovery and carries no NORST support-recovery or tracking-delay guarantee.",
    ),
    "RobustSubspaceMonitor": _entry(
        "RobustSubspaceMonitor", "Monitoring", "robustcov_composite",
        "Reference-versus-batch monitoring using robust PCA score and orthogonal distances.",
        ("hubert_etal_2005", "bhatia_2007"),
        "Calibration, rolling comparison, scale/permutation invariance, drift summaries, and plotting integration.",
        "The monitoring workflow is package-specific and combines established robust PCA and matrix-geometry concepts.",
    ),
    "ConformalAlertCalibrator": _entry(
        "ConformalAlertCalibrator", "Monitoring", "literature_adaptation",
        "Split-conformal conversion of arbitrary anomaly scores into conservative finite-sample p-values and alert labels.",
        ("vovk_etal_2005", "bashari_etal_2025"),
        "Score-model-agnostic sklearn-style calibration, explicit p-value resolution diagnostics, deterministic tie handling, and monitoring-example integration.",
        "The implementation provides ordinary split-conformal marginal calibration under exchangeability; it does not implement active cleaning, shift correction, online error control, or conditional guarantees.",
    ),
    "SubspaceStability": _entry(
        "SubspaceStability", "Uncertainty/stability", "robustcov_composite",
        "Bootstrap diagnostics for PCA loadings, eigenvalues, explained variance, and principal angles.",
        ("timmerman_etal_2007", "kunsch_1989", "politis_romano_1994"),
        "IID, moving-block, circular-block, stationary, and cluster bootstrap designs around arbitrary PCA estimators.",
        "The class is a package integration of established bootstrap and subspace-stability ideas.",
    ),
    "FeatureGeometry": _entry(
        "FeatureGeometry", "Robust geometry", "robustcov_composite",
        "Robust Mahalanobis geometry for learned feature spaces.",
        ("bhatia_2007", "pennec_etal_2006", "maronna_etal_2019"),
        "Unified fitting, robust distances, whitening, similarity kernels, SPD safeguards, and OOD/monitoring workflows.",
        "The geometry workflow is package-specific; its component scatter and SPD operations are established methods.",
    ),
    "ClassConditionalFeatureGeometry": _entry(
        "ClassConditionalFeatureGeometry", "Robust geometry", "robustcov_composite",
        "Class-conditional robust feature geometry and nearest-class robust distances.",
        ("bhatia_2007", "maronna_etal_2019"),
        "Per-class estimator composition, distance aggregation, OOD scoring, and common API.",
        "This is a robustcov workflow over established class-conditional Mahalanobis geometry.",
    ),
    "RobustInputMetric": _entry(
        "RobustInputMetric", "Kernel/input geometry", "robustcov_composite",
        "Reusable robust Mahalanobis input metric for kernels, Gaussian processes, and similarity methods.",
        ("bhatia_2007", "pennec_etal_2006"),
        "Estimator-to-metric adapter, pairwise distance implementation, sklearn/GPyTorch integration, and scale handling.",
        "This is software infrastructure connecting robust scatter to downstream metric-based models.",
    ),
    "RobustMedianImputer": _entry(
        "RobustMedianImputer", "Preprocessing", "robustcov_utility",
        "Columnwise median imputation for missing values.",
        ("maronna_etal_2019",),
        "Small dependency-free transformer with sklearn-compatible parameter and fitted-state behavior.",
        "Median imputation is a standard preprocessing rule; no methodological novelty is claimed.",
    ),
    "joint_diagonalize_symmetric": _entry(
        "joint_diagonalize_symmetric", "Numerical algorithms", "literature_adaptation",
        "Jacobi-style approximate joint diagonalization of real symmetric matrices.",
        ("cardoso_souloumiac_1996",),
        "Numerically matched Python and C++ backends, convergence diagnostics, and a complete-estimator acceleration gate.",
        "The Jacobi method is established; the native implementation and package API are robustcov engineering contributions.",
    ),
    "minimum_distance_index": _entry(
        "minimum_distance_index", "Source-separation metrics", "literature_implementation",
        "Permutation/scale-aware minimum-distance index for ICA/BSS recovery.",
        ("ilmonen_etal_2010",),
        "NumPy implementation integrated with robustcov source-separation benchmarks and diagnostics.",
        "No methodological novelty is claimed.",
    ),
    "amari_index": _entry(
        "amari_index", "Source-separation metrics", "literature_implementation",
        "Amari-style permutation/scale-aware source-separation performance index.",
        ("amari_1995",),
        "NumPy implementation exposed alongside the preferred minimum-distance index.",
        "No methodological novelty is claimed.",
    ),
}


_ALIAS_TO_CANONICAL = {
    alias: name
    for name, entry in METHOD_PROVENANCE.items()
    for alias in entry.aliases
}


EXPERIMENTAL_ESTIMATOR_PROVENANCE_NAMES = (
    "DistributionallyRobustPCA",
    "OnlineRobustSubspaceTracker",
)


PUBLIC_ESTIMATOR_PROVENANCE_NAMES = (
    "FastMCD", "MRCD", "KMRCD", "DetS", "DetMM", "TylerShape",
    "RegularizedTyler", "IterativeMScatter", "StudentTScatter",
    "RegularizedCauchy", "KLRegularizedTyler", "WieselTyler",
    "HellingerRegularizedTyler", "CellMCD", "CellRCov", "MatrixMCD",
    "RobustPCA", "DensityPowerRobustPCA", "CellPCA", "SparseCellPCA",
    "RobustMultilinearPCA", "RobustGraphicalLasso", "SGLASSO",
    "TwoScatterICA", "SOBI", "RobustSOBI", "RobustFactorModel",
    "RobustOutlierDetector", "AutoRobustAnomalyDetector",
    "ClusterRobustOutlierDetector", "AutoRobustScatter",
    "RobustSubspaceMonitor", "ConformalAlertCalibrator",
    "SubspaceStability", "FeatureGeometry",
    "ClassConditionalFeatureGeometry", "RobustInputMetric",
    "RobustMedianImputer",
)


def canonical_method_name(method: Any) -> str:
    """Return the canonical provenance name for a name, class, or instance."""

    if isinstance(method, str):
        name = method
    elif isinstance(method, type):
        name = method.__name__
    elif callable(method) and hasattr(method, "__name__"):
        name = method.__name__
    else:
        name = type(method).__name__
    return _ALIAS_TO_CANONICAL.get(name, name)


def get_method_provenance(method: Any) -> MethodProvenance:
    """Return provenance metadata for a public estimator or algorithm.

    Parameters
    ----------
    method : str, class, or estimator instance
        Canonical name, public alias, class, or fitted/unfitted instance.
    """

    name = canonical_method_name(method)
    try:
        return METHOD_PROVENANCE[name]
    except KeyError as exc:
        raise KeyError(f"no robustcov method provenance is registered for {name!r}") from exc


def iter_method_provenance(*, family: str | None = None) -> tuple[MethodProvenance, ...]:
    """Return registered provenance entries, optionally restricted by family."""

    entries = tuple(METHOD_PROVENANCE.values())
    if family is not None:
        entries = tuple(entry for entry in entries if entry.family == family)
    return tuple(sorted(entries, key=lambda entry: (entry.family.lower(), entry.name.lower())))


def attach_method_provenance(namespace: Mapping[str, Any]) -> None:
    """Attach read-only-style provenance attributes to public classes/functions.

    The registry remains the source of truth.  Attributes are attached for API
    discoverability and autodoc output; callers should use
    :func:`get_method_provenance` when they need the complete record.
    """

    for public_name, value in namespace.items():
        canonical = _ALIAS_TO_CANONICAL.get(public_name, public_name)
        entry = METHOD_PROVENANCE.get(canonical)
        if entry is None or not (isinstance(value, type) or callable(value)):
            continue
        setattr(value, "canonical_method_name", entry.name)
        setattr(value, "method_status", entry.status)
        setattr(value, "method_references", entry.references)
        setattr(value, "robustcov_contribution", entry.robustcov_contribution)
        setattr(value, "implementation_notes", entry.implementation_notes)
        setattr(value, "method_provenance", entry)


__all__ = [
    "Reference",
    "MethodProvenance",
    "STATUS_LABELS",
    "REFERENCE_CATALOG",
    "METHOD_PROVENANCE",
    "PUBLIC_ESTIMATOR_PROVENANCE_NAMES",
    "EXPERIMENTAL_ESTIMATOR_PROVENANCE_NAMES",
    "canonical_method_name",
    "get_method_provenance",
    "iter_method_provenance",
    "attach_method_provenance",
]
