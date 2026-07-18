# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""robustcov MVP: efficient robust covariance and shape estimators."""

from .covariance import FastMCD, MinCovDet, TylerShape, RegularizedTyler
from .mrcd import MinimumRegularizedCovarianceDeterminant, MRCD, MinRegularizedCovDet
from .kernel_mrcd import KernelMinimumRegularizedCovarianceDeterminant, KernelMRCD, KMRCD
from .s_estimators import DeterministicSEstimator, DeterministicMMEstimator, DetS, DetMM
from .mmcd import MatrixMinimumCovarianceDeterminant, MatrixMCD, MMCD
from .cellmcd import CellwiseMinimumCovarianceDeterminant, CellMCD, CellwiseMCD
from .cellpca import CellwiseRobustPCA, CellPCA, CasewiseCellwisePCA
from .sparse_cellpca import (
    SparseCellwiseRobustPCA,
    SparseCellPCA,
    SparseCasewiseCellwisePCA,
)
from .cellrcov import CellwiseRegularizedCovariance, CellRCov, CellwiseRobustCovariance
from .sparse_precision import RobustGraphicalLasso, SparseRobustPrecision
from .stability import SubspaceStability
from .m_estimators import (
    IterativeMScatter,
    StudentTScatter,
    RegularizedCauchy,
    KLRegularizedTyler,
    WieselTyler,
    HellingerRegularizedTyler,
)
from .outliers import RobustOutlierDetector, AutoRobustAnomalyDetector
from .auto import AutoRobustScatter
from .multimodal import ClusterRobustOutlierDetector
from .preprocessing import RobustMedianImputer
from .diagnostics import diagnostic_report, RobustDiagnosticReport
from .parallel import has_openmp, get_num_threads, set_num_threads, thread_limit
from .external import top_k_mask, scores_to_submission
from .metrics import RobustInputMetric, pairwise_mahalanobis_squared
from .kernels import robust_rbf_kernel, robust_matern_kernel
from .plotting import (
    plot_mahalanobis_diagnostics,
    plot_mahalanobis_qq,
    plot_distance_histogram,
    plot_covariance_heatmap,
    plot_benchmark_curve,
    plot_benchmark_bars,
    plot_anomaly_scatter_2d,
    plot_distance_scatter_2d,
    plot_speed_accuracy_pareto,
    plot_robust_distance_profile,
    plot_robust_distance_panel,
    plot_cluster_robust_distances,
    plot_robust_pca_outlier_map,
    plot_subspace_monitor_history,
    plot_matrix_outlier_contributions,
    plot_cellwise_residual_map,
    plot_cellpca_outlier_map,
    plot_sparse_cellpca_loadings,
    plot_partial_correlation_network,
    plot_subspace_stability,
)

__all__ = [
    "RobustPCA",
    "SubspaceStability",
    "FeatureGeometry",
    "ClassConditionalFeatureGeometry",
    "RobustSubspaceMonitor",
    "SubspaceDriftResult",
    "CellwiseRegularizedCovariance",
    "CellRCov",
    "CellwiseRobustCovariance",
    "SparseCellwiseRobustPCA",
    "SparseCellPCA",
    "SparseCasewiseCellwisePCA",
    "CellwiseRobustPCA",
    "CellPCA",
    "CasewiseCellwisePCA",
    "RobustGraphicalLasso",
    "SparseRobustPrecision",
    "CellwiseMinimumCovarianceDeterminant",
    "CellMCD",
    "CellwiseMCD",
    "MatrixMinimumCovarianceDeterminant",
    "MatrixMCD",
    "MMCD",
    "KernelMinimumRegularizedCovarianceDeterminant",
    "KernelMRCD",
    "KMRCD",
    "DeterministicSEstimator",
    "DeterministicMMEstimator",
    "DetS",
    "DetMM",
    "MinimumRegularizedCovarianceDeterminant",
    "MRCD",
    "MinRegularizedCovDet",
    "FastMCD",
    "MinCovDet",
    "TylerShape",
    "RegularizedTyler",
    "RobustOutlierDetector",
    "AutoRobustAnomalyDetector",
    "RobustMedianImputer",
    "IterativeMScatter",
    "StudentTScatter",
    "RegularizedCauchy",
    "KLRegularizedTyler",
    "WieselTyler",
    "HellingerRegularizedTyler",
    "AutoRobustScatter",
    "ClusterRobustOutlierDetector",
    "diagnostic_report",
    "RobustDiagnosticReport",
    "plot_mahalanobis_diagnostics",
    "plot_mahalanobis_qq",
    "plot_distance_histogram",
    "plot_covariance_heatmap",
    "plot_benchmark_curve",
    "plot_benchmark_bars",
    "plot_anomaly_scatter_2d",
    "plot_distance_scatter_2d",
    "plot_speed_accuracy_pareto",
    "plot_robust_distance_profile",
    "plot_robust_distance_panel",
    "plot_cluster_robust_distances",
    "plot_robust_pca_outlier_map",
    "plot_subspace_monitor_history",
    "plot_matrix_outlier_contributions",
    "plot_cellwise_residual_map",
    "plot_cellpca_outlier_map",
    "plot_sparse_cellpca_loadings",
    "plot_partial_correlation_network",
    "plot_subspace_stability",
    "has_openmp",
    "get_num_threads",
    "set_num_threads",
    "thread_limit",
    "top_k_mask",
    "scores_to_submission",
    "RobustInputMetric",
    "pairwise_mahalanobis_squared",
    "robust_rbf_kernel",
    "robust_matern_kernel",
]

__version__ = "0.0.2"

from .features import FeatureGeometry, ClassConditionalFeatureGeometry
from .pca import RobustPCA

from .monitoring import RobustSubspaceMonitor, SubspaceDriftResult
