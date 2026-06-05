"""robustcov MVP: efficient robust covariance and shape estimators."""

from .covariance import FastMCD, MinCovDet, TylerShape, RegularizedTyler
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
)

__all__ = [
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
    "has_openmp",
    "get_num_threads",
    "set_num_threads",
    "thread_limit",
    "top_k_mask",
    "scores_to_submission",
]

__version__ = "0.0.1"
