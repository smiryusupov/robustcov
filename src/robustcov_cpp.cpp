// Copyright 2026 Shohruh Miryusupov
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

static int get_max_threads_cpp() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

static void set_num_threads_cpp(int n_threads) {
#ifdef _OPENMP
    if (n_threads < 1) n_threads = 1;
    omp_set_num_threads(n_threads);
#else
    (void)n_threads;
#endif
}

static bool has_openmp_cpp() {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

#ifdef _OPENMP
static int effective_threads_cpp(int tasks, int min_tasks_per_thread = 128) {
    int mx = omp_get_max_threads();
    int by_work = std::max(1, tasks / std::max(1, min_tasks_per_thread));
    return std::max(1, std::min(mx, by_work));
}
#endif

static int checked_dimension(py::ssize_t value, const char* name, py::ssize_t minimum = 1) {
    if (value < minimum) {
        std::ostringstream message;
        message << name << " must be at least " << minimum;
        throw std::invalid_argument(message.str());
    }
    if (value > static_cast<py::ssize_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(std::string(name) + " exceeds the native integer range");
    }
    return static_cast<int>(value);
}

static size_t checked_product(size_t left, size_t right, const char* name) {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
        throw std::overflow_error(std::string(name) + " is too large");
    }
    return left * right;
}

static size_t checked_matrix_elements(int rows, int columns, const char* name) {
    if (rows < 0 || columns < 0) {
        throw std::invalid_argument(std::string(name) + " has negative dimensions");
    }
    return checked_product(static_cast<size_t>(rows), static_cast<size_t>(columns), name);
}

static int checked_int_product(int left, int right, const char* name) {
    if (left < 0 || right < 0) {
        throw std::invalid_argument(std::string(name) + " has negative dimensions");
    }
    const size_t result = checked_product(
        static_cast<size_t>(left), static_cast<size_t>(right), name
    );
    if (result > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(std::string(name) + " exceeds the native integer range");
    }
    return static_cast<int>(result);
}

static void require_all_finite(const double* values, size_t count, const char* name) {
    for (size_t index = 0; index < count; ++index) {
        if (!std::isfinite(values[index])) {
            throw std::invalid_argument(std::string(name) + " contains NaN or infinity");
        }
    }
}

struct Mat {
    int n = 0, p = 0;
    std::vector<double> a;
    Mat() = default;
    Mat(int n_, int p_, double v = 0.0)
        : n(n_), p(p_), a(checked_matrix_elements(n_, p_, "native matrix"), v) {}
    double& operator()(int i, int j) { return a[static_cast<size_t>(i) * p + j]; }
    double operator()(int i, int j) const { return a[static_cast<size_t>(i) * p + j]; }
};

static Mat numpy_to_mat(py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    auto b = arr.request();
    if (b.ndim != 2) throw std::invalid_argument("X must be a 2D float64 array");
    const int n = checked_dimension(b.shape[0], "X rows", 2);
    const int p = checked_dimension(b.shape[1], "X columns");
    const size_t count = checked_product(
        static_cast<size_t>(n), static_cast<size_t>(p), "X element count"
    );
    Mat X(n, p);
    const double* ptr = static_cast<const double*>(b.ptr);
    require_all_finite(ptr, count, "X");
    std::copy(ptr, ptr + count, X.a.begin());
    return X;
}

static py::array_t<double> vec_to_numpy(const std::vector<double>& v) {
    py::array_t<double> out(v.size());
    std::copy(v.begin(), v.end(), static_cast<double*>(out.request().ptr));
    return out;
}

static py::array_t<double> mat_to_numpy(const Mat& M) {
    py::array_t<double> out({M.n, M.p});
    std::copy(M.a.begin(), M.a.end(), static_cast<double*>(out.request().ptr));
    return out;
}

static std::vector<double> column_mean(const Mat& X, const std::vector<int>* idx = nullptr) {
    int n = idx ? static_cast<int>(idx->size()) : X.n;
    std::vector<double> m(X.p, 0.0);
#ifdef _OPENMP
    int nt = effective_threads_cpp(n, 256);
    std::vector<std::vector<double>> locals(static_cast<size_t>(nt), std::vector<double>(X.p, 0.0));
#pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num();
        auto& lm = locals[static_cast<size_t>(tid)];
#pragma omp for nowait
        for (int rr = 0; rr < n; ++rr) {
            int i = idx ? (*idx)[rr] : rr;
            for (int j = 0; j < X.p; ++j) lm[j] += X(i, j);
        }
    }
    for (const auto& lm : locals) for (int j = 0; j < X.p; ++j) m[j] += lm[j];
#else
    for (int rr = 0; rr < n; ++rr) {
        int i = idx ? (*idx)[rr] : rr;
        for (int j = 0; j < X.p; ++j) m[j] += X(i, j);
    }
#endif
    for (double& x : m) x /= std::max(1, n);
    return m;
}

static Mat covariance_from_indices(const Mat& X, const std::vector<int>& idx, const std::vector<double>& loc, double ridge = 1e-9) {
    int h = static_cast<int>(idx.size());
    Mat C(X.p, X.p, 0.0);
    if (h <= 1) throw std::runtime_error("not enough points for covariance");
#ifdef _OPENMP
    int nt = effective_threads_cpp(h, 128);
    std::vector<Mat> locals;
    locals.reserve(static_cast<size_t>(nt));
    for (int t = 0; t < nt; ++t) locals.emplace_back(X.p, X.p, 0.0);
#pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num();
        Mat& LC = locals[static_cast<size_t>(tid)];
#pragma omp for nowait
        for (int rr = 0; rr < h; ++rr) {
            int i = idx[rr];
            for (int j = 0; j < X.p; ++j) {
                double xj = X(i, j) - loc[j];
                for (int k = 0; k <= j; ++k) LC(j, k) += xj * (X(i, k) - loc[k]);
            }
        }
    }
    for (const Mat& LC : locals) for (size_t q = 0; q < C.a.size(); ++q) C.a[q] += LC.a[q];
#else
    for (int rr = 0; rr < h; ++rr) {
        int i = idx[rr];
        for (int j = 0; j < X.p; ++j) {
            double xj = X(i, j) - loc[j];
            for (int k = 0; k <= j; ++k) C(j, k) += xj * (X(i, k) - loc[k]);
        }
    }
#endif
    double denom = static_cast<double>(std::max(1, h - 1));
    double average_variance = 0.0;
    for (int j = 0; j < X.p; ++j) {
        for (int k = 0; k <= j; ++k) {
            C(j, k) /= denom;
            C(k, j) = C(j, k);
        }
        average_variance += C(j, j);
    }
    average_variance /= static_cast<double>(std::max(1, X.p));
    // Treat ridge as relative to the data scale.  A fixed absolute ridge makes
    // covariance estimates depend on the choice of measurement units and can
    // dominate otherwise valid small-valued data.  Keep the historical absolute
    // fallback only for an exactly degenerate subset.
    double effective_ridge = ridge;
    if (std::isfinite(average_variance) && average_variance > std::numeric_limits<double>::min()) {
        effective_ridge = ridge * average_variance;
    }
    for (int j = 0; j < X.p; ++j) C(j, j) += effective_ridge;
    return C;
}

static Mat covariance_all(const Mat& X, const std::vector<double>& loc, double ridge = 1e-9) {
    std::vector<int> idx(X.n);
    std::iota(idx.begin(), idx.end(), 0);
    return covariance_from_indices(X, idx, loc, ridge);
}

static bool cholesky_lower(const Mat& A, Mat& L) {
    int p = A.n;
    L = Mat(p, p, 0.0);
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = A(i, j);
            for (int k = 0; k < j; ++k) sum -= L(i, k) * L(j, k);
            if (i == j) {
                if (sum <= 0.0 || !std::isfinite(sum)) return false;
                L(i, j) = std::sqrt(sum);
            } else {
                L(i, j) = sum / L(j, j);
            }
        }
    }
    return true;
}

static Mat inverse_spd(Mat A) {
    int p = A.n;
    Mat L;
    double matrix_scale = 0.0;
    for (int j = 0; j < p; ++j) matrix_scale += std::abs(A(j, j));
    matrix_scale /= static_cast<double>(std::max(1, p));
    if (!std::isfinite(matrix_scale) || matrix_scale <= std::numeric_limits<double>::min()) matrix_scale = 1.0;
    double ridge = 1e-10 * matrix_scale;
    for (int tries = 0; tries < 8; ++tries) {
        if (cholesky_lower(A, L)) break;
        for (int j = 0; j < p; ++j) A(j, j) += ridge;
        ridge *= 10.0;
    }
    if (!cholesky_lower(A, L)) throw std::runtime_error("matrix is not positive definite");

    Mat inv(p, p, 0.0);
    for (int col = 0; col < p; ++col) {
        std::vector<double> y(p, 0.0), x(p, 0.0);
        for (int i = 0; i < p; ++i) {
            double rhs = (i == col) ? 1.0 : 0.0;
            for (int k = 0; k < i; ++k) rhs -= L(i, k) * y[k];
            y[i] = rhs / L(i, i);
        }
        for (int i = p - 1; i >= 0; --i) {
            double rhs = y[i];
            for (int k = i + 1; k < p; ++k) rhs -= L(k, i) * x[k];
            x[i] = rhs / L(i, i);
        }
        for (int i = 0; i < p; ++i) inv(i, col) = x[i];
    }
    for (int i = 0; i < p; ++i) for (int j = 0; j < i; ++j) {
        double s = 0.5 * (inv(i, j) + inv(j, i));
        inv(i, j) = inv(j, i) = s;
    }
    return inv;
}

static double logdet_spd(Mat A) {
    Mat L;
    double matrix_scale = 0.0;
    for (int j = 0; j < A.n; ++j) matrix_scale += std::abs(A(j, j));
    matrix_scale /= static_cast<double>(std::max(1, A.n));
    if (!std::isfinite(matrix_scale) || matrix_scale <= std::numeric_limits<double>::min()) matrix_scale = 1.0;
    double ridge = 1e-10 * matrix_scale;
    for (int tries = 0; tries < 8; ++tries) {
        if (cholesky_lower(A, L)) {
            double v = 0.0;
            for (int j = 0; j < A.n; ++j) v += 2.0 * std::log(std::max(L(j, j), 1e-300));
            return v;
        }
        for (int j = 0; j < A.n; ++j) A(j, j) += ridge;
        ridge *= 10.0;
    }
    return std::numeric_limits<double>::infinity();
}

static std::vector<double> mahalanobis2(const Mat& X, const std::vector<double>& loc, const Mat& precision) {
    std::vector<double> d(X.n, 0.0);
#ifdef _OPENMP
    int nt = effective_threads_cpp(X.n, 256);
#pragma omp parallel for schedule(static) num_threads(nt) if(nt > 1)
#endif
    for (int i = 0; i < X.n; ++i) {
        std::vector<double> tmp(X.p, 0.0);
        for (int j = 0; j < X.p; ++j) {
            double xj = X(i, j) - loc[j];
            for (int k = 0; k < X.p; ++k) tmp[k] += precision(k, j) * xj;
        }
        double s = 0.0;
        for (int k = 0; k < X.p; ++k) s += (X(i, k) - loc[k]) * tmp[k];
        d[i] = std::max(0.0, s);
    }
    return d;
}

static double trace(const Mat& A) {
    double t = 0.0;
    for (int i = 0; i < std::min(A.n, A.p); ++i) t += A(i, i);
    return t;
}

static py::dict fit_tyler_cpp(py::array_t<double, py::array::c_style | py::array::forcecast> arr,
                              int max_iter, double tol, double regularization,
                              bool assume_centered) {
    Mat X = numpy_to_mat(arr);
    int n = X.n, p = X.p;
    if (max_iter < 1) throw std::invalid_argument("max_iter must be positive");
    if (!(tol > 0.0) || !std::isfinite(tol))
        throw std::invalid_argument("tol must be positive and finite");
    if (!std::isfinite(regularization) || regularization < 0.0 || regularization >= 1.0)
        throw std::invalid_argument("regularization must be finite and in [0, 1)");
    if (regularization <= 0.0 && n <= p) throw std::invalid_argument("Unregularized Tyler requires n_samples > n_features");

    std::vector<double> loc(p, 0.0);
    if (!assume_centered) loc = column_mean(X);
    Mat S = covariance_all(X, loc, 1e-6);
    double tr = trace(S);
    if (tr <= 0) tr = 1.0;
    for (double& v : S.a) v *= static_cast<double>(p) / tr;

    bool converged = false;
    int it = 0;
    for (; it < max_iter; ++it) {
        Mat P = inverse_spd(S);
        Mat Snew(p, p, 0.0);
#ifdef _OPENMP
        int nt = effective_threads_cpp(n, 128);
        std::vector<Mat> locals;
        locals.reserve(static_cast<size_t>(nt));
        for (int t = 0; t < nt; ++t) locals.emplace_back(p, p, 0.0);
#pragma omp parallel num_threads(nt)
        {
            int tid = omp_get_thread_num();
            Mat& LS = locals[static_cast<size_t>(tid)];
#pragma omp for nowait
            for (int i = 0; i < n; ++i) {
                double d2 = 0.0;
                for (int j = 0; j < p; ++j) for (int k = 0; k < p; ++k)
                    d2 += (X(i, j) - loc[j]) * P(j, k) * (X(i, k) - loc[k]);
                if (!std::isfinite(d2) || d2 <= std::numeric_limits<double>::min()) continue;
                double w = static_cast<double>(p) / d2;
                for (int j = 0; j < p; ++j) {
                    double xj = X(i, j) - loc[j];
                    for (int k = 0; k <= j; ++k) LS(j, k) += w * xj * (X(i, k) - loc[k]);
                }
            }
        }
        for (const Mat& LS : locals) for (size_t q = 0; q < Snew.a.size(); ++q) Snew.a[q] += LS.a[q];
#else
        for (int i = 0; i < n; ++i) {
            double d2 = 0.0;
            for (int j = 0; j < p; ++j) for (int k = 0; k < p; ++k)
                d2 += (X(i, j) - loc[j]) * P(j, k) * (X(i, k) - loc[k]);
            if (!std::isfinite(d2) || d2 <= std::numeric_limits<double>::min()) continue;
            double w = static_cast<double>(p) / d2;
            for (int j = 0; j < p; ++j) {
                double xj = X(i, j) - loc[j];
                for (int k = 0; k <= j; ++k) Snew(j, k) += w * xj * (X(i, k) - loc[k]);
            }
        }
#endif
        for (int j = 0; j < p; ++j) {
            for (int k = 0; k <= j; ++k) {
                Snew(j, k) /= static_cast<double>(n);
                Snew(k, j) = Snew(j, k);
            }
        }
        if (regularization > 0.0) {
            for (double& v : Snew.a) v *= (1.0 - regularization);
            for (int j = 0; j < p; ++j) Snew(j, j) += regularization;
        }
        double trn = trace(Snew);
        for (double& v : Snew.a) v *= static_cast<double>(p) / std::max(trn, 1e-300);
        double diff = 0.0, base = 0.0;
        for (size_t q = 0; q < S.a.size(); ++q) {
            double e = Snew.a[q] - S.a[q];
            diff += e * e;
            base += S.a[q] * S.a[q];
        }
        S = std::move(Snew);
        if (std::sqrt(diff / std::max(base, 1e-300)) < tol) { converged = true; ++it; break; }
    }
    Mat P = inverse_spd(S);
    auto d = mahalanobis2(X, loc, P);
    py::dict out;
    out["location"] = vec_to_numpy(loc);
    out["shape"] = mat_to_numpy(S);
    out["covariance"] = mat_to_numpy(S);
    out["precision"] = mat_to_numpy(P);
    out["distances"] = vec_to_numpy(d);
    out["n_iter"] = it;
    out["converged"] = converged;
    return out;
}


struct MCDCandidate {
    double logdet = std::numeric_limits<double>::infinity();
    int iterations = 0;
    bool converged = false;
    std::vector<int> idx;
    std::vector<double> loc;
    Mat cov;
};

static std::vector<int> smallest_indices(const std::vector<double>& dist, int h) {
    std::vector<int> order(static_cast<int>(dist.size()));
    std::iota(order.begin(), order.end(), 0);
    h = std::max(0, std::min(h, static_cast<int>(order.size())));
    if (h < static_cast<int>(order.size())) {
        std::nth_element(order.begin(), order.begin() + h, order.end(),
            [&](int a, int b) { return dist[a] < dist[b]; });
        order.resize(h);
    }
    return order;
}

static MCDCandidate run_c_steps(const Mat& X,
                                std::vector<int> idx,
                                int h,
                                int max_steps,
                                double tol,
                                double ridge) {
    if (max_steps < 1) throw std::invalid_argument("C-step count must be positive");

    MCDCandidate cand;
    std::vector<double> loc = column_mean(X, &idx);
    Mat cov = covariance_from_indices(X, idx, loc, ridge);
    double prev_ld = logdet_spd(cov);

    int step = 0;
    bool converged = false;
    for (; step < max_steps; ++step) {
        Mat P;
        try { P = inverse_spd(cov); }
        catch (...) { break; }
        std::vector<double> dist = mahalanobis2(X, loc, P);
        std::vector<int> next_idx = smallest_indices(dist, h);
        std::vector<double> next_loc = column_mean(X, &next_idx);
        Mat next_cov = covariance_from_indices(X, next_idx, next_loc, ridge);
        double ld = logdet_spd(next_cov);
        if (!std::isfinite(ld)) break;

        idx = std::move(next_idx);
        loc = std::move(next_loc);
        cov = std::move(next_cov);

        if (std::abs(prev_ld - ld) <= tol * (1.0 + std::abs(prev_ld))) {
            converged = true;
            ++step;
            break;
        }
        // The C-step should not increase determinant. Small numerical increases can happen because
        // of the ridge term, so do not abort; simply continue from the new subset.
        prev_ld = ld;
    }
    cand.logdet = logdet_spd(cov);
    cand.iterations = step;
    cand.converged = converged;
    cand.idx = std::move(idx);
    cand.loc = std::move(loc);
    cand.cov = std::move(cov);
    return cand;
}


static void scale_matrix_inplace(Mat& A, double scale) {
    if (!std::isfinite(scale) || scale <= 0.0) return;
    for (double& x : A.a) x *= scale;
}

static std::vector<int> deterministic_median_start(const Mat& X, int h) {
    int n = X.n, p = X.p;
    std::vector<double> med(p, 0.0);
    for (int j = 0; j < p; ++j) {
        std::vector<double> col(n);
        for (int i = 0; i < n; ++i) col[i] = X(i, j);
        std::nth_element(col.begin(), col.begin() + n/2, col.end());
        med[j] = col[n/2];
    }
    std::vector<double> d(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = 0; j < p; ++j) {
            double e = X(i, j) - med[j];
            s += e * e;
        }
        d[i] = s;
    }
    return smallest_indices(d, h);
}

struct FastMCDParameters {
    int h;
    int n_best;
};

struct FastMCDResult {
    std::vector<double> location;
    Mat covariance;
    Mat precision;
    std::vector<double> distances;
    std::vector<unsigned char> support;
    Mat raw_covariance;
    std::vector<double> raw_distances;
    std::vector<unsigned char> raw_support;
    double consistency_factor;
    bool reweighted;
};

static FastMCDParameters validate_fast_mcd_parameters(
    const Mat& X,
    double support_fraction,
    int n_init,
    int max_iter,
    double tol,
    int n_best,
    int initial_c_steps,
    double raw_consistency_factor,
    double reweight_cutoff,
    double reweight_consistency_factor
) {
    const int n = X.n;
    const int p = X.p;
    if (!(support_fraction == -1.0 ||
          (std::isfinite(support_fraction) && support_fraction > 0.0 && support_fraction <= 1.0))) {
        throw std::invalid_argument("support_fraction must be -1 or finite and in (0, 1]");
    }
    if (!(tol > 0.0) || !std::isfinite(tol))
        throw std::invalid_argument("tol must be positive and finite");
    if (n <= p) throw std::invalid_argument("FastMCD requires n_samples > n_features for this MVP");
    if (n_init < 1) throw std::invalid_argument("n_init must be positive");
    if (max_iter < 1) throw std::invalid_argument("max_iter must be positive");
    if (n_best < 1) throw std::invalid_argument("n_best must be positive");
    if (initial_c_steps < 1) throw std::invalid_argument("initial_c_steps must be positive");
    if (!std::isfinite(raw_consistency_factor) || raw_consistency_factor <= 0.0)
        throw std::invalid_argument("raw_consistency_factor must be finite and positive");
    if (!std::isfinite(reweight_cutoff) || reweight_cutoff <= 0.0)
        throw std::invalid_argument("reweight_cutoff must be finite and positive");
    if (!std::isfinite(reweight_consistency_factor) || reweight_consistency_factor <= 0.0)
        throw std::invalid_argument("reweight_consistency_factor must be finite and positive");

    int h;
    if (support_fraction < 0.0) h = (n + p + 1) / 2;
    else h = static_cast<int>(std::floor(support_fraction * n));
    h = std::max(p + 1, std::min(h, n));

    const int maximum_best = n_init == std::numeric_limits<int>::max()
        ? n_init
        : n_init + 1;
    return {h, std::min(n_best, maximum_best)};
}

static void retain_mcd_candidate(
    std::vector<MCDCandidate>& pool,
    MCDCandidate&& candidate,
    int h,
    int n_best
) {
    if (!std::isfinite(candidate.logdet) || static_cast<int>(candidate.idx.size()) != h) return;
    pool.push_back(std::move(candidate));
    std::sort(pool.begin(), pool.end(), [](const MCDCandidate& left, const MCDCandidate& right) {
        return left.logdet < right.logdet;
    });
    if (static_cast<int>(pool.size()) > n_best) pool.resize(n_best);
}

static std::vector<MCDCandidate> initialize_mcd_candidates(
    const Mat& X,
    int h,
    int n_init,
    int n_best,
    int initial_c_steps,
    double tol,
    std::uint64_t seed
) {
    std::vector<MCDCandidate> pool;
    pool.reserve(checked_product(static_cast<size_t>(n_init), 1, "n_init") + 4);

    // Deterministic start: nearest observations to coordinate-wise median. This improves
    // reproducibility and helps easy contamination cases without sacrificing speed.
    try {
        retain_mcd_candidate(
            pool,
            run_c_steps(X, deterministic_median_start(X, h), h, initial_c_steps, tol, 1e-7),
            h,
            n_best
        );
    } catch (...) {}

    // Random elemental subsets. Starting from p+1 points gives a much higher probability
    // of drawing an uncontaminated candidate than starting from a full h-subset.
    // Starts are generated serially for deterministic random_state behavior, then
    // evaluated independently; OpenMP can parallelize this expensive phase.
    std::mt19937_64 rng(seed);
    std::vector<int> all(X.n);
    std::iota(all.begin(), all.end(), 0);
    const int elemental_size = std::min(X.n, X.p + 1);
    std::vector<std::vector<int>> random_starts;
    random_starts.reserve(static_cast<size_t>(n_init));
    for (int init = 0; init < n_init; ++init) {
        std::shuffle(all.begin(), all.end(), rng);
        random_starts.emplace_back(all.begin(), all.begin() + elemental_size);
    }

    std::vector<MCDCandidate> random_candidates(static_cast<size_t>(n_init));
    std::vector<unsigned char> random_ok(static_cast<size_t>(n_init), 0);
#ifdef _OPENMP
    const int init_threads = effective_threads_cpp(n_init, 4);
#pragma omp parallel for schedule(dynamic) num_threads(init_threads) if(init_threads > 1)
#endif
    for (int init = 0; init < n_init; ++init) {
        try {
            MCDCandidate candidate = run_c_steps(
                X,
                random_starts[static_cast<size_t>(init)],
                h,
                initial_c_steps,
                tol,
                1e-7
            );
            if (std::isfinite(candidate.logdet) && static_cast<int>(candidate.idx.size()) == h) {
                random_candidates[static_cast<size_t>(init)] = std::move(candidate);
                random_ok[static_cast<size_t>(init)] = 1;
            }
        } catch (...) {
            // Singular elemental starts are expected occasionally; skip them.
        }
    }
    for (int init = 0; init < n_init; ++init) {
        if (random_ok[static_cast<size_t>(init)]) {
            retain_mcd_candidate(
                pool,
                std::move(random_candidates[static_cast<size_t>(init)]),
                h,
                n_best
            );
        }
    }

    if (pool.empty()) throw std::runtime_error("FastMCD failed to find a valid initial subset");
    return pool;
}

static MCDCandidate polish_mcd_candidates(
    const Mat& X,
    const std::vector<MCDCandidate>& pool,
    int h,
    int max_iter,
    double tol
) {
    std::vector<MCDCandidate> polished_pool(pool.size());
    std::vector<unsigned char> polished_ok(pool.size(), 0);
#ifdef _OPENMP
    const int polish_threads = effective_threads_cpp(static_cast<int>(pool.size()), 1);
#pragma omp parallel for schedule(dynamic) num_threads(polish_threads) if(polish_threads > 1)
#endif
    for (int candidate_index = 0; candidate_index < static_cast<int>(pool.size()); ++candidate_index) {
        try {
            polished_pool[static_cast<size_t>(candidate_index)] = run_c_steps(
                X,
                pool[static_cast<size_t>(candidate_index)].idx,
                h,
                max_iter,
                tol,
                1e-9
            );
            polished_ok[static_cast<size_t>(candidate_index)] = 1;
        } catch (...) {}
    }

    MCDCandidate best;
    for (size_t candidate_index = 0; candidate_index < polished_pool.size(); ++candidate_index) {
        if (!polished_ok[candidate_index]) continue;
        MCDCandidate& polished = polished_pool[candidate_index];
        if (polished.logdet < best.logdet) best = std::move(polished);
    }
    if (best.idx.empty()) throw std::runtime_error("FastMCD failed during final C-steps");
    return best;
}

static FastMCDResult calibrate_and_reweight_mcd(
    const Mat& X,
    const MCDCandidate& best,
    bool reweight,
    double raw_consistency_factor,
    double reweight_cutoff,
    double reweight_consistency_factor
) {
    // Python computes exact Gaussian consistency constants with SciPy. Keeping
    // distribution functions out of the native kernel makes the calibration
    // auditable while C++ remains responsible for subset search and covariance work.
    Mat raw_covariance = best.cov;
    scale_matrix_inplace(raw_covariance, raw_consistency_factor);

    Mat raw_precision = inverse_spd(raw_covariance);
    std::vector<double> raw_distances = mahalanobis2(X, best.loc, raw_precision);
    std::vector<unsigned char> raw_support(X.n, 0);
    for (int index : best.idx) raw_support[index] = 1;

    FastMCDResult result;
    result.location = best.loc;
    result.covariance = raw_covariance;
    result.support = raw_support;
    result.raw_covariance = raw_covariance;
    result.raw_distances = std::move(raw_distances);
    result.raw_support = raw_support;
    result.consistency_factor = raw_consistency_factor;
    result.reweighted = false;

    if (reweight) {
        // Classical MCD reweighting: retain observations within the exact
        // chi-square cutoff supplied by Python, then correct the truncated
        // covariance back to Gaussian consistency.
        std::vector<int> reweighted_indices;
        reweighted_indices.reserve(static_cast<size_t>(X.n));
        for (int i = 0; i < X.n; ++i) {
            if (result.raw_distances[static_cast<size_t>(i)] <= reweight_cutoff) {
                reweighted_indices.push_back(i);
            }
        }
        if (static_cast<int>(reweighted_indices.size()) >= X.p + 1) {
            result.location = column_mean(X, &reweighted_indices);
            result.covariance = covariance_from_indices(X, reweighted_indices, result.location, 1e-9);
            scale_matrix_inplace(result.covariance, reweight_consistency_factor);
            result.consistency_factor = reweight_consistency_factor;
            result.reweighted = true;
            std::fill(result.support.begin(), result.support.end(), 0);
            for (int index : reweighted_indices) result.support[index] = 1;
        }
    }

    result.precision = inverse_spd(result.covariance);
    result.distances = mahalanobis2(X, result.location, result.precision);
    return result;
}

static py::array_t<bool> support_to_numpy(const std::vector<unsigned char>& support) {
    py::array_t<bool> output(static_cast<py::ssize_t>(support.size()));
    auto buffer = output.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(support.size()); ++i) {
        buffer(i) = support[static_cast<size_t>(i)] != 0;
    }
    return output;
}

static py::dict fast_mcd_result_to_dict(
    const MCDCandidate& best,
    const FastMCDResult& result,
    int h,
    bool reweight,
    double raw_consistency_factor,
    double reweight_cutoff
) {
    py::dict output;
    output["location"] = vec_to_numpy(result.location);
    output["shape"] = mat_to_numpy(result.covariance);
    output["covariance"] = mat_to_numpy(result.covariance);
    output["precision"] = mat_to_numpy(result.precision);
    output["distances"] = vec_to_numpy(result.distances);
    output["support"] = support_to_numpy(result.support);
    output["raw_location"] = vec_to_numpy(best.loc);
    output["raw_covariance"] = mat_to_numpy(result.raw_covariance);
    output["raw_scale"] = raw_consistency_factor;
    output["raw_consistency_factor"] = raw_consistency_factor;
    output["consistency_factor"] = result.consistency_factor;
    if (reweight) output["reweight_threshold"] = reweight_cutoff;
    else output["reweight_threshold"] = py::none();
    output["reweighted"] = result.reweighted;
    output["raw_distances"] = vec_to_numpy(result.raw_distances);
    output["raw_support"] = support_to_numpy(result.raw_support);
    output["h"] = h;
    output["c_step_objective_value"] = best.logdet;
    output["raw_objective_value"] = logdet_spd(result.raw_covariance);
    output["objective_value"] = logdet_spd(result.covariance);
    output["n_iter"] = best.iterations;
    output["converged"] = best.converged;
    return output;
}

static py::dict fit_fast_mcd_cpp(py::array_t<double, py::array::c_style | py::array::forcecast> arr,
                                 double support_fraction, int n_init, int max_iter,
                                 double tol, bool reweight, std::uint64_t seed,
                                 int n_best, int initial_c_steps,
                                 double raw_consistency_factor, double reweight_cutoff,
                                 double reweight_consistency_factor) {
    Mat X = numpy_to_mat(arr);
    const FastMCDParameters parameters = validate_fast_mcd_parameters(
        X,
        support_fraction,
        n_init,
        max_iter,
        tol,
        n_best,
        initial_c_steps,
        raw_consistency_factor,
        reweight_cutoff,
        reweight_consistency_factor
    );
    std::vector<MCDCandidate> initial_candidates = initialize_mcd_candidates(
        X,
        parameters.h,
        n_init,
        parameters.n_best,
        initial_c_steps,
        tol,
        seed
    );
    MCDCandidate best = polish_mcd_candidates(
        X,
        initial_candidates,
        parameters.h,
        max_iter,
        tol
    );
    FastMCDResult result = calibrate_and_reweight_mcd(
        X,
        best,
        reweight,
        raw_consistency_factor,
        reweight_cutoff,
        reweight_consistency_factor
    );
    return fast_mcd_result_to_dict(
        best,
        result,
        parameters.h,
        reweight,
        raw_consistency_factor,
        reweight_cutoff
    );
}


static std::vector<double> solve_spd_vector(Mat A, const std::vector<double>& rhs) {
    if (A.n != A.p || static_cast<int>(rhs.size()) != A.n) {
        throw std::invalid_argument("SPD solve dimension mismatch");
    }
    Mat L;
    double ridge = 1e-12;
    for (int tries = 0; tries < 8; ++tries) {
        if (cholesky_lower(A, L)) break;
        for (int j = 0; j < A.n; ++j) A(j, j) += ridge;
        ridge *= 10.0;
    }
    if (!cholesky_lower(A, L)) throw std::runtime_error("weighted normal equations are not positive definite");
    std::vector<double> y(A.n, 0.0), x(A.n, 0.0);
    for (int i = 0; i < A.n; ++i) {
        double value = rhs[static_cast<size_t>(i)];
        for (int k = 0; k < i; ++k) value -= L(i, k) * y[static_cast<size_t>(k)];
        y[static_cast<size_t>(i)] = value / L(i, i);
    }
    for (int i = A.n - 1; i >= 0; --i) {
        double value = y[static_cast<size_t>(i)];
        for (int k = i + 1; k < A.n; ++k) value -= L(k, i) * x[static_cast<size_t>(k)];
        x[static_cast<size_t>(i)] = value / L(i, i);
    }
    return x;
}

static py::array_t<double> mahalanobis2_batch_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> X_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> location_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> precision_arr) {
    auto xb = X_arr.request();
    auto lb = location_arr.request();
    auto pb = precision_arr.request();
    if (xb.ndim != 2 || lb.ndim != 1 || pb.ndim != 2) {
        throw std::invalid_argument("X must be 2D, location must be 1D, and precision must be 2D");
    }
    if (lb.shape[0] != xb.shape[1] ||
        pb.shape[0] != xb.shape[1] || pb.shape[1] != xb.shape[1]) {
        throw std::invalid_argument("Mahalanobis dimensions do not match");
    }
    const int n = checked_dimension(xb.shape[0], "X rows");
    const int p = checked_dimension(xb.shape[1], "X columns");
    const size_t x_count = checked_product(
        static_cast<size_t>(n), static_cast<size_t>(p), "X element count"
    );
    const size_t precision_count = checked_matrix_elements(p, p, "precision element count");
    const double* X = static_cast<const double*>(xb.ptr);
    const double* location = static_cast<const double*>(lb.ptr);
    const double* precision = static_cast<const double*>(pb.ptr);
    require_all_finite(X, x_count, "X");
    require_all_finite(location, static_cast<size_t>(p), "location");
    require_all_finite(precision, precision_count, "precision");

    py::array_t<double> out(static_cast<py::ssize_t>(n));
    double* distances = static_cast<double*>(out.request().ptr);
    {
        py::gil_scoped_release release;
#ifdef _OPENMP
        const int nt = effective_threads_cpp(n, 256);
#pragma omp parallel for schedule(static) num_threads(nt) if(nt > 1)
#endif
        for (int i = 0; i < n; ++i) {
            const double* row = X + static_cast<size_t>(i) * static_cast<size_t>(p);
            double distance = 0.0;
            // Pair off-diagonal terms so the result is the full quadratic form
            // even when precision is only numerically symmetric.
            for (int j = 0; j < p; ++j) {
                const double xj = row[j] - location[j];
                distance += precision[static_cast<size_t>(j) * p + j] * xj * xj;
                for (int k = 0; k < j; ++k) {
                    const double xk = row[k] - location[k];
                    const double pair = precision[static_cast<size_t>(j) * p + k]
                                      + precision[static_cast<size_t>(k) * p + j];
                    distance += pair * xj * xk;
                }
            }
            distances[i] = std::max(0.0, distance);
        }
    }
    return out;
}

static py::array_t<double> matrix_mahalanobis2_batch_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> X_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> location_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> row_precision_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> column_precision_arr) {
    auto xb = X_arr.request();
    auto mb = location_arr.request();
    auto rb = row_precision_arr.request();
    auto cb = column_precision_arr.request();
    if (xb.ndim != 3 || mb.ndim != 2 || rb.ndim != 2 || cb.ndim != 2) {
        throw std::invalid_argument("X must be 3D and matrix arguments must be 2D");
    }
    if (mb.shape[0] != xb.shape[1] || mb.shape[1] != xb.shape[2] ||
        rb.shape[0] != xb.shape[1] || rb.shape[1] != xb.shape[1] ||
        cb.shape[0] != xb.shape[2] || cb.shape[1] != xb.shape[2]) {
        throw std::invalid_argument("matrix Mahalanobis dimensions do not match");
    }
    const int n = checked_dimension(xb.shape[0], "X samples");
    const int r = checked_dimension(xb.shape[1], "X rows");
    const int c = checked_dimension(xb.shape[2], "X columns");
    const size_t sample_size = checked_matrix_elements(r, c, "matrix sample size");
    const size_t x_count = checked_product(
        static_cast<size_t>(n), sample_size, "X element count"
    );
    const size_t row_precision_count = checked_matrix_elements(r, r, "row precision element count");
    const size_t column_precision_count = checked_matrix_elements(c, c, "column precision element count");
    const double* X = static_cast<const double*>(xb.ptr);
    const double* M = static_cast<const double*>(mb.ptr);
    const double* RP = static_cast<const double*>(rb.ptr);
    const double* CP = static_cast<const double*>(cb.ptr);
    require_all_finite(X, x_count, "X");
    require_all_finite(M, sample_size, "location");
    require_all_finite(RP, row_precision_count, "row_precision");
    require_all_finite(CP, column_precision_count, "column_precision");

    py::array_t<double> out(static_cast<py::ssize_t>(n));
    double* result = static_cast<double*>(out.request().ptr);
    {
        py::gil_scoped_release release;
#ifdef _OPENMP
        int nt = effective_threads_cpp(n, 32);
#pragma omp parallel for schedule(static) num_threads(nt) if(nt > 1)
#endif
        for (int i = 0; i < n; ++i) {
            std::vector<double> left(sample_size, 0.0);
            for (int a = 0; a < r; ++a) {
                for (int b = 0; b < c; ++b) {
                    double value = 0.0;
                    for (int k = 0; k < r; ++k) {
                        const double residual = X[(static_cast<size_t>(i) * r + k) * c + b] - M[static_cast<size_t>(k) * c + b];
                        value += RP[static_cast<size_t>(a) * r + k] * residual;
                    }
                    left[static_cast<size_t>(a) * c + b] = value;
                }
            }
            double distance = 0.0;
            for (int a = 0; a < r; ++a) {
                for (int b = 0; b < c; ++b) {
                    double transformed = 0.0;
                    for (int k = 0; k < c; ++k) transformed += left[static_cast<size_t>(a) * c + k] * CP[static_cast<size_t>(k) * c + b];
                    const double residual = X[(static_cast<size_t>(i) * r + a) * c + b] - M[static_cast<size_t>(a) * c + b];
                    distance += residual * transformed;
                }
            }
            result[i] = std::max(0.0, distance);
        }
    }
    return out;
}

static py::array_t<double> weighted_tucker_scores_2d_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> X_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> weights_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> center_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> row_components_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> column_components_arr,
    double ridge) {
    auto xb = X_arr.request();
    auto wb = weights_arr.request();
    auto mb = center_arr.request();
    auto ub = row_components_arr.request();
    auto vb = column_components_arr.request();
    if (xb.ndim != 3 || wb.ndim != 3 || mb.ndim != 2 || ub.ndim != 2 || vb.ndim != 2) {
        throw std::invalid_argument("weighted Tucker inputs have invalid dimensions");
    }
    if (wb.shape[0] != xb.shape[0] || wb.shape[1] != xb.shape[1] || wb.shape[2] != xb.shape[2] ||
        mb.shape[0] != xb.shape[1] || mb.shape[1] != xb.shape[2] ||
        ub.shape[0] != xb.shape[1] || vb.shape[0] != xb.shape[2]) {
        throw std::invalid_argument("weighted Tucker dimensions do not match");
    }
    const int n = checked_dimension(xb.shape[0], "X samples");
    const int r = checked_dimension(xb.shape[1], "X rows");
    const int c = checked_dimension(xb.shape[2], "X columns");
    const int q1 = checked_dimension(ub.shape[1], "row component rank");
    const int q2 = checked_dimension(vb.shape[1], "column component rank");
    const int q = checked_int_product(q1, q2, "Tucker core size");
    if (!(ridge > 0.0) || !std::isfinite(ridge))
        throw std::invalid_argument("ridge must be positive and finite");

    const size_t sample_size = checked_matrix_elements(r, c, "matrix sample size");
    const size_t x_count = checked_product(
        static_cast<size_t>(n), sample_size, "X element count"
    );
    const size_t row_component_count = checked_matrix_elements(r, q1, "row component element count");
    const size_t column_component_count = checked_matrix_elements(c, q2, "column component element count");
    const double* X = static_cast<const double*>(xb.ptr);
    const double* W = static_cast<const double*>(wb.ptr);
    const double* M = static_cast<const double*>(mb.ptr);
    const double* U = static_cast<const double*>(ub.ptr);
    const double* V = static_cast<const double*>(vb.ptr);
    require_all_finite(X, x_count, "X");
    require_all_finite(W, x_count, "weights");
    require_all_finite(M, sample_size, "center");
    require_all_finite(U, row_component_count, "row_components");
    require_all_finite(V, column_component_count, "column_components");

    py::array_t<double> out({
        static_cast<py::ssize_t>(n),
        static_cast<py::ssize_t>(q1),
        static_cast<py::ssize_t>(q2),
    });
    double* scores = static_cast<double*>(out.request().ptr);
    {
        py::gil_scoped_release release;
#ifdef _OPENMP
        int nt = effective_threads_cpp(n, 8);
#pragma omp parallel for schedule(static) num_threads(nt) if(nt > 1)
#endif
        for (int i = 0; i < n; ++i) {
            Mat gram(q, q, 0.0);
            std::vector<double> rhs(static_cast<size_t>(q), 0.0);
            std::vector<double> design(static_cast<size_t>(q), 0.0);
            for (int a = 0; a < r; ++a) {
                for (int b = 0; b < c; ++b) {
                    const size_t offset = (static_cast<size_t>(i) * r + a) * c + b;
                    const double w = W[offset];
                    if (!(w > 0.0)) continue;
                    const double y = X[offset] - M[static_cast<size_t>(a) * c + b];
                    for (int u = 0; u < q1; ++u) {
                        for (int v = 0; v < q2; ++v) {
                            const int idx = u * q2 + v;
                            design[static_cast<size_t>(idx)] = U[static_cast<size_t>(a) * q1 + u] * V[static_cast<size_t>(b) * q2 + v];
                        }
                    }
                    for (int j = 0; j < q; ++j) {
                        rhs[static_cast<size_t>(j)] += w * design[static_cast<size_t>(j)] * y;
                        for (int k = 0; k <= j; ++k) gram(j, k) += w * design[static_cast<size_t>(j)] * design[static_cast<size_t>(k)];
                    }
                }
            }
            for (int j = 0; j < q; ++j) {
                gram(j, j) += ridge;
                for (int k = 0; k < j; ++k) gram(k, j) = gram(j, k);
            }
            std::vector<double> solution = solve_spd_vector(std::move(gram), rhs);
            for (int u = 0; u < q1; ++u) for (int v = 0; v < q2; ++v) {
                scores[(static_cast<size_t>(i) * q1 + u) * q2 + v] = solution[static_cast<size_t>(u * q2 + v)];
            }
        }
    }
    return out;
}


static py::tuple joint_diagonalize_symmetric_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> matrices_arr,
    int max_sweeps,
    double tol) {
    auto buffer = matrices_arr.request();
    if (buffer.ndim != 3 || buffer.shape[1] != buffer.shape[2]) {
        throw std::invalid_argument("matrices must have shape (n_matrices, p, p)");
    }
    const int n_matrices = checked_dimension(buffer.shape[0], "number of matrices");
    const int p = checked_dimension(buffer.shape[1], "matrix dimension");
    if (max_sweeps < 1) throw std::invalid_argument("max_sweeps must be positive");
    if (!(tol > 0.0) || !std::isfinite(tol)) throw std::invalid_argument("tol must be positive and finite");

    const double* input = static_cast<const double*>(buffer.ptr);
    const size_t matrix_size = checked_matrix_elements(p, p, "joint diagonalization matrix size");
    const size_t total_size = checked_product(
        static_cast<size_t>(n_matrices), matrix_size,
        "joint diagonalization element count"
    );
    std::vector<double> arrays(total_size, 0.0);
    for (int k = 0; k < n_matrices; ++k) {
        for (int i = 0; i < p; ++i) {
            for (int j = 0; j < p; ++j) {
                const double left = input[static_cast<size_t>(k) * matrix_size + static_cast<size_t>(i) * p + j];
                const double right = input[static_cast<size_t>(k) * matrix_size + static_cast<size_t>(j) * p + i];
                if (!std::isfinite(left) || !std::isfinite(right)) {
                    throw std::invalid_argument("matrices contain NaN or infinity");
                }
                arrays[static_cast<size_t>(k) * matrix_size + static_cast<size_t>(i) * p + j] = 0.5 * (left + right);
            }
        }
    }
    std::vector<double> rotation(matrix_size, 0.0);
    for (int i = 0; i < p; ++i) rotation[static_cast<size_t>(i) * p + i] = 1.0;

    auto off_diagonal_energy = [&]() {
        double energy = 0.0;
        for (int k = 0; k < n_matrices; ++k) {
            const size_t base = static_cast<size_t>(k) * matrix_size;
            for (int i = 0; i < p; ++i) {
                for (int j = 0; j < p; ++j) {
                    if (i != j) {
                        const double value = arrays[base + static_cast<size_t>(i) * p + j];
                        energy += value * value;
                    }
                }
            }
        }
        return energy;
    };

    const double initial_energy = off_diagonal_energy();
    bool converged = false;
    int sweeps = 0;
    {
        py::gil_scoped_release release;
        for (int sweep = 1; sweep <= max_sweeps; ++sweep) {
            double largest_sine = 0.0;
            for (int left = 0; left < p - 1; ++left) {
                for (int right = left + 1; right < p; ++right) {
                    double gram11 = 0.0, gram12 = 0.0, gram22 = 0.0;
                    for (int k = 0; k < n_matrices; ++k) {
                        const size_t base = static_cast<size_t>(k) * matrix_size;
                        const double difference = arrays[base + static_cast<size_t>(left) * p + left]
                                                - arrays[base + static_cast<size_t>(right) * p + right];
                        const double cross = 2.0 * arrays[base + static_cast<size_t>(left) * p + right];
                        gram11 += difference * difference;
                        gram12 += difference * cross;
                        gram22 += cross * cross;
                    }
                    const double phi = 0.5 * std::atan2(2.0 * gram12, gram11 - gram22);
                    double cosine_twice = std::cos(phi);
                    double sine_twice = std::sin(phi);
                    if (cosine_twice < 0.0) {
                        cosine_twice = -cosine_twice;
                        sine_twice = -sine_twice;
                    }
                    const double angle = 0.5 * std::atan2(sine_twice, cosine_twice);
                    const double cosine = std::cos(angle);
                    const double sine = std::sin(angle);
                    if (std::abs(sine) <= tol) continue;

                    for (int k = 0; k < n_matrices; ++k) {
                        const size_t base = static_cast<size_t>(k) * matrix_size;
                        for (int row = 0; row < p; ++row) {
                            const size_t row_base = base + static_cast<size_t>(row) * p;
                            const double value_left = arrays[row_base + left];
                            const double value_right = arrays[row_base + right];
                            arrays[row_base + left] = cosine * value_left + sine * value_right;
                            arrays[row_base + right] = -sine * value_left + cosine * value_right;
                        }
                        for (int column = 0; column < p; ++column) {
                            const size_t left_index = base + static_cast<size_t>(left) * p + column;
                            const size_t right_index = base + static_cast<size_t>(right) * p + column;
                            const double value_left = arrays[left_index];
                            const double value_right = arrays[right_index];
                            arrays[left_index] = cosine * value_left + sine * value_right;
                            arrays[right_index] = -sine * value_left + cosine * value_right;
                        }
                    }
                    for (int row = 0; row < p; ++row) {
                        const size_t row_base = static_cast<size_t>(row) * p;
                        const double value_left = rotation[row_base + left];
                        const double value_right = rotation[row_base + right];
                        rotation[row_base + left] = cosine * value_left + sine * value_right;
                        rotation[row_base + right] = -sine * value_left + cosine * value_right;
                    }
                    largest_sine = std::max(largest_sine, std::abs(sine));
                }
            }
            sweeps = sweep;
            if (largest_sine <= tol) {
                converged = true;
                break;
            }
        }
    }

    // Remove tiny asymmetry introduced by sequential in-place rotations.
    for (int k = 0; k < n_matrices; ++k) {
        const size_t base = static_cast<size_t>(k) * matrix_size;
        for (int i = 0; i < p; ++i) {
            for (int j = 0; j < i; ++j) {
                const double value = 0.5 * (
                    arrays[base + static_cast<size_t>(i) * p + j]
                    + arrays[base + static_cast<size_t>(j) * p + i]
                );
                arrays[base + static_cast<size_t>(i) * p + j] = value;
                arrays[base + static_cast<size_t>(j) * p + i] = value;
            }
        }
    }
    const double final_energy = off_diagonal_energy();

    py::array_t<double> rotation_out({p, p});
    std::copy(rotation.begin(), rotation.end(), static_cast<double*>(rotation_out.request().ptr));
    py::array_t<double> matrices_out({n_matrices, p, p});
    std::copy(arrays.begin(), arrays.end(), static_cast<double*>(matrices_out.request().ptr));
    py::dict info;
    info["converged"] = converged;
    info["n_sweeps"] = sweeps;
    info["initial_off_diagonal_energy"] = initial_energy;
    info["off_diagonal_energy"] = final_energy;
    return py::make_tuple(rotation_out, matrices_out, info);
}

PYBIND11_MODULE(_robustcov_cpp, m) {
    m.doc() = "C++ kernels for robustcov MVP";
    // Increment this whenever Python and native semantics must change together.
    // The Python loader rejects missing or mismatched revisions so stale editable
    // builds cannot silently run newer wrappers against an older binary.
    m.attr("__robustcov_native_api__") = 2;
    m.def("has_openmp", &has_openmp_cpp);
    m.def("get_num_threads", &get_max_threads_cpp);
    m.def("set_num_threads", &set_num_threads_cpp, py::arg("n_threads"));
    m.def("fit_tyler", &fit_tyler_cpp, py::arg("X"), py::arg("max_iter")=500,
          py::arg("tol")=1e-7, py::arg("regularization")=0.0, py::arg("assume_centered")=false);
    m.def("mahalanobis2_batch", &mahalanobis2_batch_cpp,
          py::arg("X"), py::arg("location"), py::arg("precision"));
    m.def("matrix_mahalanobis2_batch", &matrix_mahalanobis2_batch_cpp,
          py::arg("X"), py::arg("location"), py::arg("row_precision"), py::arg("column_precision"));
    m.def("joint_diagonalize_symmetric", &joint_diagonalize_symmetric_cpp,
          py::arg("matrices"), py::arg("max_sweeps")=100, py::arg("tol")=1e-10);
    m.def("weighted_tucker_scores_2d", &weighted_tucker_scores_2d_cpp,
          py::arg("X"), py::arg("weights"), py::arg("center"),
          py::arg("row_components"), py::arg("column_components"), py::arg("ridge")=1e-8);
    m.def("fit_fast_mcd", &fit_fast_mcd_cpp, py::arg("X"), py::arg("support_fraction")=-1.0,
          py::arg("n_init")=100, py::arg("max_iter")=50, py::arg("tol")=1e-6,
          py::arg("reweight")=true, py::arg("random_state")=0,
          py::arg("n_best")=10, py::arg("initial_c_steps")=2,
          py::arg("raw_consistency_factor")=-1.0,
          py::arg("reweight_cutoff")=-1.0,
          py::arg("reweight_consistency_factor")=-1.0);
}
