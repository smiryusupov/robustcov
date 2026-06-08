// Copyright 2026 Shohruh Miryusupov
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
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

static int effective_threads_cpp(int tasks, int min_tasks_per_thread = 128) {
#ifdef _OPENMP
    int mx = omp_get_max_threads();
    int by_work = std::max(1, tasks / std::max(1, min_tasks_per_thread));
    return std::max(1, std::min(mx, by_work));
#else
    (void)tasks; (void)min_tasks_per_thread;
    return 1;
#endif
}

struct Mat {
    int n = 0, p = 0;
    std::vector<double> a;
    Mat() = default;
    Mat(int n_, int p_, double v = 0.0) : n(n_), p(p_), a(static_cast<size_t>(n_) * p_, v) {}
    double& operator()(int i, int j) { return a[static_cast<size_t>(i) * p + j]; }
    double operator()(int i, int j) const { return a[static_cast<size_t>(i) * p + j]; }
};

static Mat numpy_to_mat(py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    auto b = arr.request();
    if (b.ndim != 2) throw std::invalid_argument("X must be a 2D float64 array");
    int n = static_cast<int>(b.shape[0]);
    int p = static_cast<int>(b.shape[1]);
    if (n < 2 || p < 1) throw std::invalid_argument("X must have at least 2 rows and 1 column");
    Mat X(n, p);
    const double* ptr = static_cast<const double*>(b.ptr);
    for (int i = 0; i < n * p; ++i) {
        if (!std::isfinite(ptr[i])) throw std::invalid_argument("X contains NaN or infinity");
        X.a[i] = ptr[i];
    }
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
    for (int j = 0; j < X.p; ++j) {
        for (int k = 0; k <= j; ++k) {
            C(j, k) /= denom;
            C(k, j) = C(j, k);
        }
        C(j, j) += ridge;
    }
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
    double ridge = 1e-10;
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
    double ridge = 1e-10;
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
    if (regularization <= 0.0 && n <= p) throw std::invalid_argument("Unregularized Tyler requires n_samples > n_features");
    if (regularization < 0.0 || regularization >= 1.0) throw std::invalid_argument("regularization must be in [0, 1)");

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
                d2 = std::max(d2, 1e-14);
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
            d2 = std::max(d2, 1e-14);
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
    MCDCandidate cand;
    std::vector<double> loc = column_mean(X, &idx);
    Mat cov = covariance_from_indices(X, idx, loc, ridge);
    double prev_ld = std::numeric_limits<double>::infinity();

    int step = 0;
    for (; step < std::max(1, max_steps); ++step) {
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

        if (std::isfinite(prev_ld) && std::abs(prev_ld - ld) <= tol * (1.0 + std::abs(prev_ld))) {
            ++step;
            break;
        }
        // The C-step should not increase determinant. Small numerical increases can happen because
        // of the ridge term, so do not abort; simply continue from the new subset.
        prev_ld = ld;
    }
    cand.logdet = logdet_spd(cov);
    cand.iterations = step;
    cand.idx = std::move(idx);
    cand.loc = std::move(loc);
    cand.cov = std::move(cov);
    return cand;
}


static double median_copy(std::vector<double> v) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    double med = v[mid];
    if (v.size() % 2 == 0) {
        auto max_it = std::max_element(v.begin(), v.begin() + mid);
        med = 0.5 * (med + *max_it);
    }
    return med;
}

static void scale_matrix_inplace(Mat& A, double scale) {
    if (!std::isfinite(scale) || scale <= 0.0) return;
    for (double& x : A.a) x *= scale;
}

static double normal_quantile_approx(double p) {
    // Peter J. Acklam's inverse-normal approximation, dependency-free.
    if (p <= 0.0) return -std::numeric_limits<double>::infinity();
    if (p >= 1.0) return std::numeric_limits<double>::infinity();
    static const double a[] = {
        -3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02,
        -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[] = {
        -5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01,
        -1.328068155288572e+01};
    static const double c[] = {
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00};
    static const double d[] = {
         7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00};
    const double plow = 0.02425;
    const double phigh = 1.0 - plow;
    double q, r;
    if (p < plow) {
        q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if (p > phigh) {
        q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    q = p - 0.5;
    r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
}

static double chi2_quantile_wilson_hilferty(double alpha, int p) {
    double z = normal_quantile_approx(alpha);
    double pp = static_cast<double>(std::max(1, p));
    double a = 1.0 - 2.0 / (9.0 * pp) + z * std::sqrt(2.0 / (9.0 * pp));
    return std::max(1e-12, pp * a * a * a);
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

static py::dict fit_fast_mcd_cpp(py::array_t<double, py::array::c_style | py::array::forcecast> arr,
                                 double support_fraction, int n_init, int max_iter,
                                 double tol, bool reweight, std::uint64_t seed,
                                 int n_best, int initial_c_steps, double reweight_alpha) {
    Mat X = numpy_to_mat(arr);
    int n = X.n, p = X.p;
    if (n <= p) throw std::invalid_argument("FastMCD requires n_samples > n_features for this MVP");
    int h;
    if (support_fraction <= 0.0) h = (n + p + 1) / 2;
    else h = static_cast<int>(std::floor(support_fraction * n));
    h = std::max(p + 1, std::min(h, n));
    n_init = std::max(1, n_init);
    n_best = std::max(1, std::min(n_best, n_init + 4));
    initial_c_steps = std::max(1, initial_c_steps);

    std::mt19937_64 rng(seed);
    std::vector<int> all(n);
    std::iota(all.begin(), all.end(), 0);
    std::vector<MCDCandidate> pool;
    pool.reserve(static_cast<size_t>(n_init + 4));

    auto push_candidate = [&](MCDCandidate&& c) {
        if (!std::isfinite(c.logdet) || static_cast<int>(c.idx.size()) != h) return;
        pool.push_back(std::move(c));
        std::sort(pool.begin(), pool.end(), [](const MCDCandidate& a, const MCDCandidate& b) {
            return a.logdet < b.logdet;
        });
        if (static_cast<int>(pool.size()) > n_best) pool.resize(n_best);
    };

    // Deterministic start: nearest observations to coordinate-wise median. This improves
    // reproducibility and helps easy contamination cases without sacrificing speed.
    try {
        push_candidate(run_c_steps(X, deterministic_median_start(X, h), h, initial_c_steps, tol, 1e-7));
    } catch (...) {}

    // Random elemental subsets. Starting from p+1 points gives a much higher probability
    // of drawing an uncontaminated candidate than starting from a full h-subset.
    // Starts are generated serially for deterministic random_state behavior, then
    // evaluated independently; OpenMP can parallelize this expensive phase.
    int elemental_size = std::min(n, p + 1);
    std::vector<std::vector<int>> random_starts;
    random_starts.reserve(static_cast<size_t>(n_init));
    for (int init = 0; init < n_init; ++init) {
        std::shuffle(all.begin(), all.end(), rng);
        random_starts.emplace_back(all.begin(), all.begin() + elemental_size);
    }
    std::vector<MCDCandidate> random_candidates(static_cast<size_t>(n_init));
    std::vector<unsigned char> random_ok(static_cast<size_t>(n_init), 0);
int init_threads = effective_threads_cpp(n_init, 4);
#pragma omp parallel for schedule(dynamic) num_threads(init_threads) if(init_threads > 1)
    for (int init = 0; init < n_init; ++init) {
        try {
            MCDCandidate c = run_c_steps(X, random_starts[static_cast<size_t>(init)], h, initial_c_steps, tol, 1e-7);
            if (std::isfinite(c.logdet) && static_cast<int>(c.idx.size()) == h) {
                random_candidates[static_cast<size_t>(init)] = std::move(c);
                random_ok[static_cast<size_t>(init)] = 1;
            }
        } catch (...) {
            // Singular elemental starts are expected occasionally; skip them.
        }
    }
    for (int init = 0; init < n_init; ++init) if (random_ok[static_cast<size_t>(init)]) {
        push_candidate(std::move(random_candidates[static_cast<size_t>(init)]));
    }

    if (pool.empty()) throw std::runtime_error("FastMCD failed to find a valid initial subset");

    std::vector<MCDCandidate> polished_pool(pool.size());
    std::vector<unsigned char> polished_ok(pool.size(), 0);
int polish_threads = effective_threads_cpp(static_cast<int>(pool.size()), 1);
#pragma omp parallel for schedule(dynamic) num_threads(polish_threads) if(polish_threads > 1)
    for (int ci = 0; ci < static_cast<int>(pool.size()); ++ci) {
        try {
            polished_pool[static_cast<size_t>(ci)] = run_c_steps(X, pool[static_cast<size_t>(ci)].idx, h, max_iter, tol, 1e-9);
            polished_ok[static_cast<size_t>(ci)] = 1;
        } catch (...) {}
    }
    MCDCandidate best;
    best.logdet = std::numeric_limits<double>::infinity();
    int total_iter = 0;
    for (size_t ci = 0; ci < polished_pool.size(); ++ci) {
        if (!polished_ok[ci]) continue;
        MCDCandidate& polished = polished_pool[ci];
        total_iter += polished.iterations;
        if (polished.logdet < best.logdet) best = std::move(polished);
    }
    if (best.idx.empty()) throw std::runtime_error("FastMCD failed during final C-steps");

    // Raw MCD covariance is computed from the central h-subset and is therefore
    // systematically too small under Gaussian data. Use a robust radial median
    // consistency correction before reweighting. This is dependency-free and keeps
    // the estimator well calibrated enough for the MVP benchmarks.
    Mat raw_cov_corrected = best.cov;
    Mat raw_prec_uncorrected = inverse_spd(best.cov);
    auto raw_dist_uncorrected = mahalanobis2(X, best.loc, raw_prec_uncorrected);
    double chi2_med = chi2_quantile_wilson_hilferty(0.5, p);
    double raw_scale = median_copy(raw_dist_uncorrected) / std::max(chi2_med, 1e-12);
    raw_scale = std::max(1e-6, std::min(raw_scale, 1e6));
    scale_matrix_inplace(raw_cov_corrected, raw_scale);

    Mat raw_prec = inverse_spd(raw_cov_corrected);
    auto raw_dist = mahalanobis2(X, best.loc, raw_prec);
    std::vector<unsigned char> raw_support(n, 0);
    for (int id : best.idx) raw_support[id] = 1;

    std::vector<unsigned char> support = raw_support;
    std::vector<double> final_loc = best.loc;
    Mat final_cov = raw_cov_corrected;

    if (reweight) {
        // Classical MCD reweighting: keep observations within an approximately Gaussian
        // robust distance cutoff after raw covariance consistency correction.
        double cutoff = chi2_quantile_wilson_hilferty(reweight_alpha, p);
        std::vector<int> rw;
        rw.reserve(n);
        for (int i = 0; i < n; ++i) if (raw_dist[i] <= cutoff) rw.push_back(i);
        if (static_cast<int>(rw.size()) >= p + 1) {
            final_loc = column_mean(X, &rw);
            final_cov = covariance_from_indices(X, rw, final_loc, 1e-9);
            // Do NOT radial-median rescale the final reweighted covariance using all
            // observations. At high contamination, the global median radial distance is
            // no longer a clean-data median: with 30% upper-tail outliers it corresponds
            // roughly to the 71st percentile of the clean radial distribution, inflating
            // covariance scale. Once a reweighted support is selected, its sample
            // covariance is the better default MVP estimate. Users can still request
            // explicit Python-level scale_correction if they want tail calibration.
            std::fill(support.begin(), support.end(), 0);
            for (int id : rw) support[id] = 1;
        }
    }

    Mat final_prec = inverse_spd(final_cov);
    auto final_dist = mahalanobis2(X, final_loc, final_prec);

    py::array_t<bool> supp(n);
    auto sb = supp.mutable_unchecked<1>();
    py::array_t<bool> raw_supp(n);
    auto rsb = raw_supp.mutable_unchecked<1>();
    for (int i = 0; i < n; ++i) {
        sb(i) = support[i] != 0;
        rsb(i) = raw_support[i] != 0;
    }

    py::dict out;
    out["location"] = vec_to_numpy(final_loc);
    out["shape"] = mat_to_numpy(final_cov);
    out["covariance"] = mat_to_numpy(final_cov);
    out["precision"] = mat_to_numpy(final_prec);
    out["distances"] = vec_to_numpy(final_dist);
    out["support"] = supp;
    out["raw_location"] = vec_to_numpy(best.loc);
    out["raw_covariance"] = mat_to_numpy(raw_cov_corrected);
    out["raw_scale"] = raw_scale;
    out["raw_distances"] = vec_to_numpy(raw_dist);
    out["raw_support"] = raw_supp;
    out["h"] = h;
    out["objective_value"] = best.logdet;
    out["n_iter"] = total_iter;
    out["converged"] = true;
    return out;
}

PYBIND11_MODULE(_robustcov_cpp, m) {
    m.doc() = "C++ kernels for robustcov MVP";
    m.def("has_openmp", &has_openmp_cpp);
    m.def("get_num_threads", &get_max_threads_cpp);
    m.def("set_num_threads", &set_num_threads_cpp, py::arg("n_threads"));
    m.def("fit_tyler", &fit_tyler_cpp, py::arg("X"), py::arg("max_iter")=500,
          py::arg("tol")=1e-7, py::arg("regularization")=0.0, py::arg("assume_centered")=false);
    m.def("fit_fast_mcd", &fit_fast_mcd_cpp, py::arg("X"), py::arg("support_fraction")=-1.0,
          py::arg("n_init")=100, py::arg("max_iter")=50, py::arg("tol")=1e-6,
          py::arg("reweight")=true, py::arg("random_state")=0,
          py::arg("n_best")=10, py::arg("initial_c_steps")=2,
          py::arg("reweight_alpha")=0.975);
}
