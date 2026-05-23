#include "qr.h"

#include <cmath>
#include <omp.h>

static inline double& at(std::vector<double>& M, int n, int i, int j) {
    return M[static_cast<std::size_t>(i) * n + j];
}

static inline double at_c(const std::vector<double>& M, int n, int i, int j) {
    return M[static_cast<std::size_t>(i) * n + j];
}

void qr_decompose(const std::vector<double>& A,
                  int n,
                  int threads,
                  std::vector<double>& Q,
                  std::vector<double>& R) {
    // Pin team size to whatever the caller asked for. With dynamic adjustment
    // on (the default on some libgomp builds) the runtime is free to give us
    // fewer threads than requested, which destroys the speedup numbers in a
    // hard-to-debug way.
    omp_set_dynamic(0);
    if (threads > 0) {
        omp_set_num_threads(threads);
    }

    Q.assign(static_cast<std::size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) at(Q, n, i, i) = 1.0;
    R = A;

    std::vector<double> v(static_cast<std::size_t>(n));

    for (int k = 0; k < n - 1; ++k) {
        // Build the Householder vector for column k.
        double sigma = 0.0;
        for (int i = k; i < n; ++i) {
            v[i - k] = at_c(R, n, i, k);
            sigma += v[i - k] * v[i - k];
        }
        const double nrm = std::sqrt(sigma);
        v[0] += (v[0] >= 0.0 ? nrm : -nrm);

        double nv2 = 0.0;
        for (int i = 0; i < n - k; ++i) nv2 += v[i] * v[i];
        if (nv2 < 1e-14) continue;
        const double inv = 1.0 / nv2;

        // Apply the reflection to the remaining columns of R.
        // Each column j is independent — split across threads.
        #pragma omp parallel for schedule(static)
        for (int j = k; j < n; ++j) {
            double d = 0.0;
            for (int i = k; i < n; ++i) d += v[i - k] * at_c(R, n, i, j);
            d *= 2.0 * inv;
            for (int i = k; i < n; ++i) at(R, n, i, j) -= d * v[i - k];
        }

        // Accumulate the reflection into Q. Rows are independent.
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            double d = 0.0;
            for (int j = k; j < n; ++j) d += at_c(Q, n, i, j) * v[j - k];
            d *= 2.0 * inv;
            for (int j = k; j < n; ++j) at(Q, n, i, j) -= d * v[j - k];
        }
    }
}
