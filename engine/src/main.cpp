// qr-engine — CLI wrapper around qr_decompose for use as a worker subprocess.
//
// Input  (stdin, binary, little-endian):
//   int32  n
//   int32  threads          (0 = use OpenMP default)
//   double matrix[n*n]      (row-major)
//
// Output (stdout, binary, little-endian):
//   double elapsed_ms
//   int32  n
//   int32  threads_requested
//   int32  omp_max_threads        (omp_get_max_threads after omp_set_num_threads)
//   int32  omp_num_procs          (omp_get_num_procs — what the runtime sees)
//   int32  observed_team_size     (omp_get_num_threads from inside #pragma omp parallel)
//   double diag_R[n]              (diagonal entries of R, for verification)

#include "qr.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <omp.h>
#include <vector>

int main() {
    int32_t n = 0;
    int32_t threads = 0;
    if (std::fread(&n, sizeof(int32_t), 1, stdin) != 1) return 1;
    if (std::fread(&threads, sizeof(int32_t), 1, stdin) != 1) return 1;
    if (n <= 0 || n > 16384) return 2;

    const std::size_t count = static_cast<std::size_t>(n) * n;
    std::vector<double> A(count);
    if (std::fread(A.data(), sizeof(double), count, stdin) != count) return 3;

    std::vector<double> Q, R;
    const auto t0 = std::chrono::high_resolution_clock::now();
    qr_decompose(A, n, threads, Q, R);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Diagnostic readouts so the API can show whether the team size we
    // actually got matches the request — useful when OpenMP behaves weirdly
    // inside Docker / WSL2.
    const int32_t omp_max = omp_get_max_threads();
    const int32_t omp_procs = omp_get_num_procs();
    int32_t observed_team = 0;
    #pragma omp parallel
    {
        #pragma omp single
        observed_team = omp_get_num_threads();
    }

    std::fwrite(&elapsed_ms, sizeof(double), 1, stdout);
    std::fwrite(&n, sizeof(int32_t), 1, stdout);
    std::fwrite(&threads, sizeof(int32_t), 1, stdout);
    std::fwrite(&omp_max, sizeof(int32_t), 1, stdout);
    std::fwrite(&omp_procs, sizeof(int32_t), 1, stdout);
    std::fwrite(&observed_team, sizeof(int32_t), 1, stdout);

    std::vector<double> diag(n);
    for (int i = 0; i < n; ++i) diag[i] = R[static_cast<std::size_t>(i) * n + i];
    std::fwrite(diag.data(), sizeof(double), n, stdout);

    return 0;
}
