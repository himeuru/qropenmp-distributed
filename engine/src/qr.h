#pragma once
#include <vector>

// QR decomposition A = Q * R via Householder reflections.
// Matrices are row-major n*n stored as flat std::vector<double>.
// If threads > 0, sets the OpenMP team size for this call.
void qr_decompose(const std::vector<double>& A,
                  int n,
                  int threads,
                  std::vector<double>& Q,
                  std::vector<double>& R);
