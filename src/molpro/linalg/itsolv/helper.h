#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITERATIVESOLVER_HELPER_H_
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITERATIVESOLVER_HELPER_H_
#include <complex>
#include <cstddef>
#include <list>
#include <molpro/iostream.h>
#include <molpro/linalg/array/Span.h>
#include <vector>
#include <span>

namespace molpro::linalg::itsolv {
template <typename T>
struct is_complex : std::false_type {};

template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};

//! Stores a singular value and corresponding left and right singular vectors
template <typename T>
struct SVD {
  using value_type = T;
  value_type value;
  std::vector<value_type> u; //!< left singular vector
  std::vector<value_type> v; //!< right singular vector
};

int eigensolver_lapacke_dsyev(std::span<const double> matrix, std::span<double> eigenvectors,
                              std::span<double> eigenvalues, const size_t dimension);

std::list<SVD<double>> eigensolver_lapacke_dsyev(size_t dimension, std::span<const double> matrix);

template <typename value_type>
size_t get_rank(std::vector<value_type> eigenvalues, value_type threshold);

template <typename value_type>
size_t get_rank(std::list<SVD<value_type>> svd_system, value_type threshold);

/*!
 * @brief Performs singular value decomposition and returns SVD objects for singular values less than threshold, sorted
 * in ascending order
 * @tparam value_type
 * @param nrows number of rows in the matrix
 * @param ncols number of columns in the matrix
 * @param m row-wise data buffer for a matrix
 * @param threshold singular values less than threshold will be returned
 */
template <typename value_type, typename std::enable_if_t<!is_complex<value_type>{}, std::nullptr_t> = nullptr>
std::list<SVD<value_type>> svd_system(size_t nrows, size_t ncols, const array::Span<value_type>& m, double threshold,
                                      bool hermitian = false, bool reduce_to_rank = false);
template <typename value_type, typename std::enable_if_t<is_complex<value_type>{}, int> = 0>
std::list<SVD<value_type>> svd_system(size_t nrows, size_t ncols, const array::Span<value_type>& m, double threshold,
                                      bool hermitian = false, bool reduce_to_rank = false);

template <typename value_type>
void printMatrix(const std::vector<value_type>&, size_t rows, size_t cols, std::string title = "",
                 std::ostream& s = molpro::cout);

template <typename value_type, typename std::enable_if_t<is_complex<value_type>{}, int> = 0>
void eigenproblem(std::vector<value_type>& eigenvectors, std::vector<value_type>& eigenvalues,
                  const std::vector<value_type>& matrix, const std::vector<value_type>& metric, size_t dimension,
                  bool hermitian, double svdThreshold, int verbosity);

template <typename value_type, typename std::enable_if_t<!is_complex<value_type>{}, std::nullptr_t> = nullptr>
void eigenproblem(std::vector<value_type>& eigenvectors, std::vector<value_type>& eigenvalues,
                  const std::vector<value_type>& matrix, const std::vector<value_type>& metric, size_t dimension,
                  bool hermitian, double svdThreshold, int verbosity,
				  std::vector<std::pair<std::size_t, value_type>> *imag_eval_parts = nullptr);

template <typename value_type, typename std::enable_if_t<is_complex<value_type>{}, int> = 0>
void solve_LinearEquations(std::vector<value_type>& solution, std::vector<value_type>& eigenvalues,
                           const std::vector<value_type>& matrix, const std::vector<value_type>& metric,
                           const std::vector<value_type>& rhs, size_t dimension, size_t nroot, double augmented_hessian,
                           double svdThreshold, int verbosity);

template <typename value_type, typename std::enable_if_t<!is_complex<value_type>{}, std::nullptr_t> = nullptr>
void solve_LinearEquations(std::vector<value_type>& solution, std::vector<value_type>& eigenvalues,
                           const std::vector<value_type>& matrix, const std::vector<value_type>& metric,
                           const std::vector<value_type>& rhs, size_t dimension, size_t nroot, double augmented_hessian,
                           double svdThreshold, int verbosity);

template <typename value_type, typename std::enable_if_t<is_complex<value_type>{}, int> = 0>
void solve_DIIS(std::vector<value_type>& solution, const std::vector<value_type>& matrix, size_t dimension,
                double svdThreshold, int verbosity = 0);
template <typename value_type, typename std::enable_if_t<!is_complex<value_type>{}, std::nullptr_t> = nullptr>
void solve_DIIS(std::vector<value_type>& solution, const std::vector<value_type>& matrix, size_t dimension,
                double svdThreshold, int verbosity = 0);

/*
 * Explicit instantiation of double type
 */

extern template void printMatrix<double>(const std::vector<double>&, size_t rows, size_t cols, std::string title,
                                         std::ostream& s);

extern template std::list<SVD<double>> svd_system(size_t nrows, size_t ncols, const array::Span<double>& m,
                                                  double threshold, bool hermitian, bool reduce_to_rank);

extern template void eigenproblem<double>(std::vector<double>& eigenvectors, std::vector<double>& eigenvalues,
                                          const std::vector<double>& matrix, const std::vector<double>& metric,
                                          const size_t dimension, bool hermitian, double svdThreshold, int verbosity,
                                          std::vector<std::pair<std::size_t, double>> *imag_eval_parts);

extern template void solve_LinearEquations<double>(std::vector<double>& solution, std::vector<double>& eigenvalues,
                                                   const std::vector<double>& matrix, const std::vector<double>& metric,
                                                   const std::vector<double>& rhs, size_t dimension, size_t nroot,
                                                   double augmented_hessian, double svdThreshold, int verbosity);

extern template void solve_DIIS<double>(std::vector<double>& solution, const std::vector<double>& matrix,
                                        const size_t dimension, double svdThreshold, int verbosity);

/*
 * Explicit instantiation of std::complex<double> type
 */
extern template void printMatrix<std::complex<double>>(const std::vector<std::complex<double>>&, size_t rows,
                                                       size_t cols, std::string title, std::ostream& s);

extern template std::list<SVD<std::complex<double>>> svd_system(size_t nrows, size_t ncols,
                                                                const array::Span<std::complex<double>>& m,
                                                                double threshold, bool hermitian, bool reduce_to_rank);

extern template void eigenproblem<std::complex<double>>(std::vector<std::complex<double>>& eigenvectors,
                                                        std::vector<std::complex<double>>& eigenvalues,
                                                        const std::vector<std::complex<double>>& matrix,
                                                        const std::vector<std::complex<double>>& metric,
                                                        const size_t dimension, bool hermitian, double svdThreshold,
                                                        int verbosity);

extern template void solve_LinearEquations<std::complex<double>>(
    std::vector<std::complex<double>>& solution, std::vector<std::complex<double>>& eigenvalues,
    const std::vector<std::complex<double>>& matrix, const std::vector<std::complex<double>>& metric,
    const std::vector<std::complex<double>>& rhs, size_t dimension, size_t nroot, double augmented_hessian,
    double svdThreshold, int verbosity);

extern template void solve_DIIS<std::complex<double>>(std::vector<std::complex<double>>& solution,
                                                      const std::vector<std::complex<double>>& matrix,
                                                      const size_t dimension, double svdThreshold, int verbosity);
} // namespace molpro::linalg::itsolv
#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITERATIVESOLVER_HELPER_H_
