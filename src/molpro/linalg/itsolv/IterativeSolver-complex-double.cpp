#include <complex>
#include <molpro/linalg/itsolv/helper-implementation.h>
namespace {
using value_type = std::complex<double>;
}
namespace molpro::linalg::itsolv {

template void printMatrix<value_type>(const std::vector<value_type>&, size_t rows, size_t cols, std::string title,
                                      std::ostream& s);

template std::list<SVD<value_type>> svd_system<value_type>(size_t nrows, size_t ncols, const array::Span<value_type>& m,
                                                           double threshold, bool hermitian,
                                                           bool reduce_to_rank);

template void eigenproblem<value_type>(std::vector<value_type>& eigenvectors, std::vector<value_type>& eigenvalues,
                                       const std::vector<value_type>& matrix, const std::vector<value_type>& metric,
                                       size_t dimension, bool hermitian, double svdThreshold, int verbosity);

template void solve_LinearEquations<value_type>(std::vector<value_type>& solution, std::vector<value_type>& eigenvalues,
                                                const std::vector<value_type>& matrix,
                                                const std::vector<value_type>& metric,
                                                const std::vector<value_type>& rhs, size_t dimension, size_t nroot,
                                                double augmented_hessian, double svdThreshold, int verbosity);

//template void solve_DIIS<value_type>(std::vector<value_type>& solution, const std::vector<value_type>& matrix,
//                                     size_t dimension, double svdThreshold, int verbosity);
} // namespace molpro::linalg::itsolv
