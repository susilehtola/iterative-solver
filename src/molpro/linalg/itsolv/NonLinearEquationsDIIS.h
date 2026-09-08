#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_NONLINEAREQUATIONSDIIS_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_NONLINEAREQUATIONSDIIS_H
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <molpro/linalg/itsolv/CastOptions.h>
#include <molpro/linalg/itsolv/DSpaceResetter.h>
#include <molpro/linalg/itsolv/IterativeSolverTemplate.h>
#include <molpro/linalg/itsolv/propose_rspace.h>
#include <molpro/linalg/itsolv/subspace/Matrix.h>
#include <molpro/linalg/itsolv/subspace/SubspaceSolverDIIS.h>
#include <molpro/linalg/itsolv/subspace/XSpace.h>
#include <molpro/profiler/Profiler.h>

namespace molpro::linalg::itsolv {
/*!
 * @brief A class that optimises a function using a Quasi-Newton or other method.
 *
 * @todo Explain some of the theory
 *
 * @tparam R The class encapsulating solution and residual vectors
 * @tparam Q Used internally as a class for storing vectors on backing store
 */
template <class R, class Q = R, class P = std::map<size_t, typename R::value_type>>
class NonLinearEquationsDIIS : public IterativeSolverTemplate<NonLinearEquations, R, Q, P> {
  bool m_converged;

public:
  using SolverTemplate = IterativeSolverTemplate<NonLinearEquations, R, Q, P>;
  using SolverTemplate ::report;
  using typename SolverTemplate::scalar_type;
  using typename SolverTemplate::value_type;
  using typename SolverTemplate::value_type_abs;

  explicit NonLinearEquationsDIIS(const std::shared_ptr<ArrayHandlers<R, Q, P>>& handlers,
                                  const std::shared_ptr<Logger>& logger_ = std::make_shared<Logger>())
      : SolverTemplate(std::make_shared<subspace::XSpace<R, Q, P>>(handlers, logger_),
                       std::static_pointer_cast<subspace::ISubspaceSolver<R, Q, P>>(
                           std::make_shared<subspace::SubspaceSolverDIIS<R, Q, P>>(logger_, m_converged)),
                       handlers, std::make_shared<Statistics>(), logger_), m_converged(false) {
    auto xspace = std::dynamic_pointer_cast<subspace::XSpace<R, Q, P>>(this->m_xspace);
    xspace->set_hermiticity(true);
    xspace->set_action_action();
  }

  bool nonlinear() const override { return true; }

private:
  std::pair<size_t, value_type> least_important_vector(const subspace::Matrix<value_type>& H) {
    const auto He = Eigen::Map<const Eigen::MatrixXd>(H.data().data(), H.rows(), H.cols());
    std::pair<size_t, value_type> result{0, std::numeric_limits<value_type>::max()};
    if (He.cols() < 2)
      return result;
    auto evs = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>();
    evs.compute(He);
    //    std::cout << "Eigenvalues: "<<evs.eigenvalues().transpose()<<std::endl;
    //    std::cout << "Eigenvectors:\n"<<evs.eigenvectors()<<std::endl;
    //    result.first = He.cols() - 1;
    //    return result;
    value_type evmax = 0;
    for (Eigen::Index i = 0; i < He.cols(); ++i) {
      evmax = std::max(evmax, evs.eigenvalues()(i));
      if (evs.eigenvalues()(i) < result.second) {
        result.second = evs.eigenvalues()(i);
        result.first = 1;
        for (Eigen::Index j = 1; j < He.rows(); ++j) {
          if (std::abs(evs.eigenvectors().col(i)(j)) > std::abs(evs.eigenvectors().col(i)(result.first)))
            result.first = j;
        }
      }
    }
    result.second /= evmax;
    if (result.second > m_svd_thresh)
      result = {He.cols() - 1, std::numeric_limits<value_type>::max()};
    //    std::cout << "least important vector " << result.first << " : " << result.second << std::endl;
    return result;
  }

public:
  int add_vector(R& parameters, R& residual, value_type value) override {
    auto prof = this->profiler()->push("itsolv::add_vector");
    auto error = std::sqrt(this->m_handlers->rr().dot(residual, residual));
    m_converged = error < this->m_convergence_threshold;
    using namespace subspace;
    auto& xspace = this->m_xspace;
    const auto& H = xspace->data[EqnData::H];
    //        std::cout << "H " << as_string(H) << std::endl;
    for (std::pair<size_t, value_type> deleter = least_important_vector(H);
         xspace->size() >= size_t(this->m_max_size_qspace) or deleter.second < m_svd_thresh;
         deleter = least_important_vector(H)) {
      //            std::cout << "deleter " << deleter.first << " : " << deleter.second << std::endl;
      xspace->eraseq(deleter.first);
    }
    //            std::cout << "H after delete Q "<<as_string(H)<<std::endl;

    int nwork = IterativeSolverTemplate<NonLinearEquations, R, Q, P>::add_vector(parameters, residual);
    this->m_errors.front() = error;
    return nwork;
  }
  size_t end_iteration(const VecRef<R>& parameters, const VecRef<R>& action) override {
    auto prof = this->profiler()->push("itsolv::end_iteration");
    this->solution_params(this->m_working_set, parameters);
    this->m_end_iteration_needed = false;
    if (this->m_errors.front() < this->m_convergence_threshold) {
      this->m_working_set.clear();
      return 0;
    }
    this->m_working_set.assign(1, 0);
    bool precon = true; // TODO implement
    if (precon) { // action is expected to hold the preconditioned residual, and here we should add it to parameters
      this->m_handlers->rr().axpy(-1, action.front(), parameters.front());
    } else { // residual not used, simply leave parameters alone
    }
    this->m_stats->iterations++;
    return 1;
  }

  size_t end_iteration(std::vector<R>& parameters, std::vector<R>& action) override {
    auto result = end_iteration(wrap(parameters), wrap(action));
    //    std::cout << ", max q" << this->m_max_size_qspace << std::endl; // get_max_size_qspace();
    return result;
  }

  //! Set threshold on the norm of parameters that should be considered null
  void set_norm_thresh(double thresh) { m_norm_thresh = thresh; }
  double get_norm_thresh() const { return m_norm_thresh; }
  //! Set the smallest singular value in the subspace that can be allowed when
  //! constructing the working set. Smaller singular values will lead to deletion of parameters
  void set_svd_thresh(double thresh) { m_svd_thresh = thresh; }
  double get_svd_thresh() const { return m_svd_thresh; }
  //! Set a limit on the maximum size of Q space. This does not include the size of the working space (R) and the D
  //! space
  void set_max_size_qspace(int n) { m_max_size_qspace = n; }
  int get_max_size_qspace() const { return m_max_size_qspace; }

  void set_options(const Options& options) override {
    SolverTemplate::set_options(options);
    auto opt = CastOptions::NonLinearEquationsDIIS(options);
    if (opt.max_size_qspace)
      set_max_size_qspace(opt.max_size_qspace.value());
    if (opt.norm_thresh)
      set_norm_thresh(opt.norm_thresh.value());
    if (opt.svd_thresh)
      set_svd_thresh(opt.svd_thresh.value());
  }

  std::shared_ptr<Options> get_options() const override {
    auto opt = std::make_shared<NonLinearEquationsDIISOptions>();
    opt->copy(*SolverTemplate::get_options());
    opt->max_size_qspace = get_max_size_qspace();
    opt->norm_thresh = get_norm_thresh();
    opt->svd_thresh = get_svd_thresh();
    return opt;
  }

  void report(std::ostream& cout, bool endl = true) const override {
    SolverTemplate::report(cout, endl);
    //    auto& err = this->m_errors;
    //    std::copy(begin(err), end(err), std::ostream_iterator<value_type_abs>(cout, ", "));
    //    cout << std::defaultfloat;
    //    cout << ", max q" << this->m_max_size_qspace; // get_max_size_qspace();
    //    if (endl)
    //      cout << std::endl;
  }

  size_t end_iteration(R& parameters, R& actions) override {
    auto wparams = std::vector<std::reference_wrapper<R>>{std::ref(parameters)};
    auto wactions = std::vector<std::reference_wrapper<R>>{std::ref(actions)};
    return end_iteration(wparams, wactions);
  }

protected:
  // for non-linear problems, actions already contains the residual
  void construct_residual(const std::vector<int>& roots, const CVecRef<R>& params, const VecRef<R>& actions) override {}

  double m_norm_thresh = 1e-10; //!< vectors with norm less than threshold can be considered null.
  double m_svd_thresh = 1e-12;  //!< svd values smaller than this mark the null space
  int m_max_size_qspace = std::numeric_limits<int>::max(); //!< maximum size of Q space
};

} // namespace molpro::linalg::itsolv
#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_NONLINEAREQUATIONSDIIS_H
