#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_OPTIMIZESD_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_OPTIMIZESD_H
#include <molpro/linalg/itsolv/CastOptions.h>
#include <molpro/linalg/itsolv/DSpaceResetter.h>
#include <molpro/linalg/itsolv/IterativeSolverTemplate.h>
#include <molpro/linalg/itsolv/propose_rspace.h>
#include <molpro/linalg/itsolv/subspace/SubspaceSolverOptSD.h>
#include <molpro/linalg/itsolv/subspace/XSpace.h>

namespace molpro::linalg::itsolv {
/*!
 * @brief A class that optimises a function using a simple steepest-descent method
 *
 * @todo Explain some of the theory
 *
 * @tparam R The class encapsulating solution and residual vectors
 * @tparam Q Used internally as a class for storing vectors on backing store
 */
template <class R, class Q = R, class P = std::map<size_t, typename R::value_type>>
class OptimizeSD : public IterativeSolverTemplate<Optimize, R, Q, P> {
public:
  using SolverTemplate = IterativeSolverTemplate<Optimize, R, Q, P>;
  using SolverTemplate ::report;
  using typename SolverTemplate::scalar_type;
  using typename SolverTemplate::value_type;
  using typename SolverTemplate::value_type_abs;
  using SubspaceSolver = subspace::SubspaceSolverOptSD<R, Q, P>;

  explicit OptimizeSD(const std::shared_ptr<ArrayHandlers<R, Q, P>>& handlers,
                      const std::shared_ptr<Logger>& logger_ = std::make_shared<Logger>())
      : SolverTemplate(std::make_shared<subspace::XSpace<R, Q, P>>(handlers, logger_),
                       std::static_pointer_cast<subspace::ISubspaceSolver<R, Q, P>>(
                           std::make_shared<subspace::SubspaceSolverOptSD<R, Q, P>>(logger_)),
                       handlers, std::make_shared<Statistics>(), logger_) {}

  bool nonlinear() const override { return true; }

  size_t end_iteration(const VecRef<R>& parameters, const VecRef<R>& action) override {
    this->solution_params(this->m_working_set, parameters);
    if (this->m_errors.front() < this->m_convergence_threshold) {
      this->m_working_set.clear();
      return 0;
    }
    this->m_working_set.assign(1, 0);
    this->m_handlers->rr().axpy(-1, action.front(), parameters.front());
    this->m_stats->iterations++;
    return 1;
  }

  size_t end_iteration(std::vector<R>& parameters, std::vector<R>& action) override {
    auto result = end_iteration(wrap(parameters), wrap(action));
    return result;
  }

  void set_value_errors() override {
    auto& Value = this->m_xspace->data[subspace::EqnData::value];
    this->m_value_errors.assign(1, std::numeric_limits<double>::max());
    if (this->m_xspace->size() > 1 and Value(0, 0) < Value(1, 0))
      this->m_value_errors.front() = Value(1, 0) - Value(0, 0);
  }

  void set_options(const Options& options) override {
    SolverTemplate::set_options(options);
    if (auto opt2 = dynamic_cast<const molpro::linalg::itsolv::OptimizeSDOptions*>(&options)) {
      auto opt = CastOptions::OptimizeSD(options);
    }
  }

  std::shared_ptr<Options> get_options() const override {
    auto opt = std::make_shared<OptimizeSDOptions>();
    opt->copy(*SolverTemplate::get_options());
    return opt;
  }

  void report(std::ostream& cout, bool endl = true) const override {
    SolverTemplate::report(cout, false);
    cout << ", value " << this->value();
    if (endl)
      cout << std::endl;
  }

  int add_vector(R& parameters, R& residual, value_type value) override {
    using namespace subspace;
    auto& xspace = this->m_xspace;
    auto& xdata = xspace->data;
    const auto n = this->m_xspace->dimensions().nX;
    xdata[EqnData::value].resize({n + 1, 1});
    xdata[EqnData::value](0, 0) = value;
    auto nwork = IterativeSolverTemplate<Optimize, R, Q, P>::add_vector(parameters, residual);
    return nwork;
  }

  size_t end_iteration(R& parameters, R& actions) override {
    auto wparams = std::vector<std::reference_wrapper<R>>{std::ref(parameters)};
    auto wactions = std::vector<std::reference_wrapper<R>>{std::ref(actions)};
    return end_iteration(wparams, wactions);
  }

protected:
  // for non-linear problems, actions already contains the residual
  void construct_residual(const std::vector<int>& roots, const CVecRef<R>& params, const VecRef<R>& actions) override {}
};

} // namespace molpro::linalg::itsolv
#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_OPTIMIZESD_H
