#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_OPTIMIZEBFGS_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_OPTIMIZEBFGS_H
#include <molpro/linalg/itsolv/CastOptions.h>
#include <molpro/linalg/itsolv/DSpaceResetter.h>
#include <molpro/linalg/itsolv/Interpolate.h>
#include <molpro/linalg/itsolv/IterativeSolverTemplate.h>
#include <molpro/linalg/itsolv/propose_rspace.h>
#include <molpro/linalg/itsolv/subspace/SubspaceSolverOptBFGS.h>
#include <molpro/linalg/itsolv/subspace/XSpace.h>

#include <cmath>
#include <map>
#include <memory>

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
class OptimizeBFGS : public IterativeSolverTemplate<Optimize, R, Q, P> {
public:
  using SolverTemplate = IterativeSolverTemplate<Optimize, R, Q, P>;
  using SolverTemplate ::report;
  using typename SolverTemplate::scalar_type;
  using typename SolverTemplate::value_type;
  using typename SolverTemplate::value_type_abs;
  using SubspaceSolver = subspace::SubspaceSolverOptBFGS<R, Q, P>;

  explicit OptimizeBFGS(const std::shared_ptr<ArrayHandlers<R, Q, P>>& handlers,
                        const std::shared_ptr<Logger>& logger_ = std::make_shared<Logger>())
      : SolverTemplate(std::make_shared<subspace::XSpace<R, Q, P>>(handlers, logger_),
                       std::static_pointer_cast<subspace::ISubspaceSolver<R, Q, P>>(
                           std::make_shared<subspace::SubspaceSolverOptBFGS<R, Q, P>>(logger_)),
                       handlers, std::make_shared<Statistics>(), logger_) {}

  bool nonlinear() const override { return true; }

  int add_vector(R& parameters, R& residual, value_type value) override {
    auto prof = this->profiler()->push("itsolv::add_vector");
    using namespace subspace;
    auto& xspace = this->m_xspace;
    auto& xdata = xspace->data;
    const auto& H = xdata[EqnData::H];
    const auto& S = xdata[EqnData::S];
    auto& Value = xdata[EqnData::value];
    //    std::cout << "updated value to" << as_string(Value) << std::endl;

    //    std::cout << "H "<<as_string(H)<<std::endl;
    //    std::cout << "Value "<<as_string(Value)<<std::endl;
    while (xspace->size() >= size_t(this->m_max_size_qspace)) {
      //      std::cout << "delete Q" << std::endl;
      xspace->eraseq(xspace->size() - 1);
    }
    //    std::cout << "H after delete Q "<<as_string(H)<<std::endl;
    //    std::cout << "Value after delete Q "<<as_string(Value)<<std::endl;

    // augment space with current point
    auto oldValue = Value;
    Value.resize({xspace->size() + 1, 1});
    if (xspace->size() > 0)
      Value.slice({1, 0}, {xspace->size() + 1, 1}) = oldValue.slice();
    Value(0, 0) = value;
    auto nwork = IterativeSolverTemplate<Optimize, R, Q, P>::add_vector(parameters, residual);
    //    std::cout << "H after add_vector "<<as_string(H)<<std::endl;
    //    std::cout << "Value after add_vector "<<as_string(Value)<<std::endl;

    if (xspace->size() > 1) {      // see whether a line search is needed
      auto fprev = Value(1, 0);    // the previous point
      auto fcurrent = Value(0, 0); // the current point
      auto gprev = H(0, 1) - H(1, 1);
      auto gcurrent = H(0, 0) - H(1, 0);
      bool Wolfe_1 = fcurrent <= fprev + m_Wolfe_1 * gprev;
      bool Wolfe_2 = m_strong_Wolfe ? gcurrent >= m_Wolfe_2 * gprev : std::abs(gcurrent) <= m_Wolfe_2 * std::abs(gprev);
      auto step = S(0, 0) - S(1, 0) - S(0, 1) + S(1, 1);
      if (false) {
        molpro::cout << "Size of Q=" << xspace->size() << std::endl;
        molpro::cout << "step=" << step << std::endl;
        molpro::cout << "fprev=" << fprev << std::endl;
        molpro::cout << "fcurrent=" << fcurrent << std::endl;
        molpro::cout << " m_Wolfe_1 =" << m_Wolfe_1 << std::endl;
        molpro::cout << " m_Wolfe_1 * gprev=" << m_Wolfe_1 * gprev << std::endl;
        molpro::cout << "fprev + m_Wolfe_1 * gprev=" << fprev + m_Wolfe_1 * gprev << std::endl;
        molpro::cout << "gprev=" << gprev << std::endl;
        molpro::cout << "gcurrent=" << gcurrent << std::endl;
        molpro::cout << "m_convergence_threshold=" << this->m_convergence_threshold << std::endl;
        molpro::cout << "Wolfe conditions: " << Wolfe_1 << Wolfe_2 << std::endl;
      }
      if (
          //          std::abs(gcurrent) < this->m_convergence_threshold or
          (Wolfe_1 && Wolfe_2))
        goto accept;
      //      molpro::cout << "evaluating line search" << std::endl;
      Interpolate inter({-1, fprev, gprev}, {0, fcurrent, gcurrent});
      auto [x, f, g, h] = inter.minimize(-1 - this->m_linesearch_grow_factor, this->m_linesearch_grow_factor);
      //      molpro::cout << "interpolation" << x << " f=" << f << " g=" << g << " h=" << h << std::endl;
      if (std::isnan(x))
        throw std::runtime_error("NaN in line search");
      if (std::abs(x) > m_linesearch_tolerance) {
        //        molpro::cout << "taking line search" << std::endl;
        this->m_logger->info("Line search step taken");
        this->m_handlers->rr().scal(1 + x, parameters);
        this->m_handlers->rq().axpy(-x, xspace->paramsq()[1], parameters);
        auto erased = fprev < fcurrent ? 0 : 1;
        //        std::cout << "Value before erasure: "<<as_string(Value)<<std::endl;
        //        std::cout << "erased="<<erased<<"; removing point with value "<<Value(erased,0)<<std::endl;
        xspace->eraseq(erased);
        //        std::cout << "Value after erasure: "<<as_string(Value)<<std::endl;
        m_linesearch = true;
        if (false) {
          while (xspace->size() >= 2) {
            std::cout << "delete Q because line searching" << std::endl;
            xspace->eraseq(xspace->size() - 1);
          }
        }
        return -1;
      }
    }

  accept:
    m_linesearch = false;
    this->m_logger->info("Quasi-Newton step taken");
    //    this->m_errors.front() = std::sqrt(this->m_handlers->rr().dot(residual,residual));
    for (size_t a = 0; a < xspace->size() - 1; a++) {
      if (std::abs(H(a, a) - H(a, a + 1) - H(a + 1, a) + H(a + 1, a + 1)) <
          std::max(5e-14 * std::abs(H(a, a)), 1e-15)) {
        xspace->eraseq(a + 1);
        this->m_logger->info("Erase redundant Q");
        goto accept;
      }
    }
    BFGS_update_1(residual, xspace, H);
    return nwork;
  }

  void BFGS_update_1(R& residual, std::shared_ptr<const subspace::IXSpace<R, Q, P>> xspace,
                     const subspace::Matrix<double>& H) {
    m_BFGS_update_alpha.resize(xspace->size() - 1);
    const auto& q = xspace->paramsq();
    const auto& u = xspace->actionsq();
    for (size_t a = 0; a < m_BFGS_update_alpha.size(); a++) {
      m_BFGS_update_alpha[a] =
          (this->m_handlers->rq().dot(residual, q[a]) - this->m_handlers->rq().dot(residual, q[a + 1])) /
          (H(a, a) - H(a, a + 1) - H(a + 1, a) + H(a + 1, a + 1));
      this->m_handlers->rq().axpy(-m_BFGS_update_alpha[a], u[a], residual);
      this->m_handlers->rq().axpy(m_BFGS_update_alpha[a], u[a + 1], residual);
    }
  }

  void BFGS_update_2(R& z, std::shared_ptr<const subspace::IXSpace<R, Q, P>> xspace,
                     const subspace::Matrix<double>& H) {
    const auto& q = xspace->paramsq();
    const auto& u = xspace->actionsq();
    for (int a = m_BFGS_update_alpha.size() - 1; a >= 0; a--) {
      auto beta = (this->m_handlers->rq().dot(z, u[a]) - this->m_handlers->rq().dot(z, u[a + 1])) /
                  (H(a, a) - H(a, a + 1) - H(a + 1, a) + H(a + 1, a + 1));
      this->m_handlers->rq().axpy(+m_BFGS_update_alpha[a] - beta, q[a], z);
      this->m_handlers->rq().axpy(-m_BFGS_update_alpha[a] + beta, q[a + 1], z);
    }
  }

  size_t end_iteration(const VecRef<R>& parameters, const VecRef<R>& action) override {
    auto prof = this->profiler()->push("itsolv::end_iteration");
    this->m_working_set = {0};
    this->m_end_iteration_needed = false;
    if (not m_linesearch) { // action is expected to hold the preconditioned residual
      m_last_iteration_linesearching = false;
      this->solution_params(this->m_working_set, parameters);
      if (this->m_errors.front() < this->m_convergence_threshold) {
        this->m_working_set.clear();
        return 0;
      }
      this->m_working_set.assign(1, 0);
      auto& z = action.front();
      const auto& xspace = this->m_xspace;
      auto& xdata = xspace->data;
      const auto& H = xdata[subspace::EqnData::H];
      BFGS_update_2(z, xspace, H);
      auto len_z = std::sqrt(this->m_handlers->rr().dot(z,z));
      if (len_z > this->m_quasinewton_maximum_step)
        this->m_handlers->rr().scal(this->m_quasinewton_maximum_step / len_z, z);
      this->m_handlers->rr().axpy(-1, z, parameters.front());
    } else {
      this->m_stats->line_search_steps++;
      if (not m_last_iteration_linesearching)
        this->m_stats->line_searches++;
      m_last_iteration_linesearching = true;
    }
    this->m_stats->iterations++;
    return this->errors().front() < this->m_convergence_threshold ? 0 : 1;
  }

  size_t end_iteration(std::vector<R>& parameters, std::vector<R>& action) override {
    auto result = end_iteration(wrap(parameters), wrap(action));
    return result;
  }

  size_t end_iteration(R& parameters, R& actions) override {
    auto wparams = std::vector<std::reference_wrapper<R>>{std::ref(parameters)};
    auto wactions = std::vector<std::reference_wrapper<R>>{std::ref(actions)};
    return end_iteration(wparams, wactions);
  }

  void set_value_errors() override {
    auto& Value = this->m_xspace->data[subspace::EqnData::value];
    this->m_value_errors.assign(1, std::numeric_limits<double>::max());
    if (this->m_xspace->size() > 1 and Value(0, 0) < Value(1, 0))
      this->m_value_errors.front() = Value(1, 0) - Value(0, 0);
  }

  //! Set a limit on the maximum size of Q space. This does not include the size of the working space (R) and the D
  //! space
  void set_max_size_qspace(int n) { m_max_size_qspace = n; }
  int get_max_size_qspace() const { return m_max_size_qspace; }

  void set_options(const Options& options) override {
    SolverTemplate::set_options(options);
    if (auto opt2 = dynamic_cast<const molpro::linalg::itsolv::OptimizeBFGSOptions*>(&options)) {
      auto opt = CastOptions::OptimizeBFGS(options);
      if (opt.max_size_qspace)
        set_max_size_qspace(opt.max_size_qspace.value());
      if (opt.strong_Wolfe)
        m_strong_Wolfe = opt.strong_Wolfe.value();
      if (opt.Wolfe_1)
        m_Wolfe_1 = opt.Wolfe_1.value();
      if (opt.Wolfe_2)
        m_Wolfe_2 = opt.Wolfe_2.value();
      if (opt.linesearch_tolerance)
        m_linesearch_tolerance = opt.linesearch_tolerance.value();
      if (opt.linesearch_grow_factor)
        m_linesearch_grow_factor = opt.linesearch_grow_factor.value();
      if (opt.quasinewton_maximum_step)
        m_quasinewton_maximum_step = opt.quasinewton_maximum_step.value();
    }
  }

  std::shared_ptr<Options> get_options() const override {
    auto opt = std::make_shared<OptimizeBFGSOptions>();
    opt->copy(*SolverTemplate::get_options());
    opt->max_size_qspace = get_max_size_qspace();
    opt->strong_Wolfe = m_strong_Wolfe;
    opt->Wolfe_1 = m_Wolfe_1;
    opt->Wolfe_2 = m_Wolfe_2;
    opt->linesearch_tolerance = m_linesearch_tolerance;
    opt->linesearch_grow_factor = m_linesearch_grow_factor;
    opt->quasinewton_maximum_step = m_quasinewton_maximum_step;
    return opt;
  }

  void report(std::ostream& cout, bool endl = true) const override {
    SolverTemplate::report(cout, false);
    cout << ", value " << this->value() << (m_linesearch ? ", line-searching" : ", quasi-Newton step");
    if (endl)
      cout << std::endl;
  }

protected:
  std::vector<double> m_BFGS_update_alpha;
  bool m_linesearch;
  bool m_last_iteration_linesearching = false;

protected:
  // for non-linear problems, actions already contains the residual
  void construct_residual(const std::vector<int>& roots, const CVecRef<R>& params, const VecRef<R>& actions) override {}

  int m_max_size_qspace = std::numeric_limits<int>::max(); //!< maximum size of Q space
  bool m_strong_Wolfe = true;                              //!< Whether to use strong or weak Wolfe conditions
  double m_Wolfe_1 = 1e-4; //!< Acceptance parameter for function value; recommended value Nocedal and Wright p142
  double m_Wolfe_2 = 0.9;  //!< Acceptance parameter for function gradient; recommended value Nocedal and Wright p142
  double m_linesearch_tolerance = .2; //!< If the predicted line search is within tolerance of the recently-evaluated
                                      //!< point, don't bother taking it, but proceed to Quasi-Newton instead
  double m_linesearch_grow_factor =
      2; //!< If the predicted line search step is extrapolation, limit the step to this factor times the current step
  double m_quasinewton_maximum_step = std::numeric_limits<double>::max();
};

} // namespace molpro::linalg::itsolv
#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_OPTIMIZEBFGS_H
