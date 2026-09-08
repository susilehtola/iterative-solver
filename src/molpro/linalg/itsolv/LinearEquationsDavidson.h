#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_LINEAREQUATIONSDAVIDSON_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_LINEAREQUATIONSDAVIDSON_H
#include <molpro/linalg/itsolv/CastOptions.h>
#include <molpro/linalg/itsolv/DSpaceResetter.h>
#include <molpro/linalg/itsolv/IterativeSolverTemplate.h>
#include <molpro/linalg/itsolv/propose_rspace.h>
#include <molpro/linalg/itsolv/qspace_options.h>
#include <molpro/linalg/itsolv/rspace_options.h>
#include <molpro/linalg/itsolv/subspace/SubspaceSolverLinEig.h>
#include <molpro/linalg/itsolv/subspace/XSpace.h>

namespace molpro::linalg::itsolv {
/*!
 * @brief Solves a system of linear equation, A x = b
 *
 * The equations are solved using a Krylov subspace projection method with the P and D space. This is the same approach
 * as LinearEigenvalue.
 *
 * Residual
 * --------
 * The residual is scaled down by the norm of the RHS vector so that thresholds are consistent.
 *
 * Preconditioner
 * -------------
 * @todo Explain some of the theory
 *
 * Augmented Hessian
 * -----------------
 * @todo Explain some of the theory
 */
template <class R, class Q = R, class P = std::map<size_t, typename R::value_type>>
class LinearEquationsDavidson : public IterativeSolverTemplate<LinearEquations, R, Q, P> {
public:
  using SolverTemplate = IterativeSolverTemplate<LinearEquations, R, Q, P>;
  using SolverTemplate ::report;
  using typename SolverTemplate::value_type;
  using typename SolverTemplate::value_type_abs;

  explicit LinearEquationsDavidson(const std::shared_ptr<ArrayHandlers<R, Q, P>>& handlers,
                                   const std::shared_ptr<Logger>& logger_ = std::make_shared<Logger>())
      : SolverTemplate(std::make_shared<subspace::XSpace<R, Q, P>>(handlers, logger_),
                       std::static_pointer_cast<subspace::ISubspaceSolver<R, Q, P>>(
                           std::make_shared<subspace::SubspaceSolverLinEig<R, Q, P>>(logger_)),
                       handlers, std::make_shared<Statistics>(), logger_) {
    set_hermiticity(m_hermiticity);
    this->m_normalise_solution = false;
  }

  bool solve(const VecRef<R>& parameters, const VecRef<R>& actions, const Problem<R>& problem,
             bool generate_initial_guess = false) override {
    R& vector = actions[0];
    for (unsigned int instance = 0; problem.RHS(vector, instance); ++instance)
      add_equations(vector);
    return IterativeSolverTemplate<LinearEquations, R, Q, P>::solve(parameters, actions, problem,
                                                                    generate_initial_guess);
  }

  bool nonlinear() const override { return false; }

  size_t end_iteration(const VecRef<R>& parameters, const VecRef<R>& action) override {
    auto prof = this->profiler()->push("itsolv::end_iteration");
    if (m_dspace_resetter.do_reset(this->m_stats->iterations, this->m_xspace->dimensions())) {
      this->m_working_set =
          m_dspace_resetter.run(parameters, *this->m_xspace, this->m_subspace_solver->solutions(),
                                rspace_opts.norm_thresh, rspace_opts.svd_thresh, *this->m_handlers, *this->m_logger);
    } else {
      this->m_working_set =
          detail::propose_rspace(*this, parameters, action, *this->m_xspace, *this->m_subspace_solver,
                                 *this->m_handlers, *this->m_logger, rspace_opts, qspace_opts, *this->profiler());
    }
    this->m_stats->iterations++;
    this->m_end_iteration_needed = false;
    return this->working_set().size();
  }

  // FIXME move this to the template
  size_t end_iteration(std::vector<R>& parameters, std::vector<R>& action) override {
    return end_iteration(wrap(parameters), wrap(action));
  }
  size_t end_iteration(R& parameters, R& actions) override {
    auto wparams = std::vector<std::reference_wrapper<R>>{std::ref(parameters)};
    auto wactions = std::vector<std::reference_wrapper<R>>{std::ref(actions)};
    return end_iteration(wparams, wactions);
  }

  void add_equations(const CVecRef<R>& rhs) override {
    auto prof = this->profiler()->push("itsolv::add_equations");
    auto xspace = std::static_pointer_cast<subspace::XSpace<R, Q, P>>(this->m_xspace);
    xspace->add_rhs_equations(rhs);
    this->set_n_roots(xspace->dimensions().nRHS);
  }

  void add_equations(const R& rhs) override { add_equations(cwrap_arg(rhs)); }
  void add_equations(const std::vector<R>& rhs) override { add_equations(cwrap(rhs)); }

  CVecRef<Q> rhs() const override {
    auto xspace = std::static_pointer_cast<subspace::XSpace<R, Q, P>>(this->m_xspace);
    return xspace->rhs();
  }

  //! Set threshold on the norm of parameters that should be considered null
  void set_norm_thresh(double thresh) { rspace_opts.norm_thresh = thresh; }
  double get_norm_thresh() const { return rspace_opts.norm_thresh; }
  //! Set the smallest singular value in the subspace that can be allowed when
  //! constructing the working set. Smaller singular values will lead to deletion of parameters
  void set_svd_thresh(double thresh) { rspace_opts.svd_thresh = thresh; }
  double get_svd_thresh() const { return rspace_opts.svd_thresh; }
  //! Set the period in iterations for resetting the D space
  void set_reset_D(size_t n) { m_dspace_resetter.set_nreset(n); }
  size_t get_reset_D() const { return m_dspace_resetter.get_nreset(); }
  //! Set the maximum size of Q space after resetting the D space
  void set_reset_D_maxQ_size(size_t n) { m_dspace_resetter.set_max_Qsize(n); }
  int get_reset_D_maxQ_size() const { return m_dspace_resetter.get_max_Qsize(); }
  //! Set a limit on the maximum size of Q space. This does not include the size of the working space (R) and the D
  //! space
  void set_max_size_qspace(std::size_t n) {
    qspace_opts.max_size = n;
    if (m_dspace_resetter.get_max_Qsize() > qspace_opts.max_size)
      m_dspace_resetter.set_max_Qsize(qspace_opts.max_size);
  }
  std::size_t get_max_size_qspace() const { return qspace_opts.max_size; }
  void set_min_size_qspace(std::size_t n) {
    qspace_opts.min_size = n;
  }
  std::size_t get_min_size_qspace() const { return qspace_opts.min_size; }
  void set_hermiticity(bool hermitian) override {
    m_hermiticity = hermitian;
    auto xspace = std::dynamic_pointer_cast<subspace::XSpace<R, Q, P>>(this->m_xspace);
    xspace->set_hermiticity(hermitian);
    auto subspace_solver = std::dynamic_pointer_cast<subspace::SubspaceSolverLinEig<R, Q, P>>(this->m_subspace_solver);
    subspace_solver->set_hermiticity(hermitian);
  }
  bool get_hermiticity() const override { return m_hermiticity; }
  //!@copydoc subspace::SubspaceSolverLinEig::set_augmented_hessian()
  void set_augmented_hessian(const double parameter) {
    auto subspace_solver = std::dynamic_pointer_cast<subspace::SubspaceSolverLinEig<R, Q, P>>(this->m_subspace_solver);
    subspace_solver->set_augmented_hessian(parameter);
  }
  //!@copydoc subspace::SubspaceSolverLinEig::get_augmented_hessian()
  double get_augmented_hessian() const {
    auto subspace_solver = std::dynamic_pointer_cast<subspace::SubspaceSolverLinEig<R, Q, P>>(this->m_subspace_solver);
    return subspace_solver->get_augmented_hessian();
  }

  void set_options(const Options& options) override {
    SolverTemplate::set_options(options);
    auto opt = CastOptions::LinearEquations(options);
    if (opt.reset_D)
      set_reset_D(opt.reset_D.value());
    if (opt.reset_D_max_Q_size)
      set_reset_D_maxQ_size(opt.reset_D_max_Q_size.value());
    if (opt.max_size_qspace)
      set_max_size_qspace(opt.max_size_qspace.value());
    if (opt.min_size_qspace)
      set_min_size_qspace(opt.min_size_qspace.value());
    if (opt.contrib_thresh)
      qspace_opts.contrib_thresh = opt.contrib_thresh.value();
    if (opt.norm_thresh)
      set_norm_thresh(opt.norm_thresh.value());
    if (opt.svd_thresh)
      set_svd_thresh(opt.svd_thresh.value());
    if (opt.hermiticity)
      set_hermiticity(opt.hermiticity.value());
    if (opt.augmented_hessian)
      set_augmented_hessian(opt.augmented_hessian.value());
  }

  std::shared_ptr<Options> get_options() const override {
    auto opt = std::make_shared<LinearEquationsDavidsonOptions>();
    opt->copy(*SolverTemplate::get_options());
    opt->reset_D = get_reset_D();
    opt->reset_D_max_Q_size = get_reset_D_maxQ_size();
    opt->max_size_qspace = get_max_size_qspace();
    opt->min_size_qspace = get_min_size_qspace();
    opt->contrib_thresh = qspace_opts.contrib_thresh;
    opt->norm_thresh = get_norm_thresh();
    opt->svd_thresh = get_svd_thresh();
    opt->hermiticity = get_hermiticity();
    opt->augmented_hessian = get_augmented_hessian();
    return opt;
  }

  void report(std::ostream& cout, bool endl = true) const override {
    SolverTemplate::report(cout, false);
    cout << ", errors " << std::scientific;
    auto& err = this->m_errors;
    std::copy(begin(err), end(err), std::ostream_iterator<value_type_abs>(cout, ", "));
    cout << std::defaultfloat;
    if (endl)
      cout << std::endl;
  }

protected:
  // FIXME The scale is fixed by the norm of RHS, but if RHS=0 there is no reference. We could use the norm of params
  void construct_residual(const std::vector<int>& roots, const CVecRef<R>& params, const VecRef<R>& actions) override {
    assert(params.size() >= roots.size());
    const auto& norm = std::dynamic_pointer_cast<subspace::XSpace<R, Q, P>>(this->m_xspace)->rhs_norm();
    for (size_t i = 0; i < roots.size(); ++i) {
      const auto ii = roots[i];
      this->m_handlers->rq().axpy(-1, rhs().at(ii), actions.at(i));
      if (norm.at(ii) != 0) {
        auto scal = 1 / norm[ii];
        this->m_handlers->rr().scal(scal, actions.at(i));
      }
    }
  }

  RSpaceOptions rspace_opts;                   //!< Options concerning R-space handling
  QSpaceOptions qspace_opts;                   //!< Options concerning Q-space handling
  detail::DSpaceResetter<Q> m_dspace_resetter; //!< resets D space
  bool m_hermiticity = true;                   //!< whether the problem is hermitian or not
};

} // namespace molpro::linalg::itsolv
#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_LINEAREQUATIONSDAVIDSON_H
