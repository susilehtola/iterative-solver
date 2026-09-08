#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_LINEAREIGENSYSTEMDAVIDSON_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_LINEAREIGENSYSTEMDAVIDSON_H
#include <molpro/Profiler.h>
#include <molpro/linalg/itsolv/CastOptions.h>
#include <molpro/linalg/itsolv/DSpaceResetter.h>
#include <molpro/linalg/itsolv/IterativeSolverTemplate.h>
#include <molpro/linalg/itsolv/Logger.h>
#include <molpro/linalg/itsolv/helper.h>
#include <molpro/linalg/itsolv/propose_rspace.h>
#include <molpro/linalg/itsolv/qspace_options.h>
#include <molpro/linalg/itsolv/rspace_options.h>
#include <molpro/linalg/itsolv/subspace/SubspaceSolverLinEig.h>
#include <molpro/linalg/itsolv/subspace/XSpace.h>

#include <algorithm>
#include <cassert>
#include <iterator>
#include <map>

namespace molpro::linalg::itsolv {

namespace log {

template< typename value_type >
struct ComplexRootsDavidson : ContextBase<ComplexRootsDavidson<value_type>, true, std::vector<std::pair<std::size_t, value_type>>> {
	static const char *name;
};
template<typename value_type>
const char *ComplexRootsDavidson<value_type>::name = "ComplexRootsDavidson";
static_assert(context<ComplexRootsDavidson<double>>);

}

/*!
 * @brief One specific implementation of LinearEigensystem using Davidson's algorithm
 * with modifications to manage near linear dependencies, and consequent numerical noise, in candidate expansion
 * vectors.
 *
 * TODO add more documentation and examples
 *
 * @tparam R
 * @tparam Q
 * @tparam P
 */
template <class R, class Q = R, class P = std::map<size_t, typename R::value_type>>
class LinearEigensystemDavidson : public IterativeSolverTemplate<LinearEigensystem, R, Q, P> {
public:
  using SolverTemplate = IterativeSolverTemplate<LinearEigensystem, R, Q, P>;
  using typename SolverTemplate::scalar_type;
  using IterativeSolverTemplate<LinearEigensystem, R, Q, P>::report;

  explicit LinearEigensystemDavidson(const std::shared_ptr<ArrayHandlers<R, Q, P>>& handlers =
                                         std::make_shared<molpro::linalg::itsolv::ArrayHandlers<R, Q, P>>(),
                                     const std::shared_ptr<Logger>& logger_ = std::make_shared<Logger>())
      : SolverTemplate(std::make_shared<subspace::XSpace<R, Q, P>>(handlers, logger_),
                       std::static_pointer_cast<subspace::ISubspaceSolver<R, Q, P>>(
                           std::make_shared<subspace::SubspaceSolverLinEig<R, Q, P>>(logger_)),
                       handlers, std::make_shared<Statistics>(), logger_) {
    set_hermiticity(m_hermiticity);
    this->m_normalise_solution = false;
  }

  bool nonlinear() const override { return false; }

  /*!
   * \brief Proposes new parameters for the subspace from the preconditioned residuals.
   *
   * After add_vector solves the subspace problem it returns the new solution and residual. The residual should be
   * preconditioned, i.e. using Davidson method, to accelerate convergence. The updated residual is used in this
   * function to propose new Q space parameters orthonormal to the old space. They are returned in parameters so that
   * corresponding actions can be calculated and used in add_vector in the next iteration.
   *
   * Every n_reset_D iterations the D space has to be reset. If working set does not cover all solutions than resetting
   * is done over multiple iterations. During this time, there are no deletions from the Q space.
   * This is done by temporarily increasing maximum allowed size of Q space.
   *
   * @param parameters output new parameters for the subspace.
   * @param residual preconditioned residuals.
   * @return number of significant parameters to calculate the action for
   */
  size_t end_iteration(const VecRef<R>& parameters, const VecRef<R>& action) override {
    auto prof = this->profiler();
    auto m_prof = prof->push("itsolv::end_iteration");
    if (m_dspace_resetter.do_reset(this->m_stats->iterations, this->m_xspace->dimensions())) {
      m_resetting_in_progress = true;
      this->m_working_set =
          m_dspace_resetter.run(parameters, *this->m_xspace, this->m_subspace_solver->solutions(),
                                rspace_opts.norm_thresh, rspace_opts.svd_thresh, *this->m_handlers, *this->m_logger);
    } else {
      m_resetting_in_progress = false;
      prof->start("end_iteration (propose_rspace)");
      this->m_working_set =
          detail::propose_rspace(*this, parameters, action, *this->m_xspace, *this->m_subspace_solver,
                                 *this->m_handlers, *this->m_logger, rspace_opts, qspace_opts, *this->profiler().get());
      prof->stop();
    }
    this->m_stats->iterations++;
    read_handler_counts(this->m_stats, this->m_handlers);
    this->m_end_iteration_needed = false;
    return this->working_set().size();
  }
  size_t end_iteration(std::vector<R>& parameters, std::vector<R>& action) override {
    return end_iteration(wrap(parameters), wrap(action));
  }
  size_t end_iteration(R& parameters, R& actions) override {
    auto wparams = std::vector<std::reference_wrapper<R>>{std::ref(parameters)};
    auto wactions = std::vector<std::reference_wrapper<R>>{std::ref(actions)};
    return end_iteration(wparams, wactions);
  }

  void finalize() override {
    if constexpr (is_complex<typename R::value_type>::value) {
      return;
    }

    auto subspace_solver = std::dynamic_pointer_cast<subspace::SubspaceSolverLinEig<R, Q, P>>(this->m_subspace_solver);
    assert(subspace_solver);
    auto imag_eigval_components = subspace_solver->imag_eigval_components();
    if (imag_eigval_components.empty()) {
      return;
    }

    // Convert to 1-based indexing for printout
    for (auto& pair : imag_eigval_components) {
      pair.first += 1;
    }

    this->m_logger->template warn<log::ComplexRootsDavidson<typename R::value_type>>(
        "The following roots are complex-valued. Associated eigenvectors are the real and imaginary part "
        "of the pairs and the imaginary parts of the eigenvalues are ",
        imag_eigval_components);
  }

  //! Applies the Davidson preconditioner
  void precondition(std::vector<R>& parameters, std::vector<R>& action) const {}

  std::vector<scalar_type> eigenvalues() const override { return this->m_subspace_solver->eigenvalues(); }

  virtual std::vector<scalar_type> working_set_eigenvalues() const override {
    auto eval = std::vector<scalar_type>{};
    for (auto i : this->working_set()) {
      eval.push_back(i < this->m_subspace_solver->eigenvalues().size() ? this->m_subspace_solver->eigenvalues().at(i)
                                                                       : 0);
    }
    return eval;
  }

  void set_value_errors() override {
    auto current_values = this->m_subspace_solver->eigenvalues();
    this->m_value_errors.assign(current_values.size(), std::numeric_limits<scalar_type>::max());
    for (size_t i = 0; i < std::min(m_last_values.size(), current_values.size()); i++)
      this->m_value_errors[i] = std::abs(current_values[i] - m_last_values[i]);
    if (!m_resetting_in_progress)
      m_last_values = current_values;
  }

  void report(std::ostream& cout, bool endl = true) const override {
    SolverTemplate::report(cout);
    cout << "errors " << std::scientific;
    auto& err = this->m_errors;
    std::copy(begin(err), end(err), std::ostream_iterator<scalar_type>(cout, ", "));
    cout << std::endl;
    cout << "eigenvalues ";
    auto ev = eigenvalues();
    cout << std::fixed << std::setprecision(14);
    std::copy(begin(ev), end(ev), std::ostream_iterator<scalar_type>(cout, ", "));
    cout << std::defaultfloat;
    if (endl)
      cout << std::endl;
  }

  //! Set the period in iterations for resetting the D space
  void set_reset_D(size_t n) { m_dspace_resetter.set_nreset(n); }
  size_t get_reset_D() const { return m_dspace_resetter.get_nreset(); }
  //! Set the maximum size of Q space after resetting the D space
  void set_reset_D_maxQ_size(size_t n) { m_dspace_resetter.set_max_Qsize(n); }
  int get_reset_D_maxQ_size() const { return m_dspace_resetter.get_max_Qsize(); }
  std::size_t get_max_size_qspace() const { return qspace_opts.max_size; }
  void set_max_size_qspace(std::size_t n) {
    qspace_opts.max_size = n;
    if (m_dspace_resetter.get_max_Qsize() > qspace_opts.max_size)
      m_dspace_resetter.set_max_Qsize(qspace_opts.max_size);
  }
  std::size_t get_min_size_qspace() const { return qspace_opts.min_size; }
  void set_min_size_qspace(std::size_t n) { qspace_opts.min_size = n; }
  void set_hermiticity(bool hermitian) override {
    m_hermiticity = hermitian;
    auto xspace = std::dynamic_pointer_cast<subspace::XSpace<R, Q, P>>(this->m_xspace);
    xspace->set_hermiticity(hermitian);
    auto subspace_solver = std::dynamic_pointer_cast<subspace::SubspaceSolverLinEig<R, Q, P>>(this->m_subspace_solver);
    subspace_solver->set_hermiticity(hermitian);
  }
  bool get_hermiticity() const override { return m_hermiticity; }

  void set_options(const Options& options) override {
    SolverTemplate::set_options(options);
    auto opt = CastOptions::LinearEigensystem(options);
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
      rspace_opts.norm_thresh = opt.norm_thresh.value();
    if (opt.svd_thresh)
      rspace_opts.svd_thresh = opt.svd_thresh.value();
    if (opt.hermiticity)
      set_hermiticity(opt.hermiticity.value());
  }

  std::shared_ptr<Options> get_options() const override {
    auto opt = std::make_shared<LinearEigensystemDavidsonOptions>();
    opt->copy(*SolverTemplate::get_options());
    opt->reset_D = get_reset_D();
    opt->reset_D_max_Q_size = get_reset_D_maxQ_size();
    opt->max_size_qspace = get_max_size_qspace();
    opt->min_size_qspace = get_min_size_qspace();
    opt->contrib_thresh = qspace_opts.contrib_thresh;
    opt->norm_thresh = rspace_opts.norm_thresh;
    opt->svd_thresh = rspace_opts.svd_thresh;
    opt->hermiticity = get_hermiticity();
    return opt;
  }

protected:
  void construct_residual(const std::vector<int>& roots, const CVecRef<R>& params, const VecRef<R>& actions) override {
    auto prof = this->profiler()->push("itsolv::construct_residual");
    assert(params.size() >= roots.size());
    const auto& eigvals = eigenvalues();

    auto subspace_solver = std::dynamic_pointer_cast<subspace::SubspaceSolverLinEig<R, Q, P>>(this->m_subspace_solver);
    assert(subspace_solver);
    const auto& imag_eigval_components = subspace_solver->imag_eigval_components();

    for (size_t i = 0; i < roots.size(); ++i) {
      this->m_handlers->rr().axpy(-eigvals.at(roots[i]), params.at(i), actions.at(i));

      auto it = std::ranges::find(imag_eigval_components, roots[i], [](const auto& pair) { return pair.first; });

      if (it != imag_eigval_components.end()) {
        // Ref.: https://doi.org/10.1063/1.2755681 (Appendix)
        // The idea is the following: We make use of the fact that the real and imaginary parts of the
        // eigenvectors belonging to the complex root pair are spanned by the same basis. This allows
        // us to split the associated residual vectors into a part corresponding to the real and a part
        // corresponding to the imaginary part as well. The actual (complex-valued) residual can then
        // always be reconstructed from this and therefore we don't lose any information. The formulas are
        // |(R_real)_i> = sum_j [ (|A_j> - (e_real)_i |b_j>) (c_real)_{ji} + (e_imag) |b_j> (c_imag)_{ji} ]
        // |(R_imag)_i> = sum_j [ (|A_j> - (e_real)_i |b_j>) (c_imag)_{ji} - (e_imag) |b_j> (c_real)_{ji} ]
        // where |A_j> is the j-th action and |b_j> the j-th trial vector. e are the complex conjugate
        // eigenvalues and c the associated eigenvectors.
        // Note: At this point, params and actions already contain the vectors transformed into the
        // eigenbasis (aka.: multiplied with c). Due to the way we process these complex-valued vectors,
        // the first vector in a pair is the one that has been transformed with the real and the second
        // with the imaginary part of c.
        // The above code has already dealt with the bulk of the necessary expression and we only need
        // to add the part with the imaginary components of the eigenvalue pair. We use the knowledge
        // about the ordering of vectors (which has been transformed with what eigenvector component)
        // to compute the right thing without explicit access to the eigenvectors.
        const auto distance = std::ranges::distance(imag_eigval_components.begin(), it);
        const int offset = (distance % 2) == 0 ? 1 : -1;

        auto root_it = std::ranges::find(roots, roots[i] + offset);
        if (root_it == roots.end()) {
          // We can only do this, if we have both components of a complex eigenvalue/eigenvector pair
          // available here.
          this->m_logger->warn(
              "Complex conjugate eigenvalue pair incomplete in construct_residual (this can lead to poor convergence)");
          continue;
        }

        const std::size_t param_idx = std::ranges::distance(roots.begin(), root_it);

        // Note: The minus sign in above expression is implicitly taken into account as the
        // imaginary parts of the complex conjugate eigenvalue pair has flipped signs
        this->m_handlers->rr().axpy(it->second, params.at(param_idx), actions.at(i));
      }
    }
  }

  detail::DSpaceResetter<Q> m_dspace_resetter; //!< resets D space
  bool m_hermiticity = false;                  //!< whether the problem is hermitian or not
  std::vector<double> m_last_values;           //!< The values from the previous iteration
  bool m_resetting_in_progress = false;        //!< whether D space resetting is in progress
  RSpaceOptions rspace_opts;                   //!< Options concerning R-space handling
  QSpaceOptions qspace_opts;                   //!< Options concerning Q-space handling
};

} // namespace molpro::linalg::itsolv

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_LINEAREIGENSYSTEMDAVIDSON_H
