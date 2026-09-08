#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_ITERATIVESOLVERTEMPLATE_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_ITERATIVESOLVERTEMPLATE_H
#include <molpro/Profiler.h>
#include <molpro/iostream.h>
#include <molpro/linalg/itsolv/IterativeSolver.h>
#include <molpro/linalg/itsolv/Logger.h>
#include <molpro/linalg/itsolv/subspace/ISubspaceSolver.h>
#include <molpro/linalg/itsolv/subspace/IXSpace.h>
#include <molpro/linalg/itsolv/subspace/Matrix.h>
#include <molpro/linalg/itsolv/subspace/util.h>
#include <molpro/linalg/itsolv/util.h>
#include <molpro/linalg/itsolv/wrap.h>
#include <molpro/profiler/Profiler.h>

#include <cassert>
#include <cmath>
#include <format>
#include <iostream>
#include <optional>
#include <sstream>

namespace molpro::linalg::itsolv {
namespace detail {

inline std::vector<std::pair<size_t, size_t>> parameter_batches(const size_t nsol, const size_t nparam) {
  auto batches = std::vector<std::pair<size_t, size_t>>{};
  if (nparam && nsol) {
    auto n_batch = nsol / nparam + (nsol % nparam ? 1 : 0);
    for (size_t ib = 0, start_sol = 0, end_sol = 0; ib < n_batch; ++ib, start_sol = end_sol) {
      end_sol = std::min(start_sol + nparam, nsol);
      batches.emplace_back(start_sol, end_sol);
    }
  }
  return batches;
}

template <class R, class Q, class P>
void construct_solution(const VecRef<R>& params, const std::vector<int>& roots,
                        const subspace::Matrix<double>& solutions,
                        const std::vector<std::reference_wrapper<P>>& pparams,
                        const std::vector<std::reference_wrapper<Q>>& qparams,
                        const std::vector<std::reference_wrapper<Q>>& dparams, size_t oP, size_t oQ, size_t oD,
                        ArrayHandlers<R, Q, P>& handlers) {
  auto prof = molpro::Profiler::single();
  prof->start("get rd_mat");
  if (roots.empty())
    return;
  assert(params.size() >= roots.size());
  for (size_t i = 0; i < roots.size(); ++i) {
    handlers.rr().fill(0, params.at(i));
  }
  subspace::Matrix<double> rp_mat(std::make_pair(pparams.size(), roots.size())),
      rq_mat(std::make_pair(qparams.size(), roots.size())), rd_mat(std::make_pair(dparams.size(), roots.size()));
  for (size_t i = 0; i < roots.size(); ++i) {
    for (size_t j = 0; j < pparams.size(); ++j) {
      rp_mat(j, i) = solutions(roots[i], oP + j);
    }
    for (size_t j = 0; j < qparams.size(); ++j) {
      rq_mat(j, i) = solutions(roots[i], oQ + j);
    }
    for (size_t j = 0; j < dparams.size(); ++j) {
      rd_mat(j, i) = solutions(roots[i], oD + j);
    }
  }
  prof->stop();
  handlers.rp().gemm_outer(rp_mat, cwrap(pparams), params);
  handlers.rq().gemm_outer(rq_mat, cwrap(qparams), params);
  handlers.rq().gemm_outer(rd_mat, cwrap(dparams), params);
}

template <typename T>
std::vector<std::vector<T>> construct_vectorP(const std::vector<int>& roots, const subspace::Matrix<T>& solutions,
                                              const size_t oP, const size_t nP) {
  auto vectorP = std::vector<std::vector<T>>{};
  for (auto root : roots) {
    vectorP.emplace_back();
    for (size_t j = 0; j < nP; ++j)
      vectorP.back().push_back(solutions(root, oP + j));
  }
  return vectorP;
}

template <class R>
void normalise(const size_t n_roots, const VecRef<R>& params, const VecRef<R>& actions,
               array::ArrayHandler<R, R>& handler, Logger& logger) {
  assert(params.size() >= n_roots && actions.size() >= n_roots);
  for (size_t i = 0; i < n_roots; ++i) {
    auto dot = handler.dot(params.at(i), params.at(i));
    dot = std::sqrt(std::abs(dot));
    if (dot > 1.0e-14) {
      handler.scal(1. / dot, params.at(i));
      handler.scal(1. / dot, actions.at(i));
    } else {
      logger.warn("solution parameter's length is too small, dot = " + std::format("{:.2e}", dot));
    }
  }
}

template <class R, typename T>
void update_errors(std::vector<T>& errors, const CVecRef<R>& residual, array::ArrayHandler<R, R>& handler) {
  assert(residual.size() >= errors.size());
  for (size_t i = 0; i < errors.size(); ++i) {
    auto a = handler.dot(residual[i], residual[i]);
    errors[i] = std::sqrt(std::abs(a));
  }
}

template <typename T>
std::vector<int> select_working_set(const size_t nw, const std::vector<T>& errors, const T threshold,
                                    const std::vector<T>& value_errors, const T value_threshold) {
  auto ordered_errors = std::multimap<T, size_t, std::greater<T>>{};
  for (size_t i = 0; i < errors.size(); ++i) {
    if (errors[i] > threshold or (i < value_errors.size() and value_errors[i] > value_threshold))
      ordered_errors.emplace(errors[i], i);
  }
  auto working_set = std::vector<int>{};
  auto end = (ordered_errors.size() < nw ? ordered_errors.end() : next(begin(ordered_errors), nw));
  std::transform(begin(ordered_errors), end, std::back_inserter(working_set), [](const auto& el) { return el.second; });
  std::sort(working_set.begin(), working_set.end());
  return working_set;
}

} // namespace detail

namespace log {

/*!
 * Context used when logging about the beginning of a new iteration
 */
struct NewIteration : ContextBase<NewIteration, true, int, std::vector<double>> {
  static constexpr const char *name = "NewIteration";
  static constexpr std::size_t iter = 0;
  static constexpr std::size_t errors = 1;
};
static_assert(context<NewIteration>);

/*!
 * Context used when logging a per-iteration report message
 */
struct IterationReport : ContextBase<IterationReport, true> {
  static constexpr const char *name = "IterationReport";
};
static_assert(context<IterationReport>);

/*!
 * Context used when logging a summary report message
 */
struct SummaryReport : ContextBase<SummaryReport, true> {
  static constexpr const char *name = "SummaryReport";
};
static_assert(context<SummaryReport>);

}

/*!
 * @brief Implements IterativeSolver interface that is common to all solvers
 *
 * @tparam Solver one of the solvers that inherits from IterativeSolver.
 */
template <template <class, class, class> class Solver, class R, class Q, class P>
class IterativeSolverTemplate : public Solver<R, Q, P> {
public:
  using typename Solver<R, Q, P>::fapply_on_p_type;
  using typename Solver<R, Q, P>::scalar_type;
  using typename Solver<R, Q, P>::value_type;
  using typename Solver<R, Q, P>::VectorP;

  IterativeSolverTemplate() = delete;
  IterativeSolverTemplate(const IterativeSolverTemplate<Solver, R, Q, P>&) = delete;
  IterativeSolverTemplate(IterativeSolverTemplate<Solver, R, Q, P>&&) noexcept = default;
  IterativeSolverTemplate<Solver, R, Q, P>& operator=(const IterativeSolverTemplate<Solver, R, Q, P>&) = delete;
  IterativeSolverTemplate<Solver, R, Q, P>& operator=(IterativeSolverTemplate<Solver, R, Q, P>&&) noexcept = default;

  void set_logger(std::shared_ptr<Logger> logger) override {
    assert(logger);
    m_logger = std::move(logger);
    m_xspace->set_logger(m_logger);
    m_subspace_solver->set_logger(m_logger);
  }

  Logger &logger() override { return *m_logger; }

  int add_vector(const VecRef<R>& parameters, const VecRef<R>& actions) override {
    auto outer = profiler()->push("itsolv::add_vector");
    auto prof = molpro::Profiler::single();
    m_logger->trace("IterativeSolverTemplate::add_vector  iteration = ", m_stats->iterations);
    m_logger->debug("IterativeSolverTemplate::add_vector  size of {params, actions, working_set} = ",
                      parameters.size(), actions.size(), m_working_set.size());
    if (m_xspace->dimensions().nP != 0 && !m_apply_p)
      throw std::runtime_error(
          "Solver contains P space but no valid apply_p function. Make sure add_p was called correctly.");
    auto nW = std::min(m_working_set.size(), parameters.size());
    auto cwparams = cwrap(begin(parameters), begin(parameters) + nW);
    auto cwactions = cwrap(begin(actions), begin(actions) + nW);
    m_stats->r_creations += nW;
    prof->start("update_qspace");
    m_xspace->update_qspace(cwparams, cwactions);
    m_stats->q_creations += 2 * nW;
    prof->stop();
    prof->start("solve_and_generate_working_set");
    auto working_set = solve_and_generate_working_set(parameters, actions);
    prof->stop();
    read_handler_counts(m_stats, m_handlers);
    this->m_end_iteration_needed = true;
    return working_set;
  }

  int add_vector(std::vector<R>& parameters, std::vector<R>& actions) override {
    return add_vector(wrap(parameters), wrap(actions));
  }
  int add_vector(R& parameters, R& actions, value_type value = 0) override {
    return add_vector(wrap_arg(parameters), wrap_arg(actions));
  }

  // FIXME Currently only works if called on an empty subspace. Either enforce it or generalise.
  size_t add_p(const CVecRef<P>& pparams, const array::Span<value_type>& pp_action_matrix, const VecRef<R>& parameters,
               const VecRef<R>& actions, fapply_on_p_type apply_p) override {
    auto prof = profiler()->push("itsolv::add_p");
    if (not pparams.empty() and pparams.size() < n_roots())
      throw std::runtime_error("P space must be empty or at least as large as number of roots sought");
    if (apply_p)
      m_apply_p = std::move(apply_p);
    m_xspace->update_pspace(pparams, pp_action_matrix);
    auto working_set = solve_and_generate_working_set(parameters, actions);
    read_handler_counts(m_stats, m_handlers);
    return working_set;
  };

  void clearP() override {}

  void solution(const std::vector<int>& roots, const VecRef<R>& parameters, const VecRef<R>& residual) override {
    // auto prof = profiler()->push("itsolv::solution"); // FIXME two profilers
    auto prof = molpro::Profiler::single();
    check_consistent_number_of_roots_and_solutions(roots, parameters.size());
    prof->start("construct_solution (parameters)");
    detail::construct_solution(parameters, roots, m_subspace_solver->solutions(), m_xspace->paramsp(),
                               m_xspace->paramsq(), m_xspace->paramsd(), m_xspace->dimensions().oP,
                               m_xspace->dimensions().oQ, m_xspace->dimensions().oD, *m_handlers);
    prof->stop();
    prof->start("construct_solution (residual)");
    detail::construct_solution(residual, roots, m_subspace_solver->solutions(), {}, m_xspace->actionsq(),
                               m_xspace->actionsd(), m_xspace->dimensions().oP, m_xspace->dimensions().oQ,
                               m_xspace->dimensions().oD, *m_handlers);
    prof->stop();
    prof->start("apply");
    auto pvectors = detail::construct_vectorP(roots, m_subspace_solver->solutions(), m_xspace->dimensions().oP,
                                              m_xspace->dimensions().nP);
    if (m_normalise_solution)
      detail::normalise(roots.size(), parameters, residual, m_handlers->rr(), *m_logger);
    if (m_apply_p)
      m_apply_p(pvectors, m_xspace->cparamsp(), residual);
    construct_residual(roots, cwrap(parameters), residual);
    read_handler_counts(m_stats, m_handlers);
    prof->stop();
  };

  void solution(const std::vector<int>& roots, std::vector<R>& parameters, std::vector<R>& residual) override {
    return solution(roots, wrap(parameters), wrap(residual));
  }
  void solution(R& parameters, R& residual) override {
    return solution(std::vector<int>(1, 0), wrap_arg(parameters), wrap_arg(residual));
  }

  void solution_params(const std::vector<int>& roots, std::vector<R>& parameters) override {
    return solution_params(roots, wrap(parameters));
  }

  void solution_params(const std::vector<int>& roots, const VecRef<R>& parameters) override {
    check_consistent_number_of_roots_and_solutions(roots, parameters.size());
    detail::construct_solution(parameters, roots, m_subspace_solver->solutions(), m_xspace->paramsp(),
                               m_xspace->paramsq(), m_xspace->paramsd(), m_xspace->dimensions().oP,
                               m_xspace->dimensions().oQ, m_xspace->dimensions().oD, *m_handlers);
  };

  void solution_params(R& parameters) override { return solution_params(std::vector<int>(1, 0), wrap_arg(parameters)); }

  // TODO Implement this
  std::vector<size_t> suggest_p(const CVecRef<R>& solution, const CVecRef<R>& residual, size_t max_number,
                                double threshold) override {
    return {};
  }

  const std::vector<int>& working_set() const override { return m_working_set; }

  size_t n_roots() const override { return m_nroots; }

  void set_n_roots(size_t roots) override {
    m_nroots = roots;
    m_working_set.resize(roots);
    std::iota(begin(m_working_set), end(m_working_set), (int)0);
  }

  void set_options(const Options& options) override {
    if (options.n_roots)
      set_n_roots(options.n_roots.value());
    if (options.convergence_threshold)
      set_convergence_threshold(options.convergence_threshold.value());
    if (options.verbosity)
      set_verbosity(options.verbosity.value());
    if (options.max_iter)
      set_max_iter(options.max_iter.value());
    if (options.max_p)
      set_max_p(options.max_p.value());
    if (options.p_threshold)
      set_p_threshold(options.p_threshold.value());
  }

  std::shared_ptr<Options> get_options() const override {
    auto options = std::make_shared<Options>();
    options->n_roots = n_roots();
    options->convergence_threshold = convergence_threshold();
    options->verbosity = get_verbosity();
    options->max_iter = get_max_iter();
    options->max_p = get_max_p();
    options->p_threshold = get_p_threshold();
    return options;
  }

  const std::vector<scalar_type>& errors() const override { return m_errors; }

  const Statistics& statistics() const override { return *m_stats; }

  void report(std::ostream& cout, bool endl = true) const override {
    cout << "iteration " << m_stats->iterations;
    if (not m_errors.empty()) {
      auto it_max_error = std::max_element(m_errors.cbegin(), m_errors.cend());
      if (n_roots() > 1)
        cout << ", |residual[" << std::distance(m_errors.cbegin(), it_max_error) << "]| = ";
      else
        cout << ", |residual| = ";
      cout << std::scientific << *it_max_error << std::defaultfloat;
    }
    if (endl)
      cout << std::endl;
  }

  void iteration_report() const {
    std::stringstream sstream;
    report(sstream, false);
    this->m_logger->info<log::IterationReport>(sstream.str());
  }

  void summary_report() const {
    std::stringstream sstream;
    report(sstream, false);
    this->m_logger->info<log::SummaryReport>(sstream.str());
  }

  void report() const override {
    // We don't know what kind of report this is supposed to be -> default to summary
    summary_report();
  }

  void set_convergence_threshold(double thresh) override { m_convergence_threshold = thresh; }
  double convergence_threshold() const override { return m_convergence_threshold; }
  void set_convergence_threshold_value(double thresh) override { m_convergence_threshold_value = thresh; }
  double convergence_threshold_value() const override { return m_convergence_threshold_value; }
  void set_verbosity(Verbosity v) override { m_verbosity = v; }
  void set_verbosity(int v) override {
    if (v == 0) {
      m_logger->set_verbosity(log::Verbosity::None);
      m_logger->set_min_severity(log::Severity::Error);
    } else if (v == 1) {
      m_logger->set_verbosity(log::Verbosity::None);
      m_logger->set_min_severity(log::Severity::Warning);
    } else if (v == 2) {
      m_logger->set_verbosity(log::Verbosity::Info);
      m_logger->set_min_severity(log::Severity::Normal);
    } else if (v == 3) {
      m_logger->set_verbosity(log::Verbosity::Debug);
      m_logger->set_min_severity(log::Severity::Normal);
    } else {
      m_logger->set_verbosity(log::Verbosity::Trace);
      m_logger->set_min_severity(log::Severity::Normal);
    }
  }
  Verbosity get_verbosity() const override { return m_verbosity.value_or(Verbosity::None); }
  void set_max_iter(int n) override { m_max_iter = n; }
  int get_max_iter() const override { return m_max_iter; }
  void set_max_p(int n) override { m_max_p = n; }
  int get_max_p() const override { return m_max_p; }
  void set_p_threshold(double threshold) override { m_p_threshold = threshold; }
  double get_p_threshold() const override { return m_p_threshold; }
  //! Access dimensions of the subspace
  const subspace::Dimensions& dimensions() const override { return m_xspace->dimensions(); }
  scalar_type value() const override {
    return m_xspace->data.count(subspace::EqnData::value) > 0 ? m_xspace->data[subspace::EqnData::value](0, 0)
                                                              : nan("molpro::linalg::itsolv::IterativeSolver::value");
  }
  //  void set_profiler(molpro::profiler::Profiler& profiler) override { m_profiler.reset(&profiler); }
  void set_profiler(molpro::profiler::Profiler& profiler) override {
    m_profiler = std::shared_ptr<molpro::profiler::Profiler>(&profiler);
  }
  const std::shared_ptr<molpro::profiler::Profiler>& profiler() const override { return m_profiler; }

  bool solve(const VecRef<R>& parameters, const VecRef<R>& actions, const Problem<R>& problem,
             bool generate_initial_guess = false) override {
    if (parameters.empty())
      throw std::runtime_error("Empty container passed to IterativeSolver::solve()");
    if (parameters.size() != actions.size())
      throw std::runtime_error("Inconsistent container sizes in IterativeSolver::solve()");
    if (this->m_verbosity == Verbosity::None) {
      // Backwards compatibility
      this->m_logger->set_verbosity(log::Verbosity::None);
    }
    if (this->m_verbosity == Verbosity::Detailed) {
      this->m_logger->set_verbosity(log::Verbosity::Trace);
      this->m_logger->enable_data_dumps(true);
    }
    bool use_diagonals = problem.diagonals(actions.at(0));
    std::unique_ptr<Q> diagonals;
    if (use_diagonals)
      diagonals.reset(new Q{m_handlers->qr().copy(actions.at(0))});
//    std::cout << "solve() generate_initial_guess "<<generate_initial_guess<<", roots "<<this->n_roots()<<std::endl;
    if (generate_initial_guess) {
      if (not use_diagonals)
        throw std::runtime_error("Default initial guess requested, but diagonal elements are not available");
      auto guess = m_handlers->qq().select(parameters.size(), *diagonals);
      size_t root = 0;
      if (!this->m_verbosity.has_value() || this->m_verbosity >= Verbosity::Summary) {
        this->m_logger->info("Initial guess generated from diagonal elements");
      }
      for (const auto& g : guess) {
        m_handlers->rp().copy(parameters[root], P{{g.first, 1}});
        if (!this->m_verbosity.has_value() || this->m_verbosity >= Verbosity::Detailed) {
          m_logger->debug("-> initial guess with index ", g.first);
        }
        root++;
      }
    }
    int nwork = parameters.size();
    std::vector<P> pspace;
    if (use_diagonals and m_max_p > 0) {
      auto selectp = m_handlers->qq().select(m_max_p, *diagonals);
      if (!selectp.empty()) {
        // selectp is keyed by index, not value; find the smallest selected
        // diagonal explicitly before applying the threshold.
        auto min_val = std::min_element(selectp.begin(), selectp.end(),
                                        [](const auto& a, const auto& b) { return a.second < b.second; })
                           ->second;
        for (auto s = selectp.begin(); s != selectp.end();) {
          if (s->second > min_val + m_p_threshold)
            s = selectp.erase(s);
          else
            ++s;
        }
      }
      if (!selectp.empty() && (!this->m_verbosity.has_value() || this->m_verbosity >= Verbosity::Summary)) {
        this->m_logger->info("P-space dimension, threshold and limit", selectp.size(), m_p_threshold, m_max_p);
      }
      if (!this->m_verbosity.has_value() || this->m_verbosity >= Verbosity::Detailed) {
        for (const auto& s : selectp) {
          this->m_logger->debug(std::format("P space element {}: {}", s.first, s.second));
        }
      }
      for (const auto& s : selectp)
        pspace.emplace_back((P){{s.first, 1}});
      fapply_on_p_type apply_on_p = [&problem](const std::vector<std::vector<value_type>>& pcoeff,
                                               const CVecRef<P>& pparams, const VecRef<R>& actions) {
        problem.p_action(pcoeff, pparams, actions);
      };
      auto action_matrix = problem.pp_action_matrix(pspace);
      nwork = add_p(cwrap(pspace), array::Span<value_type>(action_matrix.data(), action_matrix.size()), parameters,
                    actions, apply_on_p);
    }
    for (auto iter = 0; iter < this->m_max_iter && nwork > 0; iter++) {
      m_logger->info<log::NewIteration>("Iteration, Current errors", iter, m_errors);
      value_type value;
      if (this->nonlinear()) {
        value = problem.residual(*parameters.begin(), *actions.begin());
        nwork = this->add_vector(*parameters.begin(), *actions.begin(), value);
      } else if (iter > 0 or pspace.empty()) {
        problem.action(cwrap(parameters.begin(), parameters.begin() + nwork),
                       wrap(actions.begin(), actions.begin() + nwork));
        nwork = this->add_vector(parameters, actions);
      }
      //      std::cout << "** nwork="<<nwork<<"use_diagonals="<<use_diagonals<<std::endl;
      while (this->end_iteration_needed()) {
        if (nwork > 0) {
          if (use_diagonals) {
            m_handlers->rq().copy(parameters.at(0), *diagonals);
            problem.precondition(wrap(actions.begin(), actions.begin() + nwork), this->working_set_eigenvalues(),
                                 parameters.at(0));
          } else
            problem.precondition(wrap(actions.begin(), actions.begin() + nwork), this->working_set_eigenvalues());
        }
        nwork = this->end_iteration(parameters, actions);
      }
      if (!this->m_verbosity.has_value() || this->m_verbosity >= Verbosity::Iteration) {
        iteration_report();
      }
    }

    this->finalize();

    if (!this->m_verbosity.has_value() || this->m_verbosity >= Verbosity::Summary) {
      summary_report();
    }
    if (!this->m_verbosity.has_value() || this->m_verbosity >= Verbosity::Summary) {
      if (*std::max_element(m_errors.begin(), m_errors.end()) > m_convergence_threshold) {
        this->m_logger->warn("Solver has not converged to threshold ", m_convergence_threshold);
      } else {
        this->m_logger->info("Solver converged");
      }
    }
    return nwork == 0 and *std::max_element(m_errors.begin(), m_errors.end()) <= m_convergence_threshold;
  }

  bool solve(R& parameters, R& actions, const Problem<R>& problem, bool generate_initial_guess = false) override {
    auto wparams = std::vector<std::reference_wrapper<R>>{std::ref(parameters)};
    auto wactions = std::vector<std::reference_wrapper<R>>{std::ref(actions)};
    return solve(wparams, wactions, problem, generate_initial_guess);
  }
  bool solve(std::vector<R>& parameters, std::vector<R>& actions, const Problem<R>& problem,
             bool generate_initial_guess = false) override {
    return solve(wrap(parameters), wrap(actions), problem, generate_initial_guess);
  }

  bool test_problem(const Problem<R>& problem, R& v0, R& v1, int verbosity, double threshold) const override {
    bool success = true;
    double step=1e-4;
    if (this->nonlinear()) {
      if (!problem.test_parameters(0, v0))
        return true;
      auto value0 = problem.residual(v0, v1);
      if (verbosity > 1) {
        std::cout << "value0 " << value0 << std::endl;
        //        std::cout << "parameters0 " << v0[0] << "," << v0[1] << ",..." << std::endl;
        //        std::cout << "residual0 " << v1[0] << "," << v1[1] << ",..." << std::endl;
      }
      Q parameters0 = m_handlers->qr().copy(v0);
      Q residual0 = m_handlers->qr().copy(v1);
      for (int instance = 1; problem.test_parameters(instance, v0); ++instance) {
        m_handlers->rq().axpy(-1.0, parameters0, v0);
        m_handlers->rr().scal(1 / std::sqrt(m_handlers->rr().dot(v0, v0)), v0);
        Q step1 = m_handlers->qr().copy(v0);
        auto residual_analytic = m_handlers->rq().dot(v0, residual0);
        m_handlers->rq().copy(v0, parameters0);
        m_handlers->rq().axpy(-2*step, step1, v0);
        auto valuem2 = problem.residual(v0, v1);
        m_handlers->rq().copy(v0, parameters0);
        m_handlers->rq().axpy(-1*step, step1, v0);
        auto valuem1 = problem.residual(v0, v1);
        m_handlers->rq().copy(v0, parameters0);
        m_handlers->rq().axpy(+1*step, step1, v0);
        auto valuep1 = problem.residual(v0, v1);
        m_handlers->rq().copy(v0, parameters0);
        m_handlers->rq().axpy(+2*step, step1, v0);
        auto valuep2 = problem.residual(v0, v1);
        auto residual_numerical = (valuem2 - 8*valuem1 + 8*valuep1 - valuep2) / (12*step);
        if (verbosity > 1)
          std::cout << "testing problem class, instance: " << instance << ", numerical: " << residual_numerical <<", analytical: "<<residual_analytic <<", difference: "<<residual_numerical-residual_analytic << std::endl;
        success = success && std::abs(residual_numerical - residual_analytic) < threshold;
      }
    } else {
      for (int instance = 0; problem.test_parameters(instance, v0); ++instance) {
        problem.action({v0}, {v1});
        Q residual = m_handlers->qr().copy(v1);
        auto norm2_residual = std::sqrt(m_handlers->rr().dot(v1, v1));
        constexpr double scale_factor{10.0};
        m_handlers->rr().scal(scale_factor, v0);
        problem.action({v0}, {v1});
        m_handlers->rq().axpy(-scale_factor, residual, v1);
        auto norm2 = std::sqrt(m_handlers->rr().dot(v1, v1));
        success = success && std::abs(norm2 / norm2_residual) < threshold;
        if (verbosity > 0 or (verbosity > -1 and not success))
          std::cout << "Length of residual: " << norm2_residual << ", scaling defect: " << norm2 << std::endl;
      }
    }
    return success;
  }

protected:
  IterativeSolverTemplate(std::shared_ptr<subspace::IXSpace<R, Q, P>> xspace,
                          std::shared_ptr<subspace::ISubspaceSolver<R, Q, P>> solver,
                          std::shared_ptr<ArrayHandlers<R, Q, P>> handlers, std::shared_ptr<Statistics> stats,
                          std::shared_ptr<Logger> logger)
      : m_handlers(std::move(handlers)), m_xspace(std::move(xspace)), m_subspace_solver(std::move(solver)),
        m_stats(std::move(stats)), m_logger(std::move(logger)), m_profiler(molpro::Profiler::single()),
        m_profiler_saved_depth(m_profiler->get_max_depth()) {
    set_n_roots(1);
    m_profiler->set_max_depth(options()->parameter("PROFILER_DEPTH", 0));
  }

  virtual ~IterativeSolverTemplate() {
    if (molpro::mpi::rank_global() == 0) {
      auto file = options()->parameter("PROFILER_OUTPUT", "");
      if (profiler()->get_max_depth() > 0 and
          std::find_if(file.begin(), file.end(), [](unsigned char ch) { return !std::isspace(ch); }) != file.end())
        std::ofstream(file) << *profiler() << std::endl;
    }
    if (molpro::mpi::rank_global() == 0) {
      auto file = options()->parameter("PROFILER_DOTGRAPH", "");
      if (profiler()->get_max_depth() > 0 and
          std::find_if(file.begin(), file.end(), [](unsigned char ch) { return !std::isspace(ch); }) != file.end())
        profiler()->dotgraph(file, options()->parameter("PROFILER_THRESHOLD", .01));
    }
    molpro::Profiler::single()->set_max_depth(m_profiler_saved_depth);
  }

  //! Implementation class should overload this to set errors in the current values (e.g. change in eigenvalues)
  virtual void set_value_errors() {}
  //! Constructs residual for given roots provided their parameters and actions
  virtual void construct_residual(const std::vector<int>& roots, const CVecRef<R>& params,
                                  const VecRef<R>& actions) = 0;
  virtual bool linearEigensystem() const { return false; }
  /*!
   * @brief Solves the subspace problems and selects the working set of roots, returning their parameters and residual
   * in parameters and action
   * @param parameters container for storing parameters of the working set
   * @param action container for storing the residual of the working set
   * @param apply_p function that accumulates action from the P space projection of parameters
   * @return size of the working set
   */
  size_t solve_and_generate_working_set(const VecRef<R>& parameters, const VecRef<R>& action) {
    auto prof = profiler();
    auto p = prof->push("itsolv::solve_and_generate_working_set");
    prof->start("itsolv::solve");
    m_subspace_solver->solve(*m_xspace, n_roots());
    prof->stop();
    prof->start("itsolv::temp_solutions");
    auto nsol = m_subspace_solver->size();
    std::vector<std::pair<Q, Q>> temp_solutions{};
    const auto batches = detail::parameter_batches(nsol, parameters.size());
    for (const auto& batch : batches) {
      auto [start_sol, end_sol] = batch;
      auto roots = std::vector<int>(end_sol - start_sol);
      std::iota(begin(roots), end(roots), start_sol);
      solution(roots, parameters, action);
      auto errors = std::vector<scalar_type>(roots.size(), 0);
      detail::update_errors(errors, cwrap(action), m_handlers->rr());
      if (batches.size() > 1) {
        for (size_t i = 0; i < roots.size(); ++i)
          temp_solutions.emplace_back(m_handlers->qr().copy(parameters[i]), m_handlers->qr().copy(action[i]));
        m_stats->q_creations += 2 * roots.size();
      }
      m_subspace_solver->set_error(roots, errors);
    }
    prof->stop();
    set_value_errors();
    m_errors = m_subspace_solver->errors();
    m_working_set = detail::select_working_set(parameters.size(), m_errors, m_convergence_threshold, m_value_errors,
                                               m_convergence_threshold_value);
    for (size_t i = 0; i < m_working_set.size(); ++i) {
      size_t root = m_working_set[i];
      if (batches.size() > 1) {
        m_handlers->rq().copy(parameters[i], temp_solutions.at(root).first);
        m_handlers->rq().copy(action[i], temp_solutions.at(root).second);
      } else {
        if (root < i)
          throw std::logic_error("incorrect ordering of roots");
        if (root > i && root < parameters.size()) {
          m_handlers->rr().copy(parameters[i], parameters[root]);
          m_handlers->rr().copy(action[i], action[root]);
        }
      }
    }
    m_logger->trace("add_vector::errors = ", m_errors);
    return m_working_set.size();
  }

  template <typename TTT>
  void check_consistent_number_of_roots_and_solutions(const std::vector<TTT>& roots, const size_t nparams) {
    if (roots.size() > nparams)
      throw std::runtime_error("asking for more roots than parameters");
    if (!roots.empty() &&
        size_t(*std::max_element(roots.begin(), roots.end())) >= m_subspace_solver->solutions().size())
      throw std::runtime_error("asking for more roots than there are solutions");
  }

  bool end_iteration_needed() override { return m_end_iteration_needed; }

  void finalize() override {}


  std::shared_ptr<ArrayHandlers<R, Q, P>> m_handlers;                    //!< Array handlers
  std::shared_ptr<subspace::IXSpace<R, Q, P>> m_xspace;                  //!< manages the subspace and associated data
  std::shared_ptr<subspace::ISubspaceSolver<R, Q, P>> m_subspace_solver; //!< solves the subspace problem
  std::vector<double> m_errors;                                          //!< errors from the most recent solution
  std::vector<double> m_value_errors;                                    //!< value errors from the most recent solution
  std::vector<int> m_working_set;                                        //!< indices of roots in the working set
  size_t m_nroots{0};                     //!< number of roots the solver is searching for
  double m_convergence_threshold{1.0e-8}; //!< residual norms less than this mark a converged solution
  double m_convergence_threshold_value{
      std::numeric_limits<double>::max()};      //!< value changes less than this mark a converged solution
  std::shared_ptr<Statistics> m_stats;          //!< accumulates statistics of operations performed by the solver
  std::shared_ptr<Logger> m_logger;             //!< logger
  bool m_normalise_solution = false;            //!< whether to normalise the solutions
  fapply_on_p_type m_apply_p = {};              //!< function that evaluates effect of action on the P space projection
  std::optional<Verbosity> m_verbosity = {};    //!< how much output to print in solve()
  int m_max_iter = 100;                         //!< maximum number of iterations in solve()
  size_t m_max_p = 0;                           //!< maximum size of P space
  double m_p_threshold = std::numeric_limits<double>::max(); //!< threshold for selecting P space
private:
  mutable std::shared_ptr<molpro::profiler::Profiler> m_profiler;
  int m_profiler_saved_depth; //!< max_depth of molpro::Profiler::single() before this object changed it
protected:
  bool m_end_iteration_needed = true; //!< whether end_iteration should be called after any preconditioner
};

} // namespace molpro::linalg::itsolv

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_ITERATIVESOLVERTEMPLATE_H
