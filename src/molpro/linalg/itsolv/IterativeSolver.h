#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_ITERATIVESOLVER_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_ITERATIVESOLVER_H
#include <molpro/linalg/array/ArrayHandler.h>
#include <molpro/linalg/array/Span.h>
#include <molpro/linalg/itsolv/Options.h>
#include <molpro/linalg/itsolv/Statistics.h>
#include <molpro/linalg/itsolv/subspace/Dimensions.h>
#include <molpro/linalg/itsolv/wrap.h>
#include <molpro/linalg/itsolv/Logger.h>

#include <molpro/linalg/array/DistrArray.h>
#include <molpro/linalg/array/util/Distribution.h>

#include <memory>
#include <ostream>
#include <vector>

namespace molpro::profiler {
class Profiler;
}
namespace molpro::linalg::itsolv {

template <typename T>
struct has_iterator {
  template <typename C>
  constexpr static std::true_type test(typename C::iterator*);

  template <typename>
  constexpr static std::false_type test(...);

  constexpr static bool value =
      std::is_same<std::true_type, decltype(test<typename std::remove_reference<T>::type>(0))>::value;
};

template <typename T, typename = std::enable_if_t<std::is_base_of<molpro::linalg::array::DistrArray, T>::value>>
void precondition_default(const VecRef<T>& action, const std::vector<double>& shift, const T& diagonals) {
  auto diagonals_local_buffer = diagonals.local_buffer();
  for (size_t k = 0; k < action.size(); k++) {
    auto action_local_buffer = action[k].get().local_buffer();
    auto& distribution = action[k].get().distribution();
    auto range = distribution.range(molpro::mpi::rank_global());
    for (auto i = range.first; i < range.second; i++)
      (*action_local_buffer)[i - range.first] /= ((*diagonals_local_buffer)[i - range.first] - shift[k] + 1e-15);
  }
}

template <class T>
void precondition_default(const VecRef<T>& action, const std::vector<double>& shift, const T& diagonals,
                          typename T::iterator* = nullptr // SFINAE
) {
  for (size_t k = 0; k < action.size(); k++) {
    auto& a = action[k].get();
    std::transform(diagonals.begin(), diagonals.end(), a.begin(), a.begin(),
                   [shift, k](const auto& first, const auto& second) { return second / (first - shift[k] + 1e-15); });
  }
}

template <typename T, typename = std::enable_if_t<!std::is_base_of<molpro::linalg::array::DistrArray, T>::value>,
          class = void>
void precondition_default(const VecRef<T>& action, const std::vector<double>& shift, const T& diagonals,
                          typename std::enable_if<!has_iterator<T>::value, void*>::type = nullptr // SFINAE
) {
  throw std::logic_error("Unimplemented preconditioner");
}

/*!
 * @example ExampleProblem.h
 * Simple example of a problem-defining class
 * @example ExampleProblemDistrArray.h
 * Example of a problem-defining class with distributed data
 */
/*!
 * @brief Abstract class defining the problem-specific interface for the simplified solver
 * interface to IterativeSolver
 * @tparam R the type of container for solutions and residuals
 */
template <typename R, typename P = std::map<size_t, typename R::value_type>>
class Problem {
public:
  Problem() = default;
  virtual ~Problem() = default;
  using container_t = R;
  using p_container_t = P;
  using value_t = typename R::value_type;

  /*!
   * @brief Calculate the residual vector. Used by non-linear solvers (NonLinearEquations,
   * Optimize) only.
   * @param parameters The trial solution for which the residual is to be calculated
   * @param residual The residual vector
   * @return In the case where the residual is an exact differential, the corresponding
   * function value. Used by Optimize but not NonLinearEquations.
   */
  virtual value_t residual(const R& parameters, R& residual) const { return 0; }

  /*!
   * @brief Calculate the action of the kernel matrix on a set of parameters. Used by
   * linear solvers, but not by the non-linear solvers (NonLinearEquations, Optimize).
   * @param parameters The trial solutions for which the action is to be calculated
   * @param action The action vectors
   */
  virtual void action(const CVecRef<R>& parameters, const VecRef<R>& action) const { return; }

  /*!
   * @brief Optionally provide the diagonal elements of the underlying kernel. If
   * implemented and returning true, the provided diagonals will be used by
   * IterativeSolver for preconditioning (and therefore the precondition() function does
   * not need to be implemented), and, in the case of linear problems, for selection of
   * the P space. Otherwise, preconditioning will be done with precondition(), and any P
   * space has to be provided manually.
   * @param d On exit, contains the diagonal elements
   * @return Whether diagonals have been provided.
   */
  virtual bool diagonals(container_t& d) const { return false; }

  /*!
   * @brief Apply preconditioning to a residual vector in order to predict a step towards
   * the solution
   * @param residual On entry, assumed to be the residual. On exit, the negative of the
   * predicted step.
   * @param shift When called from LinearEigensystem, contains the corresponding current
   * eigenvalue estimates for each of the parameter vectors in the set. All other solvers
   * pass a vector of zeroes.
   */
  virtual void precondition(const VecRef<R>& residual, const std::vector<value_t>& shift) const { return; }

  /*!
   * @brief Apply preconditioning to a residual vector in order to predict a step towards
   * the solution
   * @param residual On entry, assumed to be the residual. On exit, the negative of the
   * predicted step.
   * @param shift When called from LinearEigensystem, contains the corresponding current
   * eigenvalue estimates for each of the parameter vectors in the set. All other solvers
   * pass a vector of zeroes.
   * @param diagonals The diagonal elements of the underlying kernel
   */
  virtual void precondition(const VecRef<R>& residual, const std::vector<value_t>& shift, const R& diagonals) const {
    precondition_default(residual, shift, diagonals);
  }

  /*!
   * @brief Return the inhomogeneous part of a linear equation system.
   * @param RHS On return, will contain the requested right-hand-side of equations if available.
   * @param instance Which RHS is required. If there are N sets of equations to solve, then 0,1,...N-1 number the instances.
   * The solver will call this function repeatedly with instance=0,1,2,... until it receives a negative response.
   * @return whether the requested RHS is available and has been provided
   */
  virtual bool RHS(R& RHS, unsigned int instance) const { return false;}

  /*!
   * @brief Calculate the kernel matrix in the P space
   * @param pparams Specification of the P space
   * @return
   */
  virtual std::vector<double> pp_action_matrix(const std::vector<P>& pparams) const {
    if (not pparams.empty())
      throw std::logic_error("P-space unavailable: unimplemented pp_action_matrix() in Problem class");
    return std::vector<double>(0);
  }

  /*!
   * @brief Calculate the action of the kernel matrix on a set of vectors in the P space
   * @param p_coefficients The projection of the vectors onto to the P space
   * @param pparams Specification of the P space
   * @param actions On exit, the computed action has been added to the original contents
   */
  virtual void p_action(const std::vector<std::vector<value_t>>& p_coefficients, const CVecRef<P>& pparams,
                        const VecRef<container_t>& actions) const {
    if (not pparams.empty())
      throw std::logic_error("P-space unavailable: unimplemented p_action() in Problem class");
  }

  /*!
   * @brief Provide values of R vectors for testing the problem class.
   * For use in a non-linear solver, the first vector (instance=0) should be a reference point, and the remainder
   * (instance>0) should be close to it, such that meaningful numerical differentation can be done to test the residual
   * function.
   * @param instance
   * @param parameters
   * @return true if a vector has been provided
   */
  virtual bool test_parameters(unsigned int instance, R& parameters) const { return false; }
};

/*!
 * @brief Base class defining the interface common to all iterative solvers
 *
 * As well as through the interface, some behaviour (profiling, and tuning of BLAS operations)
 * is influenced by the contents of
 * a global molpro::Options object, which defaults to molpro::Options("ITERATIVE-SOLVER", "").
 *
 * @tparam R container for "working-set" vectors. These are typically implemented in
 * memory, and are created by the client program. R vectors are never created inside
 * IterativeSolver.
 * @tparam Q container for other vectors. These are typically implemented on backing store
 * and/or distributed across processors.  IterativeSolver constructs a number of instances
 * of Q containers to store history.
 * @tparam P a class that specifies the definition of a single P-space vector, which is a
 * strictly sparse vector in the underlying space.
 */
template <class R, class Q, class P>
class IterativeSolver {
public:
  using value_type = typename R::value_type;                          ///< The underlying type of elements of vectors
  using scalar_type = typename array::ArrayHandler<R, Q>::value_type; ///< The type of scalar products of
                                                                      ///< vectors
  using value_type_abs = typename array::ArrayHandler<R, R>::value_type_abs;
  using VectorP = std::vector<value_type>; //!< type for vectors projected on to P space,
                                           //!< each element is a coefficient for the
                                           //!< corresponding P space parameter
  //! Function type for applying matrix to the P space vectors and accumulating result in
  //! a residual
  using fapply_on_p_type = std::function<void(const std::vector<VectorP>&, const CVecRef<P>&, const VecRef<R>&)>;

  virtual ~IterativeSolver() = default;
  IterativeSolver() = default;
  IterativeSolver(const IterativeSolver<R, Q, P>&) = delete;
  IterativeSolver<R, Q, P>& operator=(const IterativeSolver<R, Q, P>&) = delete;
  IterativeSolver(IterativeSolver<R, Q, P>&&) noexcept = default;
  IterativeSolver<R, Q, P>& operator=(IterativeSolver<R, Q, P>&&) noexcept = default;

  /*!
   * @example LinearEigensystemExample.cpp
   * Example for solving linear eigensystem
   * @example LinearEigensystemMultirootExample.cpp
   * Example for solving linear eigensystem, simultaneously tracking multiple roots
   * @example LinearEigensystemDistrArrayExample.cpp
   * Example for solving linear eigensystem, history vectors on disk, optional MPI
   * @example LinearEquationsExample.cpp
   * Example for solving inhomogeneous linear equations
   * @example NonLinearEquationsExample.cpp
   * Example for solving non-linear equations
   * @example OptimizeExample.cpp
   * Example for minimising a function
   * @example OptimizeDistrArrayExample.cpp
   * Example for minimising a function, history vectors on disk, optional MPI
   */
  /*!
   * @brief Simplified one-call solver
   * @param parameters A set of scratch vectors. On entry, these vectors should be filled
   * with starting guesses. Where possible, the number of vectors should be equal to the
   * number of solutions sought, but a smaller array is permitted.
   * @param actions A set of scratch vectors. It should have the same size as parameters.
   * @param problem A Problem object defining the problem to be solved
   * @param generate_initial_guess Whether to start with a guess based on diagonal
   * elements (true) or on the contents of parameters (false)
   * @return true if the solution was found
   */
  virtual bool solve(const VecRef<R>& parameters, const VecRef<R>& actions, const Problem<R>& problem,
                     bool generate_initial_guess = false) = 0;
  virtual bool solve(R& parameters, R& actions, const Problem<R>& problem, bool generate_initial_guess = false) = 0;
  virtual bool solve(std::vector<R>& parameters, std::vector<R>& actions, const Problem<R>& problem,
                     bool generate_initial_guess = false) = 0;

  /*!
   * \brief Take, typically, a current solution and residual, and add it to the solution
   * space. \param parameters On input, the current solution or expansion vector. On exit,
   * undefined. \param actions On input, the residual for parameters (non-linear), or
   * action of matrix on parameters (linear). On exit, a vector set that should be
   * preconditioned before returning to end_iteration(). \param value The value of the
   * objective function for parameters. Used only in Optimize classes. \return The size of
   * the new working set. In non-linear optimisation, the special value -1 can also be
   * returned, indicating that preconditioning should not be carried out on action.
   */
  virtual int add_vector(const VecRef<R>& parameters, const VecRef<R>& actions) = 0;

  // FIXME this should be removed in favour of VecRef interface
  virtual int add_vector(std::vector<R>& parameters, std::vector<R>& action) = 0;
  virtual int add_vector(R& parameters, R& action, value_type value = 0) = 0;

  /*!
   * \brief Add P-space vectors to the expansion set for linear methods.
   * \note the apply_p function is stored and used by the solver internally.
   * \param Pparams the vectors to add. Each Pvector specifies a sparse vector in the
   * underlying space. The size of the P space must be at least the number of roots to be sought.
   * \param pp_action_matrix Matrix projected onto the existing+new, new
   * P space. It should be provided as a 1-dimensional array, with the existing+new index
   * running fastest.
   * \param parameters Used as scratch working space
   * \param action  On exit, the  residual of the interpolated solution. The contribution from the new, and
   * any existing, P parameters is missing, and should be added in subsequently.
   * \param apply_p A function that evaluates the action of the matrix on vectors in the P space
   * \return The number of vectors contained in parameters, action, parametersP
   */
  virtual size_t add_p(const CVecRef<P>& pparams, const array::Span<value_type>& pp_action_matrix,
                       const VecRef<R>& parameters, const VecRef<R>& action, fapply_on_p_type apply_p) = 0;

  // FIXME Is this needed?
  virtual void clearP() = 0;

  //! Construct solution and residual for a given set of roots
  virtual void solution(const std::vector<int>& roots, const VecRef<R>& parameters, const VecRef<R>& residual) = 0;

  //! Constructs parameters of selected roots
  virtual void solution_params(const std::vector<int>& roots, const VecRef<R>& parameters) = 0;

  //! Behaviour depends on the solver
  virtual size_t end_iteration(const VecRef<R>& parameters, const VecRef<R>& residual) = 0;

  /*!
   * Callback invoked once the solver iterations are done but before the solve function returns
   * to the caller. Behaviour depends on the solver.
   */
  virtual void finalize() = 0;

  /*!
   * @brief signal whether end_iteration should be called
   */
  virtual bool end_iteration_needed() = 0;

  /*!
   * \brief Get the solver's suggestion of which degrees of freedom would be best
   * to add to the P-space.
   * \param solution Current solution
   * \param residual Current residual
   * \param max_number Suggest no more than this number
   * \param threshold Suggest only axes for which the current residual and update
   * indicate an energy improvement in the next iteration of this amount or more.
   * \return
   */
  virtual std::vector<size_t> suggest_p(const CVecRef<R>& solution, const CVecRef<R>& residual, size_t max_number,
                                        double threshold) = 0;

  virtual void solution(const std::vector<int>& roots, std::vector<R>& parameters, std::vector<R>& residual) = 0;
  virtual void solution(R& parameters, R& residual) = 0;
  virtual void solution_params(const std::vector<int>& roots, std::vector<R>& parameters) = 0;
  virtual void solution_params(R& parameters) = 0;
  virtual size_t end_iteration(std::vector<R>& parameters, std::vector<R>& action) = 0;
  virtual size_t end_iteration(R& parameters, R& action) = 0;

  /*!
   * @brief Working set of roots that are not yet converged
   */
  virtual const std::vector<int>& working_set() const = 0;
  //! The calculated eigenvalues for roots in the working set (eigenvalue problems) or
  //! zero (otherwise)
  virtual std::vector<scalar_type> working_set_eigenvalues() const {
    return std::vector<scalar_type>(working_set().size(), 0);
  }
  //! Total number of roots we are solving for, including the ones that are already
  //! converged
  virtual size_t n_roots() const = 0;
  virtual void set_n_roots(size_t nroots) = 0;
  virtual const std::vector<scalar_type>& errors() const = 0;
  virtual const Statistics& statistics() const = 0;
  //! Writes a report to cout output stream
  virtual void report(std::ostream& cout, bool endl = true) const = 0;
  //! Writes a report to std::cout
  virtual void report() const = 0;

  //! Sets the convergence threshold
  virtual void set_convergence_threshold(double thresh) = 0;
  //! Reports the convergence threshold
  virtual double convergence_threshold() const = 0;
  //! Sets the value convergence threshold
  virtual void set_convergence_threshold_value(double thresh) = 0;
  //! Reports the value convergence threshold
  virtual double convergence_threshold_value() const = 0;
  [[deprecated("Set the verbosity on the logger directly")]]
  virtual void set_verbosity(Verbosity v) = 0;
  virtual void set_verbosity(int v) = 0;
  [[deprecated("Query the logger directly")]]
  virtual Verbosity get_verbosity() const = 0;
  virtual void set_max_iter(int n) = 0;
  virtual int get_max_iter() const = 0;
  virtual void set_max_p(int n) = 0;
  virtual int get_max_p() const = 0;
  virtual void set_p_threshold(double thresh) = 0;
  virtual double get_p_threshold() const = 0;
  virtual const subspace::Dimensions& dimensions() const = 0;
  // FIXME Missing parameters: SVD threshold
  //! Set all spcecified options. This is no different than using setters, but can be used
  //! with forward declaration.
  virtual void set_options(const Options& options) = 0;
  //! Return all options. This is no different than using getters, but can be used with
  //! forward declaration.
  virtual std::shared_ptr<Options> get_options() const = 0;
  /*!
   * @brief Report the function value for the current optimum solution
   * @return
   */
  virtual scalar_type value() const = 0;
  /*!
   * @brief Report whether the class is a non-linear solver
   * @return
   */
  virtual bool nonlinear() const = 0;
  /*!
   * @brief Attach a profiler in order to collect performance data
   * @param profiler
   */
  virtual void set_profiler(molpro::profiler::Profiler& profiler) = 0;
  virtual const std::shared_ptr<molpro::profiler::Profiler>& profiler() const = 0;
  /*!
   * @brief Set the logger instance that shall be used
   * @param logger
   */
  virtual void set_logger(std::shared_ptr<Logger> logger) = 0;
  virtual Logger &logger() = 0;
  /*!
   * @brief Test a supplied problem class
   * @param problem
   * @param v0
   * @param v1
   * @param verbosity
   * @param threshold
   * @return false if errors were found, otherwise true
   */
  virtual bool test_problem(const Problem<R>& problem, R& v0, R& v1, int verbosity = 0,
                            double threshold = 1e-5) const = 0;
};

/*!
 * @brief Interface for a specific iterative solver, it can add special member functions
 * or variables.
 */
template <class R, class Q, class P>
class LinearEigensystem : public IterativeSolver<R, Q, P> {
public:
  using typename IterativeSolver<R, Q, P>::scalar_type;
  //! The calculated eigenvalues of the subspace matrix
  virtual std::vector<scalar_type> eigenvalues() const = 0;
  //! Sets hermiticity of kernel
  virtual void set_hermiticity(bool hermitian) = 0;
  //! Gets hermiticity of kernel, if true than it is hermitian, otherwise it is not
  virtual bool get_hermiticity() const = 0;
};

template <class R, class Q, class P>
class LinearEquations : public IterativeSolver<R, Q, P> {
public:
  using typename IterativeSolver<R, Q, P>::scalar_type;
  virtual void add_equations(const CVecRef<R>& rhs) = 0;
  virtual void add_equations(const std::vector<R>& rhs) = 0;
  virtual void add_equations(const R& rhs) = 0;
  virtual CVecRef<Q> rhs() const = 0;
  //! Sets hermiticity of kernel
  virtual void set_hermiticity(bool hermitian) = 0;
  //! Gets hermiticity of kernel, if true than it is hermitian, otherwise it is not
  virtual bool get_hermiticity() const = 0;
};

//! Optimises to a stationary point using methods such as L-BFGS
template <class R, class Q, class P>
class Optimize : public IterativeSolver<R, Q, P> {};

//! Solves non-linear system of equations using methods such as DIIS
template <class R, class Q, class P>
class NonLinearEquations : public IterativeSolver<R, Q, P> {};

} // namespace molpro::linalg::itsolv

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_ITERATIVESOLVER_H
