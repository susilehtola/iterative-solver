#include "IterativeSolverC.h"
#include "molpro/Profiler.h"
#include <memory>
#include <stack>
#include <tuple>
#ifdef LINEARALGEBRA_ARRAY_GA
#include "ga-mpi.h"
#include "ga.h"
#endif
#include <molpro/mpi.h>

#ifdef LINEARALGEBRA_ARRAY_HDF5
#include <molpro/linalg/array/DistrArrayHDF5.h>
#else
#include <molpro/linalg/array/DistrArrayFile.h>
#endif
#include <molpro/linalg/itsolv/Logger.h>
#include <molpro/linalg/array/DistrArrayMPI3.h>
#include <molpro/linalg/array/DistrArraySpan.h>
#include <molpro/linalg/array/Span.h>
#include <molpro/linalg/array/util/Distribution.h>
#include <molpro/linalg/array/util/gather_all.h>
#include <molpro/linalg/itsolv/LinearEigensystemDavidson.h>
#include <molpro/linalg/itsolv/LinearEquationsDavidson.h>
#include <molpro/linalg/itsolv/SolverFactory.h>
#include <molpro/linalg/itsolv/wrap.h>
#include <molpro/linalg/itsolv/Logger.h>

using molpro::Profiler;
using molpro::linalg::array::Span;
using molpro::linalg::array::util::Distribution;
using molpro::linalg::array::util::gather_all;
using molpro::linalg::array::util::make_distribution_spread_remainder;
using molpro::linalg::itsolv::ArrayHandlers;
using molpro::linalg::itsolv::cwrap;
using molpro::linalg::itsolv::IterativeSolver;
using molpro::linalg::itsolv::LinearEigensystem;
using molpro::linalg::itsolv::LinearEigensystemDavidson;
using molpro::linalg::itsolv::LinearEquations;
using molpro::linalg::itsolv::LinearEquationsDavidson;
using molpro::linalg::itsolv::NonLinearEquations;
using molpro::linalg::itsolv::Optimize;
using molpro::linalg::itsolv::wrap;

// using Rvector = molpro::linalg::array::DistrArrayMPI3;
using Rvector = molpro::linalg::array::DistrArraySpan;
#ifdef LINEARALGEBRA_Q_HDF5
using Qvector = molpro::linalg::array::DistrArrayHDF5;
#else
using Qvector = molpro::linalg::array::DistrArrayFile;
#endif
using Pvector = std::map<size_t, double>;
// instantiate the factory
#include <molpro/linalg/itsolv/SolverFactory-implementation.h>
template class molpro::linalg::itsolv::SolverFactory<Rvector, Qvector, Pvector>;

using vectorP = std::vector<double>;
using molpro::linalg::itsolv::CVecRef;
using molpro::linalg::itsolv::VecRef;

// FIXME Only top solver is active at any one time. This should be documented somewhere.

namespace {
struct Instance {
  Instance(std::unique_ptr<IterativeSolver<Rvector, Qvector, Pvector>> solver, std::shared_ptr<Profiler> prof,
           size_t dimension, MPI_Comm comm)
      : solver(std::move(solver)), prof(std::move(prof)), dimension(dimension), comm(comm){};
  std::unique_ptr<IterativeSolver<Rvector, Qvector, Pvector>> solver;
  std::shared_ptr<Profiler> prof;
  apply_on_p_t apply_on_p_fort;
  size_t dimension;
  MPI_Comm comm;
  std::unique_ptr<Qvector> diagonals;
  bool has_values = false;
  bool has_eigenvalues = false;
};
std::stack<Instance> instances;
} // namespace

std::pair<size_t, size_t> DistrArrayDefaultRange() {
  auto& instance = instances.top();
  int mpi_rank, comm_size;
  MPI_Comm_rank(instance.comm, &mpi_rank);
  MPI_Comm_size(instance.comm, &comm_size);
  auto d = make_distribution_spread_remainder<size_t>(instance.dimension, comm_size);
  auto range = d.range(mpi_rank);
  return range;
}

std::vector<Rvector> CreateDistrArray(size_t nvec, double* data) {
  auto& instance = instances.top();
  MPI_Comm ccomm = instance.comm;
  int mpi_rank, comm_size;
  MPI_Comm_rank(instance.comm, &mpi_rank);
  MPI_Comm_size(instance.comm, &comm_size);
  auto distr = make_distribution_spread_remainder<size_t>(instance.dimension, comm_size);
  auto range = distr.range(mpi_rank);
  auto rn = range.second - range.first;
  std::vector<Rvector> c;
  c.reserve(nvec);
  for (size_t ivec = 0; ivec < nvec; ivec++) {
    c.emplace_back(std::make_unique<Distribution<Rvector::index_type>>(distr),
                   Span<typename Rvector::value_type>(&data[ivec * instance.dimension + range.first], rn), ccomm);
  }
  return c;
}

std::vector<Rvector> CreateDistrArray(size_t nvec, const double* data) {
  auto& instance = instances.top();
  MPI_Comm ccomm = instance.comm;
  int mpi_rank, comm_size;
  MPI_Comm_rank(instance.comm, &mpi_rank);
  MPI_Comm_size(instance.comm, &comm_size);
  auto distr = make_distribution_spread_remainder<size_t>(instance.dimension, comm_size);
  auto range = distr.range(mpi_rank);
  auto rn = range.second - range.first;
  std::vector<Rvector> c;
  c.reserve(nvec);
  for (size_t ivec = 0; ivec < nvec; ivec++) {
    c.emplace_back(
        std::make_unique<Distribution<Rvector::index_type>>(distr),
        Span<typename Rvector::value_type>(&const_cast<double*>(data)[ivec * instance.dimension + range.first], rn),
        ccomm);
  }
  return c;
}

void DistrArraySynchronize(size_t nvec, std::vector<Rvector>& c, double* data) {
  auto& instance = instances.top();
  MPI_Comm ccomm = instance.comm;
  for (size_t ivec = 0; ivec < nvec; ivec++) {
    gather_all(c[ivec].distribution(), ccomm, &data[ivec * instance.dimension]);
  }
}

std::pair<size_t, size_t> DistrArrayGetRange(Rvector& rvec) {
  auto& instance = instances.top();
  int mpi_rank;
  MPI_Comm_rank(instance.comm, &mpi_rank);
  auto range = rvec.distribution().range(mpi_rank);
  return range;
}

extern "C" void IterativeSolverRange(size_t* range_begin, size_t* range_end) {
  std::tie(*range_begin, *range_end) = DistrArrayDefaultRange();
}

void apply_on_p_c(const std::vector<vectorP>& pvectors, const CVecRef<Pvector>& pspace, const VecRef<Rvector>& action) {
  auto& instance = instances.top();
  std::vector<size_t> ranges;
  size_t update_size = pvectors.size();
  ranges.reserve(update_size * 2);
  for (size_t k = 0; k < update_size; ++k) {
    ranges.push_back(DistrArrayGetRange(action[k].get()).first);
    ranges.push_back(DistrArrayGetRange(action[k].get()).second);
  }
  std::vector<double> pvecs_to_send;
  for (size_t i = 0; i < update_size; i++) {
    for (auto j : pvectors[i]) {
      pvecs_to_send.push_back(j);
    }
  }
  instance.apply_on_p_fort(pvecs_to_send.data(), &(*action.front().get().local_buffer())[0], update_size,
                           ranges.data());
}

extern "C" void IterativeSolverLinearEigensystemInitialize(size_t nQ, size_t nroot, size_t* range_begin,
                                                           size_t* range_end, double thresh, double thresh_value,
                                                           int hermitian, int verbosity, const char* fname,
                                                           int64_t fcomm, const char* algorithm, const char* options) {
  std::shared_ptr<Profiler> profiler = nullptr;
  profiler = molpro::Profiler::single("MainProfiler");
  std::string pname(fname);
  MPI_Comm comm = MPI_Comm_f2c(fcomm);
  if (!pname.empty()) {
    profiler = Profiler::single(pname);
  }
  instances.emplace(Instance{molpro::linalg::itsolv::create_LinearEigensystem<Rvector, Qvector, Pvector>(algorithm, options),
                             profiler, nQ, comm});
  auto& instance = instances.top();
  instance.solver->set_n_roots(nroot);
  instance.solver->set_verbosity(verbosity);
  instance.has_eigenvalues = true;
  LinearEigensystemDavidson<Rvector, Qvector, Pvector>* solver =
      dynamic_cast<LinearEigensystemDavidson<Rvector, Qvector, Pvector>*>(instance.solver.get());
  if (solver) {
    solver->set_hermiticity(hermitian);
    solver->set_convergence_threshold(thresh);
    solver->set_convergence_threshold_value(thresh_value);
    //    solver_cast->propose_rspace_norm_thresh = 1.0e-14;
    //    solver_cast->set_max_size_qspace(10);
    //    solver_cast->set_reset_D(50);
    solver->logger().set_verbosity(
        verbosity > 3 ? molpro::linalg::itsolv::log::Verbosity::Trace
                      : (verbosity > 2 ? molpro::linalg::itsolv::log::Verbosity::Debug : molpro::linalg::itsolv::log::Verbosity::Info));
    solver->logger().set_min_severity(
        verbosity > 1 ? molpro::linalg::itsolv::log::Severity::Warning : molpro::linalg::itsolv::log::Severity::Error);
    solver->logger().enable_data_dumps(verbosity > 0);
  }
  std::tie(*range_begin, *range_end) = DistrArrayDefaultRange();
}

extern "C" void IterativeSolverLinearEquationsInitialize(size_t n, size_t nroot, size_t* range_begin, size_t* range_end,
                                                         const double* rhs, double aughes, double thresh,
                                                         double thresh_value, int hermitian, int verbosity,
                                                         const char* fname, int64_t fcomm, const char* algorithm,
                                                         const char* options) {
  std::shared_ptr<Profiler> profiler = nullptr;
  std::string pname(fname);
  MPI_Comm comm = MPI_Comm_f2c(fcomm);
  if (!pname.empty()) {
    profiler = Profiler::single(pname);
  }
  instances.emplace(
      Instance{molpro::linalg::itsolv::create_LinearEquations<Rvector, Qvector, Pvector>(algorithm, options), profiler,
               n, comm});
  auto& instance = instances.top();
  auto rr = CreateDistrArray(nroot, rhs);
  auto solver = dynamic_cast<LinearEquationsDavidson<Rvector, Qvector, Pvector>*>(instance.solver.get());
  if (!solver)
    throw std::runtime_error("IterativeSolverLinearEquationsInitialize: solver factory returned an unexpected type for algorithm \"" +
                             std::string(algorithm ? algorithm : "") + "\"");
  solver->set_augmented_hessian(aughes);
  solver->set_hermiticity(hermitian);
  solver->set_n_roots(nroot);
  solver->add_equations(rr);
  solver->set_convergence_threshold(thresh);
  solver->set_convergence_threshold_value(thresh_value);
  solver->logger().set_verbosity(
      verbosity > 3 ? molpro::linalg::itsolv::log::Verbosity::Trace
                    : (verbosity > 2 ? molpro::linalg::itsolv::log::Verbosity::Debug : molpro::linalg::itsolv::log::Verbosity::Info));
  solver->logger().set_min_severity(
      verbosity > 1 ? molpro::linalg::itsolv::log::Severity::Warning : molpro::linalg::itsolv::log::Severity::Error);
  solver->logger().enable_data_dumps(verbosity > 0);
  // instance.solver->m_verbosity = verbosity;
  instance.solver->set_verbosity(verbosity);
  std::tie(*range_begin, *range_end) = DistrArrayDefaultRange();
}

extern "C" void IterativeSolverNonLinearEquationsInitialize(size_t n, size_t* range_begin, size_t* range_end,
                                                            double thresh, int verbosity, const char* fname,
                                                            int64_t fcomm, const char* algorithm, const char* options) {
  std::shared_ptr<Profiler> profiler = nullptr;
  std::string pname(fname);
  MPI_Comm comm = MPI_Comm_f2c(fcomm);
  if (!pname.empty()) {
    profiler = Profiler::single(pname);
  }
  instances.emplace(
      Instance{molpro::linalg::itsolv::create_NonLinearEquations<Rvector, Qvector, Pvector>(algorithm, options),
               profiler, n, comm});
  auto& instance = instances.top();
  instance.solver->set_convergence_threshold(thresh);
  // instance.solver->m_verbosity = verbosity;
  instance.solver->set_verbosity(verbosity);
  std::tie(*range_begin, *range_end) = DistrArrayDefaultRange();
  molpro::linalg::itsolv::NonLinearEquationsDIIS<Rvector, Qvector, Pvector>* solver =
      dynamic_cast<molpro::linalg::itsolv::NonLinearEquationsDIIS<Rvector, Qvector, Pvector>*>(instance.solver.get());
  if (!solver)
    throw std::runtime_error(
        "IterativeSolverNonLinearEquationsInitialize: solver factory returned an unexpected type for algorithm \"" +
        std::string(algorithm ? algorithm : "") + "\"");
  solver->logger().set_verbosity(
     verbosity > 3 ? molpro::linalg::itsolv::log::Verbosity::Trace
                   : (verbosity > 2 ? molpro::linalg::itsolv::log::Verbosity::Debug : molpro::linalg::itsolv::log::Verbosity::Info));
  solver->logger().set_min_severity(
      verbosity > 1 ? molpro::linalg::itsolv::log::Severity::Warning : molpro::linalg::itsolv::log::Severity::Error);
  solver->logger().enable_data_dumps(verbosity > 0);
}

extern "C" void IterativeSolverOptimizeInitialize(size_t n, size_t* range_begin, size_t* range_end, double thresh,
                                                  double thresh_value, int verbosity, int minimize, const char* fname,
                                                  int64_t fcomm, const char* algorithm, const char* options) {
  std::shared_ptr<Profiler> profiler = nullptr;
  std::string pname(fname);
  MPI_Comm comm = MPI_Comm_f2c(fcomm);
  if (!pname.empty()) {
    profiler = Profiler::single(pname);
  }
  instances.emplace(Instance{molpro::linalg::itsolv::create_Optimize<Rvector, Qvector, Pvector>(algorithm, options),
                             profiler, n, comm});
  auto& instance = instances.top();
  instance.solver->set_n_roots(1);
  instance.solver->set_convergence_threshold(thresh);
  instance.solver->set_convergence_threshold_value(thresh_value);
  instance.solver->set_verbosity(verbosity);
  molpro::linalg::itsolv::OptimizeBFGS<Rvector, Qvector, Pvector>* solver =
      dynamic_cast<molpro::linalg::itsolv::OptimizeBFGS<Rvector, Qvector, Pvector>*>(instance.solver.get());
  if (!solver)
    throw std::runtime_error(
        "IterativeSolverOptimizeInitialize: BFGS-specific configuration requested but the factory returned a different algorithm for \"" +
        std::string(algorithm ? algorithm : "") + "\"");
  solver->logger().set_verbosity(
     verbosity > 3 ? molpro::linalg::itsolv::log::Verbosity::Trace
                   : (verbosity > 2 ? molpro::linalg::itsolv::log::Verbosity::Debug : molpro::linalg::itsolv::log::Verbosity::Info));
  solver->logger().set_min_severity(
      verbosity > 1 ? molpro::linalg::itsolv::log::Severity::Warning : molpro::linalg::itsolv::log::Severity::Error);
  solver->logger().enable_data_dumps(verbosity > 0);

  instance.has_values = true;
  std::tie(*range_begin, *range_end) = DistrArrayDefaultRange();
}

extern "C" void IterativeSolverFinalize() { instances.pop(); }

extern "C" void IterativeSolverFinalizeAll() { for (; !instances.empty(); instances.pop()); }

extern "C" void IterativeSolverAddEquation(double* rhs) {
  if (instances.empty())
    throw std::runtime_error("IterativeSolver not initialised properly");
  auto& instance = instances.top();
  if (instance.prof != nullptr)
    instance.prof->start("AddEquation");
  auto ccc = CreateDistrArray(1, rhs);
  auto* solver =
      dynamic_cast<molpro::linalg::itsolv::LinearEquationsDavidson<Rvector, Qvector, Pvector>*>(instance.solver.get());
  if (!solver)
    throw std::runtime_error("IterativeSolverAddEquation: current solver is not LinearEquationsDavidson");
  solver->add_equations(ccc[0]);
  if (instance.prof != nullptr) {
    instance.prof->stop();
  }
}

extern "C" size_t IterativeSolverAddValue(double value, double* parameters, double* action, int sync) {
  if (instances.empty())
    throw std::runtime_error("IterativeSolver not initialised properly");
  auto& instance = instances.top();
  if (instance.prof != nullptr)
    instance.prof->start("AddValue");
  auto ccc = CreateDistrArray(1, parameters);
  auto ggg = CreateDistrArray(1, action);
  if (instance.prof != nullptr)
    instance.prof->start("AddValue:Call");
  auto* solver = dynamic_cast<molpro::linalg::itsolv::Optimize<Rvector, Qvector, Pvector>*>(instance.solver.get());
  if (!solver)
    throw std::runtime_error("IterativeSolverAddValue: current solver is not an Optimize solver");
  size_t working_set_size = solver->add_vector(ccc[0], ggg[0], value) > 0 ? 1 : 0;
  if (instance.prof != nullptr) {
    instance.prof->stop();
    instance.prof->start("AddValue:Sync");
  }
  if (sync) {
    DistrArraySynchronize(1, ccc, parameters);
    DistrArraySynchronize(1, ggg, action);
  }
  if (instance.prof != nullptr) {
    instance.prof->stop();
    instance.prof->stop();
  }
  return working_set_size;
}

extern "C" size_t IterativeSolverAddVector(size_t buffer_size, double* parameters, double* action, int sync) {
  if (instances.empty())
    throw std::runtime_error("IterativeSolver not initialised properly");
  auto& instance = instances.top();
  if (instance.prof != nullptr)
    instance.prof->start("AddVector");
  auto cc = CreateDistrArray(buffer_size, parameters);
  auto gg = CreateDistrArray(buffer_size, action);
  if (instance.prof != nullptr)
    instance.prof->start("AddVector:Call");
  auto working_set_size = instance.solver->add_vector(cc, gg);
  if (instance.prof != nullptr) {
    instance.prof->stop();
    instance.prof->start("AddVector:Sync");
  }
  if (sync) {
    DistrArraySynchronize(instance.solver->working_set().size(), cc, parameters);
    DistrArraySynchronize(instance.solver->working_set().size(), gg, action);
  }
  if (instance.prof != nullptr) {
    instance.prof->stop();
    instance.prof->stop();
  }
  return working_set_size;
}

extern "C" void IterativeSolverSolution(int nroot, int* roots, double* parameters, double* action, int sync) {
  if (instances.empty())
    throw std::runtime_error("IterativeSolver not initialised properly");
  auto& instance = instances.top();
  if (instance.prof != nullptr)
    instance.prof->start("Solution");
  auto cc = CreateDistrArray(nroot, parameters);
  auto gg = CreateDistrArray(nroot, action);
  std::vector<int> croots;
  for (int i = 0; i < nroot; i++) {
    croots.push_back(*(roots + i));
  }
  if (instance.prof != nullptr)
    instance.prof->start("Solution:Call");
  instance.solver->solution(croots, cc, gg);
  if (instance.prof != nullptr) {
    instance.prof->stop();
    instance.prof->start("Solution:Sync");
  }
  if (sync) {
    DistrArraySynchronize(nroot, cc, parameters);
    DistrArraySynchronize(nroot, gg, action);
  }
  if (instance.prof != nullptr) {
    instance.prof->stop();
    instance.prof->stop();
  }
}

extern "C" size_t IterativeSolverEndIteration(size_t buffer_size, double* solution, double* residual, int sync) {
  if (instances.empty())
    throw std::runtime_error("IterativeSolver not initialised properly");
  auto& instance = instances.top();
  if (instance.prof != nullptr)
    instance.prof->start("EndIter");
  auto cc = CreateDistrArray(buffer_size, solution);
  auto gg = CreateDistrArray(buffer_size, residual);
  if (instance.prof != nullptr)
    instance.prof->start("EndIter:Call");
  auto result = instance.solver->end_iteration(cc, gg);
  if (instance.prof != nullptr) {
    instance.prof->stop();
    instance.prof->start("EndIter:Sync");
  }
  if (sync) {
    DistrArraySynchronize(instance.solver->working_set().size(), cc, solution);
    DistrArraySynchronize(instance.solver->working_set().size(), gg, residual);
  }
  if (instance.prof != nullptr) {
    instance.prof->stop();
    instance.prof->stop();
  }
  return result;
}

extern "C" int IterativeSolverEndIterationNeeded(){
  if (instances.empty())
    throw std::runtime_error("IterativeSolver not initialised properly");
  auto& instance = instances.top();
  return instance.solver->end_iteration_needed() ? 1:0;
}

extern "C" size_t IterativeSolverAddP(size_t buffer_size, size_t nP, const size_t* offsets, const size_t* indices,
                                      const double* coefficients, const double* pp, double* parameters, double* action,
                                      int sync, apply_on_p_t func) {
  if (instances.empty())
    throw std::runtime_error("IterativeSolver not initialised properly");
  auto& instance = instances.top();
  instance.apply_on_p_fort = func;
  if (instance.prof != nullptr)
    instance.prof->start("AddP");
  auto cc = CreateDistrArray(buffer_size, parameters);
  auto gg = CreateDistrArray(buffer_size, action);
  std::vector<Pvector> Pvectors;
  Pvectors.reserve(nP);
  for (size_t p = 0; p < nP; p++) {
    Pvector ppp;
    for (size_t k = offsets[p]; k < offsets[p + 1]; k++)
      ppp.insert(std::pair<size_t, Rvector::value_type>(indices[k], coefficients[k]));
    Pvectors.emplace_back(ppp);
  }
  using vectorP = std::vector<double>;
  using molpro::linalg::itsolv::CVecRef;
  using molpro::linalg::itsolv::VecRef;
  std::function<void(const std::vector<vectorP>&, const CVecRef<Pvector>&, const VecRef<Rvector>&)> apply_on_p =
      apply_on_p_c;
  if (instance.prof != nullptr)
    instance.prof->start("AddP:Call");
  size_t working_set_size = instance.solver->add_p(
      cwrap(Pvectors),
      Span<Rvector::value_type>(&const_cast<double*>(pp)[0], (instance.solver->dimensions().oP + nP) * nP), wrap(cc),
      wrap(gg), apply_on_p);
  if (instance.prof != nullptr) {
    instance.prof->stop();
    instance.prof->start("AddP:Sync");
  }
  if (sync) {
    DistrArraySynchronize(working_set_size, cc, parameters);
    DistrArraySynchronize(working_set_size, gg, action);
  }
  if (instance.prof != nullptr) {
    instance.prof->stop();
    instance.prof->stop();
  }
  return working_set_size;
}

namespace {
void require_instance() {
  if (instances.empty())
    throw std::runtime_error("IterativeSolver not initialised properly");
}
} // namespace

extern "C" void IterativeSolverErrors(double* errors) {
  require_instance();
  auto& instance = instances.top();
  size_t k = 0;
  for (const auto& e : instance.solver.get()->errors())
    errors[k++] = e;
  return;
}

extern "C" void IterativeSolverEigenvalues(double* eigenvalues) {
  require_instance();
  auto& instance = instances.top();
  size_t k = 0;
  LinearEigensystem<Rvector, Qvector, Pvector>* solver_cast =
      dynamic_cast<LinearEigensystem<Rvector, Qvector, Pvector>*>(instance.solver.get());
  if (solver_cast) {
    for (const auto& e : solver_cast->eigenvalues())
      eigenvalues[k++] = e;
  }
}

extern "C" void IterativeSolverWorkingSetEigenvalues(double* eigenvalues) {
  require_instance();
  auto& instance = instances.top();
  size_t k = 0;
  LinearEigensystemDavidson<Rvector, Qvector, Pvector>* solver_cast =
      dynamic_cast<LinearEigensystemDavidson<Rvector, Qvector, Pvector>*>(instance.solver.get());
  if (solver_cast) {
    for (const auto& e : solver_cast->working_set_eigenvalues())
      eigenvalues[k++] = e;
  }
}

extern "C" size_t IterativeSolverSuggestP(const double* solution, const double* residual, size_t maximumNumber,
                                          double threshold, size_t* indices) {
  require_instance();
  auto& instance = instances.top();
  if (instance.prof != nullptr)
    instance.prof->start("SuggestP");
  auto cc = CreateDistrArray(instance.solver->n_roots(), solution);
  auto gg = CreateDistrArray(instance.solver->n_roots(), residual);
  auto result = instance.solver->suggest_p(cwrap(cc), molpro::linalg::itsolv::cwrap(gg), maximumNumber, threshold);
  for (size_t i = 0; i < result.size(); i++) {
    indices[i] = result[i];
  }
  if (instance.prof != nullptr)
    instance.prof->stop();
  return result.size();
}

extern "C" void IterativeSolverPrintStatistics() {
  require_instance();
  molpro::cout << instances.top().solver->statistics() << std::endl;
}

extern "C" int IterativeSolverConverged() {
  require_instance();
  return instances.top().solver->working_set().empty() ? 1 : 0;
}

int IterativeSolverNonLinear() {
  require_instance();
  return instances.top().solver->nonlinear() ? 1 : 0;
}
int IterativeSolverHasValues() {
  require_instance();
  return instances.top().has_values ? 1 : 0;
}
int IterativeSolverHasEigenvalues() {
  require_instance();
  return instances.top().has_eigenvalues ? 1 : 0;
}

void IterativeSolverSetDiagonals(const double* diagonals) {
  require_instance();
  instances.top().diagonals.reset(new Qvector(CreateDistrArray(1, diagonals).front()));
}
void IterativeSolverDiagonals(double* diagonals) {
  require_instance();
  CreateDistrArray(1, diagonals).front().copy(*instances.top().diagonals);
}
double IterativeSolverValue() {
  require_instance();
  return instances.top().solver->value();
}
int IterativeSolverVerbosity() {
  require_instance();
  auto verbosity = instances.top().solver->logger().verbosity();
  auto min_severity = instances.top().solver->logger().min_severity();
  switch (verbosity) {
    using namespace molpro::linalg::itsolv;
    case log::Verbosity::None:
      return min_severity == log::Severity::Error ? 0 : 1;
    case log::Verbosity::Info:
      return 2;
    case log::Verbosity::Debug:
      return 3;
    case log::Verbosity::Trace:
      return 4;
  }

  return -1;
}
int IterativeSolverMaxIter() {
  require_instance();
  return instances.top().solver->get_max_iter();
}
void IterativeSolverSetMaxIter(int max_iter) {
  require_instance();
  instances.top().solver->set_max_iter(max_iter);
}
/*!
 * @brief C binding of mpi::comm_global(), suitable for calling from Fortran
 */
extern "C" int64_t IterativeSolver_mpicomm_global() { return (int64_t)MPI_Comm_c2f(molpro::mpi::comm_global()); }

/*!
 * @brief C binding of mpi::comm_self(), suitable for calling from Fortran
 */
extern "C" int64_t IterativeSolver_mpicomm_self() { return (int64_t)MPI_Comm_c2f(molpro::mpi::comm_self()); }

/*!
 * @brief C binding of mpi::size_global(), suitable for calling from Fortran
 */
extern "C" int64_t IterativeSolver_mpi_size_global() { return molpro::mpi::size_global(); }

/*!
 * @brief C binding of mpi::rank_global(), suitable for calling from Fortran
 */
extern "C" int64_t IterativeSolver_mpi_rank_global() { return molpro::mpi::rank_global(); }

/*!
 * @brief C binding of mpi::init(), suitable for calling from Fortran
 */
extern "C" int IterativeSolver_mpi_init() { return molpro::mpi::init(); }

/*!
 * @brief C binding of mpi::finalize(), suitable for calling from Fortran
 */
extern "C" int IterativeSolver_mpi_finalize() { return molpro::mpi::finalize(); }
