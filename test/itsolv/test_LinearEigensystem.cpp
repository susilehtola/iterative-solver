#include "test.h"

#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <molpro/iostream.h>
#include <numeric>
#include <regex>
#include <vector>

#include "vector_types.h"
#include <molpro/linalg/itsolv/CastOptions.h>
#include <molpro/linalg/itsolv/SolverFactory.h>
#include <molpro/linalg/itsolv/helper.h>

// Find lowest eigensolution of a matrix obtained from an external file
// Storage of vectors in-memory via class Rvector

using molpro::linalg::array::Span;
using molpro::linalg::itsolv::CastOptions;
using molpro::linalg::itsolv::CVecRef;
using molpro::linalg::itsolv::cwrap;
using molpro::linalg::itsolv::LinearEigensystem;
using molpro::linalg::itsolv::VecRef;
using molpro::linalg::itsolv::wrap;
#ifndef NOFORTRAN
extern "C" int test_lineareigensystemf(double *matrix, size_t n, size_t np, size_t nroot, int hermitian,
                                       double *eigenvalues);
#endif

struct LinearEigensystemF : ::testing::Test {
  using MatrixXdr = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using MatrixXdc = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  using vectorP = std::vector<scalar>;

  size_t n = 0;
  size_t verbosity = 0;
  MatrixXdc hmat;

  void load_matrix(int dimension, const std::string &type = "", double param = 1, double non_hermiticity = 0) {
    n = dimension;
    hmat.resize(n, n);
    hmat.fill(1);
    for (size_t i = 0; i < n; i++)
      hmat(i, i) = i * param;
    if (non_hermiticity != 0)
      for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < i; j++)
          hmat(i, j) *= 1 - non_hermiticity;
  }

  void load_matrix(const std::string &file, double degeneracy_split = 0) {
    std::ifstream f(std::string{"./"} + file + ".hamiltonian");
    f >> n;
    hmat.resize(n, n);
    for (size_t i = 0; i < n; i++)
      for (size_t j = 0; j < n; j++)
        f >> hmat(i, j);
    // split degeneracies
    for (size_t i = 0; i < n; i++)
      hmat(i, i) += degeneracy_split * i;
  }

  void action(const std::vector<Rvector> &psx, std::vector<Rvector> &outputs) { action(wrap(psx), wrap(outputs)); }

  void action(const CVecRef<Rvector> &psx, const VecRef<Rvector> &outputs) {
    for (size_t k = 0; k < psx.size(); k++) {
      auto x = Eigen::Map<const Eigen::VectorXd>(psx.at(k).get().data(), psx.at(k).get().size());
      auto r = Eigen::Map<Eigen::VectorXd>(outputs.at(k).get().data(), psx.at(k).get().size());
      r.fill(0);
      r += hmat * x;
    }
  }

  template <class scalar>
  scalar dot(const std::vector<scalar> &a, const std::vector<scalar> &b) {
    return std::inner_product(std::begin(a), std::end(a), std::begin(b), scalar(0));
  }

  void residual(const std::vector<Rvector> &psx, std::vector<Rvector> &actions, const std::vector<double> &evals) {
    action(psx, actions);
    for (size_t k = 0; k < evals.size(); k++) {
      auto x = Eigen::Map<const Eigen::VectorXd>(psx.at(k).data(), psx.at(k).size());
      auto r = Eigen::Map<Eigen::VectorXd>(actions.at(k).data(), psx.at(k).size());
      r -= evals.at(k) * x;
    }
  }

  void update(std::vector<Rvector> &psg, const std::vector<scalar> &shift = std::vector<scalar>()) {
    update(wrap(psg), shift);
  }

  void update(const VecRef<Rvector> &psg, const std::vector<scalar> &shift = std::vector<scalar>()) {
    for (size_t k = 0; k < psg.size() && k < shift.size(); k++) {
      auto &g = psg[k].get();
      for (size_t i = 0; i < n; i++) {
        g[i] *= -1. / (1e-12 - shift[k] + hmat(i, i));
      }
    }
  }

  std::tuple<std::map<double, std::vector<double>>, std::vector<double>> solve_full_problem(bool hermitian) {
    std::map<double, std::vector<double>> expected_eigensolutions;
    std::vector<double> expected_eigenvalues, eigenvector;
    std::vector<double> hmat_row(n * n);
    std::vector<double> metric(n * n);
    for (size_t i = 0; i < n; ++i)
      metric[i * (n + 1)] = 1;
    for (size_t i = 0, ij = 0; i < n; ++i)
      for (size_t j = 0; j < n; ++j, ++ij)
        hmat_row[ij] = hmat(i, j);
    molpro::linalg::itsolv::eigenproblem(eigenvector, expected_eigenvalues, hmat_row, metric, n, hermitian, 1.0e-14, 0);
    for (size_t i = 0; i < n; i++)
      for (size_t j = 0; j < n; j++)
        expected_eigensolutions[expected_eigenvalues[i]].push_back(
            eigenvector[n * i + j]); // won't work for degenerate eigenvalues!
    std::sort(expected_eigenvalues.begin(), expected_eigenvalues.end());
    return std::make_tuple(expected_eigensolutions, expected_eigenvalues);
  }

  // testing the test: that sort of eigenvalues is the same thing as std::map sorting of keys
  void check_eigenvectors_map(const std::map<double, std::vector<double>> &expected_eigensolutions,
                              const std::vector<double> &expected_eigenvalues) {
    std::vector<double> testing;
    for (const auto &ee : expected_eigensolutions)
      testing.push_back(ee.first);
    ASSERT_THAT(expected_eigenvalues, ::testing::Pointwise(::testing::DoubleEq(), testing));
  }

  auto create_containers(const size_t nW) {
    std::vector<Rvector> g;
    std::vector<Rvector> x;
    for (size_t i = 0; i < nW; i++) {
      x.emplace_back(n, 0);
      g.emplace_back(n);
    }
    return std::make_tuple(x, g);
  }

  bool repeated_guess = false;
  auto initial_guess(std::vector<Rvector> &x) {
    const auto n_roots = x.size();
    std::vector<size_t> guess;
    std::vector<double> diagonals;
    for (size_t i = 0; i < n; i++) {
      diagonals.push_back(hmat(i, i));
    }
    for (size_t root = 0; root < n_roots; root++) {
      guess.push_back(std::min_element(diagonals.begin(), diagonals.end()) - diagonals.begin()); // initial guess
      if (! repeated_guess) *std::min_element(diagonals.begin(), diagonals.end()) = 1e99;
      x[root][guess.back()] = 1; // initial guess
    }
  }

  void apply_p(const std::vector<vectorP> &pvectors, const CVecRef<Pvector> &pspace, const VecRef<Rvector> &action) {
    for (size_t i = 0; i < pvectors.size(); i++) {
      auto &actioni = (action[i]).get();
      for (size_t pi = 0; pi < pspace.size(); pi++) {
        const auto &p = pspace[pi].get();
        for (const auto &pel : p)
          for (size_t j = 0; j < n; j++)
            actioni[j] += hmat(j, pel.first) * pel.second * pvectors[i][pi];
      }
    }
  }

  auto initial_pspace(const size_t np) {
    std::vector<scalar> PP;
    std::vector<Pvector> pspace;
    if (np > 0) {
      std::vector<double> diagonals;
      for (size_t i = 0; i < n; i++)
        diagonals.push_back(hmat(i, i));
      for (size_t p = 0; p < np; p++) {
        std::map<size_t, scalar> pp;
        pp.insert({std::min_element(diagonals.begin(), diagonals.end()) - diagonals.begin(), 1});
        pspace.push_back(pp);
        *std::min_element(diagonals.begin(), diagonals.end()) = 1e99;
      }
      PP.reserve(np * np);
      for (const auto &i : pspace)
        for (const auto &j : pspace) {
          const size_t kI = i.begin()->first;
          const size_t kJ = j.begin()->first;
          PP.push_back(hmat(kI, kJ));
        }
    }
    return std::make_tuple(pspace, PP);
  }

  auto set_options(std::unique_ptr<LinearEigensystem<Rvector, Qvector, Pvector>> &solver, const int nroot, const int np,
                   bool hermitian) {
    auto options = CastOptions::LinearEigensystem(solver->get_options());
    options->n_roots = nroot;
    options->convergence_threshold = 1.0e-8;
    //    options->norm_thresh = 1.0e-14;
    //    options->svd_thresh = 1.0e-10;
    options->max_size_qspace = std::max(6 * nroot, std::min(int(n), std::min(1000, 6 * nroot)) - np);
    options->reset_D = 8;
    options->hermiticity = hermitian;
    options->verbosity = molpro::linalg::itsolv::Verbosity::Summary;
    solver->set_options(*options);
    options = CastOptions::LinearEigensystem(solver->get_options());
    molpro::cout << "convergence threshold = " << options->convergence_threshold.value()
                 << ", svd thresh = " << options->svd_thresh.value()
                 << ", norm thresh = " << options->norm_thresh.value()
                 << ", max size of Q = " << options->max_size_qspace.value()
                 << ", reset D = " << options->reset_D.value() << std::endl;
    //    logger->max_trace_level = molpro::linalg::itsolv::Logger::None;
    //    logger->max_warn_level = molpro::linalg::itsolv::Logger::Error;
    //    logger->data_dump = false;
    return options;
  }

  bool check_mat_hermiticity() {
    auto d = (hmat - hmat.transpose()).norm();
    return d < 1e-10;
  }

  // Create initial subspace. This is iteration 0.
  auto initialize_subspace(const size_t np, const size_t nroot, const size_t n_working_vectors_max,
                           std::unique_ptr<LinearEigensystem<Rvector, Qvector, Pvector>> &solver) {
    auto [x, g] = create_containers(nroot);
    if (np) {
      auto apply_p_wrapper = [this](const auto &pvectors, const auto &pspace, const auto &action) {
        return this->apply_p(pvectors, pspace, action);
      };
      auto [pspace, PP] = initial_pspace(np);
      solver->add_p(cwrap(pspace), Span<double>(PP.data(), PP.size()), wrap(x), molpro::linalg::itsolv::wrap(g),
                    apply_p_wrapper);
    } else {
      initial_guess(x);
      action(x, g);
      solver->add_vector(x, g);
    }
    auto n_working_vectors_guess = std::min(
      g.size(), std::max(nroot, n_working_vectors_max > 0 ? n_working_vectors_max : nroot));
    x.resize(n_working_vectors_guess);
    g.resize(n_working_vectors_guess);
    update(g, solver->working_set_eigenvalues());
    solver->end_iteration(x, g);
    return std::make_tuple(x, g);
  };

  void test_eigen(const std::string &title = "", const int n_working_vectors_max = 0, const bool simple = false) {
    bool hermitian = check_mat_hermiticity();
    auto [expected_eigensolutions, expected_eigenvalues] = solve_full_problem(hermitian);
    check_eigenvectors_map(expected_eigensolutions, expected_eigenvalues);
    if (verbosity > 0)
      std::cout << "expected eigenvalues " << expected_eigenvalues << std::endl;
    for (int nroot = 1; nroot <= int(n) && nroot <= 28; nroot += std::max(size_t{1}, n / 10)) {
      for (size_t np = 0; np <= n && np <= 100 && (hermitian or np == 0); np += std::max(nroot, int(n) / 5)) {
        molpro::cout << "\n\n*** " << title << ", " << nroot << " roots, problem dimension " << n
                     << ", pspace dimension " << np << ", n_working_vectors_max = " << n_working_vectors_max
                     << std::endl;
#ifndef NOFORTRAN
        ASSERT_NE(test_lineareigensystemf(hmat.data(), n, np, nroot, hermitian, expected_eigenvalues.data()), 0);
#endif
        auto
            //        std::shared_ptr<molpro::linalg::itsolv::ILinearEigensystem<Rvector,Qvector,Pvector>>
            solver = molpro::linalg::itsolv::create_LinearEigensystem<Rvector, Qvector, Pvector>();
        auto options = set_options(solver, nroot, np, hermitian);
        int nwork = nroot;
        auto [x, g] = initialize_subspace(np, nroot, n_working_vectors_max, solver);
        size_t n_iter = 2;
        //                if (simple) {
        //                    auto apply_r = [this](const CVecRef<Rvector> &param, const VecRef<Rvector> &action) ->
        //                    double {
        //                        this->action(param, action);
        //                        return 0.;
        //                    };
        //                    auto precondition = [&s = solver, this](const VecRef<Rvector> &residual,
        //                                                            const VecRef<Rvector> &params) {
        //                        update(residual, s->working_set_eigenvalues());
        //                    };
        //                    solver->solve_obsolete(wrap(x), wrap(g), apply_r, precondition);
        //                } else {
        for (auto iter = 1; iter < 100; iter++, ++n_iter) {
          action(x, g);
          nwork = solver->add_vector(x, g);
          if (verbosity > 0)
            std::cout << "solver.add_vector returns nwork=" << nwork << std::endl;
          if (nwork == 0)
            break;
          update(g, solver->working_set_eigenvalues());
          if (verbosity > 0)
            solver->report();
          nwork = solver->end_iteration(x, g);
          if (verbosity > 0)
            std::cout << "solver.end_iteration returns nwork=" << nwork << std::endl;
          if (nwork == 0)
            break;
        }
        //                }
        if (verbosity > 0)
          solver->report();
        std::cout << "Error={ ";
        for (const auto &e : solver->errors())
          std::cout << e << " ";
        std::cout << "} after " << solver->statistics().iterations << " iterations, "
                  << solver->statistics().r_creations << " R vectors" << std::endl;
        EXPECT_THAT(solver->errors(), ::testing::Pointwise(::testing::DoubleNear(2 * solver->convergence_threshold()),
                                                           std::vector<double>(nroot, double(0))));
        if (verbosity > 0) {
          std::cout << "expected eigenvalues "
                    << std::vector<double>(expected_eigenvalues.data(), expected_eigenvalues.data() + solver->n_roots())
                    << std::endl;
          std::cout << "obtained eigenvalues " << solver->eigenvalues() << std::endl;
        }
        EXPECT_THAT(solver->eigenvalues(),
                    ::testing::Pointwise(::testing::DoubleNear(2e-9),
                                         std::vector<double>(expected_eigenvalues.data(),
                                                             expected_eigenvalues.data() + solver->n_roots())));
        const auto nR_creations = solver->statistics().r_creations;
        if (verbosity > 0)
          std::cout << "R creations = " << nR_creations << std::endl;
        if (not simple) {
          EXPECT_LE(nR_creations, (nroot + 1) * n_iter);
        }
        std::vector<std::vector<double>> parameters, residuals;
        std::vector<int> roots;
        for (int root = 0; root < int(solver->n_roots()); root++) {
          parameters.emplace_back(n);
          residuals.emplace_back(n);
          roots.push_back(root);
        }
        solver->solution(roots, parameters, residuals);
        residual(parameters, residuals, solver->eigenvalues());
        for (const auto &r : residuals)
          EXPECT_LE(std::sqrt(dot(r, r)), options->convergence_threshold.value());
        size_t root = 0;
        for (const auto &ee : expected_eigensolutions) {
          EXPECT_NEAR(ee.first, solver->eigenvalues()[root], 1e-10);
          auto overlap_with_reference =
              std::inner_product(parameters.at(root).begin(), parameters.at(root).end(), ee.second.begin(), 0.,
                                 std::plus<double>(), std::multiplies<double>());
          EXPECT_NEAR(std::abs(overlap_with_reference), 1., options->convergence_threshold.value());
          root++;
          if (root >= solver->n_roots())
            break;
        }
      }
    }
  }
};

TEST_F(LinearEigensystemF, file_eigen) {
  for (const auto &file : std::vector<std::string>{"phenol", "bh", "hf"}) {
    load_matrix(file, file == "phenol" ? 0 : 1e-8);
    test_eigen(file);
  }
}

TEST_F(LinearEigensystemF, n_eigen) {
  size_t n = 100;
  //  for (auto param : std::vector<double>{.01, .1, 1, 10, 100}) {
  //  for (auto param : std::vector<double>{.01, .1, 1}) {
  for (auto param : std::vector<double>{1}) {
    load_matrix(n, "", param);
    test_eigen(std::to_string(n) + "/" + std::to_string(param));
  }
}

TEST_F(LinearEigensystemF, nonhermitian_eigen) {
  size_t n = 6;
  //  for (auto param : std::vector<double>{.01, .1, 1, 10, 100}) {
  //  for (auto param : std::vector<double>{.01, .1, 1}) {
  for (auto param : std::vector<double>{1, .1}) {
    for (auto non_hermiticity : std::vector<double>{0, 0.1, 0.2}) {
      load_matrix(n, "", param, non_hermiticity);
      //      std::cout << "hmat\n"<<hmat<<std::endl;
      test_eigen(std::to_string(n) + "/" + std::to_string(param) + " non-hermiticy=" + std::to_string(non_hermiticity));
    }
  }
}

TEST_F(LinearEigensystemF, small_eigen) {
  for (int n = 1; n < 5; n++) {
    double param = 1;
    load_matrix(n, "", param);
    test_eigen(std::to_string(n) + "/" + std::to_string(param));
    test_eigen(std::to_string(n) + "/" + std::to_string(param), 1);
    test_eigen(std::to_string(n) + "/" + std::to_string(param), 0, true);
  }
}

TEST_F(LinearEigensystemF, symmetry_eigen) {
  for (int n = 1; n < 6; n++) {
    double param = 1;
    load_matrix(n, "", param);
    for (auto i = 0; i < n; i++)
      for (auto j = 0; j < n; j++)
        if (((i % 3) == 0 and (j % 3) != 0) or ((j % 3) == 0 and (i % 3) != 0))
          hmat(j, i) = 0;
    if (false) {
      std::cout << "hmat " << std::endl;
      for (auto i = 0; i < n; i++) {
        std::cout << "i%3" << i % 3 << std::endl;
        for (auto j = 0; j < n; j++)
          std::cout << " " << hmat(j, i);
        std::cout << std::endl;
      }
    }
    test_eigen(std::to_string(n) + "/sym/" + std::to_string(param));
  }
}

TEST_F(LinearEigensystemF, solution) {
  size_t n = 10;
  auto solution_residual = std::vector<Rvector>(n);
  std::for_each(solution_residual.begin(), solution_residual.end(), [&n](auto &el) { el.assign(n, 0); });
  double param = 1;
  load_matrix(n, "", param);
  for (size_t nroot = 1; nroot < n; ++nroot) {
    for (size_t np = 0; np <= nroot; np += nroot) {
      for (size_t n_working_vectors_max = 0; n_working_vectors_max < 2; ++n_working_vectors_max) {
        auto solver = molpro::linalg::itsolv::create_LinearEigensystem<Rvector, Qvector, Pvector>();
        auto options = set_options(solver, nroot, np, check_mat_hermiticity());
        auto [x, g] = initialize_subspace(np, nroot, n_working_vectors_max, solver);
        auto roots = solver->working_set();
        solver->solution(roots, x, g);
        residual(x, solution_residual, solver->working_set_eigenvalues());
        for (size_t i = 0; i < roots.size(); ++i) {
          auto diff = 0.;
          for (size_t j = 0; j < n; ++j)
            diff += std::abs(g.at(i)[j] - solution_residual.at(i)[j]);
          diff = std::sqrt(diff / n);
          EXPECT_NEAR(diff, 0., 1.0e-6);
        }
      }
    }
  }
}

TEST_F(LinearEigensystemF, linearly_dependent_guess) {
  LinearEigensystemF::repeated_guess = true;
  LinearEigensystemF::verbosity = 99;
  for (const size_t n : std::vector<size_t>{2}) {
      load_matrix(n, "", (double)1);
      test_eigen(std::to_string(n));
  }
}
