#include "ipca.h"
#include "quiver.h"
#include <cmath>
#include <stdexcept>
#include <string>

namespace
{
void Require(bool condition, const std::string &message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void TestFeasibility()
{
    const FeasibilityResult balanced =
        AnalyzeGenericMLEFeasibility(2, std::vector<int>(1, 2));
    Require(balanced.generic_mle_feasible,
            "balanced one-group dimensions should be generically feasible");
    Require(balanced.canonical_summands.size() == 1,
            "balanced case should have one distinct canonical summand");
    Require(balanced.dimension_vector == std::vector<int>{2, 2},
            "feasibility result should retain the analyzed dimensions");
    Require(balanced.multiplicities[0] == 2,
            "balanced case should preserve canonical multiplicity");

    const FeasibilityResult multiple_groups =
        AnalyzeGenericMLEFeasibility(2, std::vector<int>{1, 1});
    Require(multiple_groups.generic_mle_feasible,
            "two complementary feature groups should be generically feasible");

    const FeasibilityResult unbalanced =
        AnalyzeGenericMLEFeasibility(2, std::vector<int>(1, 1));
    Require(!unbalanced.generic_mle_feasible,
            "unbalanced one-group dimensions should not be generically feasible");

    bool rejected = false;
    try
    {
        AnalyzeGenericMLEFeasibility(0, std::vector<int>(1, 1));
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    Require(rejected, "zero sample count should be rejected");
}

void TestCanonicalSolverMemorySafety()
{
    sDecom independent(2);
    independent.n_smds = 2;
    independent.smnds = std::vector<std::vector<int> >{
        std::vector<int>{1, 0}, std::vector<int>{0, 1}};
    int right_hand_side[] = {3, 4};
    Require(independent.Solve(right_hand_side) == 1,
            "independent canonical summands should have a unique integer solution");
    Require(independent.mltp[0] == 3 && independent.mltp[1] == 4,
            "canonical solver should store coefficients in multiplicities");

    sDecom dependent(2);
    dependent.n_smds = 2;
    dependent.smnds = std::vector<std::vector<int> >{
        std::vector<int>{1, 1}, std::vector<int>{2, 2}};
    int dependent_rhs[] = {3, 3};
    Require(dependent.Solve(dependent_rhs) == -1,
            "linearly dependent canonical summands should be reported");

    sDecom inconsistent(2);
    inconsistent.n_smds = 1;
    inconsistent.smnds = std::vector<std::vector<int> >{
        std::vector<int>{1, 0}};
    int inconsistent_rhs[] = {1, 1};
    Require(inconsistent.Solve(inconsistent_rhs) == 0,
            "inconsistent canonical systems should be reported");

    sDecom non_integer(2);
    non_integer.n_smds = 1;
    non_integer.smnds = std::vector<std::vector<int> >{
        std::vector<int>{2, 0}};
    int non_integer_rhs[] = {1, 0};
    Require(non_integer.Solve(non_integer_rhs) == 2,
            "non-integer canonical coefficients should be reported");
}

void TestEstimator()
{
    std::vector<std::vector<double>> identity_data;
    identity_data.push_back(std::vector<double>{1.0, 0.0});
    identity_data.push_back(std::vector<double>{0.0, 1.0});
    std::vector<Matrix> groups;
    groups.push_back(Matrix(identity_data));

    bool converged = false;
    int iterations = -1;
    const std::vector<Matrix> estimate = FlipFlop(groups, converged, iterations, 10);
    Require(converged, "identity data should converge");
    Require(iterations == 2, "iteration count should be one-based");
    Require(estimate.size() == 2, "one group should return two covariance matrices");
    Require(std::abs(estimate[0].at(0, 0) - 0.5) < 1e-12 &&
                std::abs(estimate[0].at(1, 1) - 0.5) < 1e-12,
            "sample covariance estimate should preserve the existing update scale");

    FlipFlop(groups, converged, iterations, 1);
    Require(!converged && iterations == 1,
            "non-converged runs should report the number of completed iterations");

    bool rejected = false;
    try
    {
        FlipFlop(std::vector<Matrix>(), converged, iterations, 10);
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    Require(rejected, "empty feature-group input should be rejected");
}

void TestPositiveDefiniteSolve()
{
    Matrix coefficient(2, 2);
    coefficient.set(0, 0, 4.0);
    coefficient.set(0, 1, 1.0);
    coefficient.set(1, 0, 1.0);
    coefficient.set(1, 1, 3.0);
    Matrix rhs(2, 2);
    rhs.set(0, 0, 1.0);
    rhs.set(1, 0, 2.0);
    rhs.set(0, 1, 3.0);
    rhs.set(1, 1, 4.0);

    const PositiveDefiniteDiagnostics diagnostics =
        coefficient.positive_definite_diagnostics();
    Require(diagnostics.minimum_eigenvalue > 0 &&
                diagnostics.maximum_eigenvalue > diagnostics.minimum_eigenvalue,
            "SPD diagnostics should report ordered positive eigenvalues");
    Require(diagnostics.condition_number > 1 &&
                diagnostics.relative_minimum_eigenvalue > 0,
            "SPD diagnostics should report condition and relative eigenvalue");
    Require(std::abs(diagnostics.condition_number -
                     (7.0 + std::sqrt(5.0)) / (7.0 - std::sqrt(5.0))) < 1e-12,
            "condition number should match the SPD eigenvalue ratio");

    const Matrix solution = coefficient.solve_positive_definite(rhs);
    const Matrix reconstructed = coefficient.multiply(solution);
    for (int i = 0; i < rhs.getRows(); ++i)
        for (int j = 0; j < rhs.getCols(); ++j)
            Require(std::abs(reconstructed.at(i, j) - rhs.at(i, j)) < 1e-12,
                    "Cholesky solve should satisfy A*X=B");

    Matrix scaled(2, 2);
    scaled.set(0, 0, 4e-20);
    scaled.set(0, 1, 1e-20);
    scaled.set(1, 0, 1e-20);
    scaled.set(1, 1, 3e-20);
    const PositiveDefiniteDiagnostics scaled_diagnostics =
        scaled.positive_definite_diagnostics();
    Require(std::abs(
                diagnostics.relative_minimum_eigenvalue -
                scaled_diagnostics.relative_minimum_eigenvalue) < 1e-12,
            "relative eigenvalue diagnostics should be scale invariant");
    const Matrix scaled_solution = scaled.solve_positive_definite(rhs);
    const Matrix scaled_reconstructed = scaled.multiply(scaled_solution);
    for (int i = 0; i < rhs.getRows(); ++i)
        for (int j = 0; j < rhs.getCols(); ++j)
            Require(std::abs(scaled_reconstructed.at(i, j) - rhs.at(i, j)) < 1e-12,
                    "SPD solve should not use an absolute pivot threshold");

    Matrix ill_conditioned(2, 2);
    ill_conditioned.set(0, 0, 1.0);
    ill_conditioned.set(1, 1, 1e-14);
    bool rejected = false;
    try
    {
        ill_conditioned.solve_positive_definite(rhs);
    }
    catch (const std::runtime_error &)
    {
        rejected = true;
    }
    Require(rejected,
            "SPD solve should reject matrices below the relative eigenvalue threshold");

    Matrix indefinite(2, 2);
    indefinite.set(0, 0, 1.0);
    indefinite.set(1, 1, -1.0);
    rejected = false;
    try
    {
        indefinite.solve_positive_definite(rhs);
    }
    catch (const std::runtime_error &)
    {
        rejected = true;
    }
    Require(rejected, "SPD solve should reject indefinite matrices");

    const PositiveDefiniteDiagnostics singular_diagnostics =
        Matrix(2, 2).positive_definite_diagnostics();
    Require(singular_diagnostics.minimum_eigenvalue == 0 &&
                std::isinf(singular_diagnostics.condition_number),
            "singular covariance diagnostics should report zero and infinite condition");
}

void TestNumericalRank()
{
    Matrix full_rank(3, 2);
    full_rank.set(0, 0, 1);
    full_rank.set(1, 1, 1);
    Require(full_rank.numerical_rank() == 2,
            "numerical rank should detect independent columns");

    Matrix dependent(3, 2);
    dependent.set(0, 0, 1);
    dependent.set(1, 0, 2);
    dependent.set(2, 0, 3);
    dependent.set(0, 1, 2);
    dependent.set(1, 1, 4);
    dependent.set(2, 1, 6);
    Require(dependent.numerical_rank() == 1,
            "numerical rank should detect dependent columns");
    Require(dependent.numerical_rank(1e-12) == 1,
            "rank tolerance should be configurable");
    Require(Matrix(2, 3).numerical_rank() == 0,
            "an empty matrix should have rank zero");
}

void TestIterationDiagnostics()
{
    std::vector<std::vector<double>> identity_data;
    identity_data.push_back(std::vector<double>{1.0, 0.0});
    identity_data.push_back(std::vector<double>{0.0, 1.0});
    std::vector<Matrix> groups;
    groups.push_back(Matrix(identity_data));

    std::vector<FlipFlopIterationDiagnostics> trace;
    const FlipFlopObserver observer =
        [&trace](const FlipFlopIterationDiagnostics &diagnostics) {
            trace.push_back(diagnostics);
        };
    bool converged = false;
    int iterations = -1;
    FlipFlopWithDiagnostics(groups, converged, iterations, 10, observer);
    Require(converged && iterations == 2,
            "diagnostic flip-flop should preserve estimator convergence");
    Require(trace.size() == 4,
            "trace should include sample and group covariance each iteration");
    Require(trace[0].iteration == 1 && trace[0].covariance_index == 0 &&
                trace[1].iteration == 1 && trace[1].covariance_index == 1,
            "first-iteration trace should record Sigma before Delta");

    std::vector<std::vector<double>> deficient_data;
    deficient_data.push_back(std::vector<double>{1.0});
    deficient_data.push_back(std::vector<double>{2.0});
    deficient_data.push_back(std::vector<double>{3.0});
    std::vector<Matrix> deficient_groups;
    deficient_groups.push_back(Matrix(deficient_data));
    trace.clear();
    bool rejected = false;
    try
    {
        FlipFlopWithDiagnostics(
            deficient_groups, converged, iterations, 10, observer);
    }
    catch (const std::runtime_error &)
    {
        rejected = true;
    }
    Require(rejected,
            "unregularized flip-flop should reject singular first sample covariance");
    Require(trace.size() == 1 && trace[0].iteration == 1 &&
                trace[0].covariance_index == 0,
            "trace should preserve singular first Sigma before reporting failure");
    Require(trace[0].spectrum.relative_minimum_eigenvalue <= 1e-12,
            "trace should identify a singular first sample covariance");
}

void TestRegularizedEstimator()
{
    std::vector<std::vector<double>> rank_deficient_data;
    rank_deficient_data.push_back(std::vector<double>{1.0, 2.0});
    rank_deficient_data.push_back(std::vector<double>{2.0, 4.0});
    std::vector<Matrix> groups;
    groups.push_back(Matrix(rank_deficient_data));

    bool converged = false;
    int iterations = -1;
    const std::vector<Matrix> estimate = FlipFlopRegularized(
        groups, converged, iterations, 1000, 0.001, 0.5);
    Require(converged,
            "regularized flip-flop should converge on rank-deficient data");
    Require(estimate.size() == 2,
            "regularized one-group fit should return both covariance matrices");
    for (std::size_t matrix = 0; matrix < estimate.size(); ++matrix)
    {
        Require(estimate[matrix].getRows() == estimate[matrix].getCols(),
                "regularized covariance matrices should be square");
        for (int i = 0; i < estimate[matrix].getRows(); ++i)
        {
            Require(estimate[matrix].at(i, i) > 0,
                    "regularized covariance diagonals should be positive");
            for (int j = 0; j < estimate[matrix].getCols(); ++j)
                Require(std::isfinite(estimate[matrix].at(i, j)),
                        "regularized covariance values should be finite");
        }
        const PositiveDefiniteDiagnostics diagnostics =
            estimate[matrix].positive_definite_diagnostics();
        Require(diagnostics.relative_minimum_eigenvalue > 0,
                "regularized covariance should pass the relative SPD check");
    }

    bool rejected = false;
    try
    {
        FlipFlopRegularized(groups, converged, iterations, 10, 1.0, 0.5);
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    Require(rejected, "shrinkage equal to one should be rejected");
}
}

int main()
{
    TestFeasibility();
    TestCanonicalSolverMemorySafety();
    TestEstimator();
    TestPositiveDefiniteSolve();
    TestNumericalRank();
    TestIterationDiagnostics();
    TestRegularizedEstimator();
    return 0;
}
