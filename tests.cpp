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

void TestObservedDataLikelihood()
{
    Matrix sigma(2, 2);
    sigma.set(0, 0, 2.0);
    sigma.set(0, 1, 0.5);
    sigma.set(1, 0, 0.5);
    sigma.set(1, 1, 1.0);
    Matrix delta(2, 2);
    delta.set(0, 0, 1.0);
    delta.set(0, 1, 0.2);
    delta.set(1, 0, 0.2);
    delta.set(1, 1, 3.0);

    Matrix data(2, 2);
    data.set(0, 0, 1.0);
    data.set(0, 1, -1.0);
    data.set(1, 0, 2.0);
    data.set(1, 1, std::numeric_limits<double>::quiet_NaN());
    Matrix mask(2, 2);
    mask.set(0, 0, 1);
    mask.set(0, 1, 1);
    mask.set(1, 0, 1);

    const double partial =
        ObservedDataNegativeLogLikelihood(
            std::vector<Matrix>{data}, std::vector<Matrix>{mask},
            sigma, std::vector<Matrix>{delta});
    Require(std::abs(partial - 6.04741333965131) < 1e-12,
            "partial observed likelihood should match the explicit 3-value Gaussian calculation");

    Matrix empty_mask(2, 2);
    Require(ObservedDataNegativeLogLikelihood(
                std::vector<Matrix>{data},
                std::vector<Matrix>{empty_mask},
                sigma, std::vector<Matrix>{delta}) == 0,
            "a group with no observed values should contribute zero likelihood");

    Matrix invalid_mask(2, 2);
    invalid_mask.set(0, 0, 0.5);
    bool rejected = false;
    try
    {
        ObservedDataNegativeLogLikelihood(
            std::vector<Matrix>{data},
            std::vector<Matrix>{invalid_mask},
            sigma, std::vector<Matrix>{delta});
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    Require(rejected, "non-binary observation masks should be rejected");

    Matrix many_values(1, 513);
    Matrix many_mask(1, 513);
    Matrix one_sample_covariance(1, 1);
    one_sample_covariance.set(0, 0, 1);
    Matrix large_group_covariance(513, 513);
    for (int feature = 0; feature < 513; ++feature)
    {
        many_values.set(0, feature, 0);
        many_mask.set(0, feature, 1);
        large_group_covariance.set(feature, feature, 1);
    }
    rejected = false;
    try
    {
        ObservedDataNegativeLogLikelihood(
            std::vector<Matrix>{many_values},
            std::vector<Matrix>{many_mask},
            one_sample_covariance,
            std::vector<Matrix>{large_group_covariance});
    }
    catch (const std::length_error &)
    {
        rejected = true;
    }
    Require(rejected,
            "dense observed likelihood must reject inputs beyond its documented size limit");
}

void TestStructuredObservedDataLikelihood()
{
    Matrix sigma(3, 3);
    sigma.set(0, 0, 2.0);
    sigma.set(0, 1, 0.3);
    sigma.set(0, 2, 0.2);
    sigma.set(1, 0, 0.3);
    sigma.set(1, 1, 1.5);
    sigma.set(1, 2, 0.1);
    sigma.set(2, 0, 0.2);
    sigma.set(2, 1, 0.1);
    sigma.set(2, 2, 1.0);
    Matrix delta(2, 2);
    delta.set(0, 0, 1.2);
    delta.set(0, 1, 0.25);
    delta.set(1, 0, 0.25);
    delta.set(1, 1, 0.9);
    Matrix data(3, 2);
    data.set(0, 0, 1.0);
    data.set(0, 1, -0.5);
    data.set(1, 0, 0.2);
    data.set(1, 1, 1.5);
    data.set(2, 0, -0.8);
    data.set(2, 1, std::numeric_limits<double>::quiet_NaN());
    Matrix mask(3, 2);
    mask.set(0, 0, 1);
    mask.set(0, 1, 1);
    mask.set(1, 0, 1);
    mask.set(1, 1, 1);
    mask.set(2, 0, 1);

    const std::vector<Matrix> groups{data};
    const std::vector<Matrix> masks{mask};
    const std::vector<Matrix> group_covariances{delta};
    const double dense = ObservedDataNegativeLogLikelihood(
        groups, masks, sigma, group_covariances);
    const double structured = StructuredObservedDataNegativeLogLikelihood(
        groups, masks, sigma, group_covariances);
    Require(std::abs(dense - structured) < 1e-10,
            "structured conditional likelihood should match dense observed likelihood");

    const std::vector<std::vector<double>> means{
        std::vector<double>{0.4, -0.2}};
    const double dense_with_mean = ObservedDataNegativeLogLikelihood(
        groups, masks, sigma, group_covariances, means);
    const double structured_with_mean =
        StructuredObservedDataNegativeLogLikelihood(
            groups, masks, sigma, group_covariances, means);
    Require(std::abs(dense_with_mean - structured_with_mean) < 1e-10,
            "structured likelihood should match the dense reference with a feature mean");

    Matrix all_rows_mask(3, 2);
    all_rows_mask.set(0, 0, 1);
    all_rows_mask.set(1, 0, 1);
    all_rows_mask.set(2, 0, 1);
    const std::vector<Matrix> one_column_mask{all_rows_mask};
    const double dense_one_column = ObservedDataNegativeLogLikelihood(
        groups, one_column_mask, sigma, group_covariances);
    const double structured_one_column =
        StructuredObservedDataNegativeLogLikelihood(
            groups, one_column_mask, sigma, group_covariances);
    Require(std::abs(dense_one_column - structured_one_column) < 1e-10,
            "a repeated partial-column mask should use the matrix-normal likelihood");

    Matrix missing_row_mask = mask;
    missing_row_mask.set(1, 0, 0);
    missing_row_mask.set(1, 1, 0);
    const std::vector<Matrix> mask_with_missing_row{missing_row_mask};
    const double dense_missing_row = ObservedDataNegativeLogLikelihood(
        groups, mask_with_missing_row, sigma, group_covariances);
    const double structured_missing_row =
        StructuredObservedDataNegativeLogLikelihood(
            groups, mask_with_missing_row, sigma, group_covariances);
    Require(std::abs(dense_missing_row - structured_missing_row) < 1e-10,
            "entirely missing rows should be ignored without changing the likelihood");

    Matrix unsupported_mask(3, 2);
    unsupported_mask.set(0, 0, 1);
    unsupported_mask.set(0, 1, 1);
    unsupported_mask.set(1, 0, 1);
    unsupported_mask.set(2, 1, 1);
    bool rejected = false;
    try
    {
        StructuredObservedDataNegativeLogLikelihood(
            groups, std::vector<Matrix>{unsupported_mask},
            sigma, group_covariances);
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    Require(rejected,
            "unsupported distinct row-mask patterns should be rejected");

    Matrix group14_data(576, 32);
    Matrix group14_mask(576, 32);
    Matrix group14_sigma(576, 576);
    Matrix group14_delta(32, 32);
    for (int row = 0; row < 576; ++row)
        for (int column = 0; column < 32; ++column)
            group14_mask.set(row, column, row < 39 ? 1 : 0);
    for (int row = 0; row < 576; ++row)
        group14_sigma.set(row, row, 1);
    for (int column = 0; column < 32; ++column)
        group14_delta.set(column, column, 1);
    const double group14_score =
        StructuredObservedDataNegativeLogLikelihood(
            std::vector<Matrix>{group14_data},
            std::vector<Matrix>{group14_mask},
            group14_sigma, std::vector<Matrix>{group14_delta});
    const double expected_group14_score =
        0.5 * 39 * 32 * std::log(2.0 * std::acos(-1.0));
    Require(std::abs(group14_score - expected_group14_score) < 1e-9,
            "structured likelihood should handle the workbook's 39 complete group-14 rows");
}

void TestBalancedEstimator()
{
    std::vector<std::vector<double>> data;
    data.push_back(std::vector<double>{1.0, 0.0});
    data.push_back(std::vector<double>{0.0, 1.0});
    std::vector<Matrix> groups;
    groups.push_back(Matrix(data));

    bool converged = false;
    int iterations = -1;
    const std::vector<Matrix> estimate =
        FlipFlopBalanced(groups, converged, iterations, 10);
    Require(converged && iterations >= 1 && iterations <= 2,
            "balanced unregularized estimator should converge on identity data");
    Require(std::abs(
                (estimate[0].at(0, 0) + estimate[0].at(1, 1)) / 2.0 - 1.0) <
                1e-12,
            "balanced estimator should normalize sample covariance scale");
}

void TestFrobeniusNorm()
{
    Matrix matrix(2, 2);
    matrix.set(0, 0, 3.0);
    matrix.set(0, 1, 4.0);
    Require(std::abs(matrix.Frobenius() - 5.0) < 1e-12,
            "Frobenius should return the square root of the sum of squares");
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
    const Matrix fast_solution = coefficient.solve_positive_definite(rhs, false);
    const Matrix reconstructed = coefficient.multiply(solution);
    for (int i = 0; i < rhs.getRows(); ++i)
        for (int j = 0; j < rhs.getCols(); ++j)
        {
            Require(std::abs(reconstructed.at(i, j) - rhs.at(i, j)) < 1e-12,
                    "Cholesky solve should satisfy A*X=B");
            Require(std::abs(fast_solution.at(i, j) - solution.at(i, j)) < 1e-12,
                    "fast Cholesky solve should match the checked solve");
        }

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
    rejected = false;
    try
    {
        ill_conditioned.solve_positive_definite(rhs, false);
    }
    catch (const std::runtime_error &)
    {
        rejected = true;
    }
    Require(rejected,
            "fast SPD solve should reject a matrix with an unstable Cholesky pivot");

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
    Require(std::isfinite(trace[0].relative_change) &&
                trace[0].relative_change >= 0,
            "iteration diagnostics should expose the covariance change");

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

    rejected = false;
    try
    {
        FlipFlopBalanced(
            deficient_groups, converged, iterations, 10);
    }
    catch (const std::runtime_error &)
    {
        rejected = true;
    }
    Require(rejected,
            "balanced unregularized flip-flop must still reject rank-deficient data");
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
    const double sample_scale =
        (estimate[0].at(0, 0) + estimate[0].at(1, 1)) / 2.0;
    Require(std::abs(sample_scale - 1.0) < 1e-12,
            "regularized estimates should use the documented sample-scale normalization");
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
    TestObservedDataLikelihood();
    TestStructuredObservedDataLikelihood();
    TestCanonicalSolverMemorySafety();
    TestEstimator();
    TestBalancedEstimator();
    TestFrobeniusNorm();
    TestPositiveDefiniteSolve();
    TestNumericalRank();
    TestIterationDiagnostics();
    TestRegularizedEstimator();
    return 0;
}
