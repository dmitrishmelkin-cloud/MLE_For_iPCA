#include "ipca.h"
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
    Require(estimate[0].at(0, 0) == 0.5 &&
                estimate[0].at(1, 1) == 0.5,
            "sample covariance estimate should preserve the existing update scale");

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
}

int main()
{
    TestFeasibility();
    TestEstimator();
    return 0;
}
