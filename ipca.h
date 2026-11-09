#ifndef IPCA_H
#define IPCA_H

#include "matrix.h"
#include <functional>
#include <string>
#include <vector>

struct FeasibilityResult
{
    bool generic_mle_feasible;
    std::vector<int> dimension_vector;
    std::vector<std::vector<int>> canonical_summands;
    std::vector<int> multiplicities;
    std::string explanation;
};

struct FlipFlopIterationDiagnostics
{
    int iteration;
    int covariance_index;
    PositiveDefiniteDiagnostics spectrum;
};

typedef std::function<void(const FlipFlopIterationDiagnostics &)>
    FlipFlopObserver;

FeasibilityResult AnalyzeGenericMLEFeasibility(
    int num_samples,
    const std::vector<int> &group_sizes);

std::vector<Matrix> FlipFlop(
    const std::vector<Matrix> &feature_groups,
    bool &converged,
    int &num_iterations,
    int max_iterations);

std::vector<Matrix> FlipFlopRegularized(
    const std::vector<Matrix> &feature_groups,
    bool &converged,
    int &num_iterations,
    int max_iterations,
    double shrinkage,
    double damping);

std::vector<Matrix> FlipFlopWithDiagnostics(
    const std::vector<Matrix> &feature_groups,
    bool &converged,
    int &num_iterations,
    int max_iterations,
    const FlipFlopObserver &observer);

std::vector<Matrix> FlipFlopRegularizedWithDiagnostics(
    const std::vector<Matrix> &feature_groups,
    bool &converged,
    int &num_iterations,
    int max_iterations,
    double shrinkage,
    double damping,
    const FlipFlopObserver &observer);

int RunDemo();

#endif
