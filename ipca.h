#ifndef IPCA_H
#define IPCA_H

#include "matrix.h"
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

FeasibilityResult AnalyzeGenericMLEFeasibility(
    int num_samples,
    const std::vector<int> &group_sizes);

std::vector<Matrix> FlipFlop(
    const std::vector<Matrix> &feature_groups,
    bool &converged,
    int &num_iterations,
    int max_iterations);

int RunDemo();

#endif
