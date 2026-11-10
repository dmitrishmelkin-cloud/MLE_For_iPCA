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
    double relative_change;
    PositiveDefiniteDiagnostics spectrum;
};

typedef std::function<void(const FlipFlopIterationDiagnostics &)>
    FlipFlopObserver;

FeasibilityResult AnalyzeGenericMLEFeasibility(
    int num_samples,
    const std::vector<int> &group_sizes);

double ObservedDataNegativeLogLikelihood(
    const std::vector<Matrix> &feature_groups,
    const std::vector<Matrix> &observation_masks,
    const Matrix &sample_covariance,
    const std::vector<Matrix> &group_covariances);

double ObservedDataNegativeLogLikelihood(
    const std::vector<Matrix> &feature_groups,
    const std::vector<Matrix> &observation_masks,
    const Matrix &sample_covariance,
    const std::vector<Matrix> &group_covariances,
    const std::vector<std::vector<double>> &feature_means);

double StructuredObservedDataNegativeLogLikelihood(
    const std::vector<Matrix> &feature_groups,
    const std::vector<Matrix> &observation_masks,
    const Matrix &sample_covariance,
    const std::vector<Matrix> &group_covariances);

double StructuredObservedDataNegativeLogLikelihood(
    const std::vector<Matrix> &feature_groups,
    const std::vector<Matrix> &observation_masks,
    const Matrix &sample_covariance,
    const std::vector<Matrix> &group_covariances,
    const std::vector<std::vector<double>> &feature_means);

std::vector<Matrix> FlipFlop(
    const std::vector<Matrix> &feature_groups,
    bool &converged,
    int &num_iterations,
    int max_iterations);

std::vector<Matrix> FlipFlopBalanced(
    const std::vector<Matrix> &feature_groups,
    bool &converged,
    int &num_iterations,
    int max_iterations,
    const FlipFlopObserver &observer = FlipFlopObserver());

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
