#include "ipca.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

namespace
{
int ParsePositiveInt(const std::string &text, const std::string &name)
{
    std::size_t parsed = 0;
    long long value = 0;
    try
    {
        value = std::stoll(text, &parsed);
    }
    catch (const std::exception &)
    {
        throw std::invalid_argument(name + " must be a positive integer");
    }
    if (parsed != text.size() || value <= 0 ||
        value > std::numeric_limits<int>::max())
        throw std::invalid_argument(name + " must be a positive integer");
    return static_cast<int>(value);
}

unsigned int ParseSeed(const std::string &text)
{
    std::size_t parsed = 0;
    unsigned long long value = 0;
    try
    {
        value = std::stoull(text, &parsed);
    }
    catch (const std::exception &)
    {
        throw std::invalid_argument("seed must be a non-negative integer");
    }
    if (parsed != text.size() ||
        value > std::numeric_limits<unsigned int>::max())
        throw std::invalid_argument("seed must fit in an unsigned integer");
    return static_cast<unsigned int>(value);
}

double ParseUnitInterval(const std::string &text, const std::string &name)
{
    std::size_t parsed = 0;
    double value = 0;
    try
    {
        value = std::stod(text, &parsed);
    }
    catch (const std::exception &)
    {
        throw std::invalid_argument(name + " must be a number in (0, 1]");
    }
    if (parsed != text.size() || !std::isfinite(value) ||
        value <= 0 || value > 1)
        throw std::invalid_argument(name + " must be a number in (0, 1]");
    return value;
}

double ParseNoiseScale(const std::string &text)
{
    const double value = ParseUnitInterval(text, "relative noise SD");
    return value;
}

double ParseShrinkage(const std::string &text)
{
    const double value = ParseUnitInterval(text, "shrinkage");
    if (value >= 1)
        throw std::invalid_argument("shrinkage must be a number in (0, 1)");
    return value;
}

std::string Trim(const std::string &text)
{
    const std::string whitespace = " \t\r\n";
    const std::size_t start = text.find_first_not_of(whitespace);
    if (start == std::string::npos)
        return "";
    const std::size_t end = text.find_last_not_of(whitespace);
    return text.substr(start, end - start + 1);
}

Matrix ReadCsvMatrix(const std::string &path)
{
    std::ifstream input(path.c_str());
    if (!input)
        throw std::runtime_error("cannot open CSV file: " + path);

    std::vector<std::vector<double>> values;
    std::string line;
    std::size_t expected_columns = 0;
    std::size_t line_number = 0;
    while (std::getline(input, line))
    {
        ++line_number;
        if (Trim(line).empty())
            continue;
        std::vector<double> row;
        std::stringstream line_stream(line);
        std::string cell;
        while (std::getline(line_stream, cell, ','))
        {
            cell = Trim(cell);
            if (cell.empty())
                throw std::runtime_error(
                    path + ":" + std::to_string(line_number) +
                    ": empty CSV cell");
            std::size_t parsed = 0;
            double value = 0;
            try
            {
                value = std::stod(cell, &parsed);
            }
            catch (const std::exception &)
            {
                throw std::runtime_error(
                    path + ":" + std::to_string(line_number) +
                    ": invalid numeric value");
            }
            if (parsed != cell.size() || !std::isfinite(value))
                throw std::runtime_error(
                    path + ":" + std::to_string(line_number) +
                    ": invalid or non-finite numeric value");
            row.push_back(value);
        }
        if (!line.empty() && line[line.size() - 1] == ',')
            throw std::runtime_error(
                path + ":" + std::to_string(line_number) +
                ": empty CSV cell");
        if (row.empty())
            throw std::runtime_error(
                path + ":" + std::to_string(line_number) +
                ": expected at least one feature");
        if (values.empty())
            expected_columns = row.size();
        else if (row.size() != expected_columns)
            throw std::runtime_error(
                path + ":" + std::to_string(line_number) +
                ": inconsistent number of columns");
        values.push_back(row);
    }
    if (input.bad())
        throw std::runtime_error("error reading CSV file: " + path);
    if (values.empty())
        throw std::runtime_error("CSV file contains no data: " + path);
    return Matrix(values);
}

void PrintFeasibility(const FeasibilityResult &result)
{
    std::cout << "generic_mle_feasible: "
              << (result.generic_mle_feasible ? "yes" : "no") << "\n";
    std::cout << "criterion: " << result.explanation << "\n";
    std::cout << "dimension_vector: [";
    for (std::size_t i = 0; i < result.dimension_vector.size(); ++i)
    {
        if (i)
            std::cout << ", ";
        std::cout << result.dimension_vector[i];
    }
    std::cout << "]\n";
    std::cout << "canonical_decomposition:\n";
    for (std::size_t i = 0; i < result.canonical_summands.size(); ++i)
    {
        std::cout << "  multiplicity " << result.multiplicities[i] << ": [";
        for (std::size_t j = 0; j < result.canonical_summands[i].size(); ++j)
        {
            if (j)
                std::cout << ", ";
            std::cout << result.canonical_summands[i][j];
        }
        std::cout << "]\n";
    }
}

void PrintInputRanks(const std::vector<Matrix> &groups)
{
    const int num_samples = groups[0].getRows();
    long long total_features = 0;
    for (std::size_t i = 0; i < groups.size(); ++i)
    {
        if (groups[i].getRows() != num_samples)
            throw std::invalid_argument(
                "all feature groups must have the same number of rows");
        total_features += groups[i].getCols();
    }
    if (total_features > std::numeric_limits<int>::max())
        throw std::invalid_argument("total number of features exceeds supported range");

    Matrix combined(num_samples, static_cast<int>(total_features));
    int column_offset = 0;
    for (std::size_t group = 0; group < groups.size(); ++group)
    {
        for (int i = 0; i < num_samples; ++i)
            for (int j = 0; j < groups[group].getCols(); ++j)
                combined.set(i, column_offset + j, groups[group].at(i, j));
        column_offset += groups[group].getCols();
        std::cout << "input_rank,group=" << group + 1
                  << ",rows=" << groups[group].getRows()
                  << ",columns=" << groups[group].getCols()
                  << ",rank=" << groups[group].numerical_rank() << "\n";
    }
    std::cout << "input_rank,group=combined"
              << ",rows=" << combined.getRows()
              << ",columns=" << combined.getCols()
              << ",rank=" << combined.numerical_rank() << "\n";
    std::cout << "rank_relative_tolerance: 1e-10\n";
}

void PrintIterationDiagnostic(
    const FlipFlopIterationDiagnostics &diagnostics)
{
    std::cout << diagnostics.iteration << ",";
    if (diagnostics.covariance_index == 0)
        std::cout << "Sigma";
    else
        std::cout << "Delta[" << diagnostics.covariance_index << "]";
    std::cout << "," << std::setprecision(17)
              << diagnostics.relative_change
              << ","
              << diagnostics.spectrum.minimum_eigenvalue
              << "," << diagnostics.spectrum.maximum_eigenvalue
              << "," << diagnostics.spectrum.relative_minimum_eigenvalue
              << "," << diagnostics.spectrum.condition_number << "\n";
}

int RunDiagnostics(
    const std::vector<Matrix> &groups,
    int max_iterations,
    double shrinkage = 0,
    double damping = 1)
{
    if (groups.empty())
        throw std::invalid_argument("at least one feature group is required");
    PrintInputRanks(groups);

    std::vector<int> group_sizes;
    for (std::size_t i = 0; i < groups.size(); ++i)
        group_sizes.push_back(groups[i].getCols());
    const FeasibilityResult feasibility =
        AnalyzeGenericMLEFeasibility(groups[0].getRows(), group_sizes);
    PrintFeasibility(feasibility);

    std::cout << "iteration,covariance,relative_change,lambda_min,lambda_max,"
                 "relative_lambda_min,condition_number\n";
    bool converged = false;
    int iterations = -1;
    const FlipFlopObserver observer = PrintIterationDiagnostic;
    const std::vector<Matrix> covariance = shrinkage > 0
        ? FlipFlopRegularizedWithDiagnostics(
              groups, converged, iterations, max_iterations,
              shrinkage, damping, observer)
        : FlipFlopWithDiagnostics(
              groups, converged, iterations, max_iterations, observer);
    std::cout << "converged: " << (converged ? "yes" : "no") << "\n";
    std::cout << "iterations: " << iterations << "\n";
    return converged ? 0 : 2;
}

int RunDiagnoseRegularized(int argc, char **argv)
{
    if (argc < 6)
        throw std::invalid_argument(
            "diagnose-regularized requires an iteration limit, shrinkage, "
            "damping, and at least one group CSV");
    const int max_iterations =
        ParsePositiveInt(argv[2], "maximum iterations");
    const double shrinkage = ParseShrinkage(argv[3]);
    const double damping = ParseUnitInterval(argv[4], "damping");
    std::vector<Matrix> groups;
    for (int i = 5; i < argc; ++i)
        groups.push_back(ReadCsvMatrix(argv[i]));
    std::cout << "regularization: shrinkage=" << shrinkage
              << ", damping=" << damping << "\n";
    return RunDiagnostics(groups, max_iterations, shrinkage, damping);
}

int RunDiagnose(int argc, char **argv)
{
    if (argc < 4)
        throw std::invalid_argument(
            "diagnose requires an iteration limit and at least one group CSV");
    const int max_iterations =
        ParsePositiveInt(argv[2], "maximum iterations");
    std::vector<Matrix> groups;
    for (int i = 3; i < argc; ++i)
        groups.push_back(ReadCsvMatrix(argv[i]));
    return RunDiagnostics(groups, max_iterations);
}

int RunDiagnoseGaussian(int argc, char **argv)
{
    if (argc < 6)
        throw std::invalid_argument(
            "diagnose-gaussian requires an iteration limit, sample count, "
            "seed, and at least one group size");
    const int max_iterations =
        ParsePositiveInt(argv[2], "maximum iterations");
    const int num_samples = ParsePositiveInt(argv[3], "sample count");
    const unsigned int seed = ParseSeed(argv[4]);
    std::mt19937 generator(seed);
    std::normal_distribution<double> standard_normal(0.0, 1.0);
    std::vector<Matrix> groups;
    for (int i = 5; i < argc; ++i)
    {
        const int group_size = ParsePositiveInt(argv[i], "group size");
        Matrix group(num_samples, group_size);
        for (int row = 0; row < num_samples; ++row)
            for (int column = 0; column < group_size; ++column)
                group.set(row, column, standard_normal(generator));
        groups.push_back(group);
    }
    std::cout << "random_distribution: Gaussian(0, 1)\n"
              << "random_seed: " << seed << "\n";
    return RunDiagnostics(groups, max_iterations);
}

void PrintUsage(const char *program)
{
    std::cerr << "Usage:\n"
              << "  " << program << " check <samples> <group-features>...\n"
              << "  " << program << " estimate <max-iterations> <group.csv>...\n"
              << "  " << program << " diagnose <max-iterations> <group.csv>...\n"
              << "  " << program
              << " diagnose-regularized <max-iterations> <shrinkage>"
                 " <damping> <group.csv>...\n"
              << "  " << program
              << " diagnose-gaussian <max-iterations> <samples> <seed>"
                 " <group-features>...\n"
              << "  " << program
              << " estimate-regularized <max-iterations> <shrinkage>"
                 " <damping> <group.csv>...\n"
              << "  " << program
              << " estimate-jittered <max-iterations> <relative-noise-SD>"
                 " <seed> <output-directory> <group.csv>...\n"
              << "  " << program << " demo\n";
}

int RunCheck(int argc, char **argv)
{
    if (argc < 4)
        throw std::invalid_argument("check requires sample and group dimensions");
    const int samples = ParsePositiveInt(argv[2], "sample count");
    std::vector<int> group_sizes;
    for (int i = 3; i < argc; ++i)
        group_sizes.push_back(ParsePositiveInt(argv[i], "group size"));

    const FeasibilityResult result =
        AnalyzeGenericMLEFeasibility(samples, group_sizes);
    PrintFeasibility(result);
    return result.generic_mle_feasible ? 0 : 1;
}

int RunEstimate(int argc, char **argv, bool regularized)
{
    const int first_group_argument = regularized ? 5 : 3;
    if (argc < first_group_argument + 1)
        throw std::invalid_argument(
            regularized
                ? "estimate-regularized requires an iteration limit, "
                  "shrinkage, damping, and at least one CSV file"
                : "estimate requires an iteration limit and at least one CSV file");
    const int max_iterations =
        ParsePositiveInt(argv[2], "maximum iterations");
    const double shrinkage =
        regularized ? ParseShrinkage(argv[3]) : 0;
    const double damping =
        regularized ? ParseUnitInterval(argv[4], "damping") : 1;
    std::vector<Matrix> groups;
    std::vector<int> group_sizes;
    for (int i = first_group_argument; i < argc; ++i)
    {
        groups.push_back(ReadCsvMatrix(argv[i]));
        group_sizes.push_back(groups.back().getCols());
    }
    const int samples = groups[0].getRows();
    for (std::size_t i = 1; i < groups.size(); ++i)
        if (groups[i].getRows() != samples)
            throw std::invalid_argument(
                "all group CSV files must have the same number of rows");

    const FeasibilityResult feasibility =
        AnalyzeGenericMLEFeasibility(samples, group_sizes);
    PrintFeasibility(feasibility);
    if (!feasibility.generic_mle_feasible)
        std::cout << "warning: generic MLE existence fails for these dimensions; "
                     "attempting estimation for the supplied data anyway\n";

    bool converged = false;
    int iterations = -1;
    const std::vector<Matrix> covariance = regularized
        ? FlipFlopRegularized(
              groups, converged, iterations, max_iterations,
              shrinkage, damping)
        : FlipFlop(groups, converged, iterations, max_iterations);
    if (regularized)
        std::cout << "regularization: shrinkage=" << shrinkage
                  << ", damping=" << damping << "\n"
                  << "convergence_criterion: maximum relative Frobenius change "
                     "per covariance < 1e-10\n";
    std::cout << "converged: " << (converged ? "yes" : "no") << "\n";
    std::cout << "iterations: " << iterations << "\n";
    if (!converged)
        return 2;
    std::cout << "sample_covariance:\n";
    covariance[0].print();
    for (std::size_t i = 1; i < covariance.size(); ++i)
    {
        std::cout << "group_covariance_" << i << ":\n";
        covariance[i].print();
    }
    return 0;
}

int RunEstimateJittered(int argc, char **argv)
{
    if (argc < 7)
        throw std::invalid_argument(
            "estimate-jittered requires an iteration limit, relative noise "
            "SD, seed, output directory, and at least one group CSV");
    const int max_iterations =
        ParsePositiveInt(argv[2], "maximum iterations");
    const double noise_sd = ParseNoiseScale(argv[3]);
    const unsigned int seed = ParseSeed(argv[4]);
    const std::string output_directory = argv[5];
    std::mt19937 generator(seed);
    std::normal_distribution<double> noise(0.0, noise_sd);
    std::vector<Matrix> groups;
    std::vector<int> group_sizes;
    for (int argument = 6; argument < argc; ++argument)
    {
        Matrix group = ReadCsvMatrix(argv[argument]);
        for (int column = 0; column < group.getCols(); ++column)
        {
            long double squared_sum = 0;
            for (int row = 0; row < group.getRows(); ++row)
            {
                const long double value = group.at(row, column);
                squared_sum += value * value;
            }
            double rms = std::sqrt(static_cast<double>(
                squared_sum / group.getRows()));
            if (!std::isfinite(rms))
                throw std::runtime_error(
                    std::string("feature RMS is not finite in ") +
                    argv[argument]);
            if (rms == 0)
                rms = 1;
            for (int row = 0; row < group.getRows(); ++row)
            {
                const double transformed =
                    group.at(row, column) / rms + noise(generator);
                if (!std::isfinite(transformed))
                    throw std::runtime_error(
                        std::string("jitter produced non-finite data in ") +
                        argv[argument]);
                group.set(row, column, transformed);
            }
        }

        std::ostringstream output_path;
        output_path << output_directory << "/jittered_group_";
        if (groups.size() + 1 < 10)
            output_path << '0';
        output_path << groups.size() + 1 << ".csv";
        std::ofstream output(output_path.str().c_str());
        if (!output)
            throw std::runtime_error(
                "cannot create transformed CSV (ensure output directory exists): " +
                output_path.str());
        output << std::setprecision(17);
        for (int row = 0; row < group.getRows(); ++row)
        {
            for (int column = 0; column < group.getCols(); ++column)
            {
                if (column)
                    output << ',';
                output << group.at(row, column);
            }
            output << "\n";
        }
        if (!output)
            throw std::runtime_error(
                "error writing transformed CSV: " + output_path.str());
        groups.push_back(group);
        group_sizes.push_back(group.getCols());
    }

    std::cout << "data_transform: divide each feature by its RMS, then add "
                 "independent Gaussian noise\n"
              << "relative_noise_sd: " << noise_sd << "\n"
              << "random_seed: " << seed << "\n"
              << "transformed_groups_saved_to: " << output_directory << "\n";
    const FeasibilityResult feasibility =
        AnalyzeGenericMLEFeasibility(groups[0].getRows(), group_sizes);
    PrintFeasibility(feasibility);
    if (!feasibility.generic_mle_feasible)
        std::cout << "warning: generic MLE existence fails for these dimensions; "
                     "attempting estimation for the supplied data anyway\n";

    bool converged = false;
    int iterations = -1;
    const std::vector<Matrix> covariance =
        FlipFlopBalanced(groups, converged, iterations, max_iterations);
    std::cout << "estimator: unregularized Flip-Flop on transformed data\n"
              << "convergence_criterion: maximum relative Frobenius change "
                 "per covariance < 1e-10 after covariance-scale normalization\n"
              << "converged: " << (converged ? "yes" : "no") << "\n"
              << "iterations: " << iterations << "\n";
    if (!converged)
        return 2;
    std::cout << "sample_covariance:\n";
    covariance[0].print();
    for (std::size_t i = 1; i < covariance.size(); ++i)
    {
        std::cout << "group_covariance_" << i << ":\n";
        covariance[i].print();
    }
    return 0;
}
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        PrintUsage(argv[0]);
        return 2;
    }
    try
    {
        const std::string command = argv[1];
        if (command == "check")
            return RunCheck(argc, argv);
        if (command == "estimate")
            return RunEstimate(argc, argv, false);
        if (command == "estimate-regularized")
            return RunEstimate(argc, argv, true);
        if (command == "estimate-jittered")
            return RunEstimateJittered(argc, argv);
        if (command == "diagnose")
            return RunDiagnose(argc, argv);
        if (command == "diagnose-regularized")
            return RunDiagnoseRegularized(argc, argv);
        if (command == "diagnose-gaussian")
            return RunDiagnoseGaussian(argc, argv);
        if (command == "demo" && argc == 2)
            return RunDemo();
        PrintUsage(argv[0]);
        return 2;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << "\n";
        return 2;
    }
}
