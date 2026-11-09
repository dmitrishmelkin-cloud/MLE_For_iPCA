#include "ipca.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
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

void PrintUsage(const char *program)
{
    std::cerr << "Usage:\n"
              << "  " << program << " check <samples> <group-features>...\n"
              << "  " << program << " estimate <max-iterations> <group.csv>...\n"
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

int RunEstimate(int argc, char **argv)
{
    if (argc < 4)
        throw std::invalid_argument(
            "estimate requires an iteration limit and at least one CSV file");
    const int max_iterations =
        ParsePositiveInt(argv[2], "maximum iterations");
    std::vector<Matrix> groups;
    std::vector<int> group_sizes;
    for (int i = 3; i < argc; ++i)
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
    const std::vector<Matrix> covariance =
        FlipFlop(groups, converged, iterations, max_iterations);
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
            return RunEstimate(argc, argv);
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
