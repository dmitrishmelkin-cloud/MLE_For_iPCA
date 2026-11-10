#include "ipca.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
const std::size_t kMaximumObservedValuesPerGroup = 512;

void ValidatePositiveDefinite(const Matrix &matrix, const char *name)
{
    if (matrix.getRows() <= 0 ||
        matrix.getRows() != matrix.getCols())
        throw std::invalid_argument(
            std::string(name) + " must be a non-empty square matrix");

    const int size = matrix.getRows();
    double scale = 0;
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
        {
            const double value = matrix.at(i, j);
            if (!std::isfinite(value))
                throw std::invalid_argument(
                    std::string(name) + " must contain finite values");
            scale = std::max(scale, std::abs(value));
        }
    if (scale == 0)
        throw std::invalid_argument(
            std::string(name) + " must be positive definite");

    Matrix lower(size, size);
    for (int i = 0; i < size; ++i)
        for (int j = 0; j <= i; ++j)
        {
            if (std::abs(matrix.at(i, j) - matrix.at(j, i)) >
                scale * 1e-12)
                throw std::invalid_argument(
                    std::string(name) + " must be symmetric");
            double value =
                0.5 * (matrix.at(i, j) + matrix.at(j, i));
            for (int k = 0; k < j; ++k)
                value -= lower.at(i, k) * lower.at(j, k);
            if (i == j)
            {
                if (!std::isfinite(value) || value <= scale * 1e-12)
                    throw std::invalid_argument(
                        std::string(name) +
                        " must be numerically positive definite");
                lower.set(i, j, std::sqrt(value));
            }
            else
            {
                lower.set(i, j, value / lower.at(j, j));
            }
        }
}

struct ObservedEntry
{
    int row;
    int column;
    double value;
};

double ObservedCovariance(
    const ObservedEntry &left,
    const ObservedEntry &right,
    const Matrix &sample_covariance,
    const Matrix &group_covariance)
{
    return sample_covariance.at(left.row, right.row) *
        group_covariance.at(left.column, right.column);
}

double GaussianNegativeLogLikelihood(
    const std::vector<ObservedEntry> &observed,
    const Matrix &sample_covariance,
    const Matrix &group_covariance)
{
    const std::size_t count = observed.size();
    Matrix covariance(static_cast<int>(count), static_cast<int>(count));
    for (std::size_t i = 0; i < count; ++i)
        for (std::size_t j = 0; j <= i; ++j)
        {
            const double value = ObservedCovariance(
                observed[i], observed[j],
                sample_covariance, group_covariance);
            if (!std::isfinite(value))
                throw std::runtime_error(
                    "observed covariance contains a non-finite value");
            covariance.set(static_cast<int>(i), static_cast<int>(j), value);
            covariance.set(static_cast<int>(j), static_cast<int>(i), value);
        }

    Matrix lower(static_cast<int>(count), static_cast<int>(count));
    double log_determinant = 0;
    for (std::size_t i = 0; i < count; ++i)
        for (std::size_t j = 0; j <= i; ++j)
        {
            double value = covariance.at(
                static_cast<int>(i), static_cast<int>(j));
            for (std::size_t k = 0; k < j; ++k)
                value -= lower.at(static_cast<int>(i), static_cast<int>(k)) *
                    lower.at(static_cast<int>(j), static_cast<int>(k));

            if (i == j)
            {
                if (!std::isfinite(value) || value <= 0)
                    throw std::runtime_error(
                        "observed covariance is not positive definite");
                lower.set(static_cast<int>(i), static_cast<int>(j),
                          std::sqrt(value));
                log_determinant += std::log(value);
            }
            else
            {
                const double diagonal =
                    lower.at(static_cast<int>(j), static_cast<int>(j));
                const double factor = value / diagonal;
                if (!std::isfinite(factor))
                    throw std::runtime_error(
                        "observed covariance Cholesky factor is not finite");
                lower.set(static_cast<int>(i), static_cast<int>(j), factor);
            }
        }

    std::vector<double> solved(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        double value = observed[i].value;
        for (std::size_t j = 0; j < i; ++j)
            value -= lower.at(static_cast<int>(i), static_cast<int>(j)) *
                solved[j];
        solved[i] = value /
            lower.at(static_cast<int>(i), static_cast<int>(i));
    }

    long double quadratic_form = 0;
    for (std::size_t i = 0; i < count; ++i)
        quadratic_form += static_cast<long double>(solved[i]) * solved[i];
    const double result = 0.5 * (
        static_cast<double>(count) *
            std::log(2.0 * std::acos(-1.0)) +
        log_determinant + static_cast<double>(quadratic_form));
    if (!std::isfinite(result))
        throw std::runtime_error(
            "observed-data negative log-likelihood is not finite");
    return result;
}
}

double ObservedDataNegativeLogLikelihood(
    const std::vector<Matrix> &feature_groups,
    const std::vector<Matrix> &observation_masks,
    const Matrix &sample_covariance,
    const std::vector<Matrix> &group_covariances,
    const std::vector<std::vector<double>> &feature_means)
{
    if (feature_groups.empty())
        throw std::invalid_argument("at least one feature group is required");
    if (observation_masks.size() != feature_groups.size() ||
        group_covariances.size() != feature_groups.size() ||
        feature_means.size() != feature_groups.size())
        throw std::invalid_argument(
            "each feature group requires one observation mask, covariance, and mean");

    const int num_samples = feature_groups[0].getRows();
    if (num_samples <= 0 ||
        sample_covariance.getRows() != num_samples ||
        sample_covariance.getCols() != num_samples)
        throw std::invalid_argument(
            "sample covariance dimensions must match the sample count");
    ValidatePositiveDefinite(sample_covariance, "sample covariance");

    long double total = 0;
    for (std::size_t group = 0; group < feature_groups.size(); ++group)
    {
        const Matrix &data = feature_groups[group];
        const Matrix &mask = observation_masks[group];
        const Matrix &group_covariance = group_covariances[group];
        if (data.getRows() != num_samples || data.getCols() <= 0)
            throw std::invalid_argument(
                "feature groups must have matching rows and positive columns");
        if (mask.getRows() != data.getRows() ||
            mask.getCols() != data.getCols())
            throw std::invalid_argument(
                "observation mask dimensions must match its feature group");
        if (group_covariance.getRows() != data.getCols() ||
            group_covariance.getCols() != data.getCols())
            throw std::invalid_argument(
                "group covariance dimensions must match its feature group");
        if (feature_means[group].size() !=
            static_cast<std::size_t>(data.getCols()))
            throw std::invalid_argument(
                "feature mean dimensions must match its feature group");
        for (std::size_t column = 0;
             column < feature_means[group].size(); ++column)
            if (!std::isfinite(feature_means[group][column]))
                throw std::invalid_argument(
                    "feature means must contain finite values");
        std::vector<ObservedEntry> observed;
        for (int row = 0; row < data.getRows(); ++row)
            for (int column = 0; column < data.getCols(); ++column)
            {
                const double is_observed = mask.at(row, column);
                if (!std::isfinite(is_observed) ||
                    (is_observed != 0 && is_observed != 1))
                    throw std::invalid_argument(
                        "observation masks must contain only finite zeros and ones");
                if (is_observed == 0)
                    continue;

                const double value = data.at(row, column);
                const double mean = feature_means[group][column];
                if (!std::isfinite(value) || !std::isfinite(mean))
                    throw std::invalid_argument(
                        "observed feature values and means must be finite");
                observed.push_back(
                    ObservedEntry{row, column, value - mean});
                if (observed.size() > kMaximumObservedValuesPerGroup)
                    throw std::length_error(
                        "exact dense observed likelihood is limited to 512 "
                        "observed values per group; use a structured solver "
                        "for larger problems");
            }

        ValidatePositiveDefinite(group_covariance, "group covariance");
        if (!observed.empty())
            total += GaussianNegativeLogLikelihood(
                observed, sample_covariance, group_covariance);
    }

    const double result = static_cast<double>(total);
    if (!std::isfinite(result))
        throw std::runtime_error(
            "total observed-data negative log-likelihood is not finite");
    return result;
}

double ObservedDataNegativeLogLikelihood(
    const std::vector<Matrix> &feature_groups,
    const std::vector<Matrix> &observation_masks,
    const Matrix &sample_covariance,
    const std::vector<Matrix> &group_covariances)
{
    std::vector<std::vector<double>> means;
    for (std::size_t group = 0; group < feature_groups.size(); ++group)
        means.push_back(
            std::vector<double>(feature_groups[group].getCols(), 0.0));
    return ObservedDataNegativeLogLikelihood(
        feature_groups, observation_masks, sample_covariance,
        group_covariances, means);
}
