#include "ipca.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
struct CholeskyFactor
{
    Matrix lower;
    double log_determinant;
};

CholeskyFactor FactorPositiveDefinite(const Matrix &matrix, const char *name)
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
    double log_determinant = 0;
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
                log_determinant += std::log(value);
            }
            else
            {
                lower.set(i, j, value / lower.at(j, j));
            }
        }

    CholeskyFactor result = {lower, log_determinant};
    return result;
}

Matrix Submatrix(
    const Matrix &matrix,
    const std::vector<int> &rows,
    const std::vector<int> &columns)
{
    Matrix result(
        static_cast<int>(rows.size()), static_cast<int>(columns.size()));
    for (std::size_t i = 0; i < rows.size(); ++i)
        for (std::size_t j = 0; j < columns.size(); ++j)
            result.set(
                static_cast<int>(i), static_cast<int>(j),
                matrix.at(rows[i], columns[j]));
    return result;
}

Matrix SolveLower(const Matrix &lower, const Matrix &right_hand_side)
{
    if (lower.getRows() != lower.getCols() ||
        right_hand_side.getRows() != lower.getRows())
        throw std::invalid_argument(
            "incompatible dimensions for lower-triangular solve");
    Matrix result = right_hand_side;
    for (int i = 0; i < lower.getRows(); ++i)
    {
        const double diagonal = lower.at(i, i);
        for (int column = 0; column < result.getCols(); ++column)
        {
            double value = result.at(i, column);
            for (int k = 0; k < i; ++k)
                value -= lower.at(i, k) * result.at(k, column);
            result.set(i, column, value / diagonal);
        }
    }
    return result;
}

double MatrixNormalNegativeLogLikelihood(
    const Matrix &observed,
    const Matrix &row_covariance,
    const Matrix &column_covariance)
{
    if (observed.getRows() <= 0 || observed.getCols() <= 0)
        throw std::invalid_argument(
            "matrix-normal likelihood requires non-empty observations");
    if (row_covariance.getRows() != observed.getRows() ||
        row_covariance.getCols() != observed.getRows() ||
        column_covariance.getRows() != observed.getCols() ||
        column_covariance.getCols() != observed.getCols())
        throw std::invalid_argument(
            "matrix-normal covariance dimensions do not match observations");

    const CholeskyFactor row_factor =
        FactorPositiveDefinite(row_covariance, "row covariance");
    const CholeskyFactor column_factor =
        FactorPositiveDefinite(column_covariance, "column covariance");
    const Matrix row_whitened =
        SolveLower(row_factor.lower, observed);
    const Matrix column_whitened =
        SolveLower(column_factor.lower, row_whitened.transpose());
    long double quadratic_form = 0;
    for (int i = 0; i < column_whitened.getRows(); ++i)
        for (int j = 0; j < column_whitened.getCols(); ++j)
        {
            const long double value = column_whitened.at(i, j);
            quadratic_form += value * value;
        }

    const double count =
        static_cast<double>(observed.getRows()) * observed.getCols();
    const double result = 0.5 * (
        count * std::log(2.0 * std::acos(-1.0)) +
        observed.getCols() * row_factor.log_determinant +
        observed.getRows() * column_factor.log_determinant +
        static_cast<double>(quadratic_form));
    if (!std::isfinite(result))
        throw std::runtime_error(
            "structured observed-data likelihood is not finite");
    return result;
}

Matrix SelectColumns(const Matrix &matrix, const std::vector<int> &columns)
{
    std::vector<int> rows(matrix.getRows());
    for (int i = 0; i < matrix.getRows(); ++i)
        rows[i] = i;
    return Submatrix(matrix, rows, columns);
}

Matrix CenteredSubmatrix(
    const Matrix &data,
    const std::vector<int> &rows,
    const std::vector<int> &columns,
    const std::vector<double> &means)
{
    Matrix result = Submatrix(data, rows, columns);
    for (int row = 0; row < result.getRows(); ++row)
        for (int column = 0; column < result.getCols(); ++column)
            result.set(
                row, column,
                result.at(row, column) - means[columns[column]]);
    return result;
}

Matrix ConditionalRowCovariance(
    const Matrix &sample_covariance,
    const std::vector<int> &full_rows,
    const std::vector<int> &partial_rows)
{
    const Matrix sigma_full =
        Submatrix(sample_covariance, full_rows, full_rows);
    const Matrix sigma_full_partial =
        Submatrix(sample_covariance, full_rows, partial_rows);
    const Matrix solved =
        sigma_full.solve_positive_definite(sigma_full_partial, false);
    const Matrix sigma_partial_full =
        Submatrix(sample_covariance, partial_rows, full_rows);
    const Matrix sigma_partial =
        Submatrix(sample_covariance, partial_rows, partial_rows);
    Matrix conditional = sigma_partial;
    conditional.Add(sigma_partial_full.multiply(solved), -1.0);
    for (int i = 0; i < conditional.getRows(); ++i)
        for (int j = i + 1; j < conditional.getCols(); ++j)
        {
            const double average =
                0.5 * (conditional.at(i, j) + conditional.at(j, i));
            conditional.set(i, j, average);
            conditional.set(j, i, average);
        }
    return conditional;
}

double ScoreGroup(
    const Matrix &data,
    const std::map<std::vector<int>, std::vector<int>> &rows_by_pattern,
    const Matrix &sample_covariance,
    const Matrix &group_covariance,
    const std::vector<double> &means)
{
    if (rows_by_pattern.empty())
        return 0;
    if (rows_by_pattern.size() > 2)
        throw std::invalid_argument(
            "structured likelihood supports at most two non-empty row-mask "
            "patterns per group");

    typename std::map<std::vector<int>, std::vector<int>>::const_iterator
        full_pattern = rows_by_pattern.end();
    std::vector<int> partial_columns;
    std::vector<int> partial_rows;
    for (typename std::map<std::vector<int>, std::vector<int>>::const_iterator
             pattern = rows_by_pattern.begin();
         pattern != rows_by_pattern.end(); ++pattern)
    {
        if (pattern->first.size() ==
            static_cast<std::size_t>(data.getCols()))
            full_pattern = pattern;
        else
        {
            partial_columns = pattern->first;
            partial_rows = pattern->second;
        }
    }

    if (rows_by_pattern.size() == 1)
    {
        const std::vector<int> &columns = rows_by_pattern.begin()->first;
        const std::vector<int> &rows = rows_by_pattern.begin()->second;
        return MatrixNormalNegativeLogLikelihood(
            CenteredSubmatrix(data, rows, columns, means),
            Submatrix(sample_covariance, rows, rows),
            Submatrix(group_covariance, columns, columns));
    }

    if (full_pattern == rows_by_pattern.end() ||
        partial_columns.empty() || partial_rows.empty())
        throw std::invalid_argument(
            "two-pattern structured likelihood requires full rows and one "
            "non-empty partial feature mask");

    const std::vector<int> &full_rows = full_pattern->second;
    const std::vector<int> all_columns = full_pattern->first;
    const Matrix full_data =
        CenteredSubmatrix(data, full_rows, all_columns, means);
    const double full_score = MatrixNormalNegativeLogLikelihood(
        full_data,
        Submatrix(sample_covariance, full_rows, full_rows),
        group_covariance);

    const Matrix conditional_covariance = ConditionalRowCovariance(
        sample_covariance, full_rows, partial_rows);
    const Matrix sigma_full =
        Submatrix(sample_covariance, full_rows, full_rows);
    const Matrix regression = sigma_full.solve_positive_definite(
        Submatrix(sample_covariance, full_rows, partial_rows), false)
                                 .transpose();
    const Matrix predicted_full = regression.multiply(full_data);
    Matrix residual =
        CenteredSubmatrix(data, partial_rows, partial_columns, means);
    const Matrix predicted_observed =
        SelectColumns(predicted_full, partial_columns);
    residual.Add(predicted_observed, -1.0);

    const double conditional_score = MatrixNormalNegativeLogLikelihood(
        residual,
        conditional_covariance,
        Submatrix(group_covariance, partial_columns, partial_columns));
    return full_score + conditional_score;
}
}

double StructuredObservedDataNegativeLogLikelihood(
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
    FactorPositiveDefinite(sample_covariance, "sample covariance");

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
        FactorPositiveDefinite(group_covariance, "group covariance");
        for (std::size_t column = 0;
             column < feature_means[group].size(); ++column)
            if (!std::isfinite(feature_means[group][column]))
                throw std::invalid_argument(
                    "feature means must contain finite values");

        typename std::map<std::vector<int>, std::vector<int>> rows_by_pattern;
        for (int row = 0; row < data.getRows(); ++row)
        {
            std::vector<int> observed_columns;
            for (int column = 0; column < data.getCols(); ++column)
            {
                const double observed = mask.at(row, column);
                if (!std::isfinite(observed) ||
                    (observed != 0 && observed != 1))
                    throw std::invalid_argument(
                        "observation masks must contain only finite zeros and ones");
                if (observed == 1)
                {
                    if (!std::isfinite(data.at(row, column)) ||
                        !std::isfinite(feature_means[group][column]))
                        throw std::invalid_argument(
                            "observed feature values and means must be finite");
                    observed_columns.push_back(column);
                }
            }
            if (!observed_columns.empty())
                rows_by_pattern[observed_columns].push_back(row);
        }

        if (!rows_by_pattern.empty())
            total += ScoreGroup(
                data, rows_by_pattern, sample_covariance,
                group_covariance, feature_means[group]);
    }

    const double result = static_cast<double>(total);
    if (!std::isfinite(result))
        throw std::runtime_error(
            "total structured observed-data likelihood is not finite");
    return result;
}

double StructuredObservedDataNegativeLogLikelihood(
    const std::vector<Matrix> &feature_groups,
    const std::vector<Matrix> &observation_masks,
    const Matrix &sample_covariance,
    const std::vector<Matrix> &group_covariances)
{
    std::vector<std::vector<double>> means;
    for (std::size_t group = 0; group < feature_groups.size(); ++group)
        means.push_back(
            std::vector<double>(feature_groups[group].getCols(), 0.0));
    return StructuredObservedDataNegativeLogLikelihood(
        feature_groups, observation_masks, sample_covariance,
        group_covariances, means);
}
