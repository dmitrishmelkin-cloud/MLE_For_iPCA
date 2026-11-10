#ifndef MATRIX_H
#define MATRIX_H

#include <algorithm>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <limits>

struct PositiveDefiniteDiagnostics
{
    double minimum_eigenvalue;
    double maximum_eigenvalue;
    double relative_minimum_eigenvalue;
    double condition_number;
};

class Matrix {
private:
    std::vector<std::vector<double>> data;
    int rows, cols;
    
public:
    Matrix(int r, int c) : rows(r), cols(c) {
        if (r < 0 || c < 0)
            throw std::invalid_argument("Matrix dimensions must be non-negative");
        data.resize(rows, std::vector<double>(cols, 0.0));
    }
    
    Matrix(const std::vector<std::vector<double>>& mat) : data(mat) {
        rows = data.size();
        cols = data.empty() ? 0 : data[0].size();
        for (int i = 1; i < rows; ++i)
            if (static_cast<int>(data[i].size()) != cols)
                throw std::invalid_argument("Matrix rows must have equal lengths");
    }
    
    int getRows() const { return rows; }
    int getCols() const { return cols; }

    int numerical_rank(double relative_tolerance = 1e-10) const {
        if (!std::isfinite(relative_tolerance) || relative_tolerance < 0)
            throw std::invalid_argument(
                "Rank tolerance must be finite and non-negative");
        if (rows == 0 || cols == 0)
            return 0;

        std::vector<std::vector<double>> columns(
            cols, std::vector<double>(rows, 0.0));
        double largest_column_norm = 0;
        for (int j = 0; j < cols; ++j)
        {
            double column_norm = 0;
            for (int i = 0; i < rows; ++i)
            {
                if (!std::isfinite(data[i][j]))
                    throw std::invalid_argument(
                        "Rank calculation requires finite matrix values");
                columns[j][i] = data[i][j];
                column_norm = std::hypot(column_norm, data[i][j]);
            }
            largest_column_norm = std::max(largest_column_norm, column_norm);
        }
        if (largest_column_norm == 0)
            return 0;

        const double threshold = relative_tolerance *
            std::max(rows, cols) * largest_column_norm;
        int rank = 0;
        const int steps = std::min(rows, cols);
        for (int step = 0; step < steps; ++step)
        {
            int pivot = step;
            double pivot_norm = 0;
            for (int j = step; j < cols; ++j)
            {
                double norm = 0;
                for (int i = 0; i < rows; ++i)
                    norm = std::hypot(norm, columns[j][i]);
                if (norm > pivot_norm)
                {
                    pivot = j;
                    pivot_norm = norm;
                }
            }
            if (pivot_norm <= threshold)
                break;

            std::swap(columns[step], columns[pivot]);
            for (int i = 0; i < rows; ++i)
                columns[step][i] /= pivot_norm;
            for (int j = step + 1; j < cols; ++j)
            {
                for (int pass = 0; pass < 2; ++pass)
                {
                    double projection = 0;
                    for (int i = 0; i < rows; ++i)
                        projection += columns[step][i] * columns[j][i];
                    for (int i = 0; i < rows; ++i)
                        columns[j][i] -= projection * columns[step][i];
                }
            }
            ++rank;
        }
        return rank;
    }
    
    double& at(int i, int j) { return data[i][j]; }
    double at(int i, int j) const { return data[i][j]; }
    void set(int i, int j, double val) {data[i][j] = val; }
    
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result.data[j][i] = data[i][j];
        return result;
    }

    
    Matrix multiply(const Matrix& other) const {
       if (cols != other.rows)
            throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; ++i)
           for (int k = 0; k < cols; ++k)
               for (int j = 0; j < other.cols; ++j)
                    result.data[i][j] += data[i][k] * other.data[k][j];
        return result;
    }
    
    Matrix inverse() const {
        if (rows != cols)
            throw std::invalid_argument("Only square matrices can be inverted");
        
        int n = rows;
        Matrix aug(n, 2 * n);
        
        // Create augmented matrix [A | I]
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j)
                aug.data[i][j] = data[i][j];
            aug.data[i][n + i] = 1.0;
        }
        
        // Gauss-Jordan elimination
        for (int i = 0; i < n; ++i) {
            // Find pivot
            int maxRow = i;
            for (int k = i + 1; k < n; ++k)
                if (std::abs(aug.data[k][i]) > std::abs(aug.data[maxRow][i]))
                    maxRow = k;
            
            if (std::abs(aug.data[maxRow][i]) < 1e-10)
                throw std::runtime_error("Matrix is singular");
            
            // Swap rows
            std::swap(aug.data[i], aug.data[maxRow]);
            
            // Normalize pivot row
            double pivot = aug.data[i][i];
            for (int j = i; j < 2 * n; ++j)
                aug.data[i][j] /= pivot;
            
            // Eliminate column
            for (int k = 0; k < n; ++k) {
                if (k != i) {
                    double factor = aug.data[k][i];
                    for (int j = i; j < 2 * n; ++j)
                        aug.data[k][j] -= factor * aug.data[i][j];
                }
            }
        }
        
        // Extract inverse
        Matrix inv(n, n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                inv.data[i][j] = aug.data[i][n + j];
        
        return inv;
    }

    PositiveDefiniteDiagnostics positive_definite_diagnostics() const {
        if (rows != cols)
            throw std::invalid_argument(
                "Positive-definite diagnostics require a square matrix");

        const int n = rows;
        if (n == 0)
            throw std::invalid_argument(
                "Positive-definite diagnostics require a non-empty matrix");

        Matrix symmetric = *this;
        double scale = 0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
            {
                if (!std::isfinite(data[i][j]))
                    throw std::runtime_error(
                        "Positive-definite matrix contains a non-finite value");
                scale = std::max(scale, std::abs(data[i][j]));
            }
        if (scale == 0)
        {
            PositiveDefiniteDiagnostics diagnostics;
            diagnostics.minimum_eigenvalue = 0;
            diagnostics.maximum_eigenvalue = 0;
            diagnostics.relative_minimum_eigenvalue = 0;
            diagnostics.condition_number =
                std::numeric_limits<double>::infinity();
            return diagnostics;
        }
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                symmetric.data[i][j] = data[i][j] / scale;

        const double symmetry_tolerance = 1e-12;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
            {
                if (std::abs(data[i][j] / scale - data[j][i] / scale) >
                    symmetry_tolerance)
                    throw std::runtime_error(
                        "Matrix is not symmetric within numerical tolerance");
                const double average =
                    0.5 * (data[i][j] + data[j][i]) / scale;
                symmetric.data[i][j] = average;
                symmetric.data[j][i] = average;
            }

        const double jacobi_tolerance =
            std::numeric_limits<double>::epsilon() * n;
        bool converged = false;
        for (int sweep = 0; sweep < 50; ++sweep)
        {
            double largest_off_diagonal = 0;
            for (int p = 0; p < n - 1; ++p)
            {
                for (int q = p + 1; q < n; ++q)
                {
                    const double off_diagonal = symmetric.data[p][q];
                    largest_off_diagonal =
                        std::max(largest_off_diagonal, std::abs(off_diagonal));
                    if (std::abs(off_diagonal) <= jacobi_tolerance)
                        continue;

                    const double tau =
                        (symmetric.data[q][q] - symmetric.data[p][p]) /
                        (2.0 * off_diagonal);
                    const double tangent =
                        (tau >= 0 ? 1.0 : -1.0) /
                        (std::abs(tau) + std::sqrt(1.0 + tau * tau));
                    const double cosine =
                        1.0 / std::sqrt(1.0 + tangent * tangent);
                    const double sine = tangent * cosine;
                    const double pp = symmetric.data[p][p];
                    const double qq = symmetric.data[q][q];
                    symmetric.data[p][p] = pp - tangent * off_diagonal;
                    symmetric.data[q][q] = qq + tangent * off_diagonal;
                    symmetric.data[p][q] = 0;
                    symmetric.data[q][p] = 0;

                    for (int k = 0; k < n; ++k)
                    {
                        if (k == p || k == q)
                            continue;
                        const double kp = symmetric.data[k][p];
                        const double kq = symmetric.data[k][q];
                        symmetric.data[k][p] = cosine * kp - sine * kq;
                        symmetric.data[p][k] = symmetric.data[k][p];
                        symmetric.data[k][q] = sine * kp + cosine * kq;
                        symmetric.data[q][k] = symmetric.data[k][q];
                    }
                }
            }
            if (largest_off_diagonal <= jacobi_tolerance)
            {
                converged = true;
                break;
            }
        }
        if (!converged)
            throw std::runtime_error(
                "Symmetric eigenvalue estimate did not converge");

        double minimum_eigenvalue = symmetric.data[0][0];
        double maximum_eigenvalue = symmetric.data[0][0];
        for (int i = 1; i < n; ++i)
        {
            minimum_eigenvalue =
                std::min(minimum_eigenvalue, symmetric.data[i][i]);
            maximum_eigenvalue =
                std::max(maximum_eigenvalue, symmetric.data[i][i]);
        }
        if (!std::isfinite(minimum_eigenvalue) ||
            !std::isfinite(maximum_eigenvalue))
            throw std::runtime_error("Eigenvalue estimate is not finite");

        PositiveDefiniteDiagnostics diagnostics;
        diagnostics.minimum_eigenvalue = minimum_eigenvalue * scale;
        diagnostics.maximum_eigenvalue = maximum_eigenvalue * scale;
        diagnostics.relative_minimum_eigenvalue = maximum_eigenvalue > 0
            ? minimum_eigenvalue / maximum_eigenvalue
            : -std::numeric_limits<double>::infinity();
        diagnostics.condition_number =
            minimum_eigenvalue > 0 && maximum_eigenvalue > 0
                ? maximum_eigenvalue / minimum_eigenvalue
                : std::numeric_limits<double>::infinity();
        return diagnostics;
    }

    Matrix solve_positive_definite(
        const Matrix &right_hand_side,
        bool check_condition = true) const {
        if (rows != cols)
            throw std::invalid_argument(
                "Positive-definite solve requires a square coefficient matrix");
        if (right_hand_side.rows != rows)
            throw std::invalid_argument(
                "Right-hand side has incompatible dimensions");

        const int n = rows;
        double scale = 0;
        if (check_condition)
        {
            const PositiveDefiniteDiagnostics diagnostics =
                positive_definite_diagnostics();
            if (diagnostics.minimum_eigenvalue <= 0 ||
                diagnostics.relative_minimum_eigenvalue <= 1e-12 ||
                diagnostics.condition_number > 1e12)
                throw std::runtime_error(
                    "Matrix is too ill-conditioned for a stable SPD solve "
                    "(relative minimum eigenvalue " +
                    std::to_string(diagnostics.relative_minimum_eigenvalue) +
                    ", condition number " +
                    std::to_string(diagnostics.condition_number) + ")");
            scale = diagnostics.maximum_eigenvalue;
        }
        else
        {
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                {
                    if (!std::isfinite(data[i][j]))
                        throw std::runtime_error(
                            "Positive-definite matrix contains a non-finite value");
                    scale = std::max(scale, std::abs(data[i][j]));
                }
            if (scale == 0)
                throw std::runtime_error(
                    "Cholesky factorization failed for a zero matrix");
        }
        Matrix lower(n, n);
        const double symmetry_tolerance = scale * 1e-12;
        const double pivot_tolerance = check_condition ? 0 : scale * 1e-12;
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j <= i; ++j)
            {
                if (std::abs(data[i][j] - data[j][i]) > symmetry_tolerance)
                    throw std::runtime_error(
                        "Matrix is not symmetric within numerical tolerance");
                double value = 0.5 * (data[i][j] + data[j][i]);
                for (int k = 0; k < j; ++k)
                    value -= lower.data[i][k] * lower.data[j][k];
                if (i == j)
                {
                    if (!std::isfinite(value) || value <= pivot_tolerance)
                        throw std::runtime_error(
                            "Cholesky factorization failed for SPD matrix "
                            "or detected an unstable pivot");
                    lower.data[i][j] = std::sqrt(value);
                }
                else
                {
                    lower.data[i][j] = value / lower.data[j][j];
                }
            }
        }

        Matrix solution = right_hand_side;
        for (int i = 0; i < n; ++i)
        {
            for (int k = 0; k < i; ++k)
            {
                const double factor = lower.data[i][k];
                for (int column = 0; column < right_hand_side.cols; ++column)
                    solution.data[i][column] -=
                        factor * solution.data[k][column];
            }
            const double diagonal = lower.data[i][i];
            for (int column = 0; column < right_hand_side.cols; ++column)
                solution.data[i][column] /= diagonal;
        }
        for (int i = n - 1; i >= 0; --i)
        {
            for (int k = i + 1; k < n; ++k)
            {
                const double factor = lower.data[k][i];
                for (int column = 0; column < right_hand_side.cols; ++column)
                    solution.data[i][column] -=
                        factor * solution.data[k][column];
            }
            const double diagonal = lower.data[i][i];
            for (int column = 0; column < right_hand_side.cols; ++column)
                solution.data[i][column] /= diagonal;
        }
        return solution;
    }

    Matrix inverse_positive_definite() const {
        Matrix identity(rows, rows);
        for (int i = 0; i < rows; ++i)
            identity.data[i][i] = 1;
        return solve_positive_definite(identity);
    }
    
    void print() const {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j)
                std::cout << data[i][j] << " ";
            std::cout << "\n";
        }
    }
    void Add(const Matrix& other, double coef = 1)
    {
        if (cols != other.cols || rows != other.rows)
            throw std::invalid_argument("Matrix dimensions of two summands are unequal");
        // if (cols != other.rows)
        //     throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                data[i][j] += coef*other.data[i][j];
    }
    double Frobenius() const
    {
        double retval = 0;
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                retval += data[i][j] * data[i][j];
        return std::sqrt(retval);
    }

    void by_scalar(double coef)
    {
         for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                data[i][j] *= coef;
    }
};

#endif