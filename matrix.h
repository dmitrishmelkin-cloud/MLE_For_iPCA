#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
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
            for (int j = 0; j < other.cols; ++j)
                for (int k = 0; k < cols; ++k)
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
            retval += data[i][j]*data[i][j];
        return retval;
    }

    void by_scalar(double coef)
    {
         for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                data[i][j] *= coef;
    }
};

#endif