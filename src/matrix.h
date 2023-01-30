//
// Created by jordan on 1/24/23.
//
#pragma once


#ifndef HASH_ANALYSIS_MATRIX_H
#define HASH_ANALYSIS_MATRIX_H
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <valarray>
#include <string>
#include <assert.h>

template <typename T>
class Matrix
{
public:
    Matrix<T>();
    Matrix<T>(int rows, int cols);
    Matrix<T>(int rows, int cols, const T&);
    Matrix<T>(int rows, int cols, const std::valarray<std::valarray<T>>& mat);
    Matrix<T>(const Matrix<T>& mat);

    /*
     * Access specific element by row and column
     */
    T operator()(int row, int col) const;

    long size() const;

    Matrix<T>& operator=(const Matrix<T>& mat);
    Matrix<T>& operator=(const T& value);

    Matrix<T> operator+(const Matrix<T>& mat);
    Matrix<T> operator-(const Matrix<T>& mat);
    Matrix<T> operator*(const Matrix<T>& mat);
    Matrix<T> operator/(const Matrix<T>& mat);

    Matrix<T> operator+(const T& value);
    Matrix<T> operator-(const T& value);
    Matrix<T> operator*(const T& value);
    Matrix<T> operator/(const T& value);

    /*
      Adds the row of the mat to all rows of this matrix
    */
    Matrix<T> broadcast(Matrix<T>& mat);
    /*
      Subtracts the row of the mat from all rows of this matrix
    */
    Matrix<T> bSubtract(Matrix<T>& mat);
    /*
      Multiplies the row of the mat with all rows of this matrix
    */
    Matrix<T> bMultiply(Matrix<T>& mat);

    // reverse order divide (eg. value / matrix)
    Matrix<T> div(double);

    Matrix<T>& operator+=(const Matrix<T>& mat);
    Matrix<T>& operator-=(const Matrix<T>& mat);
    Matrix<T>& operator*=(const Matrix<T>& mat);
    Matrix<T>& operator/=(const Matrix<T>& mat);

    void resize(int rows, int cols);

    /*
     * Set value to the individual position in the matrix
     */
    void set(int row, int col, const T&);

/*
	 * Set value to the all positions in the matrix
	 */
    void set(const T&);

    /*
     * Transpose of the matrix
     */
    Matrix<T> transpose() const;

    /*
     * Dot product
     */
    Matrix<T> dot(const Matrix<T>& mat) const;
    Matrix<T> dot2(const Matrix<T>& mat);

    // experimental only! 4 times slower than dot
    Matrix<T> dotTemp(const Matrix<T>& mat);

    Matrix<T> abs() const;
    Matrix<T> exp() const;
    Matrix<T> log() const;
    Matrix<T> log10() const;

    Matrix<T> pow(double) const;
    Matrix<T> sqrt() const;

    Matrix<T> sin() const;
    Matrix<T> cos() const;
    Matrix<T> tan() const;
    Matrix<T> asin() const;
    Matrix<T> acos() const;
    Matrix<T> atan() const;
    Matrix<T> atan2() const;

    Matrix<T> sinh() const;
    Matrix<T> cosh() const;
    Matrix<T> tanh() const;

    /*
      Extracts sub matrix from row_begin to row_end (including)
      returns [row_count, m_cols] matrix
    */
    Matrix<T> extract(int row_begin, int row_count) const;

    /*
      Extracts single column
      returns [m_rows, 1] matrix
    */
    Matrix<T> extract(int col) const;

    /*
     * Normalizes the matrix elements by:
     * value - min / max - min
     */
    void normalize();

    /*
     * Swaps the rows of the matrix
     */
    void swap(int row1, int row2);

    /*
     * minimum value in the matrix
     */
    T min() const;

    /*
     * maximum value in the matrix
     */
    T max() const;

    /*
     * All the elements are summed
     */
    T sum() const;

    /*
     * Mean of the whole matrix
     */
    double mean() const;

    inline int rows() const { return m_rows; }
    inline int cols() const { return m_cols; }

    void print() {
        std::cout << "rows x cols " << m_rows << " " << m_cols << std::endl;

        for(int r = 0; r < rows(); r++) {
            for (int c = 0; c < cols(); c++) {
                std::cout << (*this)(r,c) << " ";
            }
            std::cout << std::endl;
        }

    }

    void print_eqn() {
        std::cout << "z = " << (*this)(0,0) << " * x + " << (*this)(1,0) << " * y + " << (*this)(2,0) << std::endl;
    }

    Matrix<T> inverse() ;


private:
    std::valarray<T> m_data;

    int m_rows;
    int m_cols;
};

template<typename T>
Matrix<T>::Matrix()
{
}

template<typename T>
Matrix<T>::Matrix(int rows, int cols)
        : m_rows(rows), m_cols(cols)
{
    m_data.resize(rows * cols);
}

template<typename T>
Matrix<T>::Matrix(int rows, int cols, const T& value)
        : m_rows(rows), m_cols(cols)
{
    m_data.resize(rows * cols, value);
}

template<typename T>
Matrix<T>::Matrix(int rows, int cols, const std::valarray<std::valarray<T>> & mat)
        : m_rows(rows), m_cols(cols)
{
    m_data.resize(rows * cols);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            set(r, c, mat[r][c]);
        }
    }
}

template<typename T>
Matrix<T>::Matrix(const Matrix<T>& mat)
{
    m_data = mat.m_data;
    m_rows = mat.m_rows;
    m_cols = mat.m_cols;
}

template<typename T>
T Matrix<T>::operator()(int row, int col) const
{
    return m_data[row * m_cols + col];
}

template<typename T>
long Matrix<T>::size() const
{
    return m_cols * m_rows;
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& mat)
{
    m_rows = mat.m_rows;
    m_cols = mat.m_cols;

    m_data = mat.m_data;
    return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(const T& value)
{
    m_data = value;
    return *this;
}

template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& mat)
{
    assert(m_rows == mat.m_rows);
    assert(m_cols == mat.m_cols);

    Matrix<T> result(m_rows, m_cols);
    result.m_data = (m_data + mat.m_data);
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::broadcast(Matrix<T>& mat)
{
    assert(m_cols == mat.m_cols);

    Matrix<T> result(m_rows, m_cols);
    result.m_data = m_data;

    auto broadcast_valarray = mat.m_data[std::slice(0, m_cols, 1)];

    for (int i = 0; i < m_rows; i++) {
        result.m_data[std::slice(i * m_cols, m_cols, 1)] += broadcast_valarray;
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::bSubtract(Matrix<T>& mat)
{
    assert(m_cols == mat.m_cols);

    Matrix<T> result(m_rows, m_cols);
    result.m_data = m_data;

    auto broadcast_valarray = mat.m_data[std::slice(0, m_cols, 1)];

    for (int i = 0; i < m_rows; i++) {
        result.m_data[std::slice(i * m_cols, m_cols, 1)] -= broadcast_valarray;
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::bMultiply(Matrix<T>& mat)
{
    assert(m_cols == mat.m_cols);

    Matrix<T> result(m_rows, m_cols);
    result.m_data = m_data;

    auto broadcast_valarray = mat.m_data[std::slice(0, m_cols, 1)];

    for (int i = 0; i < m_rows; i++) {
        result.m_data[std::slice(i * m_cols, m_cols, 1)] *= broadcast_valarray;
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& mat)
{
    assert(m_rows == mat.m_rows);
    assert(m_cols == mat.m_cols);

    Matrix<T> result(m_rows, m_cols);
    result.m_data = (m_data - mat.m_data);
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& mat)
{
    assert(m_rows == mat.m_rows);
    assert(m_cols == mat.m_cols);

    Matrix<T> result(m_rows, m_cols);
    result.m_data = (m_data * mat.m_data);
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator/(const Matrix<T>& mat)
{
    assert(m_rows == mat.m_rows);
    assert(m_cols == mat.m_cols);

    Matrix<T> result(m_rows, m_cols);
    result.m_data = (m_data / mat.m_data);
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator+(const T& value)
{
    Matrix<T> result(m_rows, m_cols);
    result.m_data = (m_data + value);
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(const T& value)
{
    Matrix<T> result(m_rows, m_cols);
    result.m_data = (m_data - value);
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const T& value)
{
    Matrix<T> result(m_rows, m_cols);
    result.m_data = (m_data * value);
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator/(const T& value)
{
    Matrix<T> result(m_rows, m_cols);
    result.m_data = (m_data / value);
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::div(double value)
{
    Matrix<T> result(m_rows, m_cols);
    result.m_data = (value / m_data);
    return result;
}

template<typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& mat)
{
    assert(m_rows == mat.m_rows);
    assert(m_cols == mat.m_cols);

    m_data += mat.m_data;
    return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& mat)
{
    assert(m_rows == mat.m_rows);
    assert(m_cols == mat.m_cols);

    m_data -= mat.m_data;
    return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator*=(const Matrix<T>& mat)
{
    assert(m_rows == mat.m_rows);
    assert(m_cols == mat.m_cols);

    m_data *= mat.m_data;
    return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator/=(const Matrix<T>& mat)
{
    assert(m_rows == mat.m_rows);
    assert(m_cols == mat.m_cols);

    m_data /= mat.m_data;
    return *this;
}

template<typename T>
void Matrix<T>::resize(int rows, int cols)
{
    m_rows = rows;
    m_cols = cols;

    m_data.resize(rows * cols);
}

template<typename T>
void Matrix<T>::set(int row, int col, const T& value)
{
    m_data[row * m_cols + col] = value;
}

template<typename T>
void Matrix<T>::set(const T& value)
{
    m_data.fill(value);
}

template<typename T>
Matrix<T> Matrix<T>::transpose() const
{
    Matrix<T> result(m_cols, m_rows);
    for (auto r = 0; r < m_rows; r++)
    {
        for (auto c = 0; c < m_cols; c++)
        {
            //result.m_data[c * m_cols + r] = m_data[r * m_cols + c];
            T value = operator()(r, c);
            result.set(c, r, value);
        }
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::extract(int row_begin, int row_count) const
{
    int index_begin = row_begin * m_cols;
    int number_of_elements = row_count * m_cols;

    Matrix<T> result(row_count, m_cols);
    result.m_data = m_data[std::slice(index_begin, number_of_elements, 1)];
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::extract(int col) const
{
    Matrix<T> result(m_rows, 1);
    result.m_data = m_data[std::slice(col, m_rows, m_cols)];
    return result;
}

template<typename T>
void Matrix<T>::normalize()
{
    const T _min = min();
    const T _max = max();
    const T min_max_diff = _max - _min;
    for (auto r = 0; r < m_rows; r++)
    {
        for (auto c = 0; c < m_cols; c++)
        {
            T value = operator()(r, c);
            set(r, c, (value - _min)/min_max_diff);
        }
    }
}

template<typename T>
void Matrix<T>::swap(int row1, int row2)
{
    assert(m_rows > row1);
    assert(m_rows > row2);

    int ind_start = row1 * m_cols;
    int ind_end = ind_start + m_cols;
    int swp_ind_start = row2 * m_cols;
    std::swap_ranges(std::begin(m_data) + ind_start, std::begin(m_data) + ind_end, std::begin(m_data) + swp_ind_start);
}

template<typename T>
T Matrix<T>::min() const
{
    return m_data.min();
}

template<typename T>
T Matrix<T>::max() const
{
    return m_data.max();
}

template<typename T>
T Matrix<T>::sum() const
{
    return m_data.sum();
}

template<typename T>
double Matrix<T>::mean() const
{
    return (double)sum() / m_data.size();
}

template<typename T>
Matrix<T> Matrix<T>::dotTemp(const Matrix<T> &mat)
{
    Matrix<T> result = Matrix(m_rows, mat.m_cols);
    for(auto r = 0; r < m_rows; r++) {
        for(auto c = 0; c < mat.m_cols; c++) {
            std::valarray<T> row = m_data[std::slice(r * m_cols, m_cols, 1)];
            std::valarray<T> col = mat.m_data[std::slice(c, mat.m_rows, mat.m_cols)];
            std::valarray<T> mul = row * col;
            const T& sum = mul.sum();
            result.set(r, c, sum);
        }
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::dot(const Matrix<T> &mat) const
{
    Matrix<T> dest = Matrix(m_rows, mat.m_cols);

    Matrix<T> tr = mat.transpose();

    for (int r = 0; r < dest.m_rows; r++) {
        for (int c = 0; c < dest.m_cols; c++) {
            T sum = 0;
            for (int k = 0; k < m_cols; k++)
                sum += m_data[r * m_cols + k] * tr.m_data[c * tr.m_cols + k];
            //sum += m_data[r * m_cols + k] * tr(c, k);
            dest.m_data[r * dest.m_cols + c] = sum;
        }
    }
    return dest;
}

template<typename T>
Matrix<T> Matrix<T>::dot2(const Matrix<T> &mat)
{

    Matrix<T> result = Matrix(m_rows, mat.m_cols);

    for(auto r = 0; r < m_rows; r++) {
        for(auto k = 0; k < m_cols; k++) {
            double total = 0;
            for(auto c = 0; c < mat.m_cols; c++) {
                result.m_data[r * mat.m_cols + c] += (m_data[r * m_cols + k] * mat.m_data[k * mat.m_cols + c]);
            }
        }
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::abs() const
{
    Matrix<T> result(m_rows, m_cols);
    result.m_data = std::abs(m_data);
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::exp() const
{
    Matrix<T> result(m_rows, m_cols);
    result.m_data = std::exp(m_data);
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::log() const
{
    Matrix<T> result(m_rows, m_cols);
    result.m_data = std::log(m_data);
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::log10() const
{
    Matrix<T> result(m_rows, m_cols);
    result.m_data = std::log10(m_data);
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::pow(double value) const
{
    Matrix<T> result(m_rows, m_cols);
    result.m_data = std::pow(m_data, (T)value);
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::sqrt() const
{
    Matrix<T> result(m_rows, m_cols);
    result.m_data = std::sqrt(m_data);
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::sin() const
{
    Matrix<T> result(m_rows, m_cols);
    result.m_data = std::sin(m_data);
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::cos() const

{
    Matrix<T> result(m_rows, m_cols);
    result.m_data = std::cos(m_data);
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::tan() const
{
    Matrix<T> result(m_rows, m_cols);
    result.m_data = std::tan(m_data);
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::asin() const
{
    Matrix<T> result(m_rows, m_cols);
    result.m_data = std::asin(m_data);
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::acos() const
{
    Matrix<T> result(m_rows, m_cols);
    result.m_data = std::acos(m_data);
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::atan() const
{
    Matrix<T> result(m_rows, m_cols);
    result.m_data = std::atan(m_data);
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::atan2() const
{
    Matrix<T> result(m_rows, m_cols);
    result.m_data = std::atan2(m_data);
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::sinh() const
{
    Matrix<T> result(m_rows, m_cols);
    result.m_data = std::sinh(m_data);
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::cosh() const
{
    Matrix<T> result(m_rows, m_cols);
    result.m_data = std::cosh(m_data);
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::tanh() const
{
    Matrix<T> result(m_rows, m_cols);
    result.m_data = std::tanh(m_data);
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::inverse() {

    auto m = *this;

    double det = m(0, 0) * (m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2)) -
                 m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) +
                 m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));

    double invdet = 1 / det;

    Matrix<T> minv(3, 3); // inverse of matrix m
    minv.set(0, 0, ((m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2)) * invdet));
    minv.set(0, 1, ((m(0, 2) * m(2, 1) - m(0, 1) * m(2, 2)) * invdet));
    minv.set(0, 2, ((m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) * invdet));
    minv.set(1, 0, ((m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2)) * invdet));
    minv.set(1, 1, ((m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0)) * invdet));
    minv.set(1, 2, ((m(1, 0) * m(0, 2) - m(0, 0) * m(1, 2)) * invdet));
    minv.set(2, 0, ((m(1, 0) * m(2, 1) - m(2, 0) * m(1, 1)) * invdet));
    minv.set(2, 1, ((m(2, 0) * m(0, 1) - m(0, 0) * m(2, 1)) * invdet));
    minv.set(2, 2, ((m(0, 0) * m(1, 1) - m(1, 0) * m(0, 1)) * invdet));

    return minv;
}


template<typename T1, typename T2>
auto operator* (const Matrix<T1>& a, const Matrix<T2>& b) -> Matrix<decltype(T1{} * T2{})>
{
    if (a.get_height() != b.get_width())
    {
        std::stringstream ss;
        ss << "Matrix dimension mismatch: ";
        ss << a.get_height();
        ss << " x ";
        ss << a.get_width();
        ss << " times ";
        ss << b.get_height();
        ss << " x ";
        ss << b.get_width();
        ss << ".";
        throw std::runtime_error(ss.str());
    }

    using value_type = decltype(T1{} + T2{});
    Matrix<decltype(T1{} * T2{})> result(a.get_height(), b.get_width());

    for (size_t rowa = 0; rowa != a.get_height(); ++rowa)
    {
        for (size_t colb = 0; colb != b.get_width(); ++colb)
        {
            value_type sum = 0;

            for (size_t i = 0; i != a.get_width(); ++i)
            {
                sum += a[rowa][i] * b[i][colb];
            }

            result[rowa][colb] = sum;
        }
    }

    return result;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, Matrix<T> m)
{
    size_t maximum_entry_length = 0;

    for (size_t row = 0; row < m.get_height(); ++row)
    {
        for (size_t col = 0; col < m.get_width(); ++col)
        {
            std::stringstream ss;
            ss << m[row][col];
            std::string entry_text;
            ss >> entry_text;
            maximum_entry_length = std::max(maximum_entry_length,
                                            entry_text.length());
        }
    }

    for (size_t row = 0; row < m.get_height(); ++row)
    {
        for (size_t col = 0; col < m.get_width(); ++col)
        {
            os << std::setw((int) maximum_entry_length) << m[row][col];

            if (col < m.get_width() - 1)
            {
                os << ' ';
            }
        }

        if (row < m.get_height() - 1)
        {
            os << '\n';
        }
    }

    return os;
}
#endif //HASH_ANALYSIS_MATRIX_H
