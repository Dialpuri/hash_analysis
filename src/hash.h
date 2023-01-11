//
// Created by Jordan Dialpuri on 08/01/2023.
//

#include <stdio.h>
#include <fstream>
#include <string>
#include <array>
#include <cmath>
#include <clipper/clipper-contrib.h>
#include <clipper/clipper-ccp4.h>
#include <clipper/clipper-minimol.h>


#ifndef HASH_TEST_HASH_H
#define HASH_TEST_HASH_H

class SliceData {
public:

    SliceData(float data, float u, float v, float w): m_data(data), m_u(u), m_v(v), m_w(w) {}

    float data() {return m_data;}
    float u() {return m_u;}
    float v() {return m_v;}
    float w() {return m_w;}

    void print() {
        std::cout << m_u << " " << m_v << " " << m_w << " " << m_data << std::endl;
    };

private:
    float m_data, m_u, m_v, m_w;
};

class Matrix {
public:

    std::array<std::array<float, 3>, 3> m_matrix;

    void print() {
        for (auto i: m_matrix) {
            for (auto j: i) {
                std::cout << j << " ";
            }
            std::cout << std::endl;
        }
    }
};

class Hasher {
public:
    Hasher() {}

//    LOADING FUNCTIONS
    void load(std::string file_path);

//    SLICING FUNCTIONS
    void slice(float slice_index);
    void dump_slice(std::string file_name);
    void dump_slice(std::string file_name, std::vector<SliceData> data);
//    KERNELS
    float gaussian_1d(float x, float sigma);
    float gaussian_2d(float x, float y, float sigma);
    float gaussian_3d(float x, float y, float z, float sigma);
    Matrix generate_gaussian_kernel(int sigma);

//    FUNCTIONS
    float convolute(Matrix& kernel, Matrix& base);
    std::vector<SliceData> apply_gaussian(Matrix kernel);
    std::vector<SliceData> difference_of_gaussian(std::vector<SliceData>& top, std::vector<SliceData>& bottom);


private:
    clipper::Xmap<float> xmap;
    std::vector<SliceData> m_slice;
    std::vector<std::vector<float>> m_grid_values;

};


#endif //HASH_TEST_HASH_H
