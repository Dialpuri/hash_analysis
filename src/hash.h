//
// Created by Jordan Dialpuri on 08/01/2023.
//

#include <stdio.h>
#include <fstream>
#include <string>
#include <array>
#include <chrono>
#include <cmath>
#include <clipper/clipper-contrib.h>
#include <clipper/clipper-ccp4.h>
#include <clipper/clipper-minimol.h>


#ifndef HASH_TEST_HASH_H
#define HASH_TEST_HASH_H

class PixelData {
public:

    PixelData(float data, float u, float v, float w): m_data(data), m_u(u), m_v(v), m_w(w) {}

    float set_data(float& data) {m_data = data;}
    float data() {return m_data;}
    float u() {return m_u;}
    float v() {return m_v;}
    float w() {return m_w;}

    void print() {
        std::cout << m_u << " " << m_v << " " << m_w << " " << m_data << std::endl;
    };

    float m_data = 0.0f;
private:
    float m_u = 0.0f;
    float m_v = 0.0f;
    float m_w = 0.0f;
};

class Matrix_2D {
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

class Matrix_3D {
public:

    Matrix_3D(int n) {
        for (int i = 0 ; i < n; i++) {
            std::vector<std::vector<float>> tmp_j;
            for (int j = 0; j < n; j++) {
                std::vector<float> tmp_k;
                for (int k = 0; k < n; k++) {
                    tmp_k.push_back(0.0f);
                }
                tmp_j.push_back(tmp_k);
            }
            m_matrix.push_back(tmp_j);
        }
    }

//    std::array<std::array<std::array<float, 3>, 3>, 3> m_matrix;

    void print() {
        for (auto i: m_matrix) {
            for (auto j: i) {
                for(auto k: j) {
                    std::cout << k << " ";
                }
                std::cout << '\n';
            }
            std::cout << std::endl;
        }
    }

    std::vector<std::vector<std::vector<float>>> m_matrix;

};


class Hasher {

public:
    typedef std::vector<std::vector<std::vector<PixelData>>> PixelMap;
    Hasher() {}

//    LOADING FUNCTIONS
    void load_file(std::string file_path);
    void extract_data();

//    SLICING FUNCTIONS
    void slice(float slice_index);
    void dump_slice(std::string file_name);
    void dump_slice(std::string file_name, std::vector<PixelData> data);

//    KERNELS
    float gaussian_1d(float x, float sigma);
    float gaussian_2d(float x, float y, float sigma);
    float gaussian_3d(float x, float y, float z, float sigma);
    Matrix_2D generate_gaussian_kernel_2d(int sigma);
    Matrix_3D generate_gaussian_kernel_3d(int sigma, int matrix_size);

//    FUNCTIONS
    float convolute_2D(Matrix_2D& kernel, Matrix_2D& base);
    std::vector<PixelData> apply_gaussian_2d(Matrix_2D kernel);
    PixelMap apply_gaussian_3d(Matrix_3D kernel);

    void export_pixelmap(std::string file_name);
    void export_pixelmap(std::string file_name, PixelMap pixel_map);


    std::vector<PixelData> difference_of_gaussian(std::vector<PixelData>& top, std::vector<PixelData>& bottom);
    PixelMap difference_of_gaussian(PixelMap& top, PixelMap& bottom);




private:
    clipper::Xmap<float> xmap;
    std::vector<PixelData> m_slice;
    std::vector<std::vector<float>> m_grid_values;
    std::vector<std::vector<std::vector<float>>> m_grid_values_3d;
    std::vector<std::vector<std::vector<PixelData>>> m_pixel_data;

    clipper::Cell m_cell;
    clipper::Grid_sampling m_gridsampling;
    clipper::Spacegroup m_spacegroup;

};


#endif //HASH_TEST_HASH_H
