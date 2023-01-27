//
// Created by jordan on 1/27/23.
//
#include <stdio.h>
#include <fstream>
#include <string>
#include <array>
#include <chrono>
#include <cmath>
#include "/opt/xtal/ccp4-8.0/include/clipper/clipper-contrib.h"
#include "/opt/xtal/ccp4-8.0/include/clipper/clipper-minimol.h"
#include "/opt/xtal/ccp4-8.0/include/clipper/clipper-ccp4.h"
#include "gradient.h"
#include "model.h"
#include "probe.h"
#include <vector>
#include <iostream>

#ifndef HASH_ANALYSIS_HASH__H
#define HASH_ANALYSIS_HASH_H

class PixelData {
public:

    PixelData();
    PixelData(float data, float u, float v, float w): m_data(data), m_u(u), m_v(v), m_w(w) {}
    PixelData(float data, float u, float v, float w, float x, float y, float z): m_data(data), m_u(u), m_v(v), m_w(w), m_x(x), m_y(y), m_z(z) {}

    void set_data(float& data) {m_data = data;}
    float data() {return m_data;}
    float u() {return m_u;}
    float v() {return m_v;}
    float w() {return m_w;}
    void set_u(float u) { m_u = u;}
    void set_v(float v) { m_v = v;}
    void set_w(float w) { m_w = w;}

    void set_x(float x) { m_x = x;}
    void set_y(float y) { m_y = y;}
    void set_z(float z) { m_z = z;}

    void print() {
        std::cout << m_u << " " << m_v << " " << m_w << " " << m_data << std::endl;
    };

    clipper::Xmap<float>* xmap_ptr{};
    float m_data = 0.0f;
private:
    float m_u = 0.0f;
    float m_v = 0.0f;
    float m_w = 0.0f;
    float m_x = 0.0f;
    float m_y = 0.0f;
    float m_z = 0.0f;
};

struct GradientData {
    float m_angle = 0.0f;
    float m_magnitude = 0.00f;
    float m_theta = 0.0f;
    float m_psi = 0.0f;
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
            for (const auto& j: i) {
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


class Density {

public:

    friend class Model;
    friend class Gradient;
    typedef std::vector<std::vector<std::vector<PixelData>>> PixelMap;
    Density() {}


//    LOADING FUNCTIONS

    //!
    //! \param file_path
    //!
    //!Takes in mtz input file and deposits the resulting clipper::Xmap<float> to the xmap public variable xmap.
    //!
    void load_file(std::string file_path);


    //!
    //! \param xmap
    //! \return PixelMap
    //!
    //! Takes in xmap and converts to PixelMap (3D vector of PixelData)
    //! 
    PixelMap extract_data(clipper::Xmap<float> xmap);

//    SLICING FUNCTIONS
    void slice(float slice_index);
    void dump_slice(std::string file_name);
    void dump_slice(std::string file_name, std::vector<PixelData> data);

//    KERNELS
    float gaussian_1d(float x, float sigma);
    float gaussian_2d(float x, float y, float sigma);
    float gaussian_3d(float x, float y, float z, float sigma);
    Matrix_2D generate_gaussian_kernel_2d(int sigma);


    //!
    //! \param sigma
    //! \param matrix_size
    //! \return Matrix_3D
    //!
    //! Generates a 3D matrix of matrix_size size which contains a Gaussian distribution with the specified sigma
    //! /*
    Matrix_3D generate_gaussian_kernel_3d(int sigma, int matrix_size);

//    FUNCTIONS
    float convolute_2D(Matrix_2D& kernel, Matrix_2D& base);
    std::vector<PixelData> apply_gaussian_2d(Matrix_2D kernel);
    PixelMap apply_gaussian_3d(Matrix_3D kernel);

//    GRADIENT CALCULATIONS
    void calculate_gradient(clipper::MMonomer sugar);

    void export_pixelmap(std::string file_name);
    void export_pixelmap(std::string file_name, PixelMap pixel_map);


    std::vector<PixelData> difference_of_gaussian(std::vector<PixelData>& top, std::vector<PixelData>& bottom);
    PixelMap difference_of_gaussian(PixelMap& top, PixelMap& bottom);

    void set_xmap(clipper::Xmap<float>& xmap_in) {xmap = xmap_in; }
    clipper::Xmap<float> xmap;

private:
    std::vector<PixelData> m_slice;
    std::vector<std::vector<float>> m_grid_values;
    std::vector<std::vector<std::vector<float>>> m_grid_values_3d;
    std::vector<std::vector<std::vector<PixelData>>> m_pixel_data;

    clipper::Cell m_cell;
    clipper::Grid_sampling m_gridsampling;
    clipper::Spacegroup m_spacegroup;

};


class Block {

public:
    std::array<std::array<std::array<GradientData, 8>, 8>, 8> m_data;

    int i, j, k;
    bool overflowing = false;

    std::vector<std::pair<int, float>> m_theta_histogram;
    std::vector<std::pair<int, float>> m_psi_histogram;
    std::vector<std::pair<int, float>> histogram;



};

#endif //HASH_ANALYSIS_HASH__H
