//
// Created by jordan on 1/18/23.
//

#ifndef HASH_ANALYSIS_DENSITY_H
#define HASH_ANALYSIS_DENSITY_H

#include <stdio.h>
#include <fstream>
#include <string>
#include <array>
#include <chrono>
#include <cmath>
#include "/opt/xtal/ccp4-8.0/include/clipper/clipper-contrib.h"
#include "/opt/xtal/ccp4-8.0/include/clipper/clipper-minimol.h"
#include "/opt/xtal/ccp4-8.0/include/clipper/clipper-ccp4.h"
#include "hash.h"

class Gradient {

public:

    friend class Block;
    friend class Density;

    typedef std::vector<std::vector<std::vector<Block>>> Blocks;
    typedef std::vector<Block> Block_list;

    typedef std::vector<std::vector<std::vector<GradientData>>> GradientMap;

    Gradient(Density::PixelMap& image): m_image(image) {
        std::cout << image.size() << " " << image[0].size() << " " << image[0][0].size() << std::endl;
        GradientMap generated_map(image.size(), std::vector<std::vector<GradientData>>(image[0].size(), std::vector<GradientData>(image[0][0].size())));
        m_gradient_map = generated_map;
    };

    void calculate_gradient();
    Blocks transform_to_blocks();
    void calculate_histograms(Blocks& blocks);

    Gradient::Block_list calculate_histograms(Model& model, Density& dens);
    Gradient::Block_list calculate_histograms(Model& model, clipper::Xmap<float> &xmap);

    void write_histogram_data(Gradient::Block_list &blocks, std::string file_name, std::string path);

    clipper::MMonomer return_bounding_box(clipper::Xmap<float>& xmap, int lower_u, int lower_v, int lower_w, int upper_u, int upper_v, int upper_w);

private:
    Density::PixelMap& m_image;
    GradientMap m_gradient_map;

    GradientData calculate_gradient_data(int nu, int nv, int nw, int local_u, int local_v, int local_w);
};


#endif //HASH_ANALYSIS_DENSITY_H
