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

    //!
    //! \param Density::PixelMap& image
    //!
    //! Constructor which takes in a PixelMap and creates a GradientMap of the same size and deposits it into the private variable m_gradient_map.
    //! Constructor also stores PixelMap in m_image private variable
    //!
    Gradient(Density::PixelMap& image): m_image(image) {
        std::cout << image.size() << " " << image[0].size() << " " << image[0][0].size() << std::endl;
        GradientMap generated_map(image.size(), std::vector<std::vector<GradientData>>(image[0].size(), std::vector<GradientData>(image[0][0].size())));
        m_gradient_map = generated_map;
    };

    //!
    //! Takes private m_image in Gradient class and iterates through each pixel and calculates GradientData which is then deposited into m_gradient_map
    //!
    void calculate_gradient();

    //!
    //! \return Blocks (3D vector of Block)
    //!
    //! Takes m_gradient_map and creates 8x8x8 blocks, with each Block storing each GradientData point. Returns a 3D vector of Block called Blocks which contains all Blocks in a map
    //!
    Blocks transform_to_blocks();

    //! OLD - DO NOT USE
    //! \param blocks
    //!
    void calculate_histograms(Blocks& blocks);

    //!
    //! \param Model& model
    //! \param Density& dens
    //! \return Gradient::Block_list
    //!
    //! Takes a model which can contain any number of polymers and monomers and calls the xmap overload of calculate_histograms
    Gradient::Block_list calculate_histograms(Model& model, Density& dens);

    //!
    //! \param Model& model
    //! \param clipper::Xmap<float>& xmap
    //! \return Gradient::Block_list
    //!
    //! Takes a model which can contain any number of polymers and monomers and calculate a 8x8x8 cube around the central point of supplied monomer.
    //! Calculates GradientData for each point in the surrounding box and then the corresponding psi and theta histograms.
    //! Returns a list of Block called Block_list which contains all the blocks (surrounding boxes) around each monomer.
    //!
    Gradient::Block_list calculate_histograms(Model& model, clipper::Xmap<float> &xmap);

    //!
    //! \param Gradient::Block_list& blocks
    //! \param std::string file_name
    //! \param std::string path
    //!
    //! Writes three histograms (theta, psi, theta+psi) to path->[theta/psi/2d]
    //!
    void write_histogram_data(Gradient::Block_list &blocks, std::string file_name, std::string path);

    //!
    //! \param clipper::Xmap<float>& xmap
    //! \param int lower_u
    //! \param int lower_v
    //! \param int lower_w
    //! \param int upper_u
    //! \param int upper_v
    //! \param int upper_w
    //! \return clipper::MMonomer
    //!
    //! Returns a clipper::MMonomer which has hydrogen atoms at each of the bounding box points for use in visualistaion in Coot
    //!
    clipper::MMonomer return_bounding_box(clipper::Xmap<float>& xmap, int lower_u, int lower_v, int lower_w, int upper_u, int upper_v, int upper_w);

private:
    Density::PixelMap& m_image;
    GradientMap m_gradient_map;

    //!
    //! \param int nu
    //! \param int nv
    //! \param int nw
    //! \param int local_u
    //! \param int local_v
    //! \param int local_w
    //! \return GradientData
    //!
    //! Calculates GradientData (data from single point) from a specified point, uses m_image from Gradient class (must be initialised earlier)
    GradientData calculate_gradient_data(int nu, int nv, int nw, int local_u, int local_v, int local_w);
};


#endif //HASH_ANALYSIS_DENSITY_H
