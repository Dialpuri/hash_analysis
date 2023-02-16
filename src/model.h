//
// Created by jordan on 23/01/23.
//

#ifndef HASH_ANALYSIS_MODEL_H
#define HASH_ANALYSIS_MODEL_H

#include <stdio.h>
#include <fstream>
#include <string>
#include <array>
#include <chrono>
#include <cmath>
#include "/opt/xtal/ccp4-8.0/include/clipper/clipper-contrib.h"
#include "/opt/xtal/ccp4-8.0/include/clipper/clipper-minimol.h"
#include "/opt/xtal/ccp4-8.0/include/clipper/clipper-ccp4.h"
#include "/opt/xtal/ccp4-8.0/include/clipper/core/map_interp.h"

#include "vec3.h"
#include "matrix.h"


class Model {

    friend class Density;

public:
    Model(clipper::Xmap<float>* xmap): m_xmap_ptr(xmap)  {};

    void load_model(std::string file_path);
    void prepare_sugars();

    //!
    //! \param clipper::Atom_list& atoms
    //! \return Matrix<float>
    //!
    //! Calculates the plane equation in form Ax + By + C = z and returns matrix
    //! [ A , B , C ] vertically.
    Matrix<float> calculate_plane_eqn(clipper::Atom_list atoms);

    clipper::MiniMol m_model;
    clipper::Xmap<float>* m_xmap_ptr;
private:
};

#endif //HASH_ANALYSIS_MODEL_H
