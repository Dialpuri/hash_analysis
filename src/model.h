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
#include "vec3.h"
#include "matrix.h"


class Model {

    friend class Density;

public:
    Model(clipper::Xmap<float> xmap): m_xmap(xmap)  {};

    void load_model(std::string file_path);
    void prepare_sugars();

    Matrix<float> calculate_plane_eqn(clipper::Atom_list atoms);

    clipper::MiniMol m_model;
private:
    clipper::Xmap<float> m_xmap;
};

#endif //HASH_ANALYSIS_MODEL_H
