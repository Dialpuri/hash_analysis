//
// Created by jordan on 12/14/22.
//

#ifndef PROBE_POINTS_PROBE_H
#define PROBE_POINTS_PROBE_H
#include <iostream>
#include <string>
#include <fstream>
#include "/opt/xtal/ccp4-8.0/include/clipper/clipper-ccp4.h"
#include "/opt/xtal/ccp4-8.0/include/clipper/clipper-contrib.h"
#include "/opt/xtal/ccp4-8.0/include/clipper/clipper-minimol.h"
#include "/opt/xtal/ccp4-8.0/include/clipper/core/container_map.h"
#include <map>
#include <stdexcept>
#include <array>


class LibraryItem {

public:
    LibraryItem(std::string pdb_code, std::string pdb_base_dir, std::string pdb_file_ending,
                bool use_experimental_data);
    clipper::MiniMol load_pdb();

    clipper::Cell return_cell(clipper::MMonomer& monomer);

    clipper::Coord_orth calculate_center_point(std::vector<clipper::MAtom> &atoms);
    clipper::RTop_orth align_fragment(clipper::MMonomer& monomer);

    void convert_map_to_array(clipper::Xmap<float> &xmap);
    void calculate_electron_density(clipper::MiniMol& test_mol);

    void dump_minimol(clipper::MiniMol& output_model, std::string file_path, std::string file_name);
    void dump_electron_density(std::string path);
    std::vector<std::pair<clipper::MMonomer, clipper::Xmap<float>>> m_density;
    std::pair<clipper::MiniMol, clipper::Xmap<float>*> model_pair;

    std::string get_pdb_code() {return m_pdb_code;}

    void load_reflections(clipper::MiniMol &mol);

    const std::string m_pdb_base_dir;
    clipper::Xmap<float> m_xmap;
private:
    std::string m_pdb_code;

    std::string m_pdb_file_path;
    std::string m_pdb_file_ending;

};

class Library {

public:
    Library();
    Library(std::string library_file_path, std::string pdb_base_dir, bool use_experimental_data,
            std::string pdb_file_ending);

    std::vector<LibraryItem> read_library_item();

    void combine_density();
    std::vector <LibraryItem> m_library;

private:
    std::string m_library_path;
    std::string m_pdb_base_dir;
    std::string m_pdb_file_ending;
    bool m_use_experimental_data;
};

#endif //PROBE_POINTS_PROBE_H