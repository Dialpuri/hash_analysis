//
// Created by jordan on 12/14/22.
//

#include "library.h"
#include "hash.h"

LibraryItem::LibraryItem(std::string pdb_code, std::string pdb_base_dir, std::string pdb_file_ending,
                         bool use_experimental_data) {

    m_pdb_code = pdb_code;
    m_pdb_base_dir = pdb_base_dir;
    m_pdb_file_ending = pdb_file_ending;

    m_pdb_file_path = pdb_base_dir + pdb_code + m_pdb_file_ending;

//    std::cout << m_pdb_file_path << std::endl;

    if (use_experimental_data == true) { return; }

    try {
        clipper::MiniMol mol = load_pdb();
        calculate_electron_density(mol);
    }
    catch (std::runtime_error &e) {
        std::cout << pdb_code << " had an issue, skipping... - " << e.what() << std::endl;
    }

//        dump_minimol(mol, "./debug/minimol_dump/", pdb_code);
//        dump_electron_density("./debug/aligned_fragments/");
}

clipper::MiniMol LibraryItem::load_pdb() {

    std::cout << "Loading PDB File " << m_pdb_file_path << std::endl;

    clipper::MMDBfile m_file;
    clipper::MiniMol mol;
    const int mmdbflags =
            ::mmdb::MMDBF_IgnoreBlankLines | ::mmdb::MMDBF_IgnoreDuplSeqNum | ::mmdb::MMDBF_IgnoreNonCoorPDBErrors |
            ::mmdb::MMDBF_IgnoreRemarks;

    m_file.SetFlag(mmdbflags);
    m_file.read_file(m_pdb_file_path);
    m_file.import_minimol(mol);

    if (mol.size() == 0) {
        return clipper::MiniMol();
        throw std::runtime_error(m_pdb_code + " - Mol.size() == 0");
    }

    clipper::MiniMol return_mol = clipper::MiniMol(mol.spacegroup(), mol.cell());

    std::vector<std::string> search_atoms = {" C1'", " C2'", " C3'", " C4'", " C5'", " O3'", " O4'", " O5'", " P  "};

    for (int chain = 0; chain < mol.size(); chain++) {
        for (int residue = 0; residue < mol[chain].size(); residue++) {
            if (
                    mol[chain][residue].lookup(" C1'", clipper::MM::UNIQUE) >= 0 &&
                    mol[chain][residue].lookup(" C2'", clipper::MM::UNIQUE) >= 0 &&
                    mol[chain][residue].lookup(" C3'", clipper::MM::UNIQUE) >= 0 &&
                    mol[chain][residue].lookup(" C4'", clipper::MM::UNIQUE) >= 0 &&
                    mol[chain][residue].lookup(" C5'", clipper::MM::UNIQUE) >= 0 &&
                    mol[chain][residue].lookup(" O3'", clipper::MM::UNIQUE) >= 0 &&
                    mol[chain][residue].lookup(" O4'", clipper::MM::UNIQUE) >= 0 &&
                    mol[chain][residue].lookup(" O5'", clipper::MM::UNIQUE) >= 0 &&
                    mol[chain][residue].lookup(" P  ", clipper::MM::UNIQUE) >= 0
                    ) {
                int atom = mol[chain][residue].lookup(" C4'", clipper::MM::ANY);
                if (mol[chain][residue][atom].occupancy() > 0.01 &&
                    mol[chain][residue][atom].u_iso() < clipper::Util::b2u(100.0)) {

                    clipper::MPolymer mp;
                    clipper::MMonomer monomer;

                    for (int atom_string_index = 0; atom_string_index < search_atoms.size(); atom_string_index++) {
                        std::string atom_string = search_atoms[atom_string_index];
                        if (mol[chain][residue].lookup(atom_string, clipper::MM::UNIQUE) >= 0) {
                            monomer.insert(mol[chain][residue].find(atom_string));
                        }
                    }
                    monomer.set_id("A");
                    monomer.set_type("?");

//                  Center the monomer and add to the return_minimol
                    clipper::Coord_orth origin(0.0, 0.0, 0.0);
                    int number_of_atoms = monomer.size();

                    for (int atom = 0; atom < number_of_atoms; atom++) {
                        origin += monomer[atom].coord_orth();
                    }

                    origin = (-1.0 / number_of_atoms) * origin;
                    clipper::RTop_orth rtop = clipper::RTop_orth(clipper::Mat33<>::identity(), origin);
                    monomer.transform(rtop);
                    mp.insert(monomer);
                    return_mol.insert(mp);

                }
            }
        }
    }

    return return_mol;
}

clipper::Cell LibraryItem::return_cell(clipper::MMonomer &monomer) {

    float min_x = 1e20;
    float min_y = 1e20;
    float min_z = 1e20;
    float max_x = -1e20;
    float max_y = -1e20;
    float max_z = -1e20;

    for (int atom = 0; atom < monomer.size(); atom++) {
        clipper::MAtom current_atom = monomer[atom];
        float x = current_atom.coord_orth().x();
        float y = current_atom.coord_orth().y();
        float z = current_atom.coord_orth().z();

        if (x < min_x) {
            min_x = x;
        }
        if (y < min_y) {
            min_y = y;
        }
        if (z < min_z) {
            min_z = z;
        }

        if (x > max_x) {
            max_x = x;
        }
        if (y > max_y) {
            max_y = y;
        }
        if (z > max_z) {
            max_z = z;
        }
    }

//    std::cout << min_x << " " << max_x << " " << min_y << " " << max_y << " " << min_z << " " << max_z << std::endl;

    clipper::Cell_descr cell_description = clipper::Cell_descr(max_x - min_x + 10, max_y - min_y + 10,
                                                               max_z - min_z + 10);
    return clipper::Cell(cell_description);
}

clipper::Coord_orth LibraryItem::calculate_center_point(std::vector<clipper::MAtom> &atoms) {

    float center_x = 0.0f;
    float center_y = 0.0f;
    float center_z = 0.0f;

    for (int i = 0; i < atoms.size(); i++) {
        float x = atoms[i].coord_orth().x();
        float y = atoms[i].coord_orth().y();
        float z = atoms[i].coord_orth().z();

        center_x = center_x + x;
        center_y = center_y + y;
        center_z = center_z + z;
    }

    float average_x = center_x / atoms.size();
    float average_y = center_y / atoms.size();
    float average_z = center_z / atoms.size();

    clipper::Coord_orth return_coord = clipper::Coord_orth(average_x, average_y, average_z);
    return return_coord;
}

clipper::RTop_orth LibraryItem::align_fragment(clipper::MMonomer &monomer) {

//    ATOM      8  O5'   ?     0       1.070  -0.334   0.726  1.00 17.23           O
//    ATOM      9  P     ?     0       0.593  -1.729   1.377  1.00 20.98           P
//    ATOM      5  C5'   ?     0       2.318  -0.206   0.082  1.00 13.43           C

    clipper::Coord_orth ref_o5 = clipper::Coord_orth(1.070, -0.334, 0.726);
    clipper::Coord_orth ref_p = clipper::Coord_orth(0.593, -1.729, 1.377);
    clipper::Coord_orth ref_c5 = clipper::Coord_orth(2.318, -0.206, 0.082);

    clipper::Coord_orth move = clipper::Coord_orth(10, 10, 9);
    clipper::Coord_orth ref_o5_moved = ref_o5 + move;
    clipper::Coord_orth ref_p_moved = ref_p + move;
    clipper::Coord_orth ref_c5_moved = ref_c5 + move;

    std::vector<clipper::Coord_orth> reference_coords = {ref_o5_moved, ref_p_moved, ref_c5_moved};

    clipper::Coord_orth target_o5 = monomer.find("O5'").coord_orth();
    clipper::Coord_orth target_p = monomer.find("P").coord_orth();
    clipper::Coord_orth target_c5 = monomer.find("C5'").coord_orth();

    std::vector<clipper::Coord_orth> target_coords = {target_o5, target_p, target_c5};

    clipper::RTop_orth rtop = clipper::RTop_orth(target_coords, reference_coords);

    return rtop;
}

void LibraryItem::convert_map_to_array(clipper::Xmap<float> &xmap) {

    clipper::Grid_sampling grid_sampling = xmap.grid_sampling();

//    const int nu = grid_sampling.nu();
//    const int nv = grid_sampling.nv();
//    const int nw = grid_sampling.nw();

    std::vector<std::vector<std::vector<float>>> matrix;

    clipper::Coord_orth base_coord_orth = clipper::Coord_orth(0, 0, 0);
    clipper::Coord_grid base_coord = base_coord_orth.coord_frac(xmap.cell()).coord_grid(grid_sampling);
    clipper::Xmap_base::Map_reference_coord i0 = clipper::Xmap_base::Map_reference_coord(xmap, base_coord);

    clipper::Coord_orth end_coord_orth = clipper::Coord_orth(10, 10, 10);
    clipper::Coord_grid end_coord = end_coord_orth.coord_frac(xmap.cell()).coord_grid(grid_sampling);
    clipper::Xmap_base::Map_reference_coord iend = clipper::Xmap_base::Map_reference_coord(xmap, end_coord);


    clipper::Xmap_base::Map_reference_coord iu, iv, iw;
    for (iu = i0; iu.coord().u() <= iend.coord().u(); iu.next_u()) {
        for (iv = iu; iv.coord().v() <= iend.coord().v(); iv.next_v()) {
            for (iw = iv; iw.coord().w() <= iend.coord().w(); iw.next_w()) {

            }
        }
    }


}

void LibraryItem::calculate_electron_density(clipper::MiniMol &test_mol) {

    for (int n_polymer = 0; n_polymer < test_mol.size(); n_polymer++) {
        for (int n_monomer = 0; n_monomer < test_mol[n_polymer].size(); n_monomer++) {

            clipper::MMonomer tmp_monomer = test_mol[n_polymer][n_monomer];

            clipper::RTop_orth alignment_rtop = align_fragment(tmp_monomer);

            tmp_monomer.transform(alignment_rtop);

            clipper::RTop_orth move_to_center = clipper::RTop_orth(clipper::Mat33<>::identity(),
                                                                   clipper::Vec3<>(10, 10, 10));
            tmp_monomer.transform(move_to_center);

            clipper::Atom_list atom_list = tmp_monomer.atom_list();
            clipper::Cell_descr cell_descr = clipper::Cell_descr(20, 20, 20);
            clipper::Cell cell = clipper::Cell(cell_descr);
            clipper::Grid_sampling grid = clipper::Grid_sampling(10, 10, 10);

            clipper::Spacegroup spacegroup = clipper::Spacegroup(clipper::Spacegroup::P1);

            clipper::Xmap<float> xmap(spacegroup, cell, grid);

            clipper::EDcalc_iso<float> ed_calc(2);
            ed_calc(xmap, atom_list);

//            std::cout << atom_list[0].coord_orth().format() << " " << xmap.get_data(atom_list[0].coord_orth().coord_frac(cell).coord_grid(grid)) << std::endl;

            m_density.push_back(std::make_pair(tmp_monomer, xmap));
        }
    }

}

void LibraryItem::dump_minimol(clipper::MiniMol &output_model, std::string file_path, std::string file_name) {
    clipper::MMDBfile m_file;
    m_file.export_minimol(output_model);
    std::string output_path = file_path + file_name + ".pdb";
    m_file.write_file(output_path);
}

void LibraryItem::dump_electron_density(std::string path) {

    for (int i = 0; i < m_density.size(); i++) {
        std::pair<clipper::MMonomer, clipper::Xmap<float>> pair_ = m_density[i];
        clipper::CCP4MAPfile map_out;
        std::string output_path = path + "/maps/" + m_pdb_code + "_map_" + std::to_string(i) + ".map";
        map_out.open_write(output_path);
        map_out.export_xmap(pair_.second);
        map_out.close_write();

        clipper::MMDBfile m_file;
        std::string output_path_model = path + "/models/" + m_pdb_code + "_fragment_" + std::to_string(i) + ".pdb";
        clipper::MPolymer mp;
        mp.insert(pair_.first);
        clipper::MiniMol mm;
        mm.insert(mp);
        m_file.export_minimol(mm);
        m_file.write_file(output_path_model);
    }

}

void LibraryItem::load_reflections(clipper::MiniMol &mol) {
    std::string path_base = "./data/DNA_test_structures/MTZ_Files/";

    std::string pdb_code = this->get_pdb_code();

    std::string mtz_file_path = path_base + pdb_code + "_phases.mtz";

//    std::cout << "Loading reflection file for " << mtz_file_path << std::endl;

    clipper::HKL_info hkl;
    clipper::HKL_data<clipper::data32::F_phi> f_phi_data(hkl);

    clipper::CCP4MTZfile mtz_in;
    mtz_in.open_read(mtz_file_path);        // open new file
    mtz_in.import_hkl_info(hkl);     // read sg, cell, reso, hkls

    std::vector<clipper::String> mtz_columns = mtz_in.column_labels();

//    for (auto x: mtz_columns) {
//        std::cout << x << "\n";
//    }
//    std::cout << std::endl;

    std::map<std::string, std::string> column_name_map;

    column_name_map["/FOBS "] = "*/*/[FOBS,SIGFOBS]";
    column_name_map["/FP "] = "*/*/[FP,SIGFP]";
    column_name_map["/FOSC "] = "*/*/[FOSC,SIGFOSC]";
    column_name_map["/F-obs "] = "*/*/[F-obs,SIGF-obs]";
    column_name_map["/F "] = "*/*/[F,SIGF]";


    for (clipper::String column: mtz_columns) {
        std::map<std::string, std::string>::iterator it;
        for (it = column_name_map.begin(); it != column_name_map.end(); it++) {
            if (column.find(it->first) != -1) {
//                std::cout << "FOUND: " << it->first << std::endl;
                mtz_in.import_hkl_data(f_phi_data, "*/*/[FC,PHIC]");
            }
        }
    }

    mtz_in.close_read();

    clipper::Grid_sampling mygrid(hkl.spacegroup(), hkl.cell(), hkl.resolution());  // define grid
    clipper::Xmap<float> mymap(hkl.spacegroup(), hkl.cell(), mygrid);  // define map
    mymap.fft_from(f_phi_data);                  // generate map

//    std::cout << mymap.cell().format() << std::endl;

//    std::cout << "Created xmap with memadd - " << &mymap << std::endl;

//    model_pair = std::make_pair(mol, &mymap);
    m_xmap = mymap;
    this->model_pair = std::make_pair(mol, &mymap);
}

// LIBRARY CLASS IMPLEMENTATIONS

Library::Library(std::string library_file_path, std::string pdb_base_dir, bool use_experimental_data,
                 std::string pdb_file_ending)
        : m_library_path(library_file_path), m_pdb_base_dir(pdb_base_dir), m_pdb_file_ending(pdb_file_ending),
          m_use_experimental_data(use_experimental_data) {

    m_library = read_library_item();

    #pragma omp parallel for
    for (int i = 0; i < m_library.size(); i++) {
        LibraryItem *library_item = m_library[i];
        clipper::MiniMol mol = library_item->load_pdb();

        if (use_experimental_data == true) {
            library_item->load_reflections(mol);
        } else {
            library_item->calculate_electron_density(mol);
        }

//        std::cout << "Getting all positions..." << std::endl;
        Density dens = Density();

        dens.get_all_positions((m_library[i])->model_pair.first,
                               &(m_library[i])->m_xmap,
                               (m_library[i])->m_pdb_code, true);

//        std::string pdb_code = (*iter)->m_pdb_code;
//        std::string path = "./debug/32x32x32_points";
//
//        dens.write_labelled_pixel_map(map_vector, path, pdb_code);

    }

//    for (int i = 0; i < m_library.size(); i++) {
//        LibraryItem& library_item = m_library[i];
//        clipper::MiniMol mol = library_item.load_pdb();
//
//        if (use_experimental_data == true) {
//            library_item.load_reflections(mol);
//        }
//        else {
//            library_item.calculate_electron_density(mol);
//        }
//
//    }

    std::cout << "Library loaded." << std::endl;

}

std::vector<LibraryItem *> Library::read_library_item() {
//    "./data/RNA_test_structures/PDB Files/";
    std::vector<std::string> pdb_ids;

    std::fstream file_stream;
    file_stream.open(m_library_path, std::ios::in);

    std::string line, word;
    std::vector<std::string> row;
    std::vector<std::vector<std::string>> file_content;

    if (file_stream.is_open()) {
        while (std::getline(file_stream, line)) {
            std::stringstream str(line);

            while (std::getline(str, word, ',')) {
                row.push_back(word);
                file_content.push_back(row);
            }
        }
    } else {
        std::cout << "There was an error when opening the library path" << std::endl;
    }

    std::vector<LibraryItem *> return_list;


    for (std::string pdb_code: row) {
//        ORIGINAL
        LibraryItem *library_item = new LibraryItem(pdb_code, m_pdb_base_dir, m_pdb_file_ending,
                                                    m_use_experimental_data);
        return_list.push_back(library_item);

    }

    return return_list;

}

void Library::combine_density() {
//
//    clipper::Cell_descr cell_descr = clipper::Cell_descr(20,20,20);
//    clipper::Cell cell = clipper::Cell(cell_descr);
//    clipper::Grid_sampling grid = clipper::Grid_sampling(10,10,10);
//
//    clipper::Spacegroup spacegroup = clipper::Spacegroup(clipper::Spacegroup::P1);
//    clipper::Xmap<float> combined_xmap(spacegroup, cell, grid);
//
//    std::vector<std::vector<std::vector<std::pair<std::vector<int>,float>>>> array3d;
//
//    for (LibraryItem item: m_library) {
//        std::cout << item.m_pdb_base_dir << std::endl;
//        for (std::pair<clipper::MMonomer, clipper::Xmap<float>> pair_: item.m_density) {
//            combined_xmap += pair_.second;
//
//            clipper::Xmap<float> xmap = pair_.second;
//
//            clipper::Coord_orth base_coord_orth = clipper::Coord_orth(0,0,0);
//            clipper::Coord_grid base_coord = base_coord_orth.coord_frac(xmap.cell()).coord_grid(grid);
//            clipper::Xmap_base::Map_reference_coord i0 = clipper::Xmap_base::Map_reference_coord(xmap, base_coord);
//
//            clipper::Coord_orth end_coord_orth = clipper::Coord_orth(20,20,20);
//            clipper::Coord_grid end_coord = end_coord_orth.coord_frac(xmap.cell()).coord_grid(grid);
//            clipper::Xmap_base::Map_reference_coord iend = clipper::Xmap_base::Map_reference_coord(xmap, end_coord);
//
//
//            clipper::Xmap_base::Map_reference_coord iu, iv, iw;
//            for (iu = i0; iu.coord().u() <= iend.coord().u(); iu.next_u()) {
//
//                std::vector<std::vector<std::pair<std::vector<int>,float>>> array2d;
//
//                for (iv = iu; iv.coord().v() <= iend.coord().v(); iv.next_v()) {
//
//                    std::vector<std::pair<std::vector<int>,float>> array;
//
//                    for (iw = iv; iw.coord().w() <= iend.coord().w(); iw.next_w()) {
////                        std::cout << xmap[iw] << std::endl;
////                        std::cout << iw.coord().format() << std::endl;
//                        std::vector<int> coords = {iw.coord().u(), iw.coord().v(), iw.coord().w()};
//                        array.push_back(std::make_pair(coords, xmap[iw]));
//                    }
//
//                    array2d.push_back(array);
//                }
//
//                array3d.push_back(array2d);
//            }
//
//        }
//    }
//
//    std::ofstream output_csv ;
//    output_csv.open("./debug/aligned_fragments/array3d.csv");
//
//    for (auto i: array3d) {
//        for (auto j: i) {
//            for (auto k: j) {
//                output_csv << k.first[0] << " " << k.first[1] << " " << k.first[2] << " " << k.second << std::endl;
//            }
//
//        }
//    }
//
//    output_csv.close();
//
//    clipper::CCP4MAPfile map_out;
//    std::string output_path = "./debug/aligned_fragments/combined_map.map";
//    map_out.open_write(output_path);
//    map_out.export_xmap(combined_xmap);
//    map_out.close_write();

}


//int main() {
//    std::string library_path = "./data/rebuilt_filenames.txt";
//    Library lib = Library(library_path);
//    lib.combine_density();
//
////    MapReader map;
////    map._test();
//
//    return 0;
//}