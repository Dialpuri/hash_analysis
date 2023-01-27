//
// Created by jordan on 12/14/22.
//

#include "hash.h"


void Density::load_file(std::string file_path) {
    clipper::CCP4MTZfile mtz;
    mtz.set_column_label_mode( clipper::CCP4MTZfile::Legacy );

    mtz.open_read(file_path);

    clipper::HKL_info hkls;
    hkls.init( mtz.spacegroup(), mtz.cell(), mtz.resolution(), true );

    clipper::HKL_data<clipper::data32::F_sigF>  wrk_f ( hkls );
    clipper::HKL_data<clipper::data32::F_phi>   fphi( hkls );

    mtz.import_hkl_data( wrk_f ,"FP,SIGFP");
    mtz.import_hkl_data( fphi,  "FWT,PHWT" );

    mtz.close_read();

    clipper::Spacegroup cspg = hkls.spacegroup();
    clipper::Cell       cxtl = hkls.cell();
    clipper::Grid_sampling grid( cspg, cxtl, hkls.resolution() );
    clipper::Xmap<float>   xwrk( cspg, cxtl, grid );
    xwrk.fft_from( fphi );

    std::cout << std::endl;
    std::cout << " Spgr " << hkls.spacegroup().symbol_xhm() << std::endl;
    std::cout << hkls.cell().format() << std::endl;
    std::cout << " Nref " << hkls.num_reflections() << " " << fphi.num_obs() << std::endl;

    xmap = xwrk;

}


Density::PixelMap Density::extract_data(clipper::Xmap<float> xmap) {

    clipper::Xmap_base::Map_reference_coord iu, iv, iw;
    clipper::Cell cell = xmap.cell();

//    std::cout << "CELL : " << cell.format() << std::endl;

    clipper::Grid grid = clipper::Grid(cell.a(),cell.b(),cell.c());

    m_cell = xmap.cell();
    m_gridsampling = xmap.grid_sampling();
    m_spacegroup = xmap.spacegroup();

    clipper::Coord_orth base_coord_orth = clipper::Coord_orth(0,0,0);
    clipper::Coord_grid base_coord = base_coord_orth.coord_frac(xmap.cell()).coord_grid(grid);
    clipper::Xmap_base::Map_reference_coord i0 = clipper::Xmap_base::Map_reference_coord(xmap, base_coord);

    clipper::Coord_grid end_coord = clipper::Coord_grid(m_gridsampling.nu(), m_gridsampling.nv(), m_gridsampling.nw());
    clipper::Xmap_base::Map_reference_coord iend = clipper::Xmap_base::Map_reference_coord(xmap, end_coord);


    PixelMap return_map;

    for (iu = i0; iu.coord().u() < iend.coord().u(); iu.next_u()) {

        std::vector<std::vector<PixelData>> secondary_list;

        for (iv = iu; iv.coord().v() < iend.coord().v(); iv.next_v()) {

            std::vector<PixelData> tertiary_list;

            for (iw = iv; iw.coord().w() < iend.coord().w(); iw.next_w()) {

                float xmap_value = xmap[iw];

                if (xmap[iw] == 0) {
//                    std::cout << xmap[iw] << std::endl;
//                    xmap_value = 0.01;
                    xmap_value = (((float) rand() / (RAND_MAX)) + 1)/10;
                }
//                else {
//                    std::cout << xmap[iw] << std::endl;
//                }

                tertiary_list.push_back(PixelData(xmap_value, iw.coord().u(), iw.coord().v(), iw.coord().w(), iw.coord_orth().x(), iw.coord_orth().y(), iw.coord_orth().z()));

                std::cout << iw.coord().format() << " " << xmap[iw] << std::endl;
            }
            secondary_list.push_back(tertiary_list);
        }

        m_pixel_data.push_back(secondary_list);
        return_map.push_back(secondary_list);
    }


    std::cout << "Size of m_pixel_data ~ "<< m_pixel_data.size() << " " << m_pixel_data[0].size() << " " << m_pixel_data[0][0].size() << std::endl;

    return return_map;
}


void Density::slice(float slice_index) {

    std::cout << "Slicing map with " << slice_index << std::endl;

    clipper::Xmap_base::Map_reference_coord iu, iv, iw;
    clipper::Cell cell = xmap.cell();

    clipper::Grid grid = clipper::Grid(cell.a(),cell.b(),cell.c());

    clipper::Coord_orth base_coord_orth = clipper::Coord_orth(0,0,0);
    clipper::Coord_grid base_coord = base_coord_orth.coord_frac(xmap.cell()).coord_grid(grid);
    clipper::Xmap_base::Map_reference_coord i0 = clipper::Xmap_base::Map_reference_coord(xmap, base_coord);

    std::cout  << cell.format() << std::endl;
    std::cout << grid.format() << std::endl;

//    clipper::Coord_orth end_coord_orth = clipper::Coord_orth(cell.a(),cell.b(),cell.c());
//    clipper::Coord_grid end_coord = end_coord_orth.coord_frac(xmap.cell()).coord_grid(grid);

//    std::cout << "END COORD - " << end_coord_orth.format() << std::endl;
    clipper::Coord_grid end_coord = clipper::Coord_grid(cell.a(), cell.b(), cell.c());
//    clipper::Xmap_base::Map_reference_coord test = clipper::Xmap_base::Map_reference_coord(xmap, x);

    clipper::Xmap_base::Map_reference_coord iend = clipper::Xmap_base::Map_reference_coord(xmap, end_coord);

    int nu = grid.nu();
    int nv = grid.nv();

//    std::cout << "Nu " << nu << " Nv " << nv << "\n";

    for (int i = 0; i <= nu; i++) {
        std::vector<float> i_list = {};
        for (int j = 0; j <= nv; j++) {
            i_list.push_back(0.0f);
        }
        m_grid_values.push_back(i_list);
    }

//    std::cout << "m_grid " << m_grid_values.size() << " " << m_grid_values[0].size() << std::endl;

    int i_index = 0;
    for (iu = i0; iu.coord().u() < iend.coord().u(); iu.next_u()) {
        int j_index = 0;
        for (iv = iu; iv.coord().v() < iend.coord().v(); iv.next_v()) {

//            std::cout << iv.coord().u() << "/" << iend.coord().u() << " " << iv.coord().v() << "/" << iend.coord().v() << std::endl;

            for (iw = iv; iw.coord().w() <= iend.coord().w(); iw.next_w()) {
                if (iw.coord().w() == slice_index) {
                    PixelData data = PixelData(xmap[iw], iw.coord().u(), iw.coord().v(), iw.coord().w());
                    data.xmap_ptr = &xmap;
                    m_slice.push_back(data);
//                    std::cout << i_index << "/" << m_grid_values.size() << " " << j_index << "/" << m_grid_values[0].size() << std::endl;
                    m_grid_values[i_index][j_index] = xmap[iw];
                    break;
                }
            }
            j_index++;
        }
        i_index++;
    }

    std::cout << "Finished slicing..." << std::endl;
}

void Density::dump_slice(std::string file_name) {
    std::ofstream output_csv ;
    output_csv.open(file_name);

    output_csv << "u,v,data\n";
    for (PixelData data: m_slice) {
        output_csv << data.u() << "," << data.v() << "," << data.data() << "\n";
    }

    output_csv.close();
}


void Density::dump_slice(std::string file_name, std::vector<PixelData> data) {

    std::cout << "Dumping " << file_name << "\n";

    std::ofstream output_csv ;
    output_csv.open(file_name);

    output_csv << "u,v,data\n";
    for (PixelData data: data) {
        output_csv << data.u() << "," << data.v() << "," << data.data() << "\n";
    }

    output_csv.close();
}

float Density::gaussian_1d(float x, float sigma) {
    float kernel = (1 / (sigma * sqrt(2 * M_PI))) * (exp((-(pow(x,2)))/(2*(pow(sigma,2)))));
    return kernel;
}

float Density::gaussian_2d(float x, float y, float sigma) {
    float kernel = (1 / (sigma * sqrt(2 * M_PI))) * (exp((-((pow(x,2)+(pow(y,2)))))/(2*(pow(sigma,2)))));
    return kernel;
}


float Density::gaussian_3d(float x, float y, float z, float sigma) {
    float kernel = (1 / (sigma * 2 * sqrt(2 * M_PI))) * (exp((-((pow(x,2)+(pow(y,2))+(pow(z,2)))))/(2*(pow(sigma,2)))));
    return kernel;
}



Matrix_2D Density::generate_gaussian_kernel_2d(int sigma) {

    std::cout << "Generating gaussian kernel with " << sigma << std::endl;
    int matrix_dimension = 1;
    Matrix_2D kernel_matrix;

    int index_i = 0;
    for (int i = -matrix_dimension; i <= matrix_dimension; i++) {
        int index_j = 0;
        for (int j = -matrix_dimension; j <= matrix_dimension; j++) {
            kernel_matrix.m_matrix[index_i][index_j] = gaussian_2d(i, j, sigma);
            index_j++;
        }
        index_i++;
    }

    kernel_matrix.print();

    return kernel_matrix;
}


Matrix_3D Density::generate_gaussian_kernel_3d(int sigma, int matrix_size) {

    int matrix_dimension = 1;
    Matrix_3D kernel_matrix(matrix_size);

    int index_i = 0;
    for (int i = -matrix_dimension; i <= matrix_dimension; i++) {
        int index_j = 0;
        for (int j = -matrix_dimension; j <= matrix_dimension; j++) {
            int index_k = 0;

            for (int k = -matrix_dimension; k <= matrix_dimension; k++) {
                kernel_matrix.m_matrix[index_i][index_j][index_k] = gaussian_3d(i, j, k, sigma);
                index_k++;
            }
            index_j++;
        }
        index_i++;
    }

    return kernel_matrix;
}


float Density::convolute_2D(Matrix_2D& kernel, Matrix_2D& base) {

    float sum = 0.0f;

    for (int i = 0; i < base.m_matrix.size(); i++) {
        for (int j = 0; j < base.m_matrix[i].size(); j++) {
            sum = sum + (base.m_matrix[i][j] * kernel.m_matrix[i][j]);
        }
    }

    return sum;
}

std::vector<PixelData> Density::apply_gaussian_2d(Matrix_2D kernel) {

    auto m_grid_values_out = m_grid_values;
    std::vector<PixelData> m_grid_out;

    for (int i = 0; i < m_grid_values.size(); i++) {
        std::vector<float> grid = m_grid_values[i];
        for (int j = 0; j < grid.size(); j++) {

            int i_index = i;
            int j_index = j;

            Matrix_2D grid;

            int i_iter_pos = 1;
            int i_iter_neg = -1;
            int j_iter_pos = 1;
            int j_iter_neg = -1;

            if (i == m_grid_values.size() - 1) {
                i_iter_pos = -(m_grid_values.size()-1);
            }

            if (j == m_grid_values[0].size() - 1) {
                j_iter_pos = -(m_grid_values[0].size()-1);
            }

            if (i == 0) {
                i_iter_neg = m_grid_values.size()-1;
            }

            if (j == 0) {
                j_iter_neg = m_grid_values[0].size()-1;
            }

            grid.m_matrix[0][0] = m_grid_values[i_index + i_iter_neg][j_index + j_iter_neg];
            grid.m_matrix[0][1] = m_grid_values[i_index + i_iter_neg][j_index];
            grid.m_matrix[0][2] = m_grid_values[i_index + i_iter_neg][j_index + j_iter_pos];

            grid.m_matrix[1][0] = m_grid_values[i_index][j_index + j_iter_neg];
            grid.m_matrix[1][1] = m_grid_values[i_index][j_index];
            grid.m_matrix[1][2] = m_grid_values[i_index][j_index + j_iter_pos];

            grid.m_matrix[2][0] = m_grid_values[i_index + i_iter_pos][j_index + j_iter_neg];
            grid.m_matrix[2][1] = m_grid_values[i_index + i_iter_pos][j_index];
            grid.m_matrix[2][2] = m_grid_values[i_index + i_iter_pos][j_index + j_iter_pos];

            float convolute_sum = convolute_2D(kernel, grid);
            m_grid_values_out[i][j] = convolute_sum;
            m_grid_out.push_back(PixelData(convolute_sum, i, j, 0));

        }
    }

    return m_grid_out;
}


Density::PixelMap Density::apply_gaussian_3d(Matrix_3D kernel) {


    Density::PixelMap blurred_pixels = Density::PixelMap(m_pixel_data);

    for (int u = 0; u < m_pixel_data.size(); u++) {
        for (int v = 0; v < m_pixel_data[u].size(); v++) {
            for (int w = 0; w < m_pixel_data[u][v].size(); w++) {

                int m_u = m_pixel_data[u][v][w].u();
                int m_v = m_pixel_data[u][v][w].v();
                int m_w = m_pixel_data[u][v][w].w();

                float sum = 0.0f;

                for (int k_x = 0; k_x < kernel.m_matrix.size(); k_x++) {
                    for (int k_y = 0; k_y < kernel.m_matrix[k_x].size(); k_y++) {
                        for (int k_z = 0; k_z < kernel.m_matrix[k_x][k_y].size(); k_z++) {

                            int delta_k_x = k_x - floor(kernel.m_matrix.size() / 2);
                            int delta_k_y = k_y - floor(kernel.m_matrix[k_x].size() / 2);
                            int delta_k_z = k_z - floor(kernel.m_matrix[k_x][k_y].size() / 2);

                            int probe_x = m_u;
                            int probe_y = m_v;
                            int probe_z = m_w;

//                            std::cout << "kernel.m_matrix " << k_x << " " << k_y << " " << k_z << " \n";
//                            std::cout << "delta_k " << delta_k_x << " " << delta_k_y << " " << delta_k_z << " \n";
//                            std::cout << "Probe " << probe_x << " " << probe_y << " " << probe_z << " \n";
//                            std::cout << "m_pixel dims " << m_pixel_data.size() << " " << m_pixel_data[0].size() << " " << m_pixel_data[0][0].size() << std::endl;



                            if (m_u + delta_k_x < 0) {
//                                std::cout << "m_u + delta_k_x < 0\n";
                                probe_x = m_pixel_data.size() + delta_k_x;
                            }
                            else if (m_u + delta_k_x >= m_pixel_data.size()) {
//                                std::cout << "m_u + delta_k_x > size\n";

                                probe_x = m_u - m_pixel_data.size() + delta_k_x;
                            }
                            else {
//                                std::cout << "0 < m_u + delta_k_x < size\n";

                                probe_x = m_u + delta_k_x;
                            }

                            if (m_v + delta_k_y < 0) {
//                                std::cout << "m_v + delta_k_y < 0\n";

                                probe_y = m_pixel_data[u].size() + delta_k_y;
                            }
                            else if (m_v + delta_k_y >= m_pixel_data[u].size()) {
//                                std::cout << "m_v + delta_k_y > size\n";


                                probe_y = probe_y - m_pixel_data[u].size() + delta_k_y;
                            }
                            else {
//                                std::cout << "0 < m_v + delta_k_y < size\n";

                                probe_y = m_v + delta_k_y;
                            }

                            if (m_w + delta_k_z < 0) {
//                                std::cout << "m_w + delta_k_w < 0\n";

                                probe_z = m_pixel_data[u][v].size() + delta_k_z;
                            }
                            else if (m_w + delta_k_z >= m_pixel_data[u][v].size()) {
//                                std::cout << "m_w + delta_k_w > size\n";

                                probe_z = probe_z - m_pixel_data[u][v].size() + delta_k_z;
                            }
                            else {
//                                std::cout << "0 < m_w + delta_k_z < size\n";

                                probe_z = m_w + delta_k_z;
                            }

//                            std::cout << "SUM " << sum << " = " << sum + (m_pixel_data[probe_x][probe_y][probe_z].data() * kernel.m_matrix[k_x][k_y][k_z]) << std::endl;
                            sum += (m_pixel_data[probe_x][probe_y][probe_z].data() * kernel.m_matrix[k_x][k_y][k_z]);

                        }
                    }
                }

//                blurred_pixels[u][v][w].print();
                blurred_pixels[u][v][w].m_data = sum;

            }
        }
    }

    return blurred_pixels;
}


std::vector<PixelData> Density::difference_of_gaussian(std::vector<PixelData>& top, std::vector<PixelData>& bottom) {

    std::vector<PixelData> return_list;

    for (int i = 0; i < top.size(); i++) {

//        std::cout << i << std::endl;

        for (int j = 0; j < bottom.size(); j++) {

            PixelData top_slice = top[i];
            PixelData bottom_slice = bottom[i];

            if (top_slice.u() == bottom_slice.u() && top_slice.v() == bottom_slice.v()) {
                PixelData difference = PixelData(top_slice.data() - bottom_slice.data(), top_slice.u(), top_slice.v(), 1);
                return_list.push_back(difference);
                break;
            }
        }
    }

    return return_list;
}

void Density::export_pixelmap(std::string file_name ) {

    clipper::Xmap<float> map = clipper::Xmap<float>(m_spacegroup, m_cell, m_gridsampling);

    for (int i = 0; i < m_pixel_data.size(); i++) {
        for (int j = 0; j < m_pixel_data[i].size(); j++) {
            for (int k = 0; k < m_pixel_data[i][j].size(); k++) {
                clipper::Coord_grid grid_point = clipper::Coord_grid(m_pixel_data[i][j][k].u(), m_pixel_data[i][j][k].v(), m_pixel_data[i][j][k].w());
                map.set_data(grid_point, m_pixel_data[i][j][k].m_data);
            }
        }
    }

    clipper::CCP4MAPfile mapout;
    mapout.open_write(file_name);
    mapout.export_xmap(map);
    mapout.close_write();
}


void Density::export_pixelmap(std::string file_name, Density::PixelMap pixel_map) {
    clipper::Xmap<float> map = clipper::Xmap<float>(m_spacegroup, m_cell, m_gridsampling);

    for (int i = 0; i < pixel_map.size(); i++) {
        for (int j = 0; j < pixel_map[i].size(); j++) {
            for (int k = 0; k < pixel_map[i][j].size(); k++) {
                clipper::Coord_grid grid_point = clipper::Coord_grid(pixel_map[i][j][k].u(), pixel_map[i][j][k].v(), pixel_map[i][j][k].w());
                map.set_data(grid_point, pixel_map[i][j][k].m_data);
            }
        }
    }

    clipper::CCP4MAPfile mapout;
    mapout.open_write(file_name);
    mapout.export_xmap(map);
    mapout.close_write();
}


Density::PixelMap Density::difference_of_gaussian(Density::PixelMap &top, Density::PixelMap &bottom) {

    std::cout << "Computing DOG..." << std::endl;

    Density::PixelMap difference = Density::PixelMap(top);

    for (int i = 0; i < top.size(); i++) {
        for (int j = 0; j < top[i].size(); j++) {
            for (int k = 0; k < top[i][j].size(); k++) {

                if ((top[i][j][k].u() == bottom[i][j][k].u()) & (top[i][j][k].v() == bottom[i][j][k].v()) & (top[i][j][k].w() == bottom[i][j][k].w())) {
                    difference[i][j][k].m_data = top[i][j][k].m_data - bottom[i][j][k].m_data;
//                    std::cout << "DOG: Us are the same\n";
                }
                else {
                    std::cout << "DOG: Us are NOT the same\n";
                }


            }
        }
    }

    return difference;
}


int main() {


    std::cout << "Main function called" << std::endl;
//
//    Density dens;
//    dens.load_file("./data/1hr2_phases.mtz");
//    dens.extract_data();

    std::string library_path = "./data/rebuilt_filenames.txt";
    Library lib = Library(library_path);

    for (LibraryItem library_item: lib.m_library) {
        for (int i = 0; i < library_item.m_density.size(); i++) {

            std::pair<clipper::MMonomer, clipper::Xmap<float>> sugars = library_item.m_density[i];

            Density dens;

            Density::PixelMap map = dens.extract_data(sugars.second);
//            Matrix_3D kernel = dens.generate_gaussian_kernel_3d(1, 3);
//            Density::PixelMap map = dens.apply_gaussian_3d(kernel);

//            for (auto x : map) {
//                for (auto y : x) {
//                    for (auto z: y) {
//                        std::cout << z.u() << " " << z.v() << " " << z.w() << " " << z.data() << std::endl;
//                    }
//                }
//            }

//            std::cout << "Pixelmap " << map[0][0][0].data();

            clipper::MiniMol monomer_model;
            clipper::MPolymer m_polymer;

            m_polymer.insert(sugars.first);
            monomer_model.insert(m_polymer);

            Model model(sugars.second);
            model.m_model = monomer_model;

            Gradient grad(map);
//            grad.calculate_gradient();
            Gradient::Block_list blocks = grad.calculate_histograms(model, sugars.second);

            std::string file_name = library_item.get_pdb_code() + "-SI_" + std::to_string(i);
            grad.write_histogram_data(blocks, file_name);

//            std::cout << sugars.second.cell().format() << std::endl;
        }

    }
    

//    Matrix_3D kernel = dens.generate_gaussian_kernel_3d(1, 3);
//
//    Density::PixelMap map = dens.apply_gaussian_3d(kernel);
//
//    Gradient gradient(map);
//    gradient.calculate_gradient();
//    Gradient::Blocks blocks = gradient.transform_to_blocks();
//    gradient.calculate_histograms(blocks);

//    Model model(dens.xmap);
//    model.load_model("./data/1hr2.pdb");
//    model.prepare_sugars();


//    Gradient::Block_list blocks = gradient.calculate_histograms(model, dens);

//    std::cout << blocks[0].histogram.size() << std::endl;

//    gradient.write_histogram_data(blocks, "./debug/");



//    Model model;
//    model.load_model("./data/1hr2.pdb");

//
//
//    std::cout << "Running" << std::endl;
//    auto start = std::chrono::high_resolution_clock::now();
//
//    Density hash;
//
//    hash.load_file("./data/1hr2_phases.mtz");
//    hash.slice(2);
//    hash.dump_slice("./debug/original_slice.csv");
//
//    int max_sigma = 5;
//
//    for (int i = 0; i < max_sigma; i++) {
//        for (int j = 0; j < max_sigma; )
//    }
//
//
//    Matrix_2D kernel_1 = hash.generate_gaussian_kernel_2d(1);
//    std::vector<PixelData> hash_1 = hash.apply_gaussian_2d(kernel_1);
//
//    Matrix_2D kernel_2 = hash.generate_gaussian_kernel_2d(2);
//    std::vector<PixelData> hash_2 = hash.apply_gaussian_2d(kernel_2);
//
//    hash.difference_of_gaussian(hash_1, hash_2);

//
//    auto s1 = std::chrono::high_resolution_clock::now();
//    hash.load_file("./data/1hr2_phases.mtz");
//    auto e1 = std::chrono::high_resolution_clock::now();
//
//    auto s2 = std::chrono::high_resolution_clock::now();
//    hash.extract_data();
//    auto e2 = std::chrono::high_resolution_clock::now();
//
//    auto s3 = std::chrono::high_resolution_clock::now();
//    Matrix_3D kernel_3 = hash.generate_gaussian_kernel_3d(1, 3);
//    auto e3 = std::chrono::high_resolution_clock::now();
//
//    auto s4 = std::chrono::high_resolution_clock::now();
//    Density::PixelMap blurred_map_3 = hash.apply_gaussian_3d(kernel_3);
//    auto e4 = std::chrono::high_resolution_clock::now();
//
//    auto s5 = std::chrono::high_resolution_clock::now();
//    Matrix_3D kernel_5 = hash.generate_gaussian_kernel_3d(5, 3);
//    auto e5 = std::chrono::high_resolution_clock::now();
//
//    auto s6 = std::chrono::high_resolution_clock::now();
//    Density::PixelMap blurred_map_5 = hash.apply_gaussian_3d(kernel_5);
//    auto e6 = std::chrono::high_resolution_clock::now();
//
//    auto s7 = std::chrono::high_resolution_clock::now();
//    Density::PixelMap difference = hash.difference_of_gaussian(blurred_map_3, blurred_map_5);
//    auto e7 = std::chrono::high_resolution_clock::now();
//
//    auto s8 = std::chrono::high_resolution_clock::now();
//    hash.export_pixelmap("./debug/blurred_map_5.map", blurred_map_5);
//    auto e8 = std::chrono::high_resolution_clock::now();
//
//    hash.export_pixelmap("./debug/blurred_map_3.map", blurred_map_3);
//    hash.export_pixelmap("./debug/1hr2_blurred_map_diff.map", difference);
//
//    auto end = std::chrono::high_resolution_clock::now();
//
//
//    auto d1 = std::chrono::duration_cast<std::chrono::milliseconds>(e1-s1);
//    auto d2 = std::chrono::duration_cast<std::chrono::milliseconds>(e2-s2);
//    auto d3 = std::chrono::duration_cast<std::chrono::milliseconds>(e3-s3);
//    auto d4 = std::chrono::duration_cast<std::chrono::milliseconds>(e4-s4);
//    auto d5 = std::chrono::duration_cast<std::chrono::milliseconds>(e5-s5);
//    auto d6 = std::chrono::duration_cast<std::chrono::milliseconds>(e6-s6);
//    auto d7 = std::chrono::duration_cast<std::chrono::milliseconds>(e7-s7);
//    auto d8 = std::chrono::duration_cast<std::chrono::milliseconds>(e8-s8);
//    auto total = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
//
//    std::cout << "== Timings ==\n"
//        << "Loading file : " << d1.count() << "ms\n"
//        << "Extracting Data : " << d2.count() << "ms\n"
//        << "Generating Kernel sigma=3 : " << d3.count() << "ms\n"
//        << "Applying Kernel 3 : " << d4.count() << "ms\n"
//        << "Generating Kernel sigma=8 : " << d5.count() << "ms\n"
//        << "Applying Kernel 8 : " << d6.count() << "ms\n"
//        << "Calculaing DOG : " << d7.count() << "ms\n"
//        << "Exporting Pixelmap : " << d8.count() << "ms\n"
//        << "Total Time :   " << total.count() << "ms"
//        << std::endl;
//

    //
//    std::vector<PixelData> data;
//
//    auto s2 = std::chrono::high_resolution_clock::now();
//    hash.slice(3);
//    auto e2 = std::chrono::high_resolution_clock::now();
//
//    hash.dump_slice("./debug/slice_data.csv");
//
//    auto s3 = std::chrono::high_resolution_clock::now();
//    Matrix_2D kernel = hash.generate_gaussian_kernel_2d(3);
//    auto e3 = std::chrono::high_resolution_clock::now();
//
//    auto s4 = std::chrono::high_resolution_clock::now();
//    std::vector<PixelData> blur_1 = hash.apply_gaussian_2d(kernel);
//    auto e4 = std::chrono::high_resolution_clock::now();
//
//    hash.dump_slice("./debug/kernel1.csv", blur_1);
//
//    Matrix_2D kernel_2 = hash.generate_gaussian_kernel_2d(5);
//    std::vector<PixelData> blur_2 = hash.apply_gaussian_2d(kernel_2);
//
//    hash.dump_slice("./debug/kernel2.csv", blur_2);
//
//    std::vector<PixelData> difference = hash.difference_of_gaussian(blur_1, blur_2);
//
//    hash.dump_slice("./debug/kerneldifference.csv", difference);
//
//    auto d1 = std::chrono::duration_cast<std::chrono::milliseconds>(e1-s1);
//    auto d2 = std::chrono::duration_cast<std::chrono::milliseconds>(e2-s2);
//    auto d3 = std::chrono::duration_cast<std::chrono::milliseconds>(e3-s3);
//    auto d4 = std::chrono::duration_cast<std::chrono::milliseconds>(e4-s4);
//
//    std::cout << "==Timings==\n" << "Loading " << d1.count() << "ms\n"
//            << "Slicing " << d2.count() << "ms\n"
//            << "Generating Kernel " << d3.count() << "ms\n"
//            << "Applying Kernel " << d4.count() << "ms" << std::endl;


    return 0;
}
