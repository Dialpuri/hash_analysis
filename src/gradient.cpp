//
// Created by jordan on 1/18/23.
//

#include "gradient.h"


void Gradient::calculate_gradient() {

    Density::PixelMap image = m_image;

    for (int i = 0; i < image.size(); i++){
        for (int j = 0; j < image[i].size(); j++){
            for (int k = 0; k < image[i][j].size(); k++){
//                PixelData pixel = image[i][j][k];

                int neg_probe_x = i-1;
                int neg_probe_y = j-1;
                int neg_probe_z = k-1;

                int pos_probe_x = i+1;
                int pos_probe_y = j+1;
                int pos_probe_z = k+1;

                if (i == 0) {
                    neg_probe_x = image.size()-1;
                    pos_probe_x = 1;
                }

                if (i == image.size()-1) {
                    pos_probe_x = 0;
                    neg_probe_x = image.size()-2;
                }


                if (j == 0) {
                    neg_probe_y = image[i].size()-1;
                    pos_probe_y = 1;
                }

                if (j == image[i].size()-1) {
                    pos_probe_y = 0;
                    neg_probe_y = image[i].size()-2;
                }


                if (k == 0) {
                    neg_probe_z = image[i][j].size()-1;
                    pos_probe_z = 1;
                }

                if (k == image[i][j].size()-1) {
                    pos_probe_z = 0;
                    neg_probe_z = image[i][j].size()-2;
                }


                float gradient_x = image[pos_probe_x][j][k].data() - image[neg_probe_x][j][k].data();
                float gradient_y = image[i][pos_probe_y][k].data() - image[i][neg_probe_y][k].data();
                float gradient_z = image[i][j][pos_probe_z].data() - image[i][j][neg_probe_z].data();

                float gxy_mag = sqrt((pow(gradient_x, 2) + pow(gradient_y, 2)));

                float magnitude = sqrt((pow(gradient_x, 2) + pow(gradient_y, 2) + pow(gradient_z, 2)));
                float angle = atan((gradient_z/gxy_mag)) * (180.0 / M_PI);

                if (angle < 0) {
                    angle = angle + 180;
                }

                if (angle > 180) {
                    angle = angle - 180;
                }

                GradientData gradient_data;
                gradient_data.m_angle = angle;
                gradient_data.m_magnitude = magnitude;
                m_gradient_map[i][j][k] = gradient_data;
            }
        }
    }
}

Gradient::Blocks Gradient::transform_to_blocks() {
    GradientMap map = m_gradient_map;

    int block_size = 8;

    int x_dimension = m_gradient_map.size();
    int y_dimension = m_gradient_map[0].size();
    int z_dimension = m_gradient_map[0][0].size();

    int no_x = floor(x_dimension / block_size);
    int no_y = floor(y_dimension / block_size);
    int no_z = floor(z_dimension / block_size);

    int remainder_x = x_dimension % block_size;
    int remainder_y = y_dimension % block_size;
    int remainder_z = z_dimension % block_size;

    int overflow_x = block_size - remainder_x;
    int overflow_y = block_size - remainder_y;
    int overflow_z = block_size - remainder_z;

    int block_no_x = ceil(x_dimension / block_size);
    int block_no_y = ceil(y_dimension / block_size);
    int block_no_z = ceil(z_dimension / block_size);

    Gradient::Blocks blocks(block_no_x, std::vector<std::vector<Block>>(block_no_y, std::vector<Block>(block_no_z)));

    std::cout << "Xdim " << x_dimension << " no_x " << no_x << " remainder_x " << remainder_x << " overflowx " << overflow_x << std::endl;
    std::cout << "Ydim " << y_dimension << " no_y " << no_y << " remainder_y " << remainder_y << " overflowy " << overflow_y << std::endl;
    std::cout << "Zdim " << z_dimension << " no_z " << no_z << " remainder_z " << remainder_z << " overflowz " << overflow_z << std::endl;

    for (int i = 0; i < block_no_x; i++) {
        for (int j = 0; j < block_no_y; j++) {
            for (int k = 0; k < block_no_z; k++) {

                bool overflowing = false;

                if (i == block_no_x-1 && remainder_x > 0) {overflowing = true;}
                if (j == block_no_y-1 && remainder_y > 0) {overflowing = true;}
                if (k == block_no_z-1 && remainder_z > 0) {overflowing = true;}

//                std::cout << i << " " << j << " " << k << " " << overflowing << std::endl;

                if (overflowing == true) {

                    int x_counter = 0;
                    for (int u = 0; u < block_size; u++) {
                        int y_counter = 0;

                        for (int v = 0; v < block_size; v++) {
                            int z_counter = 0;
                            for (int w = 0; w < block_size; w++) {

                                int probe_x = (i * block_size) + u;
                                int probe_y = (j * block_size) + v;
                                int probe_z = (k * block_size) + w;

//                                std::cout << probe_x << " " << probe_y << " " << probe_z << std::endl;
//                                std::cout << "B4 " << i << " " << j << " " << k << " " << u << " " << v << " " << w << " " << m_gradient_map[probe_x][probe_y][probe_z].m_angle << std::endl;

                                if (u+1 >= remainder_x) {
                                    probe_x = x_counter;
//                                    std::cout << u << " " << remainder_x << " " << probe_x << std::endl;
                                    if (x_counter < overflow_x ) {
//                                        std::cout << "Changing x counter " << x_counter << " " << overflow_x << std::endl;
                                        x_counter++;
                                    }
                                }

                                if (v+1 >= remainder_y) {
//                                    std::cout << v << " " << remainder_y << " " << probe_x << std::endl;

                                    probe_y = y_counter;
                                    if (y_counter < overflow_y ) {
//                                        std::cout << "Changing y counter " << y_counter << " " << overflow_y << std::endl;
                                        y_counter++;
                                    }
                                }

                                if (w+1 >= remainder_z) {
//                                    std::cout << w << " " << remainder_z << " " << probe_x << std::endl;

                                    probe_z = z_counter;
                                    if (z_counter < overflow_z ) {
//                                        std::cout << "Changing z counter " << z_counter << " " << overflow_z << std::endl;
                                        z_counter++;
                                    }
                                }

                                blocks[i][j][k].m_data[u][v][w] = m_gradient_map[probe_x][probe_y][probe_z];

//                                std::cout << "AF " <<
//                                i << " " << j << " " << k << "  " <<
//                                u << " " << v << " " << w << "  " <<
//                                x_counter << " " << y_counter << " " << z_counter << "  " <<
//                                remainder_x << " " << remainder_y << " " << remainder_z << "  " <<
//                                probe_x << " " << probe_y << " " << probe_z << "  " <<
//                                overflow_x << " " << overflow_y << " " << overflow_z << "  " <<
//                                m_gradient_map[probe_x][probe_y][probe_z].m_angle << std::endl;

                                blocks[i][j][k].i = i;
                                blocks[i][j][k].j = j;
                                blocks[i][j][k].k = k;
                                blocks[i][j][k].overflowing = overflowing;

                            }
                        }
                    }
                    continue;
                }

                for (int u = 0; u < block_size; u++) {
                    for (int v = 0; v < block_size; v++) {
                        for (int w = 0; w < block_size; w++) {

                            int gradient_index_i = (i * block_size) + u;
                            int gradient_index_j = (j * block_size) + v;
                            int gradient_index_k = (k * block_size) + w;

                            blocks[i][j][k].m_data[u][v][w] = m_gradient_map[gradient_index_i][gradient_index_j][gradient_index_k];
                            blocks[i][j][k].i = i;
                            blocks[i][j][k].j = j;
                            blocks[i][j][k].k = k;
                            blocks[i][j][k].overflowing = overflowing;
                        }
                    }
                }
            }
        }
    }

    return blocks;

}

void Gradient::calculate_histograms(Gradient::Blocks &blocks) {

    for (auto x: blocks) {
        for (auto y: x) {
            for (auto block : y) {

                int angle_step = 20;
                int number_of_bins = 180 / angle_step;

                for (int i = 0; i < number_of_bins; i++) {
//                    Init histogram of length number_of_bins
                    block.histogram.push_back(std::make_pair((i*angle_step), 0.0f));
                }

                for (auto i: block.m_data) {
                    for (auto j: i) {
                        for (auto k: j) {
                            float angle = k.m_angle;

                            int angle_bin_index_lower  = floor(angle / angle_step);
                            int angle_bin_index_upper = ceil(angle / angle_step);

                            if (angle_bin_index_upper > block.histogram.size() - 1) { angle_bin_index_upper = 0;}

                            int lower_angle_bin = block.histogram[angle_bin_index_lower].first;

                            float lower_angle_delta = angle - lower_angle_bin;
                            float upper_angle_delta = angle_step - lower_angle_delta;

                            float lower_angle_proportion = lower_angle_delta / 20;
                            float upper_angle_proportion = upper_angle_delta / 20;

                            block.histogram[angle_bin_index_lower].second += k.m_magnitude * lower_angle_proportion;
                            block.histogram[angle_bin_index_upper].second += k.m_magnitude * upper_angle_proportion;

                        }
                    }
                }
            }
        }
    }
}

Gradient::Block_list Gradient::calculate_histograms(Model& model, Density& dens) {
    return calculate_histograms(model, dens.xmap);
}


Gradient::Block_list Gradient::calculate_histograms(Model &model, clipper::Xmap<float> &xmap) {

    clipper::MModel sugar_model = model.m_model.model();
//    Density::PixelMap map = m_image;

    Block_list blocks;

    clipper::MiniMol output_minimol;

    for (int i = 0; i < sugar_model.size(); i++) {
        clipper::MPolymer polymer = sugar_model[i];

        clipper::MPolymer output_polymer;

        for (int j = 0; j < polymer.size(); j++) {

            clipper::MMonomer monomer = polymer[j];
            clipper::Atom_list atoms = monomer.atom_list();

            float u = 0;
            float v = 0;
            float w = 0;

            for (int atom_index = 0; atom_index < atoms.size(); atom_index++) {
                clipper::Atom atom = atoms[atom_index];

                u += atom.coord_orth().coord_frac(xmap.cell()).coord_grid(xmap.grid_sampling()).u();
                v += atom.coord_orth().coord_frac(xmap.cell()).coord_grid(xmap.grid_sampling()).v();
                w += atom.coord_orth().coord_frac(xmap.cell()).coord_grid(xmap.grid_sampling()).w();
            }

            Block block;

            int center_u = round(u / atoms.size());
            int center_v = round(v / atoms.size());
            int center_w = round(w / atoms.size());

            int lower_u = center_u - 4;
            int upper_u = center_u + 4;

            int lower_v = center_v - 4;
            int upper_v = center_v + 4;

            int lower_w = center_w - 9;
            int upper_w = center_w + 9;

//            std::cout << "Center coords = " << center_u << " " << center_v << " " << center_w << std::endl;

//            clipper::MMonomer output_monomer = return_bounding_box(xmap, lower_u, lower_v, lower_w, upper_u, upper_v, upper_w);
//            clipper::MPolymer tmp_polymer;
//            tmp_polymer.insert(output_monomer);
//            output_polymer = tmp_polymer;

            int nu = xmap.grid_sampling().nu();
            int nv = xmap.grid_sampling().nv();
            int nw = xmap.grid_sampling().nw();

            int local_map_u = 0;
            for (int local_u = lower_u; local_u < upper_u; local_u++) {
                int local_map_v = 0;
                for (int local_v = lower_v; local_v < upper_v ; local_v++) {
                    int local_map_w = 0;
                    for (int local_w = lower_w; local_w < upper_w; local_w++) {

                        GradientData gradient_data = calculate_gradient_data(nu, nv, nw, local_u, local_v, local_w);

                        std::cout << gradient_data.m_psi << std::endl;

                        block.m_data[local_map_u][local_map_v][local_map_w] = gradient_data;

                        local_map_w++;
                    }
                    local_map_v++;
                }
                local_map_u++;
            }
            blocks.push_back(block);
        }

//        output_minimol.insert(output_polymer);
    }

    for (int block_index = 0; block_index < blocks.size(); block_index++) {

        int angle_step = 20;
        int number_of_bins = 180 / angle_step;

        for (int i = 0; i < number_of_bins; i++) {
            blocks[block_index].m_theta_histogram.push_back(std::make_pair((i*angle_step), 0.0f));
            blocks[block_index].m_psi_histogram.push_back(std::make_pair((i*angle_step), 0.0f));

        }

        for (auto i: blocks[block_index].m_data) {
            for (auto j: i) {
                for (auto k: j) {
                    std::cout << k.m_magnitude << " " << k.m_theta << " " << k.m_psi << std::endl;
                    float theta = k.m_theta;
                    float psi = k.m_psi;

                    int theta_angle_bin_index_lower  = floor(theta / angle_step);
                    int theta_angle_bin_index_upper = ceil(theta / angle_step);

                    int psi_angle_bin_index_lower  = floor(psi / angle_step);
                    int psi_angle_bin_index_upper = ceil(psi / angle_step);

                    if (theta_angle_bin_index_upper > blocks[block_index].m_theta_histogram.size() - 1) { theta_angle_bin_index_upper = 0;}
                    if (psi_angle_bin_index_upper > blocks[block_index].m_psi_histogram.size() - 1) { psi_angle_bin_index_upper = 0;}

                    int theta_lower_angle_bin = blocks[block_index].m_theta_histogram[theta_angle_bin_index_lower].first;
                    int psi_lower_angle_bin = blocks[block_index].m_psi_histogram[psi_angle_bin_index_lower].first;

                    float theta_lower_angle_delta = theta - theta_lower_angle_bin;
                    float theta_upper_angle_delta = angle_step - theta_lower_angle_delta;

                    float psi_lower_angle_delta = psi - psi_lower_angle_bin;
                    float psi_upper_angle_delta = angle_step - psi_lower_angle_delta;

                    float theta_lower_angle_proportion = theta_lower_angle_delta / 20;
                    float theta_upper_angle_proportion = theta_upper_angle_delta / 20;

                    float psi_lower_angle_proportion = psi_lower_angle_delta / 20;
                    float psi_upper_angle_proportion = psi_upper_angle_delta / 20;

                    blocks[block_index].m_theta_histogram[theta_angle_bin_index_lower].second += k.m_magnitude * theta_lower_angle_proportion;
                    blocks[block_index].m_theta_histogram[theta_angle_bin_index_upper].second += k.m_magnitude * theta_upper_angle_proportion;

                    blocks[block_index].m_psi_histogram[psi_angle_bin_index_lower].second += k.m_magnitude * psi_lower_angle_proportion;
                    blocks[block_index].m_psi_histogram[psi_angle_bin_index_upper].second += k.m_magnitude * psi_upper_angle_proportion;

                }
            }
        }
    }

    std::cout << "Theta Block size " << blocks[0].m_theta_histogram.size() << std::endl;
    std::cout << "Psi Block size " << blocks[0].m_psi_histogram.size() << std::endl;

//    clipper::MMDBManager::TYPE cifflag = clipper::MMDBManager::Default;
//
//    clipper::MMDBfile pdb_file;
//    pdb_file.export_minimol( output_minimol );
//    pdb_file.write_file( "./debug/output_sugar_with_bounding_box.pdb", cifflag );

    return blocks;

}

GradientData Gradient::calculate_gradient_data(int nu, int nv, int nw, int local_u, int local_v, int local_w) {
    int probe_u_upper = local_u + 1;
    int probe_u = local_u;
    int probe_u_lower = local_u-1;

    int probe_v_upper = local_v+1;
    int probe_v = local_v;
    int probe_v_lower = local_v-1;

    int probe_w_upper = local_w+1;
    int probe_w = local_w;
    int probe_w_lower = local_w-1;

    if (local_u + 1 > nu) {
        probe_u_upper = (local_u - nu) + 1;
        probe_u = local_u - nu;
        probe_u_lower = abs(nu - local_u);
    }

    if (local_u + 1 == nu) {
        probe_u_upper = 0;
        probe_u_lower = local_u - 1;
    }

    if (local_u <= 0 ) {
        probe_u_upper = abs(local_u + nu) + 1;
        probe_u = abs(local_u + nu);
        probe_u_lower = abs(local_u + nu) - 1;
    }

    if (local_v + 1 > nv) {
        probe_v_upper = (local_v - nv) + 1 ;
        probe_v = local_v - nv;
        probe_v_lower = abs(nv - local_v);
    }

    if (local_v <= 0 ) {
        probe_v_upper = abs(local_v + nv) + 1; ;
        probe_v = local_v + nv;
        probe_v_lower = abs(local_v + nv) - 1;
    }

    if (local_v + 1 == nv) {
        probe_v_upper = 1;
        probe_v_lower = local_v - 1;
    }


    if (local_w + 1> nw) {
        probe_w_upper = (local_w - nw) + 1;
        probe_w = abs(local_w - nw);
        probe_w_lower = abs(nw - local_w) ;
    }

    if (local_w <= 0 ) {
        probe_w_upper = abs(local_w + nw) + 1; ;
        probe_w = abs(local_w - nw);
        probe_w_lower = abs(local_w + nw) - 1;
    }

    if (local_w + 1 == nw) {
        probe_w_upper = 1;
        probe_w_lower = local_w - 1;
    }

    float gradient_x = m_image[probe_u_upper][probe_v][probe_w].data() - m_image[probe_u_lower][probe_v][probe_w].data();
    float gradient_y = m_image[probe_u][probe_v_upper][probe_w].data() - m_image[probe_u][probe_v_lower][probe_w].data();
    float gradient_z = m_image[probe_u][probe_v][probe_w_upper].data() - m_image[probe_u][probe_v][probe_w_lower].data();

    float gxy_mag = sqrt((pow(gradient_x, 2) + pow(gradient_y, 2)));

    float magnitude = sqrt((pow(gradient_x, 2) + pow(gradient_y, 2) + pow(gradient_z, 2)));
    float angle = atan((gradient_z/gxy_mag)) * (180.0 / M_PI);

    float theta = acos((gradient_z/magnitude)) * (180.0 / M_PI);
    float psi = atan(gradient_y/gradient_x) * (180.0 / M_PI);

    std::cout << m_image[probe_u][probe_v][probe_w].data() << " " << gradient_x << " " << gradient_y << " " << gradient_z << " " << magnitude << " " << theta << " " << psi <<  std::endl;


    if (theta < 0) {
        theta += 180;
    }

    if (theta > 180) {
        theta -= 180;
    }

    if (psi < 0) {
        psi += 180;
    }

    if (psi > 180) {
        psi -= 180;
    }

    std::cout << "Theta " << theta << " Psi " << psi << std::endl;

    GradientData gradient_data;
    gradient_data.m_magnitude = magnitude;
    gradient_data.m_angle = angle;
    gradient_data.m_theta = theta;
    gradient_data.m_psi = psi;

//    std::cout << theta << std::endl;

    return gradient_data;
}

void Gradient::write_histogram_data(Gradient::Block_list &blocks, std::string file_name) {

    std::cout << "Writing histogram data to " << file_name << std::endl;
    std::ofstream theta_file;
    theta_file.open (file_name + "theta_histogram.csv");
    theta_file << "0,20,40,60,80,100,120,140,160\n";

    std::ofstream psi_file;
    psi_file.open (file_name + "psi_histogram.csv");
    psi_file << "0,20,40,60,80,100,120,140,160\n";

    std::ofstream all_data_file;
    all_data_file.open (file_name + "theta_and_psi_histogram.csv");
    all_data_file << "Theta,Psi\n";

    for( Block block : blocks) {
        int x_no = 0;

        for (auto y: block.m_data) {
            for (auto z: y) {
                for (auto a : z) {
                    all_data_file << a.m_theta << "," << a.m_psi << "\n";
                }
            }
        }

        for (int i = 0; i < block.m_theta_histogram.size(); i++) {
            auto theta = block.m_theta_histogram[i];
            auto psi = block.m_psi_histogram[i];

            if (x_no == 8) {
                theta_file << theta.second;
                psi_file << psi.second;
            }
            else {

                theta_file << theta.second << ", ";
                psi_file << psi.second << ", ";
            }
            x_no++;
        }
        theta_file << "\n";
        psi_file << "\n";
    }

    psi_file.close();
    all_data_file.close();
    theta_file.close();

}

clipper::MMonomer
Gradient::return_bounding_box(clipper::Xmap<float>& xmap, int lower_u, int lower_v, int lower_w, int upper_u, int upper_v, int upper_w) {

        clipper::MMonomer output_monomer;
        clipper::Coord_orth corner_1 = clipper::Coord_grid(lower_u, lower_v, lower_w).coord_frac(xmap.grid_sampling()).coord_orth(xmap.cell());
        clipper::Coord_orth corner_2 = clipper::Coord_grid(upper_u, lower_v, lower_w).coord_frac(xmap.grid_sampling()).coord_orth(xmap.cell());
        clipper::Coord_orth corner_3 = clipper::Coord_grid(lower_u, upper_v, lower_w).coord_frac(xmap.grid_sampling()).coord_orth(xmap.cell());
        clipper::Coord_orth corner_4 = clipper::Coord_grid(lower_u, lower_v, upper_w).coord_frac(xmap.grid_sampling()).coord_orth(xmap.cell());
        clipper::Coord_orth corner_5 = clipper::Coord_grid(upper_u, upper_v, lower_w).coord_frac(xmap.grid_sampling()).coord_orth(xmap.cell());
        clipper::Coord_orth corner_6 = clipper::Coord_grid(lower_u, upper_v, upper_w).coord_frac(xmap.grid_sampling()).coord_orth(xmap.cell());
        clipper::Coord_orth corner_7 = clipper::Coord_grid(upper_u, lower_v, upper_w).coord_frac(xmap.grid_sampling()).coord_orth(xmap.cell());
        clipper::Coord_orth corner_8 = clipper::Coord_grid(upper_u, upper_v, upper_w).coord_frac(xmap.grid_sampling()).coord_orth(xmap.cell());


        clipper::Atom atom_1;
        atom_1.set_coord_orth(corner_1);
        atom_1.set_element("H");

        clipper::Atom atom_2;
        atom_2.set_coord_orth(corner_2);
        atom_2.set_element("H");

        clipper::Atom atom_3;
        atom_3.set_coord_orth(corner_3);
        atom_3.set_element("H");

        clipper::Atom atom_4;
        atom_4.set_coord_orth(corner_4);
        atom_4.set_element("H");

        clipper::Atom atom_5;
        atom_5.set_coord_orth(corner_5);
        atom_5.set_element("H");

        clipper::Atom atom_6;
        atom_6.set_coord_orth(corner_6);
        atom_6.set_element("H");

        clipper::Atom atom_7;
        atom_7.set_coord_orth(corner_7);
        atom_7.set_element("H");

        clipper::Atom atom_8;
        atom_8.set_coord_orth(corner_8);
        atom_8.set_element("H");

        output_monomer.insert(atom_1);
        output_monomer.insert(atom_2);
        output_monomer.insert(atom_3);
        output_monomer.insert(atom_4);
        output_monomer.insert(atom_5);
        output_monomer.insert(atom_6);
        output_monomer.insert(atom_7);
        output_monomer.insert(atom_8);

        return output_monomer;
}
