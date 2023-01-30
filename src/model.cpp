//
// Created by jordan on 23/01/23.
//

#include "model.h"


void Model::load_model(std::string file_path) {
    const int mmdbflags = ::mmdb::MMDBF_IgnoreBlankLines | ::mmdb::MMDBF_IgnoreDuplSeqNum | ::mmdb::MMDBF_IgnoreNonCoorPDBErrors | ::mmdb::MMDBF_IgnoreRemarks;
    clipper::MMDBfile mfile;
    clipper::MiniMol mol;
    mfile.SetFlag( mmdbflags );
    mfile.read_file( file_path );
    mfile.import_minimol( mol );

    if ( mol.size() == 0 ) return ;

    for ( int c = 0; c < mol.size(); c++ ) {
        clipper::MPolymer mp;
        // select monomers by occupancy
        for ( int r = 0; r < mol[c].size(); r++ ) {
            if ( mol[c][r].lookup( " C1'", clipper::MM::ANY ) >= 0 &&
                 mol[c][r].lookup( " C2'", clipper::MM::ANY ) >= 0 &&
                 mol[c][r].lookup( " C3'", clipper::MM::ANY ) >= 0 &&
                 mol[c][r].lookup( " C4'", clipper::MM::ANY ) >= 0 &&
                 mol[c][r].lookup( " C5'", clipper::MM::ANY ) >= 0 &&
                 mol[c][r].lookup( " O3'", clipper::MM::ANY ) >= 0 &&
                 mol[c][r].lookup( " O4'", clipper::MM::ANY ) >= 0 &&
                 mol[c][r].lookup( " O5'", clipper::MM::ANY ) >= 0 &&
                 mol[c][r].lookup( " P  ", clipper::MM::ANY ) >= 0 ) {
                int a = mol[c][r].lookup( " C4'", clipper::MM::ANY );
                if ( mol[c][r][a].occupancy() > 0.01 &&
                     mol[c][r][a].u_iso() < clipper::Util::b2u(100.0) )
                    mp.insert( mol[c][r] );
            }
        }

        m_model.insert(mp);
    }
}

void Model::prepare_sugars() {
    clipper::MModel sugar_model = m_model.model();

    for (int p = 0; p < sugar_model.size(); p++) {
        clipper::MPolymer polymer = sugar_model[p];

        for (int m = 0; m < polymer.size(); m++) {

            clipper::MMonomer monomer = polymer[m];

//            std::cout << "Monomer size - " << monomer.size() << std::endl;

            Matrix<float> abc = calculate_plane_eqn(monomer.atom_list());
        }

        clipper::RTop<float> rtop;
    }


    }

Matrix<float> Model::calculate_plane_eqn(clipper::Atom_list atoms) {

    float xx_sum = 0.0;
    float xy_sum = 0.0;
    float x_sum = 0.0;
    float y_sum = 0.0;
    float yy_sum = 0.0;
    float xz_sum = 0.0;
    float yz_sum = 0.0;
    float z_sum = 0.0;

    for (int atom_index = 0; atom_index < atoms.size(); atom_index++) {

        clipper::Coord_orth atom_coord_orth = atoms[atom_index].coord_orth();

        xx_sum += pow(atom_coord_orth.x(),2);
        yy_sum += pow(atom_coord_orth.y(),2);
        xy_sum += atom_coord_orth.x() + atom_coord_orth.y();
        x_sum += atom_coord_orth.x();
        y_sum += atom_coord_orth.y();
        xz_sum += atom_coord_orth.x() + atom_coord_orth.z();
        yz_sum += atom_coord_orth.y() + atom_coord_orth.z();
        z_sum += atom_coord_orth.z();

    }

    Matrix<float> A(3, 3);

    A.set(0,0,xx_sum);
    A.set(0,1,xy_sum);
    A.set(0,2,x_sum);

    A.set(1,0, xy_sum);
    A.set(1,1, yy_sum);
    A.set(1,2, y_sum);

    A.set(2,0, x_sum);
    A.set(2,1, y_sum);
    A.set(2,2, atoms.size());

    Matrix<float> B(3, 1);

    B.set(0,0,xz_sum);
    B.set(0,1,yz_sum);
    B.set(0,2,z_sum);

    Matrix<float> A_inv = A.inverse();
    Matrix<float> X = A_inv.dot(B);

    X.print();
    X.print_eqn();

    return X;
}

