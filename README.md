# Hash Analysis

This repository consists of a small program which is a test for whether a hash based approach to electron density classification is viable. This work is part of a PhD project funded by the BBSRC. 

### Development

#### Prerequisites

You must have
- Clipper
- MMDB2

installed, which come included in the CCP4. To ensure they are in your path, you must source the appropriate script. To do this run:

    source /opt/xtal/ccp4-X.X/bin/ccp4.setup-sh 
where X.X is your CCP4 version.

#### Development

To compile this, just simply run:

    make
and the executable 'hash_exec' should be created in the root directory of the project.

To run the executable run:

    ./hash_exec
    
#### Usage

To create a slice of density and output it to a csv file. 

    Density dens; 
    dens.load_file("./path/to/file.mtz")
    dens.slice(2);
    dens.dump_slice("./path/to/output.csv");
