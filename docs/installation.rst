Installation
============
First setup a conda environment from the `environment.yml` file:
```
cd CalicoST
conda config --add channels conda-forge
conda config --add channels bioconda
conda env create -f environment.yml --name calicost_env
```

Then, install CalicoST using pip by
```
conda activate calicost_env
pip install -e .
```

Install dependencies
--------------------
CalicoST depends on two additional softwares that cannot be installed using conda in the previous step. If you need to use the corresponding steps in CalicoST, you need to install them manually as follows:

1. [**Eagle2**](https://alkesgroup.broadinstitute.org/Eagle/): Eagle2 is used in the preprocessing step to parse B allele count matrix from the BAM file.
```
wget https://storage.googleapis.com/broad-alkesgroup-public/Eagle/downloads/Eagle_v2.4.1.tar.gz
tar -xzf Eagle_v2.4.1.tar.gz
```
2. [**Startle**](https://github.com/raphael-group/startle): Startle is used in the phylogeny-reconstruction step to reconstruct a phylogeny based on CalicoST-inferred CNAs and cancer clones.
```
git clone --recurse-submodules https://github.com/raphael-group/startle.git
cd startle
mkdir build; cd build
cmake -DLIBLEMON_ROOT=<lemon path>\
        -DCPLEX_INC_DIR=<cplex include path>\
        -DCPLEX_LIB_DIR=<cplex lib path>\
        -DCONCERT_INC_DIR=<concert include path>\
        -DCONCERT_LIB_DIR=<concert lib path>\
        ..
make
```