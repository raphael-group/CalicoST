Installation
============
Minimum installation
--------------------
First setup a conda environment from the `environment.yml` file:

..  code-block:: bash

        git clone https://github.com/raphael-group/CalicoST.git
        cd CalicoST
        conda env create -f environment.yml --name calicost_env


Then, install CalicoST using pip by

..  code-block:: bash

        conda activate calicost_env
        pip install -e .


Setting up the conda environments takes around 15 minutes on an HPC head node.

Additional installation for SNP parsing
---------------------------------------
CalicoST requires allele count matrices for reference-phased A and B alleles for inferring allele-specific CNAs, and provides a snakemake pipeline for obtaining the required matrices from a BAM file. Run the following commands in CalicoST directory for installing additional package, [Eagle2](https://alkesgroup.broadinstitute.org/Eagle/), for snakemake preprocessing pipeline.

..  code-block:: bash

        mkdir external
        wget --directory-prefix=external https://storage.googleapis.com/broad-alkesgroup-public/Eagle/downloads/Eagle_v2.4.1.tar.gz
        tar -xzf external/Eagle_v2.4.1.tar.gz -C external


Additional installation for reconstructing phylogeny
----------------------------------------------------
Based on the inferred cancer clones and allele-specific CNAs by CalicoST, we apply Startle to reconstruct a phylogenetic tree along the clones. Install Startle by

..  code-block:: bash

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


Prepare reference files for SNP parsing
--------------------
We followed the recommended pipeline by `Numbat <https://kharchenkolab.github.io/numbat/>`_` for parsing SNP information from BAM file(s): first genotyping using the BAM file by cellsnp-lite (included in the conda environment) and reference-based phasing by Eagle2. Download the following panels for genotyping and reference-based phasing.

* `SNP panel <https://sourceforge.net/projects/cellsnp/files/SNPlist/genome1K.phase3.SNP_AF5e4.chr1toX.hg38.vcf.gz>`_ - 0.5GB in size. You can also choose other SNP panels from `cellsnp-lite webpage <https://cellsnp-lite.readthedocs.io/en/latest/main/data.html#data-list-of-common-snps>`_.
* `Phasing panel <http://pklab.med.harvard.edu/teng/data/1000G_hg38.zip>`_ - 9.0GB in size. Unzip the panel after downloading.
