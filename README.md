# CalicoST

<p align="center">
<img src="https://github.com/raphael-group/CalicoST/blob/main/docs/_static/img/overview4_combine.png?raw=true" width="100%" height="auto"/>
</p>

CalicoST is a probabilistic model that infers allele-specific copy number aberrations and tumor phylogeography from spatially resolved transcriptomics.CalicoST has the following key features:
1. Identifies allele-specific integer copy numbers for each transcribed region, revealing events such as copy neutral loss of heterozygosity (CNLOH) and mirrored subclonal CNAs that are invisible to total copy number analysis.
2. Assigns each spot a clone label indicating whether the spot is primarily normal cells or a cancer clone with aberration copy number profile.
3. Infers a phylogeny relating the identified cancer clones as well as a phylogeography that combines genetic evolution and spatial dissemination of clones.
4. Handles normal cell admixture in SRT technologies hat are not single-cell resolution (e.g. 10x Genomics Visium) to infer more accurate allele-specific copy numbers and cancer clones.
5.  Simultaneously analyzes multiple regional or aligned SRT slices from the same tumor.

# Installation
First setup a conda environment from the `environment.yml` file:
```
cd CalicoST
conda env create -f environment.yml
```

Then install CalicoST using pip by
```
conda activate calicost_env
pip install -e .
```

# Getting started
With the input data paths and running configurations specified in `config.yaml`, you can run CalicoST by
```
snakemake --cores <number threads> --configfile config.yaml --snakefile calicost.smk all
```

# Software dependencies
CalicoST uses the following command-line packages and python for extracting the BAF information
* samtools
* cellsnp-lite
* Eagle2
* pysam
CalicoST uses the following python packages for the remaining steps to infer allele-specific copy numbers and cancer clones:
* numpy
* scipy
* pandas
* scikit-learn
* scanpy
* anndata
* numba
* tqdm
* statsmodels
* networkx
* matplotlib
* seaborn
