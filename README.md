# CalicoST

<p align="center">
<img src="https://github.com/raphael-group/CalicoST/blob/main/docs/_static/img/overview4_combine.png?raw=true" width="100%" height="auto"/>
</p>

CalicoST is a probabilistic model that infers allele-specific copy number aberrations and tumor phylogeography from spatially resolved transcriptomics.CalicoST has the following key features:
1. Identifies allele-specific integer copy numbers for each transcribed region, revealing events such as copy neutral loss of heterozygosity (CNLOH) and mirrored subclonal CNAs that are invisible to total copy number analysis.
2. Assigns each spot a clone label indicating whether the spot is primarily normal cells or a cancer clone with aberration copy number profile.
3. Infers a phylogeny relating the identified cancer clones as well as a phylogeography that combines genetic evolution and spatial dissemination of clones.
4. Handles normal cell admixture in SRT technologies hat are not single-cell resolution (e.g. 10x Genomics Visium) to infer more accurate allele-specific copy numbers and cancer clones.
5.  Simultaneously analyzes multiple regions or aligned SRT slices from the same tumor.

# System requirements
The package has tested on the following Linux operating systems: SpringdaleOpenEnterprise 9.2 (Parma) and CentOS Linux 7 (Core).

# Installation
## Minimum installation
First setup a conda environment from the `environment.yml` file:
```
git clone https://github.com/raphael-group/CalicoST.git
cd CalicoST
conda env create -f environment.yml --name calicost_env
```


Then, install CalicoST using pip by
```
conda activate calicost_env
pip install -e .
```

Setting up the conda environments takes around 15 minutes on an HPC head node.

## Additional installation for SNP parsing
CalicoST requires allele count matrices for reference-phased A and B alleles for inferring allele-specific CNAs, and provides a snakemake pipeline for obtaining the required matrices from a BAM file. Run the following commands in CalicoST directory for installing additional package, [Eagle2](https://alkesgroup.broadinstitute.org/Eagle/), for snakemake preprocessing pipeline.

```
mkdir external
wget --directory-prefix=external https://storage.googleapis.com/broad-alkesgroup-public/Eagle/downloads/Eagle_v2.4.1.tar.gz
tar -xzf external/Eagle_v2.4.1.tar.gz -C external
```

## Additional installation for reconstructing phylogeny
Based on the inferred cancer clones and allele-specific CNAs by CalicoST, we apply Startle to reconstruct a phylogenetic tree along the clones. Install Startle by
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


# Getting started
### Preprocessing: genotyping and reference-based phasing
To infer allele-specific CNAs, we generate allele count matrices in this preprocessing step. We followed the recommended pipeline by [Numbat](https://kharchenkolab.github.io/numbat/), which is designed for scRNA-seq data to infer clones and CNAs: first genotyping using the BAM file by cellsnp-lite (included in the conda environment) and reference-based phasing by Eagle2. Download the following panels for genotyping and reference-based phasing.
* [SNP panel](https://sourceforge.net/projects/cellsnp/files/SNPlist/genome1K.phase3.SNP_AF5e4.chr1toX.hg38.vcf.gz) - 0.5GB in size. You can also choose other SNP panels from [cellsnp-lite webpage](https://cellsnp-lite.readthedocs.io/en/latest/main/data.html#data-list-of-common-snps).
* [Phasing panel](http://pklab.med.harvard.edu/teng/data/1000G_hg38.zip)- 9.0GB in size. Unzip the panel after downloading.

Replace the following paths `config.yaml`:
* `region_vcf`: Replace with the path of downloaded SNP panel.
* `phasing_panel`: Replace with the unzipped directory of the downloaded phasing panel.
* `spaceranger_dir`: Replace with the spaceranger directory of your Visium data, which should contain the BAM file `possorted_genome_bam.bam`.
* `output_snpinfo`: Replace with the desired output directory.
* Replace `calicost_dir` and `eagledir` with the path to the cloned CalicoST directory and downloaded Eagle2 directory.

Then you can run preprocessing pipeline by
```
snakemake --cores <number threads> --configfile config.yaml --snakefile calicost.smk all
```

### Inferring tumor purity per spot (optional)
Replace the paths in the parameter configuration file `configuration_purity` with the corresponding data/reference file paths and run
```
OMP_NUM_THREADS=1 <CalicoST directory>/src/calicost/estimate_tumor_proportion.py -c configuration_purity
```

### Inferring clones and allele-specific CNAs
Replace the paths in parameter configuration file `configuration_cna` with the corresponding data/reference file paths and run
```
OMP_NUM_THREADS=1 python <CalicoST directory>/src/calicost/calicost_main.py -c configuration_cna
```

When jointly inferring clones and CNAs across multiple SRT slices, prepare a table with the following columns (See [`examples/example_input_filelist`](https://github.com/raphael-group/CalicoST/blob/main/examples/example_input_filelist) as an example). 
Path to BAM file | sample ID | Path to Spaceranger outs
Modify `configuration_cna_multi` with paths to the table and run
```
OMP_NUM_THREADS=1 python <CalicoST directory>/src/calicost/calicost_main.py -c configuration_cna_multi
```

### Reconstruct phylogeography

```
python <CalicoST directory>/src/calicost/phylogeny_startle.py -c <CalicoST clone and CNA output directory> -s <startle executable path> -o <output directory>
```


# Tutorials
Check out our [readthedocs](https://calicost.readthedocs.io/en/latest/) for the following tutorials:
1. [Inferring clones and allele-specific CNAs on simulated data](https://calicost.readthedocs.io/en/latest/notebooks/tutorials/simulated_data_tutorial.html)
The simulated count matrices and parameter configuration file are available from [`examples/simulated_example.tar.gz`](https://github.com/raphael-group/CalicoST/blob/main/examples/simulated_example.tar.gz). CalicoST takes about 2h to finish on this example.

2. [Inferring tumor purity, clones, allele-specific CNAs, and phylogeography on prostate cancer data](https://calicost.readthedocs.io/en/latest/notebooks/tutorials/prostate_tutorial.html)
The transcript count, allele count matrices, and running configuration fies are available from [`examples/prostate_example.tar.gz`](https://github.com/raphael-group/CalicoST/blob/main/examples/prostate_example.tar.gz). This sample contains five slices and over 10000 spots, CalicoST takes about 9h to finish on this example.

<!-- CalicoST requires a reference SNP panel and phasing panel, which can be downloaded from
* [SNP panel](https://sourceforge.net/projects/cellsnp/files/SNPlist/genome1K.phase3.SNP_AF5e4.chr1toX.hg38.vcf.gz/download). You can also choose other SNP panels from [cellsnp-lite webpage](https://cellsnp-lite.readthedocs.io/en/latest/snp_list.html).
* [Phasing panel](http://pklab.med.harvard.edu/teng/data/1000G_hg38.zip) -->

<!-- ### Run CalicoST
Untar the downloaded example data. Replace the following paths in the `example_config.yaml`  of the downloaded example data with paths on your machine
* calicost_dir: the path to CalicoST git-cloned code.
* eagledir: the path to Eagle2 directory
* region_vcf: the path to the downloaded SNP panel.
* phasing_panel: the path to the downloaded and unzipped phasing panel.

To avoid falling into local maxima in CalicoST's optimization objective, we recommend run CalicoST with multiple random initializations with a list random seed specified by `random_state` in the `example_config.yaml` file. The provided one uses five random initializations.

Then run CalicoST by
```
cd <directory of downloaded example data>
snakemake --cores 5 --configfile example_config.yaml --snakefile <calicost_dir>/calicost.smk all
```

CalicoST takes about 69 minutes to finish on this example using 5 cores on an HPC. -->

### Understanding the output
The above snakemake run will create a folder `calicost` in the directory of downloaded example data. Within this folder, each random initialization of CalicoST generates a subdirectory of `calicost/clone*`. 

CalicoST generates the following key files of each random initialization:
* clone_labels.tsv: The inferred clone labels for each spot.
* cnv_seglevel.tsv: Allele-specific copy numbers for each clone for each genome segment.
* cnv_genelevel.tsv: The projected allele-specific copy numbers from genome segments to the covered genes.
* cnv_diploid_seglevel.tsv, cnv_triploid_seglevel.tsv, cnv_tetraploid_seglevel.tsv, cnv_diploid_genelevel.tsv, cnv_triploid_genelevel.tsv, cnv_tetraploid_genelevel.tsv: Allele-specific copy numbers when enforcing a ploidy for each genome segment or each gene.

See the following examples of the key files.
```
head -10 calicost/clone3_rectangle0_w1.0/clone_labels.tsv
BARCODES        clone_label
spot_0  2
spot_1  2
spot_2  2
spot_3  2
spot_4  2
spot_5  2
spot_6  2
spot_7  2
spot_8  0
```

```
head -10 calicost/clone3_rectangle0_w1.0/cnv_seglevel.tsv
CHR     START   END     clone0 A        clone0 B        clone1 A        clone1 B        clone2 A        clone2 B
1       1001138 1616548 1       1       1       1       1       1
1       1635227 2384877 1       1       1       1       1       1
1       2391775 6101016 1       1       1       1       1       1
1       6185020 6653223 1       1       1       1       1       1
1       6785454 7780639 1       1       1       1       1       1
1       7784320 8020748 1       1       1       1       1       1
1       8026738 9271273 1       1       1       1       1       1
1       9292894 10375267        1       1       1       1       1       1
1       10398592        11922488        1       1       1       1       1       1
```

```
head -10 calicost/clone3_rectangle0_w1.0/cnv_genelevel.tsv
gene    clone0 A        clone0 B        clone1 A        clone1 B        clone2 A        clone2 B
A1BG    1       1       1       1       1       1
A1CF    1       1       1       1       1       1
A2M     1       1       1       1       1       1
A2ML1-AS1       1       1       1       1       1       1
AACS    1       1       1       1       1       1
AADAC   1       1       1       1       1       1
AADACL2-AS1     1       1       1       1       1       1
AAK1    1       1       1       1       1       1
AAMP    1       1       1       1       1       1
```

CalicoST graphs the following plots for visualizing the inferred cancer clones in space and allele-specific copy number profiles for each random initialization.
* plots/clone_spatial.pdf: The spatial distribution of inferred cancer clones and normal regions (grey color, clone 0 by default)
* plots/rdr_baf_defaultcolor.pdf: The read depth ratio (RDR) and B allele frequency (BAF) along the genome for each clone. Higher RDR indicates higher total copy numbers, and a deviation-from-0.5 BAF indicates allele imbalance due to allele-specific CNAs.
* plots/acn_genome.pdf: The default allele-specific copy numbers along the genome.
* plots/acn_genome_diploid.pdf, plots/acn_genome_triploid.pdf, plots/acn_genome_tetraploid.pdf: Allele-specific copy numbers when enforcing a ploidy.

The allele-specific copy number plots have the following color legend.
<p align="center">
<img src="https://github.com/raphael-group/CalicoST/blob/main/docs/_static/img/acn_color_palette.png?raw=true" width="20%" height="auto"/>
</p>


# Software dependencies
CalicoST uses the following command-line packages and python for extracting the BAF information
* samtools
* cellsnp-lite
* Eagle2
* pysam
* snakemake

CalicoST uses the following packages for the remaining steps to infer allele-specific copy numbers and cancer clones:
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
* snakemake


# Citations
The CalicoST manuscript is available on bioRxiv. If you use CalicoST for your work, please cite our paper.
```
@article{ma2024inferring,
  title={Inferring allele-specific copy number aberrations and tumor phylogeography from spatially resolved transcriptomics},
  author={Ma, Cong and Balaban, Metin and Liu, Jingxian and Chen, Siqi and Ding, Li and Raphael, Benjamin},
  journal={bioRxiv},
  pages={2024--03},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```