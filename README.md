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

# Run on a simulated example data
### Download data
The simulated count matrices can be downloaded from the [google drive](https://drive.google.com/drive/folders/19ZzfhyjKdEPTrQ7kh_HE6QvNfhmx4ng6?usp=drive_link).
CalicoST requires a reference SNP panel and phasing panel, which can be downloaded from
* [SNP panel](https://sourceforge.net/projects/cellsnp/files/SNPlist/genome1K.phase3.SNP_AF5e4.chr1toX.hg38.vcf.gz/download). You can also choose other SNP panels from [cellsnp-lite webpage](https://cellsnp-lite.readthedocs.io/en/latest/snp_list.html).
* [Phasing panel](http://pklab.med.harvard.edu/teng/data/1000G_hg38.zip)

### Run CalicoST
Replace the following paths in the `config.yaml` file from the downloaded google drive with the downloaded files on your machine
* calicost_dir: the path to CalicoST git-cloned code.
* eagledir: the path to Eagle2 directory
* region_vcf: the path to the downloaded SNP panel.
* phasing_panel: the path to the downloaded and unzipped phasing panel.
* hgtable_file, filtergenelist_file, filterregion_file, outputdir: the path to the corresponding files in the downloaded google drive folder.

To avoid falling into local maxima in CalicoST's optimization objective, we recommend run CalicoST with multiple random initializations with a list random seed specified by `random_state` in the `config.yaml` file. The provided one uses five random initializations.

Then run CalicoST by
```
snakemake --cores 5 --configfile config.yaml --snakefile <calicost_dir>/calicost.smk all
```

### Understanding the output
Each random initialization of CalicoST generates a folder of `<outputdir>/<output_foldername>/clone*`. 

CalicoST outputs the following key files:
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
