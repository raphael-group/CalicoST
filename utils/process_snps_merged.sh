#!/bin/bash

##### input and output data paths #####
# SAMPLE_ID is used for setting directory/file name
SAMPLE_ID="joint_H1_245_H2_1"
INPUTLIST="/u/congma/ragr-data/datasets/spatial_cna/Lundeberg_organwide/P1_snps/joint_H1_245_H2_1/bamfile_list.tsv"
BAMFILE="/u/congma/ragr-data/datasets/spatial_cna/Lundeberg_organwide/P1_snps/joint_H1_245_H2_1/possorted_genome_bam.bam"
OUTDIR="/u/congma/ragr-data/datasets/spatial_cna/Lundeberg_organwide/P1_snps/joint_H1_245_H2_1/visium_snpnew"

NTHREADS=20

##### reference file paths #####
# PHASING_PANEL is downloaded as instructed in numbat "1000G Reference Panel" and then unzipped. Link to download: wget http://pklab.med.harvard.edu/teng/data/1000G_hg38.zip
PHASING_PANEL="/u/congma/ragr-data/users/congma/references/phasing_ref/1000G_hg38/"
# REGION_VCF serves as the same purpose as "1000G SNP reference file" in numbat, but using a larger SNP set. Link to download: wget https://sourceforge.net/projects/cellsnp/files/SNPlist/genome1K.phase3.SNP_AF5e4.chr1toX.hg38.vcf.gz
REGION_VCF="/u/congma/ragr-data/users/congma/references/snplist/nocpg.genome1K.phase3.SNP_AF5e4.chr1toX.hg38.vcf.gz"
# HGTABLE_FILE specifies gene positions in the genome, for mapping SNPs to genes. Link to download: https://github.com/raphael-group/STARCH/blob/develop/hgTables_hg38_gencode.txt
HGTABLE_FILE="/u/congma/ragr-data/users/congma/Codes/STARCH_crazydev/hgTables_hg38_gencode.txt"
# there is a reference file in eagle folder
eagledir="/u/congma/ragr-data/users/congma/environments/Eagle_v2.4.1/"


##### Following are commands for calling + phasing + processing SNPs #####
# index bam file
if [[ ! -e ${BAMFILE}.bai ]]; then
    samtools index ${BAMFILE}
fi

# write required barcode list file
mkdir -p ${OUTDIR}
touch ${OUTDIR}/barcodes.txt
>${OUTDIR}/barcodes.txt
while read -r line; do
	CELLRANGER_OUT=$(echo ${line} | awk '{print $3}')
	suffix=$(echo ${line} | awk '{print $2}')
	gunzip -c ${CELLRANGER_OUT}/filtered_feature_bc_matrix/barcodes.tsv.gz | awk -v var=${suffix} '{print $0"_"var}' >> ${OUTDIR}/barcodes.txt
done < ${INPUTLIST}

# run cellsnp-lite
mkdir -p ${OUTDIR}/pileup/${SAMPLE_ID}
cellsnp-lite -s ${BAMFILE} \
             -b ${OUTDIR}/barcodes.txt \
             -O ${OUTDIR}/pileup/${SAMPLE_ID} \
             -R ${REGION_VCF} \
             -p ${NTHREADS} \
             --minMAF 0 --minCOUNT 2 --UMItag Auto --cellTAG CB

# run phasing
mkdir -p ${OUTDIR}/phasing/
SCRIPTDIR="/u/congma/ragr-data/users/congma/Codes/STARCH_crazydev/scripts"
python ${SCRIPTDIR}/filter_snps_forphasing.py ${SAMPLE_ID} ${OUTDIR}
for chr in {1..22}; do
    bgzip -f ${OUTDIR}/phasing/${SAMPLE_ID}_chr${chr}.vcf
    tabix ${OUTDIR}/phasing/${SAMPLE_ID}_chr${chr}.vcf.gz
    ${eagledir}/eagle --numThreads ${NTHREADS} \
          --vcfTarget ${OUTDIR}/phasing/${SAMPLE_ID}_chr${chr}.vcf.gz \
          --vcfRef ${PHASING_PANEL}/chr${chr}.genotypes.bcf \
          --geneticMapFile=${eagledir}/tables/genetic_map_hg38_withX.txt.gz \
          --outPrefix ${OUTDIR}/phasing/${SAMPLE_ID}_chr${chr}.phased
done


# run my pythonn to get a cell-by-gene matrix of SNP-covering UMI counts
#SCRIPTDIR=$(dirname "$0")
python ${SCRIPTDIR}/get_snp_matrix.py ${OUTDIR} ${SAMPLE_ID} ${HGTABLE_FILE} ${OUTDIR}/barcodes.txt ${CELLRANGER_OUT}/filtered_feature_bc_matrix/
