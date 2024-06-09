import numpy as np
import pandas as pd
import scipy
import calicost.arg_parse
import calicost.parse_input


rule all:
    input:
        f"{config['output_snpinfo']}/cell_snp_Aallele.npz",


rule link_or_merge_bam:
    output:
        bam="{outputdir}/possorted_genome_bam.bam",
        bai="{outputdir}/possorted_genome_bam.bam.bai",
        barcodefile="{outputdir}/barcodes.txt",
    params:
        outputdir = "{outputdir}",
        samtools_sorting_mem=config['samtools_sorting_mem']
    threads: 1
    log:
        "{outputdir}/logs/link_or_merge_bam.log"
    run:
        if "bamlist" in config:
            # merged BAM file
            shell(f"python {config['calicost_dir']}/utils/merge_bamfile.py -b {config['bamlist']} -o {params.outputdir}/ >> {log} 2>&1")
            shell(f"samtools sort -m {params.samtools_sorting_mem} -o {output.bam} {params.outputdir}/unsorted_possorted_genome_bam.bam >> {log} 2>&1")
            shell(f"samtools index {output.bam}")
            shell(f"rm -fr {params.outputdir}/unsorted_possorted_genome_bam.bam")
            
            # merged barcodes
            df_entries = pd.read_csv(config["bamlist"], sep='\t', index_col=None, header=None)
            df_barcodes = []
            for i in range(df_entries.shape[0]):
                tmpdf = pd.read_csv(f"{df_entries.iloc[i,2]}/filtered_feature_bc_matrix/barcodes.tsv.gz", header=None, index_col=None)
                tmpdf.iloc[:,0] = [f"{x}_{df_entries.iloc[i,1]}" for x in tmpdf.iloc[:,0]]
                df_barcodes.append( tmpdf )
            df_barcodes = pd.concat(df_barcodes, ignore_index=True)
            df_barcodes.to_csv(f"{output.barcodefile}", sep='\t', index=False, header=False)
        else:
            # BAM file
            assert "spaceranger_dir" in config
            print("softlink of possorted_genome_bam.bam")
            shell(f"ln -sf -T {config['spaceranger_dir']}/possorted_genome_bam.bam {output.bam}")
            shell(f"ln -sf -T {config['spaceranger_dir']}/possorted_genome_bam.bam.bai {output.bai}")
            # barcodes
            shell(f"gunzip -c {config['spaceranger_dir']}/filtered_feature_bc_matrix/barcodes.tsv.gz > {output.barcodefile}")



rule genotype:
    input:
        barcodefile="{outputdir}/barcodes.txt",
        bam="{outputdir}/possorted_genome_bam.bam",
        bai="{outputdir}/possorted_genome_bam.bam.bai"
    output:
        vcf="{outputdir}/genotyping/cellSNP.base.vcf.gz"
    params:
        outputdir="{outputdir}",
        region_vcf=config['region_vcf']
    threads: config['nthreads_cellsnplite']
    log:
        "{outputdir}/logs/genotyping.log"
    run:
        shell(f"mkdir -p {params.outputdir}/genotyping")
        command = f"cellsnp-lite -s {input.bam} " + \
             f"-b {input.barcodefile} " + \
             f"-O {params.outputdir}/genotyping/ " + \
             f"-R {params.region_vcf} " + \
             f"-p {threads} " + \
             f"--minMAF 0 --minCOUNT 2 --UMItag {config['UMItag']} --cellTAG {config['cellTAG']} --gzip >> {log} 2>&1"
        print(command)
        shell(command)
        


rule pre_phasing:
    input:
        vcf="{outputdir}/genotyping/cellSNP.base.vcf.gz"
    output:
        expand("{{outputdir}}/phasing/chr{chrname}.vcf.gz", chrname=config["chromosomes"])
    params:
        outputdir="{outputdir}",
    threads: 1
    run:
        shell(f"mkdir -p {params.outputdir}/phasing")
        print(f"python {config['calicost_dir']}/utils/filter_snps_forphasing.py -c {params.outputdir}/genotyping -o {params.outputdir}/phasing")
        shell(f"python {config['calicost_dir']}/utils/filter_snps_forphasing.py -c {params.outputdir}/genotyping -o {params.outputdir}/phasing")
        for chrname in config["chromosomes"]:
            shell(f"bgzip -f {params.outputdir}/phasing/chr{chrname}.vcf")
            shell(f"tabix -f {params.outputdir}/phasing/chr{chrname}.vcf.gz")


rule phasing:
    input:
        vcf="{outputdir}/phasing/chr{chrname}.vcf.gz"
    output:
        "{outputdir}/phasing/chr{chrname}.phased.vcf.gz"
    params:
        outputdir="{outputdir}",
        chrname="{chrname}",
    threads: 2
    log:
        "{outputdir}/logs/phasing_chr{chrname}.log",
    run:
        command = f"{config['eagledir']}/eagle --numThreads {threads} --vcfTarget {input.vcf} " + \
            f"--vcfRef {config['phasing_panel']}/chr{params.chrname}.genotypes.bcf " + \
            f"--geneticMapFile={config['eagledir']}/tables/genetic_map_hg38_withX.txt.gz "+ \
            f"--outPrefix {params.outputdir}/phasing/chr{params.chrname}.phased >> {log} 2>&1"
        shell(command)
        


rule parse_final_snp:
    input:
        "{outputdir}/genotyping/cellSNP.base.vcf.gz",
        expand("{{outputdir}}/phasing/chr{chrname}.phased.vcf.gz", chrname=config["chromosomes"]),
    output:
        "{outputdir}/cell_snp_Aallele.npz",
        "{outputdir}/cell_snp_Ballele.npz",
        "{outputdir}/unique_snp_ids.npy"
    params:
        outputdir="{outputdir}",
    threads: 1
    log:
        "{outputdir}/logs/parse_final_snp.log"
    run:
        command = f"python {config['calicost_dir']}/utils/get_snp_matrix.py " + \
            f"-c {params.outputdir}/genotyping -e {params.outputdir}/phasing -b {params.outputdir}/barcodes.txt -o {params.outputdir}/ >> {log} 2>&1"
        shell( command )

