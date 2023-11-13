import numpy as np
import pandas as pd
import scipy
import calicost.arg_parse
import calicost.parse_input


rule all:
    input:
        f"{config['outputdir']}/{config['output_foldername']}/parsed_inputs/table_bininfo.csv.gz",
        expand(f"{config['outputdir']}/{config['output_foldername']}/summary{{r}}", r=config['random_state'])


rule link_or_merge_bam:
    output:
        bam="{outputdir}/possorted_genome_bam.bam",
        bai="{outputdir}/possorted_genome_bam.bam.bai",
        barcodefile="{outputdir}/snpinfo/barcodes.txt",
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
            shell(f"{config['samtools']} sort -m {params.samtools_sorting_mem} -o {output.bam} {params.outputdir}/unsorted_possorted_genome_bam.bam >> {log} 2>&1")
            shell(f"{config['samtools']} index {output.bam}")
            shell(f"rm -fr {params.outputdir}/unsorted_possorted_genome_bam.bam")
            
            # merged barcodes
            df_entries = pd.read_csv(config["bamlist"], sep='\t', index_col=None, header=0)
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
        barcodefile="{outputdir}/snpinfo/barcodes.txt",
        bam="{outputdir}/possorted_genome_bam.bam",
        bai="{outputdir}/possorted_genome_bam.bam.bai"
    output:
        vcf="{outputdir}/snpinfo/genotyping/cellSNP.base.vcf.gz"
    params:
        outputdir="{outputdir}",
        region_vcf=config['region_vcf']
    threads: config['nthreads_cellsnplite']
    log:
        "{outputdir}/logs/genotyping.log"
    run:
        shell(f"mkdir -p {params.outputdir}/snpinfo/genotyping")
        command = f"{config['cellsnplite']} -s {input.bam} " + \
             f"-b {input.barcodefile} " + \
             f"-O {params.outputdir}/snpinfo/genotyping/ " + \
             f"-R {params.region_vcf} " + \
             f"-p {threads} " + \
             f"--minMAF 0 --minCOUNT 2 --UMItag {config['UMItag']} --cellTAG {config['cellTAG']} --gzip >> {log} 2>&1"
        print(command)
        shell(command)
        


rule pre_phasing:
    input:
        vcf="{outputdir}/snpinfo/genotyping/cellSNP.base.vcf.gz"
    output:
        expand("{{outputdir}}/snpinfo/phasing/chr{chrname}.vcf.gz", chrname=config["chromosomes"])
    params:
        outputdir="{outputdir}",
    threads: 1
    run:
        shell(f"mkdir -p {params.outputdir}/snpinfo/phasing")
        print(f"python {config['calicost_dir']}/utils/filter_snps_forphasing.py -c {params.outputdir}/snpinfo/genotyping -o {params.outputdir}/snpinfo/phasing")
        shell(f"python {config['calicost_dir']}/utils/filter_snps_forphasing.py -c {params.outputdir}/snpinfo/genotyping -o {params.outputdir}/snpinfo/phasing")
        for chrname in config["chromosomes"]:
            shell(f"{config['bgzip']} -f {params.outputdir}/snpinfo/phasing/chr{chrname}.vcf")
            shell(f"{config['tabix']} -f {params.outputdir}/snpinfo/phasing/chr{chrname}.vcf.gz")


rule phasing:
    input:
        vcf="{outputdir}/snpinfo/phasing/chr{chrname}.vcf.gz"
    output:
        "{outputdir}/snpinfo/phasing/chr{chrname}.phased.vcf.gz"
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
            f"--outPrefix {params.outputdir}/snpinfo/phasing/chr{params.chrname}.phased >> {log} 2>&1"
        shell(command)
        


rule parse_final_snp:
    input:
        "{outputdir}/snpinfo/genotyping/cellSNP.base.vcf.gz",
        expand("{{outputdir}}/snpinfo/phasing/chr{chrname}.phased.vcf.gz", chrname=config["chromosomes"]),
    output:
        "{outputdir}/snpinfo/cell_snp_Aallele.npz",
        "{outputdir}/snpinfo/cell_snp_Ballele.npz",
        "{outputdir}/snpinfo/unique_snp_ids.npy"
    params:
        outputdir="{outputdir}",
    threads: 1
    log:
        "{outputdir}/logs/parse_final_snp.log"
    run:
        command = f"python {config['calicost_dir']}/utils/get_snp_matrix.py " + \
            f"-c {params.outputdir}/snpinfo/genotyping -e {params.outputdir}/snpinfo/phasing -b {params.outputdir}/snpinfo/barcodes.txt -o {params.outputdir}/snpinfo/ >> {log} 2>&1"
        shell( command )


rule write_calicost_configfile:
    input:
        "{outputdir}/snpinfo/cell_snp_Aallele.npz",
    output:
        expand("{{outputdir}}/" + config['output_foldername'] + "/configfile{r}", r=config['random_state'])
    params:
        outputdir="{outputdir}",
    threads: 1
    run:
        if "bamlist" in config:
            calicost_config = calicost.arg_parse.get_default_config_joint()
        else:
            calicost_config = calicost.arg_parse.get_default_config_single()
        
        # update input
        calicost_config['snp_dir'] = f"{params.outputdir}/snpinfo/"
        calicost_config['output_dir'] = f"{params.outputdir}/{config['output_foldername']}"
        if 'spaceranger_dir' in calicost_config:
            assert 'spaceranger_dir' in config
            calicost_config['spaceranger_dir'] = config['spaceranger_dir']
        if 'input_filelist' in calicost_config:
            assert 'bamlist' in config
            calicost_config['input_filelist'] = config['bamlist']

        for k in calicost_config.keys():
            if k in config:
                calicost_config[k] = config[k]

        for r in config['random_state']:
            calicost_config["num_hmrf_initialization_start"] = r
            calicost_config["num_hmrf_initialization_end"] = r+1
            calicost.arg_parse.write_config_file(f"{params.outputdir}/{config['output_foldername']}/configfile{r}", calicost_config)


rule prepare_calicost_data:
    input:
        expand("{{outputdir}}/" + config['output_foldername'] + "/configfile{r}", r=config['random_state']),
    output:
        f"{{outputdir}}/{config['output_foldername']}/parsed_inputs/table_bininfo.csv.gz",
        f"{{outputdir}}/{config['output_foldername']}/parsed_inputs/table_rdrbaf.csv.gz",
        f"{{outputdir}}/{config['output_foldername']}/parsed_inputs/table_meta.csv.gz",
        f"{{outputdir}}/{config['output_foldername']}/parsed_inputs/exp_counts.pkl",
        f"{{outputdir}}/{config['output_foldername']}/parsed_inputs/adjacency_mat.npz",
        f"{{outputdir}}/{config['output_foldername']}/parsed_inputs/smooth_mat.npz"
    params:
        outputdir="{outputdir}",
    threads: 1
    log:
        "{outputdir}/logs/prepare_calicost_data.log"
    run:
        command = f"OMP_NUM_THREADS=1 python {config['calicost_dir']}/src/calicost/parse_input.py -c {input[0]} >> {log} 2>&1"
        shell(command)


rule run_calicost:
    input:
        f"{{outputdir}}/{config['output_foldername']}/configfile{{r}}",
        f"{{outputdir}}/{config['output_foldername']}/parsed_inputs/table_bininfo.csv.gz",
        f"{{outputdir}}/{config['output_foldername']}/parsed_inputs/table_rdrbaf.csv.gz",
        f"{{outputdir}}/{config['output_foldername']}/parsed_inputs/table_meta.csv.gz",
        f"{{outputdir}}/{config['output_foldername']}/parsed_inputs/exp_counts.pkl",
        f"{{outputdir}}/{config['output_foldername']}/parsed_inputs/adjacency_mat.npz",
        f"{{outputdir}}/{config['output_foldername']}/parsed_inputs/smooth_mat.npz"
    output:
        f"{{outputdir}}/{config['output_foldername']}/summary{{r}}",
    params:
        outputdir="{outputdir}",
        r="{r}"
    threads: 1
    log:
        "{outputdir}/logs/calicost_run_{r}.log"
    run:
        command = f"OMP_NUM_THREADS=1 python {config['calicost_dir']}/src/calicost/calicost_main.py -c {input[0]} >> {log} 2>&1"
        shell(command)
        shell(f"echo {command} > {output}")
