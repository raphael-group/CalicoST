Specification of running parameters of CalicoST
===============================================

Supporting reference files
--------------------------
geneticmap_file: str
    The path to genetic map file.

hgtable_file: str
    The path to the location of genes in the genome. This file should be a tab-delimited file with the following columns: gene_name, chrom, cdsStart, cdsEnd.

normalidx_file: str, optional
    The path to the file containing the indices of normal spots in the spatial transcriptomics data. Each line is a single index without header.

tumorprop_file: str, optional
    The path to inferred tumor proportions per spot. This file should be a tab-delimited file with the following columns names: barcode, Tumor.

filtergenelist_file: str, optional
    The file to a list of genes to exclude from CNA inference, based on prior knowledge.

filterregion_file: str, optional
    The file to a list of genomic regions to exclude from CNA inference in BED format. E.g., HLA regions.


Phasing parameters
------------------
logphase_shift: float, optional
    Adjustment to the strength of Markov Model self-transition in phasing. The higher the value, the higher self-transition probability. Default is -1.0.

secondary_min_umi: int, optional
    The minimum UMI count a genome segment has in pseudobulk of spots in the step of genome segmentation. Default is 300.


Clone inference parameters
--------------------------
n_clones: int
    The number of clones to infer using only BAF signals. Default is 3.

n_clones_rdr: int, optional
    The number of clones to refine for each BAF-identified clone using RDR and BAF signals. Default is 2.

min_spots_per_clone: int, optional
    The minimum number of spots required to call a clone should have. Default is 100.

min_avgumi_per_clone: int, optional
    The minimum average UMI count required for a clone. Default is 10.

nodepotential: str, optional
    One of the following two options: "max" or "weighted_sum". "max" refers to using the MLE decoding of HMM in evaluating the probability of spots being in each clone. "weighted_sum" refers to using the full HMM posterior probabilities to evaluate the probability of spots being in each clone. Default is "weighted_sum".

spatial_weight: float, optional
    The strength of spatial coherence in HMRF. The higher the value, the stronger the spatial coherence. Default is 1.0.


CNA inference parameters
------------------------
n_states: int
    The number of allele-specific copy number states in the HMM for CNA inference.

t: float, optional
    The self-transition probability of HMM. The higher the value, the higher probability that adjacent genome segments are in the same CNA state. Default is 1-1e-5.

max_iter: int, optional
    The number of Baum-Welch steps to perform in HMM. Default is 30.

tol: float, optional
    The convergence threshold to terminate Baum-Welch steps. Default is 1e-4.


Merging clones with similar CNAs
--------------------------------
np_threshold: float, optional
    The threshold of Neyman Pearson statistics to decide two clones have distinct CNA events. The higher the value, the two clones are merged more easily. Default is 1.0.

np_eventminlen: int, optional
    The minimum number of consecutive genome segments to be considered as a CN event. Default is 10.