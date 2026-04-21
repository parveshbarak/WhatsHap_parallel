#!/bin/bash
#PBS -N mec_runs
#PBS -q regular
#PBS -l nodes=1:ppn=48
#PBS -l walltime=23:59:00
#PBS -o ont_out.txt
#PBS -e ont_err.txt


source ~/.bashrc
conda activate hale
cd /home/parvesh/atcg/PP/PP_project/WhatsHap_parallel
module load gcc-9.3.0


# whatshap phase /scratch/seq/chr18_HG002_HiFi_60x_variants.vcf.gz /scratch/seq/chr18_HG002_HiFi_60x_to_GRCh38.bam --reference /scratch/references/chr18_GCA_000001405.15_GRCh38_no_alt_analysis_set.fna --only-snvs --ignore-read-groups --mec-matrix mec_matrix_down_to_15.txt -o ../experiments/chr18_HG002_HiFi_60x_variants_phased.vcf.gz

# whatshap phase /scratch/seq/chr18_HG002_HiFi_60x_variants.vcf.gz /scratch/seq/chr18_HG002_HiFi_60x_to_GRCh38.bam --reference /scratch/references/chr18_GCA_000001405.15_GRCh38_no_alt_analysis_set.fna --only-snvs --ignore-read-groups --internal-downsampling 20 --mec-matrix mec_matrix_down_to_20.txt -o ../experiments/chr18_HG002_HiFi_60x_variants_phased.vcf.gz

# whatshap phase /scratch/seq/chr18_HG002_HiFi_60x_variants.vcf.gz /scratch/seq/chr18_HG002_HiFi_60x_to_GRCh38.bam --reference /scratch/references/chr18_GCA_000001405.15_GRCh38_no_alt_analysis_set.fna --only-snvs --ignore-read-groups --internal-downsampling 25 --mec-matrix mec_matrix_down_to_25.txt -o ../experiments/chr18_HG002_HiFi_60x_variants_phased.vcf.gz

# whatshap phase /scratch/seq/chr18_HG002_HiFi_60x_variants.vcf.gz /scratch/seq/chr18_HG002_HiFi_60x_to_GRCh38.bam --reference /scratch/references/chr18_GCA_000001405.15_GRCh38_no_alt_analysis_set.fna --only-snvs --ignore-read-groups --internal-downsampling 30 --mec-matrix mec_matrix_down_to_30.txt -o ../experiments/chr18_HG002_HiFi_60x_variants_phased.vcf.gz