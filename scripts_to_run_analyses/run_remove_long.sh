#!/bin/bash
 
#SBATCH --job-name=iqt500
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=10GB
#SBATCH --time=00:40:00
##SBATCH --exclusive

module load PrgEnv-cray/8.5.0
cd $SLURM_SUBMIT_DIR

source /home/b35bw/bzxdp.b35bw/miniforge3/etc/profile.d/conda.sh

conda activate phylo_programming

python long_branch_identifier.py \
  --trees trees \
  --tip_ratio_threshold 3.0 \
  --cross_gene_ratio_threshold 5 \
  --clade_stem_ratio_threshold 4 \
  --long_branch_clades stem_to_exclude_from_quartet_checks.txt \
  --stem_to_test named_stem_branches_to_test.txt \
  --internal_branch_ratio_threshold 3.0\
  --internal_side_ratio_threshold 1.2 \
  --global_rescue 0.99 \
  --global_stem_rescue 0.95 \
  --global_internal_rescue 0.99 \
  --internal_path_consistency_threshold 0.5 \
  --internal_min_path_branches 2 \
  --hybrid_terminal \
  --hybrid_stem \
  --progress




