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

python prerun_param_optimiser.py \
  --trees trees \
  --stem_to_test named_stem_branches_to_test.txt \
  --k_values 5,4.5,4,3.75,3.5,3.25,3,2.75,2.5,2.25,2,1,0.5 \
  --percentiles 0.99,0.98,0.97,0.96,0.95,0.94,0.93,0.92,0.91,0.90,0.85,0.80,0.75 \
  --alternative_cutoffs 1,2,3,4 \
  --progress




