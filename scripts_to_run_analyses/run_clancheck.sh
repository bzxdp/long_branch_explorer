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

python clan_check.py \
  --trees trees \
  --clan_check clades_to_clan_check.txt   \
  --lbr_results_ext .out \
  --progress




