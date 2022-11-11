#!/bin/bash

#SBATCH -J 72-1less_ 
#SBATCH -o out_files/72-1less_.out
#SBATCH -n 1
#SBATCH -G 1
#SBATCH -c 1                            # one CPU core
#SBATCH -t 3-00:00:00
#SBATCH --mem=40G


# Load software
# conda init bash
source /home/x_ricup/miniconda3/etc/profile.d/conda.sh
conda activate MTLenv

# Run python script
srun python main_meta_test.py --config_exp exp_configs/meta_testing.yaml