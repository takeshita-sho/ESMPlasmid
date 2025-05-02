#!/bin/bash
#SBATCH -J graph # Job name
#SBATCH -o /nfshomes/stakeshi/esm/esm/model/graph-%j.o # Name of output file
#SBATCH -e /nfshomes/stakeshi/esm/esm/model/graph-%j.e # Name of error file
#SBATCH --time=01:00:00
#SBATCH --partition=cbcb
#SBATCH --account=cbcb
#SBATCH --qos=high
#SBATCH --mem=16gb

source /nfshomes/stakeshi/miniconda3/etc/profile.d/conda.sh
conda activate esmfold
python3 /nfshomes/stakeshi/esm/esm/model/graph.py