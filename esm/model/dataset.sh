#!/bin/bash
#SBATCH -J dataset # Job name
#SBATCH -o dataset.o # Name of output file
#SBATCH -e dataset-%j.e # Name of error file
#SBATCH --time=06:00:00
#SBATCH --mem=128gb
#SBATCH --partition=cbcb
#SBATCH --account=cbcb
#SBATCH --qos=high


source /nfshomes/stakeshi/miniconda3/etc/profile.d/conda.sh
conda activate base
python3 /nfshomes/stakeshi/esm/esm/model/make_dataset.py