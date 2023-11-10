#!/bin/bash
#SBATCH -J preprocess # Job name
#SBATCH -o preprocess.o # Name of output file
#SBATCH -e preprocess.e # Name of error file
#SBATCH --time=02:00:00
#SBATCH --partition=cbcb
#SBATCH --account=cbcb
#SBATCH --ntasks=16
#SBATCH --qos=high
#SBATCH --mem=64gb

module load Python3/3.9.6
python3 /nfshomes/stakeshi/esm/process.py
