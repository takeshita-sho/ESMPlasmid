#!/bin/bash
#SBATCH -J train # Job name
#SBATCH -o train.o # Name of output file
#SBATCH -e train.e # Name of error file
#SBATCH --time=1-00:00:00
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --mem=128gb
#SBATCH --gres=gpu:rtxa6000:8


source /nfshomes/stakeshi/anaconda3/etc/profile.d/conda.sh
conda activate esmfold
python3 -m torch.distributed.launch --nproc_per_node=8 /nfshomes/stakeshi/esm/esm/model/train.py
