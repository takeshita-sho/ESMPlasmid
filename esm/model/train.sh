#!/bin/bash
#SBATCH -J train # Job name
#SBATCH -o /nfshomes/stakeshi/esm/esm/model/train.o # Name of output file
#SBATCH -e /nfshomes/stakeshi/esm/esm/model/train-%j.e # Name of error file
#SBATCH --time=1-00:00:00
#SBATCH --partition=cbcb
#SBATCH --account=cbcb
#SBATCH --qos=highmem
#SBATCH --mem=2000gb

##SBATCH --gres=gpu:rtxa6000:1


source /nfshomes/stakeshi/miniconda3/etc/profile.d/conda.sh
conda activate esmfold
#python3 -m torch.distributed.launch --nproc_per_node=8 /nfshomes/stakeshi/esm/esm/model/train.py
python3 /nfshomes/stakeshi/esm/esm/model/train.py
