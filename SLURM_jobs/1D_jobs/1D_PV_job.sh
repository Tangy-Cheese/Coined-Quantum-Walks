#!/bin/bash

#SBATCH --job-name=1D_PV
#SBATCH --output=../../output/logs/%x_output_%j.log
#SBATCH --error=../../output/logs/%x_error_%j.log
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

module purge
module load PyTorch/2.0-Miniconda3-4.12.0-Python-3.10.4
module load GCC/11.3.0 OpenMPI/4.1.4 SciPy-bundle/2022.05
module load GCCcore/11.3.0 Python/3.10.4

j = 100
python ../../src/CQW_1D/1D_PV.py $j

 