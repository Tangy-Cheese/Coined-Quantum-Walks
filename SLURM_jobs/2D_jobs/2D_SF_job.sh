#! /bin/bash

#SBATCH --job-name=1D_PV
#SBATCH --output=/storage/physics/phugcx/output_%j.log
#SBATCH --error=/storage/physics/phugcx/error_%j.log
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=3988

module load PyTorch/2.0-Miniconda3-4.12.0-Python-3.10.4
module load GCC/11.3.0 OpenMPI/4.1.4 SciPy-bundle/2022.05
module load GCCcore/11.3.0 Python/3.10.4

python /storage/physics/phugcx/1D_Part.py 100 500