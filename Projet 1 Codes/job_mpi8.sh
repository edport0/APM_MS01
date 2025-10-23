#!/bin/bash
#SBATCH --job-name=Ejob8
#SBATCH --output=output_Ejob8.txt
#SBATCH --error=error_Ejob8.txt
#SBATCH --partition=cpu_test
#SBATCH --account=ams301
#SBATCH --ntasks=8
#SBATCH --time=00:25:00
## load modules
module load cmake
module load gcc
module load gmsh
module load openmpi
## execution
mpirun -display-map ${SLURM_SUBMIT_DIR}/a.out