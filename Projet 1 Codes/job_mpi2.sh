#!/bin/bash
#SBATCH --job-name=Ejob2
#SBATCH --output=output_Ejob2.txt
#SBATCH --error=error_Ejob2.txt
#SBATCH --partition=cpu_test
#SBATCH --account=ams301
#SBATCH --ntasks=2
#SBATCH --time=00:25:00
## load modules
module load cmake
module load gcc
module load gmsh
module load openmpi
## execution
mpirun -display-map ${SLURM_SUBMIT_DIR}/a.out
