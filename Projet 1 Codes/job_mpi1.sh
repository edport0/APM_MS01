#!/bin/bash
#SBATCH --job-name=Ejob1
#SBATCH --output=output_Ejob1.txt
#SBATCH --error=error_Ejob1.txt
#SBATCH --partition=cpu_test
#SBATCH --account=ams301
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
## load modules
module load cmake
module load gcc
module load gmsh
module load openmpi
## execution
mpirun -display-map ${SLURM_SUBMIT_DIR}/a.out
