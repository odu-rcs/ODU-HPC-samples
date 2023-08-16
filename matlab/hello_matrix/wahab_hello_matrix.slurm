#!/bin/bash
#
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -J wahab_hello_matrix
#SBATCH -o %x.out%j
#
# A sample MATLAB job on Wahab
# Updated Aug 2023 to conform to the new containerized software practice
# on Wahab cluster.

module load container_env matlab/R2022a
#module list

crun.matlab  matlab -nodisplay -nodesktop -r hello_matrix
