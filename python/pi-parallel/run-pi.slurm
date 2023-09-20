#!/bin/bash
#SBATCH -J pi-mpi
#SBATCH -o %x.o%j
#SBATCH -n 4

#module load  container_env  python3/2023.2-py310   # does not work as of 2023-09-18
module load  container_env  python3/2022.1
srun crun.python3  python3  pi-mpi.py  "${1:-1000000}"

# This also works:
#module load  container_env  tensorflow-cpu/2.10.0
#srun crun.tensorflow-cpu  python3  pi-mpi.py  "${1:-1000000}"

