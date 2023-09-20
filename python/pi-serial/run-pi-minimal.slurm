#!/bin/bash
#SBATCH -J pi-serial-minimal
#SBATCH -o %x.o%j

#module load  container_env  python3/2023.2-py39  # can also work for serial runs
module load  container_env  python3/2022.1
crun.python3  python3  pi-serial-minimal.py  1000000

