#!/bin/bash
#SBATCH --job-name calc-pi

module load container_env
module load R/4.1.3

crun.R Rscript  calc-pi.R  "${1:-1000000}"

