Matlab "Hello, Matrix" Example
==============================

This directory contains a naively simple script to demonstrate the
runing of Matlab scripts on Turing & Wahab cluster.


Files:

  * `hello.m` is the main computational workload.

  * `wahab_hello_matrix.slurm` is the job script that can run
    on both Wahab and Turing after the summer 2023 upgrade.


How to run

  * Simply submit the SLURM job script:

    ```
    sbatch wahab_hello_matrix.slurm
    ```

    and observe the output given in the `wahab_hello_matrix.outNNNN` where
    `NNNN` is the job ID given by `sbatch` upon submitting the job.


Obsoleted files:

  * `turing_hello_matrix.slurm` is an archived job script that
    shows how Matlab jobs used to be executed on Turing,
    before every software package is containerized.


