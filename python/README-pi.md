`Pi-serial` and `pi-mpi`: Estimation of pi using a Monte Carlo Method
=====================================================================


The Python scripts in the `pi-serial` and `pi-mpi` folders
implement a simple Monte Carlo method to obtain the estimate of pi.
There is a full-featured version (`pi-serial.py` and `pi-mpi.py`)
which report more information,
and a minimal version (those scripts with `-minimal` in their file names)
which report only the estimated value of pi.
Refer to [Estimation of Pi for Pedestrians][est-pi-pedestrian]
for a quick explanation of what was being done by this code.

<!-- FIXME Explain the difference between the outputs of the full-featured
and minimal versions -->

The job scripts in these folders were intended to demonstrate how to
run Python scripts in serial and MPI-parallel fashions on ODU's HPC
systems, which now provide software exclusively through AppTainer
(a.k.a. Singularity) containers.


Credits
-------

* The Python codes for the calculation of pi were taken
  from the HPC Carpentry's "Intro to HPC" lesson:

  https://github.com/carpentries-incubator/hpc-intro/raw/gh-pages/files/hpc-intro-pi-code.tar.gz
  (ref: commit decb12174647686e04b00e3ea794206f87e21139)


[est-pi-pedestrian]: https://www.hpc-carpentry.org/hpc-parallel-novice/01-estimate-of-pi/index.html
