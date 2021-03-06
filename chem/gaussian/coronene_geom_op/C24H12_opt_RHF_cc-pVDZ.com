! Coronene (C24H12) in cc-pVDZ basis
! See https://cccbdb.nist.gov/
! Starting coordinate below is for RHF/STO-3G

! If you want, please uncomment the following
! so that your scratch files go to the fast filesystem
! (on Wahab).
! See: http://gaussian.com/running/?tabid=2
!
! But please do not forget to adjust the path, and
! replace $USER with your own user ID!
!%RWF=/scratch/$USER/G09-scratch/
!%Int=/scratch/$USER/G09-scratch/
!%D2E=/scratch/$USER/G09-scratch/

! %save
%mem=4096MB

! 3/24 = Print out the Gaussian function table
! 3/33 = Print out 1B & 2B matrix elements
#RHF/cc-pVDZ  5D  7F  units=Ang opt

Coronene RHF with cc-pVDZ basis (geometry optimization)

0,1
C       0.0000000       1.4373760       0.0000000
C       1.2448050       0.7186880       0.0000000
C       1.2448050       -0.7186880      0.0000000
C       0.0000000       -1.4373760      0.0000000
C       -1.2448050      -0.7186880      0.0000000
C       -1.2448050      0.7186880       0.0000000
C       0.0000000       2.8323200       0.0000000
C       2.4528610       1.4161600       0.0000000
C       2.4528610       -1.4161600      0.0000000
C       0.0000000       -2.8323200      0.0000000
C       -2.4528610      -1.4161600      0.0000000
C       -2.4528610      1.4161600       0.0000000
C       1.2575260       3.5268640       0.0000000
C       2.4255900       2.8524820       0.0000000
C       3.6831170       0.6743820       0.0000000
C       3.6831170       -0.6743820      0.0000000
C       2.4255900       -2.8524820      0.0000000
C       1.2575260       -3.5268640      0.0000000
C       -1.2575260      -3.5268640      0.0000000
C       -2.4255900      -2.8524820      0.0000000
C       -3.6831170      -0.6743820      0.0000000
C       -3.6831170      0.6743820       0.0000000
C       -2.4255900      2.8524820       0.0000000
C       -1.2575260      3.5268640       0.0000000
H       1.2467250       4.6095750       0.0000000
H       3.3686470       3.3844830       0.0000000
H       4.6153720       1.2250920       0.0000000
H       4.6153720       -1.2250920      0.0000000
H       3.3686470       -3.3844830      0.0000000
H       1.2467250       -4.6095750      0.0000000
H       -1.2467250      -4.6095750      0.0000000
H       -3.3686470      -3.3844830      0.0000000
H       -4.6153720      -1.2250920      0.0000000
H       -4.6153720      1.2250920       0.0000000
H       -3.3686470      3.3844830       0.0000000
H       -1.2467250      4.6095750       0.0000000


