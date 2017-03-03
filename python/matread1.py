"""
matread1.py
A demo program for reading in a matrix then printing it out
"""

import sys
import numpy
from np_helper import loadmatrix1, printmatrix1

def read1(argv):
    if len(argv) < 2:
        print >> sys.stderr, "Needs an input file name on arg1"
        sys.exit(1)

    matfile = argv[1]
    M = loadmatrix1(matfile)
    #printmatrix1(M, float_fmt=" %9.6g")
    printmatrix1(M)

if __name__ == "__main__":
    read1(sys.argv)
