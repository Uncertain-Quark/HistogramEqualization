# Main python file for performing histogram equalization
import sys, numpy

sourcefile = sys.argv[1]
sourceArr = np.loadtxt(sourcefile)
