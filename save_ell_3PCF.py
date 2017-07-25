# Imports
import sys
import json
from nbodykit.lab import *

# math
import numpy as np

# Load in data
filename = str(sys.argv[1])   
result = Multipoles3PCF.load(filename)
poles = result.poles

poles.shape
#Z=np.zeros((11,10,10))
#for ell in range(0,11,1):
 #   Z[ell,:,:]=poles

# Define some variables
Numell = 11
NBins = 10 
Rmin = 10.
Rmax = 120.
delta_bin = (Rmax-Rmin)/float(NBins)

"""
1st way to import:
    Save all multipoles into separate arrays in separate .npy files
"""
Z = np.zeros((Numell,NBins,NBins)) #multipoles by Nbins by Nbins.

for ell in range(0, Numell, 1):
    Z[ell,:,:] = poles['zeta_' + str(ell)]
    np.save('Desktop/3PCF/NPY/func_redshift_space_ell_'+str(ell)+'.npy', Z)

"""
2nd way to import:
    Save all multipoles to 11x10x10 array in the same .npy file

Z = np.zeros((Numell,NBins,NBins)) #multipoles by Nbins by Nbins.
for ell in range(0, Numell, 1):
    Z[ell,:,:] = poles['zeta_' + str(ell)]
    np.save('Desktop/3PCF/NPY/func_redshift_space_all_ell.npy', Z)
"""