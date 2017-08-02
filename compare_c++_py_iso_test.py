import sys
import numpy as np
from nbodykit.lab import *

# Load in datasets (c_bins = C++ results, p_bins = Python results)
# This is for Bins x = 1-7, 0 (we do not count Bin 0,0)
cbins = str(sys.argv[1])
pbins = str(sys.argv[2])

c_bins_x0 = np.load(cbins)
p_bins_x0 = np.load(pbins)

"""
p_result = Multipoles3PCF.load(filename)
poles = p_result.poles
p_result.poles['zeta_0'][0,0]
"""

# Take Bin 1, 0
c_10 = c_bins_x0[0]
p_10 = p_bins_x0[0]

# Take Bin 2, 0
c_20 = c_bins_x0[1]
p_20 = p_bins_x0[1]

# Take Bin 3, 0
c_30 = c_bins_x0[2]
p_30 = p_bins_x0[2]

# Take Bin 4, 0
c_40 = c_bins_x0[3]
p_40 = p_bins_x0[3]

# Take Bin 5, 0
c_50 = c_bins_x0[4]
p_50 = p_bins_x0[4]

# Take Bin 6, 0
c_60 = c_bins_x0[5]
p_60 = p_bins_x0[5]

# Take Bin 7, 0
c_70 = c_bins_x0[6]
p_70 = p_bins_x0[6]


# We think Python code is missing a factor of 2l+1
two_ell_arr=np.zeros(len(p_10))
for i in range(0,11,1):
    two_ell_arr[i]=2*i+1

# Rescale Python results to account for missing factor AND normalize
p_10_rescale = p_10/p_10[0]
p_20_rescale = p_20/p_20[0]
p_30_rescale = p_30/p_30[0]
p_40_rescale = p_40/p_40[0]
p_50_rescale = p_50/p_50[0]
p_60_rescale = p_60/p_60[0]
p_70_rescale = p_70/p_70[0]

# Both datasets are normalized, ex: [1, 0.015, 1, 0.015, ...]
# The C++ dataset not normalized for l=0, so we won't use that
# Divide 2nd numbers to see how different they are 
# If we are correct, these should be off by something related to 1/pi**2
print('Python l=1 multipole divided by C++ l=1 multipole')

diff_10 = p_10_rescale[1]/c_10[1]
print('bin 1 0', p_10_rescale[1], c_10[1], diff_10)

diff_20 = p_20_rescale[1]/c_20[1]
print('bin 2 0', p_20_rescale[1], c_20[1], diff_20)

diff_30 = p_30_rescale[1]/c_30[1]
print('bin 3 0', p_30_rescale[1], c_30[1], diff_30)

diff_40 = p_40_rescale[1]/c_40[1]
print('bin 4 0', p_40_rescale[1], c_40[1], diff_40)

diff_50 = p_50_rescale[1]/c_50[1]
print('bin 5 0', p_50_rescale[1], c_50[1], diff_50)

diff_60 = p_60_rescale[1]/c_60[1]
print('bin 6 0', p_60_rescale[1], c_60[1], diff_60)

diff_70 = p_70_rescale[1]/c_70[1]
print('bin 7 0', p_70_rescale[1], c_70[1], diff_70)

print('-------------------------------')

ell = 0
print('Python l=' + str(ell) +' multipole divided by C++ l=' + str(ell) +' multipole')


diff10 = p_10[ell]/c_10[ell]
print('bin 10', diff10)
    
diff20 = p_20[ell]/c_20[ell]
print('bin 20', diff20)

diff30 = p_30[ell]/c_30[ell]
print('bin 30', diff30)

diff40 = p_40[ell]/c_40[ell]
print('bin 40', diff40)

diff50 = p_50[ell]/c_50[ell]
print('bin 50', diff50)

diff60 = p_60[ell]/c_60[ell]
print('bin 60', diff60)

diff70 = p_70[ell]/c_70[ell]
print('bin 70', diff70)
