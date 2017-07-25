"""
****CURRENTELY NOT USING CORRECT DATASET FOR PYTHON****
- make sure you are loading everything in correctly and creating the correct numpy arrays from .npy files
- other than that everything is working correctly
- just need to repeat for all other bins and then record the results!
"""
import sys
import numpy as np


# Load in datasets (c_bins = C++ results, p_bins = Python results)
# This is for Bins x = 1-7, 0 (we do not count Bin 0,0)
cbins = str(sys.argv[1])
pbins = str(sys.argv[2])

c_bins_x0 = np.load(cbins)
p_bins_x0 = np.load(pbins)

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
p_10_rescale = p_10/p_10[0]/two_ell_arr
p_20_rescale = p_20/p_20[0]/two_ell_arr
p_30_rescale = p_30/p_30[0]/two_ell_arr
p_40_rescale = p_40/p_40[0]/two_ell_arr
p_50_rescale = p_50/p_50[0]/two_ell_arr
p_60_rescale = p_60/p_60[0]/two_ell_arr
p_70_rescale = p_70/p_70[0]/two_ell_arr

# Both datasets are normalized, ex: [1, 0.015, 1, 0.015, ...]
# The C++ dataset not normalized for l=0, so we won't use that
# Divide 2nd numbers to see how different they are 
# If we are correct, these should be off by something related to 1/pi**2
print('Python l=1 multipole divided by C++ l=1 mutlipole')

diff_10 = p_10_rescale[1]/c_10[1]
print('bin 1 0', diff_10)

diff_20 = p_20_rescale[1]/c_20[1]
print('bin 2 0', diff_20)

diff_30 = p_30_rescale[1]/c_30[1]
print('bin 3 0', diff_30)

diff_40 = p_40_rescale[1]/c_40[1]
print('bin 4 0', diff_40)

diff_50 = p_50_rescale[1]/c_50[1]
print('bin 5 0', diff_50)

diff_60 = p_60_rescale[1]/c_60[1]
print('bin 6 0', diff_60)

diff_70 = p_70_rescale[1]/c_70[1]
print('bin 7 0', diff_70)

