# Imports
import sys
import json

# math
import numpy as np
import scipy.special as sp
import scipy.misc as sm

# interpolation to set up functions for 2d plots
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d

# plotting
import matplotlib.pyplot as plt
plt.matplotlib.use('PS')
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize

# for getting latex characters in plot titles
from matplotlib import rc
rc('font',size='18',family='serif')
plt.rcParams['pdf.fonttype'] = 42

# plotting a density plot (2-d)
from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from pylab import *

# Define some variables
Numell = 11
NBins = 10 
Rmin = 10.
Rmax = 120.
delta_bin = (Rmax-Rmin)/float(NBins)


# For plotting data

# Normalize the color bar
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)
    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# Function to add an inner title to each grid
def add_inner_title(ax, title, loc, size=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke
    if size is None:
        size = dict(size=plt.rcParams['legend.fontsize'])
    at = AnchoredText(title, loc=loc, prop=size, pad=0., borderpad=0.5,frameon=False, **kwargs)
    ax.add_artist(at)
    at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
    return at

# Loading .npy file(s)
"""
# .npy files with individual multipoles
for ell in range (0, 11, 1):
    Z = np.load('Desktop/3PCF/NPY/func_redshift_space_ell_'+str(ell)+'.npy')
"""
# .npy file with all multipoles
Z = np.load('Desktop/3PCF/NPY/func_redshift_space_all_ell.npy')


# Plotting multipoles (L=0 > L=9)

F = plt.figure(1, (8, 8))
grid = ImageGrid(F, 111, nrows_ncols = (5, 2), direction="row", axes_pad = 0.05, add_all=True, label_mode = "1", share_all = True, cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.05)

# putting each of the mulitpole plots on grids
i = 0
minval = np.zeros((11))
maxval = np.zeros((11))
for i in range(0, 10):
    if i==0:
        minval[i] = -1e8 #colorbar range
        maxval[i] = 1e8
    elif i==1:
        minval[i] = -3e6
        maxval[i] = 3e6
    elif i ==2:
        minval[i] = -1e6
        maxval[i] = 1e6
    elif i==3:
        minval[i] = -3e5
        maxval[i] = 3e5
    else:
        minval[i] = -1e5
        maxval[i] = 1e5
    
    
count=0
for ax in grid:
    Z_new = Z[:][count][:]
    im=ax.imshow(Z_new, norm=MidpointNormalize(midpoint=0), cmap='RdBu_r', origin='lower', vmin=minval[count], vmax=maxval[count], interpolation='none')
    count += 1


ax.cax.colorbar(im,format='%.1e') # adding colorbar
ax.cax.toggle_label(True)
grid[0].set_title(r'$\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;{\rm Multipoles}$', fontsize=24) #adding title

# For axes
extents = np.array([a.get_position().extents for a in grid])  #all axes extents
bigextents = np.empty(4)   
bigextents[:2] = extents[:,:2].min(axis=0)
bigextents[2:] = extents[:,2:].max(axis=0)

labelpad_x=0.08  #distance between the external axis and the text
labelpad_y=-0.20

xlab_t = F.text((bigextents[2]+bigextents[0])/2, bigextents[1]-labelpad_x, r'$r_1\;{\rm [Mpc}/h]$',horizontalalignment='center', verticalalignment = 'bottom', fontsize=24)
ylab_t = F.text(bigextents[0]-labelpad_y,(bigextents[3]+bigextents[1])/2,  r'$r_2\;{\rm [Mpc}/h]$',rotation='vertical', horizontalalignment = 'left', verticalalignment = 'center', fontsize=24)

# Adding internal titles
for ax, im_title in zip(grid, [r'l=0', 'l=1', 'l=2', r'l=3', 'l=4', 'l=5', r'l=6', 'l=7', 'l=8', 'l=9']):
    t = add_inner_title(ax, im_title, loc=2)
    t.patch.set_ec("none")
    t.patch.set_alpha(0.5)
    
plt.show()
    
