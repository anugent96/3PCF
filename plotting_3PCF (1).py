"""
NOTES ON CURRENT CODE:
    Dominated in upper right hand corner (not what we expected)
    Ask Nick how he ordered bins
"""

# Imports
import sys
import json

# math
import numpy as np

# plotting
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
#plt.matplotlib.use('PS')
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

# for timing
import time


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


# Convert JSON file containing data into Numpy array
json_file = str(sys.argv[1])
jf = open(json_file)
dataset = json.load(jf)
data = dataset[u'poles'][u'__data__']
d = np.array(data)

"""
# To test:

json_file = open('threepcf_uncut_N1_real_space_test.json')
dataset = json.load(json_file)
data = dataset[u'poles'][u'__data__']
d = np.array(data)
"""


# Define some variables
Numell = 11
NBins = 10 
Rmin = 10.
Rmax = 120.
delta_bin = (Rmax-Rmin)/float(NBins)
"""
Nbins: number of bins
Rmin: minimum radius
Rmax: maximum radius
delta_bin: bin width


#Find centers of bins
ctrs = [Rmin+(delta_bin/2)]
i=0
while len(ctrs) < NBins:
    ctrs.append(ctrs[i]+delta_bin)
    i += 1

rarr_sm = np.array(ctrs)
"""


# Create Z and Z_wted and rearrange dataset
Z = np.zeros((Numell,NBins,NBins)) #multipoles by Nbins by Nbins.
Z_wted = Z*0.
for i in range(0, NBins, 1):
    r1 = i*(delta_bin/2) + Rmin
    for j in range(0, NBins, 1):
        r2 = j*(delta_bin/2) + Rmin
        for ell in range(0, Numell, 1):
            Z[ell, i, j] = d[i][j][ell] # rearrange the dataset
            Z_wted[ell, i, j] = d[i][j][ell]*r1**2*r2**2 # put weights on Z
            


"""
For colors on Colormap: https://matplotlib.org/examples/color/colormaps_reference.html
"""
# Plotting multipoles (L=0 > L=10)

F = plt.figure(1, (8, 8))
grid = ImageGrid(F, 111, nrows_ncols = (5, 2), direction="row", axes_pad = 0.05, add_all=True, label_mode = "1", share_all = True, cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.05)

# putting each of the mulitpole plots on grids
#minval = -20e6 #colorbar range
#maxval = 20e6
count=0
for ax in grid:
    Z_new = Z_wted[:][count][:]
    im=ax.imshow(Z_new, norm=MidpointNormalize(midpoint=0), cmap='RdBu_r', origin='lower')#,interpolation = 'none', vmin=minval, vmax=maxval)
    count += 1


ax.cax.colorbar(im,format='%.1e') # adding colorbar
ax.cax.toggle_label(True)
grid[0].set_title(r'$\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;{\rm Multipoles}$', fontsize=24) #adding title

# For axes
extents = np.array([a.get_position().extents for a in grid])  #all axes extents
bigextents = np.empty(4)   
bigextents[:2] = extents[:,:2].min(axis=0)
bigextents[2:] = extents[:,2:].max(axis=0)

labelpad_x=0.08  #distance between the external axis and the text
labelpad_y=-0.10

xlab_t = F.text((bigextents[2]+bigextents[0])/2, bigextents[1]-labelpad_x, r'$r_1\;{\rm [Mpc}/h]$',horizontalalignment='center', verticalalignment = 'bottom', fontsize=24)
ylab_t = F.text(bigextents[0]-labelpad_y,(bigextents[3]+bigextents[1])/2,  r'$r_2\;{\rm [Mpc}/h]$',rotation='vertical', horizontalalignment = 'left', verticalalignment = 'center', fontsize=24)

# Adding internal titles
for ax, im_title in zip(grid, [r'l=0', 'l=1', 'l=2', r'l=3', 'l=4', 'l=5', r'l=6', 'l=7', 'l=8', 'l=9']):
    t = add_inner_title(ax, im_title, loc=2)
    t.patch.set_ec("none")
    t.patch.set_alpha(0.5)
    
plt.show()



"""
# For individual plots of mulitpoles

# Plot Z (L=0)
plt.imshow(Z[:][0][:], cmap = 'RdBu_r', origin='lower', norm=MidpointNormalize(midpoint=0))
plt.xlabel('r1 [Mpc/h]')
plt.ylabel('r2 [Mpc/h]')
plt.colorbar()
plt.show()

# Plot Z (L=1)
plt.imshow(Z[:][1][:], cmap = 'RdBu_r', origin='lower', norm=MidpointNormalize(midpoint=0))
plt.colorbar()
plt.xlabel('r1 [Mpc/h]')
plt.ylabel('r2 [Mpc/h]')
plt.show()

# Plot Z (L=2)
plt.imshow(Z[:][2][:], cmap = 'RdBu_r', origin='lower', norm=MidpointNormalize(midpoint=0))
plt.colorbar()
plt.xlabel('r1 [Mpc/h]')
plt.ylabel('r2 [Mpc/h]')
plt.show()

# Plot Z (L=3)
plt.imshow(Z[:][3][:], cmap = 'RdBu_r', origin='lower', norm=MidpointNormalize(midpoint=0))
plt.colorbar()
plt.xlabel('r1 [Mpc/h]')
plt.ylabel('r2 [Mpc/h]')
plt.show()

# Plot Z (L=4)
plt.imshow(Z[:][4][:], cmap = 'RdBu_r', origin='lower', norm=MidpointNormalize(midpoint=0))
plt.colorbar()
plt.xlabel('r1 [Mpc/h]')
plt.ylabel('r2 [Mpc/h]')
plt.show()

# Plot Z (L=5)
plt.imshow(Z[:][5][:], cmap = 'RdBu_r', origin='lower', norm=MidpointNormalize(midpoint=0))
plt.colorbar()
plt.xlabel('r1 [Mpc/h]')
plt.ylabel('r2 [Mpc/h]')
plt.show()

# Plot Z (L=6)
plt.imshow(Z[:][6][:], cmap = 'RdBu_r', origin='lower', norm=MidpointNormalize(midpoint=0))
plt.colorbar()
plt.xlabel('r1 [Mpc/h]')
plt.ylabel('r2 [Mpc/h]')
plt.show()

# Plot Z (L=7)
plt.imshow(Z[:][7][:], cmap = 'RdBu_r', origin='lower', norm=MidpointNormalize(midpoint=0))
plt.colorbar()
plt.xlabel('r1 [Mpc/h]')
plt.ylabel('r2 [Mpc/h]')
plt.show()

# Plot Z (L=8)
plt.imshow(Z[:][8][:], cmap = 'RdBu_r', origin='lower', norm=MidpointNormalize(midpoint=0))
plt.colorbar()
plt.xlabel('r1 [Mpc/h]')
plt.ylabel('r2 [Mpc/h]')
plt.show()

# Plot Z (L=9)
plt.imshow(Z[:][9][:], cmap = 'RdBu_r', origin='lower', norm=MidpointNormalize(midpoint=0))
plt.colorbar()
plt.xlabel('r1 [Mpc/h]')
plt.ylabel('r2 [Mpc/h]')
plt.show()

# Plot Z (L=10)
plt.imshow(Z[:][10][:], cmap = 'RdBu_r', origin='lower', norm=MidpointNormalize(midpoint=0))
plt.colorbar()
plt.xlabel('r1 [Mpc/h]')
plt.ylabel('r2 [Mpc/h]')
plt.show()

"""