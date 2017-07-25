"""
NOTES:
    - Should phi and theta be negative for rotation matrices (see: "Practical
    Computation Method for Anisotropic Redshift-Space 3PCF")?
    
    - Solve rotated positions by hand and check to see that your rotations
    are correct
        - Run "compute_3PCF_iso.py" and this code & print out first few "Positions"
        to solve

    - Make sure to add pre-rotation on uncut boxes
"""

import argparse
from nbodykit.lab import *
from nbodykit import setup_logging
import os
import numpy as np
from numpy import cos # **angles are in radians
from numpy import sin
from numpy import arctan
from numpy import arccos
from numpy import sqrt
import dask.array as da

def load_data():
    """
    Return the data catalog object
    """
    data_path = "/global/cscratch1/sd/nhand/NCutskyData/data/CutskyN%d.redshift.sel.rdz" %(ns.box)
    names = ['ra', 'dec', 'z', 'wc', 'nbar', 'wfkp']
    d = CSVCatalog(data_path, names=names, usecols=['ra', 'dec', 'z', 'nbar', 'wc'])

    # no completeness weights needed
    d['Weight'] = 1.0

    # compute Cartesian position from (ra, dec, z)
    d['Position'] = transform.SkyToCartesion(d['ra'], d['dec'], d['z'], cosmo, degrees=True)

    # Pull out x, y, z coordinates
    x = d['Position'][:,0] # first column
    y = d['Position'][:,1] # second column
    z = d['Position'][:,2] # third column
    
    # Define cos_theta, cos_phi, etc using geometry/ trig identities
    cos_phi = 1.0 / da.sqrt(((y/x)*(y/x) + 1)) 
    sin_phi = da.sqrt(1- (cos_phi*cos_phi))

    cos_theta = z/da.sqrt((x*x)+(y*y)+(z*z))
    sin_theta = da.sqrt(1.0-(cos_theta*cos_theta))

    # Define phi and theta using trig
    phi = da.arctan(y/x) 
    theta = da.arccos(z/da.sqrt((x*x)+(y*y)+(z*z)))


    # Apply R_z rotation matrix
    """
    # Using geometry
    new_x = cos_phi*x + sin_phi*y
    new_y = -sin_phi*x + cos_phi*y
    new_z = z
    """

    # Using trig
    new_x = np.cos(-phi)*x - np.sin(-phi)*y
    new_y = np.sin(-phi)*x + np.cos(-phi)*y
    new_z = z


    # Apply R_y rotation matrix
    """ 
    # Using geometry
    new2_x = cos_theta*new_x + -sin_theta*new_z
    new2_y = new_y
    new2_z = sin_theta*new_x + cos_theta*new_z
    """

    # Using trig
    new2_x = np.cos(-theta)*new_x + np.sin(-theta)*new_z
    new2_y = new_y
    new2_z = -np.sin(-theta)*new_x + np.cos(-theta)*new_z
   
    
    d['Position'] = transform.vstack(new2_x, new2_y, new2_z)

    return d

def load_randoms(sign=1.0, subsample=1):
    """
    Return the randoms catalog object

    Parameters
    ----------
    sign : float
        the sign given to the 'Weight' column
    subsample : int
        the integer value to subsample the catalog by
    """
    # and the randoms source
    # THIS RANDOMS FILE IS ALREADY DOWNWEIGHTED BY COMPLETENESS
    randoms_path = "/global/cscratch1/sd/nhand/NCutskyData/randoms/cmass_ngc_randoms_50x.dat"
    names = ['ra', 'dec', 'z', 'wc', 'nbar', 'wfkp']
    r = CSVCatalog(randoms_path, names=names, usecols=['ra', 'dec', 'z', 'nbar'], use_cache=True)

    # no completeness weights for randoms
    r['Weight'] = sign * 1.0

    # scale nbar of randoms to match n(z) of data
    r['nbar'] /= 50.

    # add position
    r['Position'] = transform.SkyToCartesion(r['ra'], r['dec'], r['z'], cosmo, degrees=True)

    if subsample > 1:
        r = r[::subsample]

    return r


# define some constants
LMAX = 10
BOXSIZE = 2600.0
NBINS = 10
RMIN = 10.
RMAX = 120.

# the radial bins
edges = numpy.linspace(RMIN, RMAX, NBINS+1)

setup_logging()

# parse the input arguments
desc = 'compute the three PCF from the N-cutsky mocks'
parser = argparse.ArgumentParser(description=desc)

h = 'the box number to compute'
parser.add_argument('box', type=int, help=h)

h = 'the type of correlation to run, either DD, DR, or RR'
parser.add_argument('type', choices=['DD', 'DR', 'RR'], help=h)

h = 'the integer factor to subsample the randoms by'
parser.add_argument('--subsample-randoms', dest='subsample', type=int, default=1, help=h)

h = 'only use the first few objects (for testing purpose)'
parser.add_argument('--test', action='store_true', help=h)

ns = parser.parse_args()

# N-series cosmology
cosmo = cosmology.Cosmology(H0=70.0, Om0=0.286, flat=True, Tcmb0=0.)

# load the data source
s = None
if ns.type != 'RR':
    s = load_data()
    if ns.test: s = s[:10] # only use small slice of catalog for testing

# load the randoms
if ns.type in ['DR', 'RR']:

    # negative random weights to do DR (Data - Randoms)
    sign = -1.0 if ns.type == 'DR' else 1.0
    r = load_randoms(sign=sign, subsample=ns.subsample)
    if ns.test: r = r[:1000] # only use small slice of catalog for testing

    # concatenate in to a single catalog
    if s is None:
        s = r
    else:
        s = transform.concatenate(s, r, columns=['Position', 'Weight'])

# compute the power
poles = list(range(0, LMAX+1, 1))
result = Multipoles3PCF(s, poles=poles, edges=edges, BoxSize=BOXSIZE, periodic=True, weight='Weight')

# and save!
output = "results/threepcf_cutskyN%d_redshift_anisotropic" %ns.box
if ns.test: output += '_test'
output += "_%s" %ns.type
result.save(output + '.json')
