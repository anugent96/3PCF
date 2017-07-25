import argparse
from nbodykit.lab import *
from nbodykit import setup_logging
import os

def load_data():
    """
    Return the data catalog object
    """
    data_path = "/global/cscratch1/sd/nhand/NSeriesPeriodic/BoxN%d.mock" %(ns.box)
    names = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    d = CSVCatalog(data_path, names=names)

    # add Position and Velocity
    d['Position'] = transform.vstack(d['x'], d['y'], d['z']) # shape is (d.size,3)
    d['Velocity'] = transform.vstack(d['vx'], d['vy'], d['vz']) # shape is (d.size, 3)
    
    # shift position into redshift space
    if ns.rsd:
        d['Position'] += d['Velocity'] * RSD_FACTOR * LOS
    
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

# define some constants
LMAX = 10
BOXSIZE = 2600.0
NBINS = 10
RMIN = 10.
RMAX = 120.

# N-series cosmology
cosmo = cosmology.Cosmology(H0=70.0, Om0=0.286, flat=True, Tcmb0=0.)

# RSD quantities
REDSHIFT = 0.5
LOS = [0, 0, 1] # the line-of-sight
RSD_FACTOR = (1+REDSHIFT) / (100*cosmo.efunc(REDSHIFT))

# the radial bins
edges = numpy.linspace(RMIN, RMAX, NBINS+1)

setup_logging()

# parse the input arguments
desc = 'compute the three PCF from the periodic N-series boxes'
parser = argparse.ArgumentParser(description=desc)

h = 'the box number to compute'
parser.add_argument('box', type=int, choices=[1,2,3,4,5,6,7], help=h)

h = 'whether to compute the results in redshift-space; default is real space'
parser.add_argument('--rsd', action='store_true', help=h)

h = 'only use the first few objects (for testing purpose)'
parser.add_argument('--test', action='store_true', help=h)

ns = parser.parse_args()

# load the data source
s = load_data()
if ns.test: s = s[:1000] # only use small slice of catalog for testing

# compute the power
poles = list(range(0, LMAX+1, 1))
result = Multipoles3PCF(s, poles=poles, edges=edges, BoxSize=BOXSIZE, periodic=True)

# and save!
output = "results/threepcf_uncut_N%d" %ns.box
if ns.rsd:
    output += '_redshift_space'
else:
    output += '_real_space'
if ns.test: output += '_test'
result.save(output + '.json')