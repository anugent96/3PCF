import argparse
from nbodykit.lab import *
from nbodykit import setup_logging
import os

def load_data():
    """
    Return the data catalog object
    """
    data_path = "/global/cscratch1/sd/anuge96/z_set_line_multipoles"
    names = ['x', 'y', 'z', 'wt']
    d = CSVCatalog(data_path, names=names)

    # no completeness weights needed
    d['Weight'] = 1.0

    # compute Cartesian position from (ra, dec, z)
    d['Position'] = transform.vstack(d['x'], d['y'], d['z'])

    return d

# define some constants
LMAX = 10
BOXSIZE = 2600.0
NBINS = 8
RMAX = 200.
RMIN = RMAX*10e-12

# the radial bins
edges = numpy.linspace(RMIN, RMAX, NBINS+1)

setup_logging()

# parse the input arguments
desc = 'compute the three PCF from the periodic N-series boxes'
parser = argparse.ArgumentParser(description=desc)

h = 'only use the first few objects (for testing purpose)'
parser.add_argument('--test', action='store_true', help=h)

ns = parser.parse_args()

# load the data source
s = load_data()
if ns.test: s = s[:1000] # only use small slice of catalog for testing

# compute the power
poles = list(range(0, 1, 1))
result = Multipoles3PCF(s, poles=poles, edges=edges, BoxSize=BOXSIZE, periodic=True)

# and save!
output = "results/3pcf_line_test_ell_0" 
result.save(output + '.json')
