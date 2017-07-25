import argparse
from nbodykit.lab import *
from nbodykit import setup_logging
import os

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
    if ns.test: s = s[:10000] # only use small slice of catalog for testing

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
output = "results/threepcfBIG_cutskyN%d_redshift" %ns.box
if ns.test: output += '_test'
result.save(output + '.json')