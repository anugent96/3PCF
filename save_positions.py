from nbodykit.lab import *
from nbodykit import setup_logging
import os
import numpy as np
import dask.array as da

# For Cutsky

def load_data():
    
    data_path = "/global/cscratch1/sd/nhand/NCutskyData/data/CutskyN1.redshift.sel.rdz" 
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
    
    # Apply R_z rotation matrix
    phi = da.arctan(y/x) # define phi using trig
    
    cos_phi = 1.0 / da.sqrt(((y/x)**2 + 1)) # define cos_phi and sin_phi using geometry
    sin_phi = da.sqrt(1- (cos_phi*cos_phi))
    
    # Using geometry
    new_x = cos_phi*x + sin_phi*y
    new_y = -sin_phi*x + cos_phi*y
    new_z = z
    
    """
    # Using trig
    new_x = da.cos(-phi)*x - da.sin(-phi)*y
    new_y = da.sin(-phi)*x + da.cos(-phi)*y
    new_z = z
    """

    # Apply R_y rotation matrix
    theta = da.arccos(z/da.sqrt((x**2)+(y**2)+(z**2))) # define theta using trig
    
    cos_theta = z/da.sqrt(x**2+y**2+z**2) # define cos_theta and sin_theta using geometry
    sin_theta = da.sqrt(1.0-(cos_theta*cos_theta))
    
    # Using geometry
    new2_x = cos_theta*new_x + -sin_theta*new_z
    new2_y = new_y
    new2_z = sin_theta*new_x + cos_theta*new_z
    
    """
    # Using trig
    new2_x = da.cos(-theta)*new_x + da.sin(-theta)*new_z
    new2_y = new_y
    new2_z = -da.sin(-theta)*new_x + da.cos(-theta)*new_z
    """
  
    d['Position'] = transform.vstack(new2_x, new2_y, new2_z)
    return d


cosmo = cosmology.Cosmology(H0=70.0, Om0=0.286, flat=True, Tcmb0=0.)
s = load_data()
sub = s[:10]

x = sub['Position'][:,0] # first column
y = sub['Position'][:,1] # second column
z = sub['Position'][:,2] # third column

np.save('/global/homes/a/anuge96/Cutsky_Positions/CP1_x1_rot.npy', x)
np.save('/global/homes/a/anuge96/Cutsky_Positions/CP1_y1_rot.npy',y)
np.save('/global/homes/a/anuge96/Cutsky_Positions/CP1_z1_rot.npy',z)







"""  
# For Uncut Boxes
def load_data():
    data_path = "/global/cscratch1/sd/nhand/NSeriesPeriodic/BoxN1.mock" 
    names = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    d = CSVCatalog(data_path, names=names)

    # add Position and Velocity
    d['Position'] = transform.vstack(d['x'], d['y'], d['z']) # shape is (d.size,3)
    d['Velocity'] = transform.vstack(d['vx'], d['vy'], d['vz']) # shape is (d.size, 3)
    
  
    x = d['Position'][:,0] # first column
    y = d['Position'][:,1] # second column
    z = d['Position'][:,2] # third column

    # Apply R_z rotation matrix
    phi = da.arctan(y/x) # define phi
    new_x = da.cos(-phi)*x - da.sin(-phi)*y
    new_y = da.sin(-phi)*x + da.cos(-phi)*y
    new_z = z

    # Apply R_y rotation matrix
    theta = da.arccos(new_z/da.sqrt((new_x**2)+(new_y**2)+(new_z**2))) # define theta
    new2_x = da.cos(-theta)*new_x + da.sin(-theta)*new_z
    new2_y = new_y
    new2_z = -da.sin(-theta)*new_x + da.cos(-theta)*new_z

  
    d['Position'] = transform.vstack(new2_x, new2_y, new2_z)
   
    
    #d['Position'] += d['Velocity'] * RSD_FACTOR * LOS

    return d

cosmo = cosmology.Cosmology(H0=70.0, Om0=0.286, flat=True, Tcmb0=0.)

# RSD quantities
REDSHIFT = 0.5
LOS = [0, 0, 1] # the line-of-sight
RSD_FACTOR = (1+REDSHIFT) / (100*cosmo.efunc(REDSHIFT))

s = load_data()
sub = s[:10]

x = sub['Position'][:,0] # first column
y = sub['Position'][:,1] # second column
z = sub['Position'][:,2] # third column
np.save('/global/homes/a/anuge96/Cutsky_Positions/UCB1_x_rot.npy', x)
np.save('/global/homes/a/anuge96/Cutsky_Positions/UCB1_y_rot.npy', y)
np.save('/global/homes/a/anuge96/Cutsky_Positions/UCB1_z_rot.npy', z)
"""