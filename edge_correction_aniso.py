"""
NOTE: I changed this so that the sigmas are just outputted as is. They do get divided by RRR_0 along with the powers, but that's as it should be because I want sigma on the counts divided by RRR_0; that just is constructing an overdensity field. 7 Sept. 2015.

Syntax: list all of the D-R files first, with the RRR file last
For example, the glob r* will put r00 before rrr.

python ./sumfiles_to_fix_normalization_zs.py ldam.1*.r0*.out ldam.10a_no.real.rrr.out >sumfiles_sept7
python ./sumfiles_to_fix_normalization_zs.py ldam.1*.r0*.out ldam.10a_no.real.rrr.out >sumfiles_may15

run above in directory where data is. note it uses same random file to do denominator for redshift space and for real space! check this with DE.
"""

# Edge Correction imports
import sys
import re
import string
import numpy as np
import scipy.misc as sm
import scipy.special as sp

# Gaunt Integrals
from sympy.physics.wigner import gaunt

#for timing
import time


# For flattening mocks' 6d tensor into a 2d matrix.
def flatten_6d_2d_mock_covar_bd(mock_covar_6d_active, bin_dict, num_of_ells, offset_r1, offset_r2):
    lbd = len(bin_dict)
    covar_2d = np.zeros((lbd*num_of_ells,lbd*num_of_ells))
    index = 0
    for i in range(0, lbd, 1):
        for ell in range(0, num_of_ells, 1):
            index_prime = 0
            for j in range(0, lbd, 1):    
                for ellprime in range(0, num_of_ells, 1):
                    print "index, index_prime, i, j = ", index, index_prime, i, j
                    if (index_prime >= index):
                        covar_2d[index, index_prime] =  mock_covar_6d_active[ell, ellprime, bin_dict[i][0]-offset_r1, bin_dict[i][1]-offset_r2, bin_dict[j][0]-offset_r1, bin_dict[j][1]-offset_r2]
                    else:
                        covar_2d[index, index_prime] = covar_2d[index_prime, index]
                    index_prime += 1
            index += 1
    return covar_2d


# Average all of the D-R files, then divide by the RRR file
nfile = 0
bins = -1
order = -1

numfiles = len(sys.argv[1:])
print "# Looking at %d files" % (numfiles)

for filename in sys.argv[1:]:
    f = open(filename,'r')

    for line in f:
        if (re.match("Bins = ",line)):
            if (bins<0):
                bins = string.atoi(line.split()[-1])
                print "# Using %d bins" % (bins)
        if (re.match("Order = ",line)):
            if (order<0):
                order = string.atoi(line.split()[-1])
                print "# Using order %d" % (order)

                pairs = np.zeros((bins,numfiles))
                power = np.zeros((bins,bins,order+1,numfiles))

        if (re.match("#", line)):
            continue
        if (re.search("=", line)):
            continue
        if (re.match("Multipole", line)):
            continue
        if (re.match("^$", line)):
            continue

        if (re.match("Pairs",line)):
        # We've found a pairs line.  Process it.
            b,cnt = (line.split()[1:-1])
            pairs[string.atoi(b),nfile] = string.atoi(cnt)
            continue

        # Otherwise, we have a power line, so process that
        s = line.split()
        b1 = string.atoi(s[0])
        b2 = string.atoi(s[1])
        p0 = string.atof(s[2])
        for p in range(order+1):
            if (p==0):
                power[b1,b2,p,nfile] = p0
            else:
                power[b1,b2,p,nfile] = p0*string.atof(s[2+p])

        # End loop over lines in a file
        nfile+=1
# End loop over files

# Sum up over the D-R files.
# Also compute the stdev so we can monitor convergence.

nfile -= 1    # Now [nfile] is RRR and the other loops can be normal


# ZS added: May 2 2015: to fix normalization issue in C++ code.
print "# fixing ell dependent normalization"
for ell in range(0,11):
    power[:,:,ell,:] *= (2.*ell+1.)/(4.*np.pi)**2 
    #ZS 14 May 2015; note that changing to from (2ell+1)/2 to  (2ell+1)/(4 pi)^2 here should make no difference here.
# END ZS addition.

pairsD = np.average(pairs[:,0:-1],axis=1)
powerD = np.average(power[:,:,:,0:-1],axis=3)

sqdof = np.sqrt(nfile)   # Convert to error on the mean
if (sqdof==0): 
    sqdof = 1e30
    
pairsDsig = np.std(pairs[:,0:-1],axis=1,ddof=1)/sqdof
powerDsig = np.std(power[:,:,:,0:-1],axis=3,ddof=1)/sqdof

pairsR = pairs[:,nfile]
powerR = power[:,:,:,nfile]

zeta = np.copy(powerD)   # Just a place-holder

# Now combine to make our desired output

print 
print "Multipole RRR correction factors (f_ell): "
f = np.copy(powerR)

for b1 in range(bins):
    for b2 in range(b1):
        f[b1,b2,1:] /= f[b1,b2,0]
        print "%2d %2d "% (b1,b2),
        np.set_printoptions(precision=5,suppress=True,linewidth=1000,formatter={'float': '{: 0.5f}'.format})
        print f[b1,b2,1:]
        np.set_printoptions()
        print

# Now we have the RRR corrections.  For each bin, these adjust 
# the final zeta_ell's.

def triplefact(j1,j2,j3):
    jhalf = (j1+j2+j3)/2.0
    return sm.factorial(jhalf) /sm.factorial(jhalf-j1) /sm.factorial(jhalf-j2) /sm.factorial(jhalf-j3)

def threej(j1,j2,j3):
    # Compute {j1,j2,j3; 0,0,0} 3j symbol
    j = j1+j2+j3
    if (j%2>0): return 0     # Must be even
    if (j1+j2<j3): return 0  # Check legal triangle
    if (j2+j3<j1): return 0  # Check legal triangle
    if (j3+j1<j2): return 0  # Check legal triangle
    return (-1)**(j/2.0) * triplefact(j1,j2,j3) / (triplefact(2*j1,2*j2,2*j3)*(j+1))**0.5
    # DJE did check this against Wolfram

"""
Notes from Zack:

I think this is mostly right but you need to address a particular element of M or you will just have a constant scalar and not a tensor.  I would define m=np.zeros((dim1,dim2,etc)) so it has the right shape, ell by ell' by m by j by j' by s. 

On the other hand it may even be better to map the 6-d to 2-d right here. So you could have M(index1=some formula, index2=some formula) inside the loops you already have, with these formulas being those we discussed to map 6 to 2-d. 

You'll still have to setup M with the right shape  but in this case it is longer in each dimension but has only two dimensions. ""

"""
# WARNING: Check this setup of the matrix!
# What does flist look like for the anisotropic case?
def Mjl_calc(j, jprime, s, ell, ellprime, m, flist):
    p = s-m
    M= np.zeros((ell, ellprime, m, j, jprime, s)) 
    for p in np.arange(1, len(flist)):
        for k in np.arange(1,len(flist[p])):
            for kprime in np.arange(1, len(flist[p][k])):
                for i in M:
                    i += gaunt(m, p, -s, ell, k, j)*gaunt(-m, -p, s, ellprime, kprime, jprime)*flist[p][k][kprime]
                return M


            
# Mock example of what Mjl will look like and how we will flatten it
Mjl = Mjl_calc(j, jprime, s, ell, ellprime, m, flist)
# bin_dict = particular bin combinations (r1, r2, triangle side lengths) we want to use 
num_of_ells = 11
offset_r1 = 2
offset_r2 = 5

x = flatten_6d_2d_mock_covar_bd(Mjl, bin_dict, num_of_ells, offset_r1, offset_r2)
print x
           
            
"""
print
print "Three-point Function (no multipole RRR corrections), with errors due to randoms: "
for b1 in range(bins):
    for b2 in range(b1):
    # Get zeta's by dividing by RRR_0
        powerD[b1,b2,:] /= powerR[b1,b2,0] #this means any constant rescaling of the amplitudes will cancel out! ZS; 14 May 2015.
        powerDsig[b1,b2,:] /= powerR[b1,b2,0]

    # Now adjust for the f_ell
    Mjl = np.zeros((order+1,order+1))
    for j in range(order+1):
        for k in range(order+1):
            Mjl[j][k] = Mjl_calc(j,k,f[b1,b2,:])

    # WARNING: The following lines may be the wrong math!!!
    geometry = np.linalg.inv(np.identity(order+1)+Mjl)
    zeta[b1,b2,:] = geometry.dot(powerD[b1,b2,:])

    # Now normalize to ell=0 for printing
    #zeta[b1,b2,1:] /= zeta[b1,b2,0] #don't do this!
    #powerD[b1,b2,1:] /= powerD[b1,b2,0] #don't do this!
    #powerDsig[b1,b2,0:] /= powerD[b1,b2,0] #don't do this either! i.e. don't divide out monopole of raws from sigma.
    print "%2d %2d %10.7f "% (b1,b2,zeta[b1,b2,0]),
    np.set_printoptions(precision=5,suppress=True,linewidth=1000,formatter={'float': '{: 0.5f}'.format})
    print zeta[b1,b2,1:], "zeta"
    print "%2d %2d %10.7f "% (b1,b2,powerD[b1,b2,0]),
    print powerD[b1,b2,1:], "raw"
    print "%2d %2d %10.7f " % (b1,b2, powerDsig[b1,b2,0]), 
    print powerDsig[b1,b2,1:], "sigma"
    print
    np.set_printoptions()

print
print "Two-point Correlation Monopole:"
xi = pairsD/pairsR
print xi

print "Two-point Correlation Monopole Error due to Randoms:"
xisig = pairsDsig/pairsR
print xisig


#now put these into numpy arrays and save them
np.save('zeta.npy',zeta)
np.save('zeta_sigma.npy',powerDsig)
np.save('f_ell.npy', f)
np.save('xi.npy', xi)
np.save('xi_sigma.npy', xisig)
np.save('f_one.npy',f[:,:,1])
"""