"""
Description of Code:
    - Gaunt Integrals/ Wigner 3j: dlmf 
    - Make a Gaunt Integral of 4 spherical Harmonics (H)
        - H = \int d\Omega Y_{l_1 m_1}(\hat{r}) * Y_{l_2 m_2}(\hat{r}) * Y_{l_3 m_3}(\hat{r}) * Y_{l_4 m_4}(\hat{r})
        - H = \sum_{LM} (-1)^M * G^{m_1 m_2 M}_{l_1 l_2 L} * G^{M m_3 m_4}_{L l_3 l_4}
        - Equation (53) in Anistropic Practical Computation
    - Then, put all Gaunt Integrals (G and H) into weights
        - Recall: All weights have same arguments
        - You can find them in Equation (40) of Anisotropic Practical Computation
    - Next, put in f-tensor (power spectrum and CAMB)- both shot noise and no shot noise: f2 and f3
    - Compute Xi_lk (lk^th Multipole of the Anisotropic 2PCF)
    - Finally, put everything together to get the full covariance matrix for aniso 3PCF
"""

# Imports
import sys
import numpy as np
from numpy import sqrt # Gaunt integrals return w/ sqrt and pi
from numpy import pi
from sympy.physics.wigner import wigner_3j
from sympy.physics.wigner import gaunt
from sympy.functions.special.tensor_functions import KroneckerDelta 
import scipy.integrate as integrate
from scipy.interpolate import interp1d
import scipy.special as special
import time
#from scipy.special import sph_jn # Spherical Bessel Function for Python 2.7
from scipy.special import spherical_jn # Spherical Bessel Function for Python 3.6
from scipy.integrate import romb # Romberg Integration

# jfn_jbar code imports
k = np.load('/Users/anya/Desktop/3PCF/Results/jfn_jbar/k.npy') #This actually should be the same grid you have the transfer function and power spectrum from CAMB on, if you run CAMB with the grid parameters that agree with those in the sBF calculation code. It may just load in a k grid, in fact.
r = np.load('/Users/anya/Desktop/3PCF/Results/jfn_jbar/r.npy')

kmin = min(k)
kmax = max(k)
num_k = 2**12+1 # need to use 2^n+1, as we put this into Romberg integral
kgrid = np.linspace(kmin, kmax, num_k)
dk = kgrid[1]-kgrid[0] # only for uniform grid
print('kgrid shape', kgrid.shape)

print('kgrid dimensions and min/max are:', k.shape, kmin, kmax) 

k2 = k*k
kgrid2 = kgrid*kgrid
ksqdk = k2*dk

rmin = min(r)
rmax = max(r)
num_r = 2**12+1 # need to use 2^n+1, as we put this into Romberg integral
rgrid = np.linspace(rmin, rmax, num_r)
dr = rgrid[1]-rgrid[0]

print('rgrid dimensions and min/max are:', r.shape, rmin, rmax)

print('have loaded jfns, k and r')

# f-tensor imports
 
f2_noshot = np.load('/Users/anya/Desktop/3PCF/Results/ftensor/f2_tensor.npy') # no shot noise
f3_noshot = np.load('/Users/anya/Desktop/3PCF/Results/ftensor/f3_tensor.npy') # no shot noise

f2_sn = np.load('/Users/anya/Desktop/3PCF/Results/ftensor/f2_tensor_sn.npy') # shot noise
f3_sn = np.load('/Users/anya/Desktop/3PCF/Results/ftensor/f3_tensor_sn.npy') # shot noise

n = 3.0*(10**(-4))
f2 = f2_noshot+(1/n)*f2_sn
f3 = f3_noshot+(1/n)*f3_sn

print('have loaded in f-tensors')

# Power Spectrum Important Imports
# Code taken from Zack Slepian's 3pcf_pred_w_rsd_and rv_v1.py
koh, delttotok2z0 = np.loadtxt('/Users/anya/Desktop/3PCF/CAMB/sep19_dan_params_newns__transfer_outz0.dat', unpack=True, usecols = (0, 6))#NOTE: this is not the Planck cosmology transfer function! Whoops.

print('have loaded in CAMB transfer function at z=0')

#define CAMB wavenumber grid and its spacing.
h = 1.0
k_camb = koh*h
k2_camb = k_camb*k_camb
dlogk_camb = 1./5000 #set in CAMB params.ini
dk_camb = k_camb*dlogk_camb

# Some cosmology inputs
b1 = 2.0 # linear bias
f = 0.74 # logarithmic derivative linear derivative wrt linear growth rate a- Lahav or Carroll 1994
beta = f/b1 # beta ~ 0.37
beta_sq = beta*beta
linear_bias = 2 
tpi=(1./(2.*np.pi**2))

cosmology = np.load('/Users/anya/Desktop/3PCF/Results/cosmology.npy')
ns = cosmology[4]
sigma8 = cosmology[9]

print('cosmology loaded')

# These values are for Gaunt Integrals in Weights
lq = np.array((0,1,2))# ell_q should be 0, 2 and 4 (lp, lq, lk)
lp = np.array((0,1,2))# ell_p
lk = np.array((0,1,2))# ell_k
#l1 = np.array((0,1,2,3))# ell_1
#l2 = np.array((0,1,2,3))# ell_2
#l1prime = np.array((0,1,2))# l'_1
#l2prime = np.array((0,1,2))# l'_2
#m = np.array((0,1,2))# m
#mprime = np.array((0,1,2))# m'
J1 =  np.array((0,2,4))# J1 <= 2*l_max+4
J2 = np.array((0,2,4))# J2
J3 = np.array((0,2,4))# J3
#r1 = np.array((0,1,2))# r1
#r2 = np.array((0,1,2))# r2
#r1prime = np.array((0,1,2))# r'_1
#r2prime = np.array((0,1,2))# r'_2


#------------------------------------------------------------------
# WEIGHTS
# NOTE: These functions take in integers NOT arrays.
# They are used in loops for arrays of their arguments.
#------------------------------------------------------------------

"""
In cov_func, we create loops of S1, S2 and S3 which means they are set values before we get to the weights.
Need to change the code to take in an S1 or S2 value. If the expected value matches the value passed in, then
you can continue computing the weight. ELSE, the weight can just go to zero. This is probably easier. 
computationally, too.
"""

# for weights, pass in J1, J2, and J3 values similar to S1 and S2 because we need to sum over them in cov func

def H1_w_1_2(weight_num, lq, l1, l1prime, m, mprime, S1): # first gaunt integral for w1 or w2
    H1 = 0.
    # lq, lp have to be 0, 2, 4
    # if l1, l1prime is even then, L must be even and then J1 must be even
    Lmin = abs(l1-l1prime) # set L
    Lmax = l1+l1prime
    if weight_num == 1: # compute first gaunt integral weight 1
        if S1 == -m-mprime:
            M = -S1
            for L in range(Lmin, Lmax+1, 1):
                J1min = abs(lq-L) # set J1
                J1max = lq+L
                for J1 in range(J1min, J1max+1, 1):
                    # H1 is the sum of the gaunt integrals within the J1 range
                    H1 += gaunt(lq, J1, L, 0, -S1, -M)*gaunt(L, l1, l1prime, M, -m, -mprime) 
            H1 *= ((-1)**(M))
        else:
            H1 = 0.
    elif weight_num == 2: # compute first gaunt integral for weight 2
        if S1 == m-mprime:
            M = -S1
            for L in range(Lmin, Lmax+1, 1):
                J1min = abs(lq-L) # set J1
                J1max = lq+L
                for J1 in range(J1min, J1max+1, 1):
                    H1 += gaunt(lq, J1, L, 0, -S1, -M)*gaunt(L, l1, l1prime, M, -m, mprime)
            H1 *= ((-1)**(M)) # M = -S1
        else:
            H1 = 0.
    else:
        raise ValueError('weight_num can only equal 1 or 2')
        
    return float(H1)

def H2_w_1_2(weight_num, lp, l2, l2prime, m, mprime, S1): # second gaunt integral for w1 or w2
    H2 = 0.
    Lmin = abs(l2-l2prime) # set L
    Lmax = l2+l2prime
    if weight_num == 1: # compute 2nd gaunt integral for w1
        if S1 == -mprime-m:
            M = -S1
            for L in range(Lmin, Lmax+1, 1):
                J2min = abs(lp-L) # set J2
                J2max = lp+L
                for J2 in range(J2min, J2max+1, 1):
                    H2 += gaunt(lp, J2, L, 0, S1, -M)*gaunt(L, l2, l2prime, M, m, mprime)
            H2 *= ((-1)**S1) # M = S1
        else:
            H2 = 0.
    elif weight_num == 2: # compute 2nd gaunt integral for w2
        if S1 == m-mprime:
            M = -S1
            for L in range(Lmin, Lmax+1, 1):
                J2min = abs(lp-L) # set J2
                J2max = lp+L
                for J2 in range(J2min, J2max+1, 1):
                    H2 += gaunt(lp, J2, L, 0, S1, -M)*gaunt(L, l2, l2prime, M, m, -mprime)
            H2 *= ((-1)**S1) # M = S1
        else:
            H2 = 0.
    else:
        raise ValueError('weight_num can only equal 1 or 2')
    
    return float(H2)


def w_3_4(weight_num, lq, lp, lk, l1, l1prime, l2, l2prime, m, mprime, S2): # weights for w3 or w4
    H = 0. # 4-spherical harmonic gaunt integral for w3 and w4
    w = 0. # product of 2 gaunt integrals and H for w3 and w4
    Lmin = abs(l2-l1prime)
    Lmax = l2+l1prime
    J1min = abs(lq-l1) # J1 is the same for w3 and w4
    J1max = lq+l1
    J3min_3 = abs(lk-l1prime) # J3 for w3
    J3max_3 = lk+l1prime 
    J3min_4 = abs(lk-l2prime) # J3 for w4
    J3max_4 = lk+l2prime
    if weight_num == 3: # compute w3
        if S2 == mprime+m:
            for L in range(Lmin, Lmax+1, 1):
                J2min = abs(lp-L)
                J2max = lp+L
                M = -S2
                for J2 in range(J2min, J2max+1, 1):    
                    H += gaunt(lp, J2, L, 0, -S2, -M)*gaunt(L, l2, l1prime, M, m, mprime)# H_gaunt
            H *= ((-1)**M)
            for J1 in range(J1min, J1max+1, 1):
                for J3 in range(J3min_3, J3max_3+1, 1):
                    # total product of gaunt integrals
                    g1 = gaunt(lq, J1, l1, 0, m, -m)
                    g2 = gaunt(lk, J3, l1prime, 0, mprime, -mprime)
                    w += g1*float(H)*g2
        else:
            w = 0.
    elif weight_num == 4: # compute w4
        if S2 == m-mprime:
            for L in range(Lmin, Lmax+1, 1):
                J2min = abs(lp-L)
                J2max = lp+L      
                M = -S2
                for J2 in range(J2min, J2max+1, 1):
                    H += gaunt(lp, J2, L, 0, -S2, -M)*gaunt(L, l2, l1prime, M, m, -mprime) # H_gaunt
            H *= ((-1)**M)
            for J1 in range(J1min, J1max+1, 1):
                for J3 in range(J3min_4, J3max_4+1, 1):
                    g1 = gaunt(lq, J1, l1, 0, m, -m)
                    g2 = gaunt(lk, J3, l2prime, 0, -mprime, mprime)
                    w += g1*H*g2
        else:
            w = 0.
    else:
            raise ValueError('weight_num can only equal 3 or 4')
    return float(w) # return weight


def w_5(lq, lp, lk, l1, l1prime, l2, l2prime, m, mprime, S1): # Compute w5
    H = 0. # 4-spherical harmonic gaunt integral for w5
    w5 = 0. # weight 5
    if S1 == mprime - m:
        M = -S1
        Lmin = abs(l1-l2prime)
        Lmax = l1+l2prime
        for L in range(Lmin, Lmax+1, 1):
            J1min = abs(lq-L)
            J1max = lq+L
            for J1 in range(J1min, J1max+1, 1):
                H += float(gaunt(lq, J1, L, 0, -S1, -M)*gaunt(L, l1, l2prime, M, -m, mprime))# H_gaunt
        H *= ((-1)**(M))
        J2min = abs(lp-l2) # set parameters for J2 and J3
        J2max = lp+l2
        J3min = abs(lk-l1prime)
        J3max = lk+l1prime
        for J2 in range(J2min, J2max+1, 1): # compute entire weight
            for J3 in range(J3min, J3max+1,1):
                g1 = gaunt(lp, J2, l2, 0, -m, m)
                g2 = gaunt(lk, J3, l1prime, 0, mprime, -mprime)
                w5 += float(H)*g1*g2
    else:
        w5 = 0
    return float(w5)

def w_6(lq, lp, lk, l1, l1prime, l2, l2prime, m, mprime, S1): # Compute w6
    H = 0. #H_gaunt for w6
    w6 = 0. # total product of gaunt integrals for w6
    if S1 == -mprime - m:
        M = -S1
        Lmin = abs(l1-l1prime)
        Lmax = l1+l1prime
        for L in range(Lmin, Lmax+1, 1):
            J1min = abs(lq-L)
            J1max = lq+L
            for J1 in range(J1min, J1max+1, 1):
                H += gaunt(lq, J1, L, 0, -S1, -M)*gaunt(L, l1, l1prime, M, -m, -mprime) # H_gaunt
        H *= ((-1)**(M))
        J2min = abs(lp-l2)
        J2max = l2+lp
        J3min = abs(lk-l1prime)
        J3max = lk+l1prime
        for J2 in range(J2min, J2max+1, 1): #compute w6
            for J3 in range(J3min, J3max+1,1):
                g1 = gaunt(lp, J2, l2, 0, -m, m)
                g2 = gaunt(lk, J3, l1prime, 0, mprime, -mprime)
                w6 += float(H)*g1*g2
    else:
        w6 = 0
    return float(w6)

"""
UNIT TEST FOR WEIGHTS

>>> H1_w_1_2(1, 2, 4, 8, 1, 1, -2)
-21*sqrt(255)/(4862*pi) - 9*sqrt(46410)/(184756*pi) + 21*sqrt(85)/(46189*pi) + 33303*sqrt(70)/(1749748*pi) + 44*sqrt(672945)/(215441*pi) + 89*sqrt(51051)/(104006*pi) + 339*sqrt(6545)/(96577*pi)
>>>
>>> w_3_4(4, 2, 2, 4, 6, 8, 4, 6, 1, 1, -2)
-0.00710673067037648*sqrt(5)/pi - 0.000204386180817201*sqrt(3927)/pi - 0.000263224626810031*sqrt(2310)/pi - 0.0021249406237028*sqrt(34)/pi - 0.0016670893031302*sqrt(51)/pi - 0.00214700895100101*sqrt(30)/pi - 6.37128271646221e-5*sqrt(23205)/pi - 0.00039043087142573*sqrt(546)/pi - 0.000684384029706081*sqrt(143)/pi - 0.000797460467514662*sqrt(91)/pi
>>>
>>> w_5(2, 2, 4, 6, 8, 4, 6, 1, 1, -2)
-0.000198392732474146*sqrt(23205)/pi - 0.0044481047661587*sqrt(34)/pi - 0.00215498243329428*sqrt(119)/pi - 0.000677280193321059*sqrt(1105)/pi - 0.0030732360202551*sqrt(51)/pi - 0.000174450958885727*sqrt(6630)/pi - 8.13769590195042e-5*sqrt(24310)/pi - 0.000237487859995696*sqrt(2618)/pi - 9.3088352063581e-5*sqrt(15470)/pi - 0.000468361880472655*sqrt(546)/pi - 0.00191359282593113*sqrt(30)/pi - 0.000122342230906874*sqrt(3927)/pi - 0.00295737073098448*sqrt(5)/pi
>>>
>>> w_6(2, 2, 4, 6, 8, 4, 6, 1, 1, -2)
0.00907569353012492*sqrt(5)/pi + 0.000375448563776427*sqrt(3927)/pi + 0.00587250757831613*sqrt(30)/pi + 0.00143732702965779*sqrt(546)/pi + 0.00028567313042697*sqrt(15470)/pi + 0.000728811917918946*sqrt(2618)/pi + 0.000249732755091107*sqrt(24310)/pi + 0.00053536184094046*sqrt(6630)/pi + 0.00943126540523083*sqrt(51)/pi + 0.00207846361776885*sqrt(1105)/pi + 0.00661329332926451*sqrt(119)/pi + 0.0136505157180973*sqrt(34)/pi + 0.000608835280499319*sqrt(23205)/pi
"""


# -------------------------------------
# Power Spectrum and j1calc
# -------------------------------------

def j1calc(k, r_fixed):
    print('computing j1 at fixed r to use in power spectrum normalization')
    start = time.clock()
    j1 = np.zeros(len(k))
    kr = k*r_fixed
    i = 0
    for kr_val in kr:
        j1[i] = spherical_jn(1,kr_val)
        i += 1
    print('time for j1 = ', time.clock() - start, ' s')
    return j1

def powerspecnormfactor(delttotok2z0, sigma8, ns, r_fixed, k, ksqdk): # power spectrum norm factor
    j1 = j1calc(k, r_fixed) 
    win=(3.0*j1)/(k*r_fixed)
    return (sigma8*sigma8)/(tpi*np.tensordot(ksqdk*k**ns*delttotok2z0*delttotok2z0, win*win, axes=([0],[0])))

A = powerspecnormfactor(delttotok2z0, sigma8, ns, 8./h, k_camb, k_camb*k_camb*dk_camb)

# define power spectrum with smoothing to avoid ringing if we FT to real space from finiteness of grid in wavenumber.

Pk_camb = A*delttotok2z0*delttotok2z0*k_camb**ns*np.exp(-k2_camb) # this just gives P_matter
P_galaxy = Pk_camb*(b1**2)*linear_bias # full power spectrum? Need to load in RSD factors
Pk_camb *= linear_bias**2

Pk_interp = interp1d(k_camb, Pk_camb)
Pk = Pk_interp(kgrid)

print('Power spectrum loaded and interpolated')

# ----------------------------------------
# Xi_lk formula
# Equation (47) of Practical Computation 
# ----------------------------------------

def xi_lk(kgrid, r): # lkth multipole of the anisotropic 2PCF
    c0 = 1.0 + (2.0*beta)/3.0 + (beta_sq)/5.0 # define Kaiser constants (Equation 53 of Practical Computation)
    c2 = (4.0*beta)/3.0 + (4.0*beta_sq)/7.0
    c4 = (8.0*beta_sq)/35.0
    kaiser_const = np.array((c0, c2, c4))
    ell_k = np.array((0, 1, 2)) # ell_k can only equal 0, 2, or 4 (Practical Computation - pg 11)
    num_ell_k = len(ell_k)
    num_r = len(r) # already defined?
    xi = np.zeros((num_ell_k, num_r))
    for ell in ell_k: # going through each index in array lk
        rcount = 0
        for r_val in r: # going through each value in r 
            xi[ell][rcount] = romb(kgrid2*Pk_camb*spherical_jn(ell, kgrid*r_val), dx=dk)
            #xi[ell][rcount] = np.sum(kgrid2*Pk*spherical_jn(ell, kgrid*r_val))*dk
            rcount += 1 
    xi_lk = kaiser_const*xi # multiply by Kaiser constants at end
    return xi_lk/(2.0*pi) # returns array

# -----------------------------------
# Full Covariance Matrix Calculation
# -----------------------------------

def D_def(J1, J2, J3): # Practical Computation Equation (43)
    if (J1+J2+J3) % 2.0 == 0:
        if (J1+J2+J3)/2.0 % 2.0 == 0:
            D = 1
        else:
            D = -1
    else:
        raise ValueError('J1+J2+J3 is odd') # result would be imaginary in this case
    return D

def C_def(J1, J2, J3): # Practical Computation Equation (43)
    C1 = (2*J1+1)*(2*J2+1)*(2*J3+1)
    C2 = C1/(4*pi)
    return sqrt(C2)

def cov_func(J1_arr, J2_arr, J3_arr, l1, l2, l1prime, l2prime, m, mprime, l_p, l_q, l_k, r1, r2, r1prime, r2prime):  
    # Equation (53) of Practical Computation
    element = 0
    integral = 0
    for J1 in J1_arr:
        for J2 in J2_arr:
            for J3 in J3_arr:
                for lp in l_p:
                    for lq in l_q:
                        for lk in l_k:
                            for S1 in range(-J1, J1+1, 1):
                                for S2 in range(-J2, J2+1, 1):
                                    S3 = -S1-S2 
                                    w1 = H1_w_1_2(1, lq, l1, l1prime, m, mprime, S1)*H2_w_1_2(1, lp, l2, l2prime, m, mprime, S1)
                                    w2 = H1_w_1_2(2, lq, l1, l1prime, m, mprime, S1)*H2_w_1_2(2, lp, l2, l2prime, m, mprime, S1)
                                    w3 = w_3_4(3, lq, lp, lk, l1, l1prime, l2, l2prime, m, mprime, S2)
                                    w4 = w_3_4(4, lq, lp, lk, l1, l1prime, l2, l2prime, m, mprime, S2)
                                    w5 = w_5(lq, lp, lk, l1, l1prime, l2, l2prime, m, mprime, S1)
                                    w6 = w_6(lq, lp, lk, l1, l1prime, l2, l2prime, m, mprime, S1)
                                    print('J1, J2, J3, S1, S2, S3 = ', J1, J2, J3, S1, S2, S3)
                                    print('w1, w2, w3, w4, w5, w6 = ', w1, w2, w3, w4, w5, w6)                           
                                    element += 1./sqrt((2*lp+1)*(2*lq+1)*(2*lk+1))*D_def(J1, J2, J3) * C_def(J1, J2, J3)*wigner_3j(J1, J2, J3, 0, 0, 0)*(xi_lk(kgrid, r)*(w1*f3[J1][l1][l1prime][r1][r1prime]*f3[J2][l2][l2prime][r2][r2prime] + w2*f3[J1][l1][l2prime][r1][r2prime]*f3[J2][l2][l1prime][r2][r1prime]) + wigner_3j(J1, J2, J3, S1, S2, S3)*(f2[J1][l1][r1]*(w3*f3[J2][l2][l2prime][r2][r2prime]*f2[J3][l1prime][r1prime]*KroneckerDelta(S1,-m)*KroneckerDelta(S3,-mprime) + w4*f3[J2][l2][l1prime][r2][r1prime]*f2[J3][l2prime][r2prime] * KroneckerDelta(S1,-m)*KroneckerDelta(S3,-mprime) + f2[J2][l2][r2]*(w5*f3[J1][l1][l2prime][r1][r2prime]*f2[J3][l1prime][r1prime]* KroneckerDelta(S2,-m)*KroneckerDelta(S3,-mprime)+w6*f3[J1][l1][l1prime][r1][r1prime]*f2[J3][l2prime][r2prime]*KroneckerDelta(S2,-m)*KroneckerDelta(S3,-mprime)))))
    
                                    integral += romb((rgrid*element, dr))
    return integral
                
       
# Values to set for covariance matrix
ellmax = 5 # sets max values for looping over ells - number is subject to change
V = 1 # set survey volume
nbins = 10 # sets max values for looping over alls r's 

"""
pass in the binned r values to covariance matrix
run with once over shot noise and once without

"""
def covariance_matrix(f2, f3):
    const_1 = ((4.0*pi)**(3.0/2.0))/V 
    cov = np.zeros((3,3,3,3,3,3,3,3))
    #cov = np.zeros((len(l1),len(l2),len(r1),len(r2),len(l1prime),len(l2prime),len(r1prime),len(r2prime)))
    for l1 in range(0, ellmax): 
        for l2 in range(0, ellmax): 
            for r1 in range(0, nbins): 
                for r2 in range(0, nbins):
                    for l1prime in range(0, ellmax):
                        for l2prime in range(0, ellmax):
                            print('l1, l2, r1, r2, l1p, l2p, r1p, r2p =', l1, l2, r1, r2, l1prime, l2prime, r1prime, r2prime) 
                            if l1 + l2 + l1prime + l2prime % 2 == 0:
                                if (l1+l2+l1prime+l2prime)/2.0 % 2.0 == 0:
                                    const_2 = 1
                                else:
                                    const_2 = -1
                            else:
                                #const_2 = 0
                                raise ValueError('l1+l2+l1prime+l2prime != 0') # result would be imaginary in this case
                            for r1prime in range(0, nbins):
                                for r2prime in range(0, nbins):
                                    for m in range(0, min(l1, l2)):
                                        for mprime in range(0, min(l1prime, l2prime)):
                                            const_3 = pow(-1, m+mprime)
                                            if l2 <= l1:
                                                if l1prime <= l1:
                                                    if l2prime <= l2:

                                                        cov[l1][l2][r1][r2][l1prime][l2prime][r1prime][r2prime] = const_2*const_3*cov_func(J1, J2, J3, l1, l2, l1prime, l2prime, m, mprime, lp, lq, lk, r1, r2, r1prime, r2prime)
                                                        print('covariance value = ', cov[l1][l2][r1][r2][l1prime][l2prime][r1prime][r2prime])
                                                        cov[l2][l1][r2][r1][l1prime][l2prime][r1prime][r2prime] = cov[l1][l2][r1][r2][l1prime][l2prime][r1prime][r2prime]
                                                        cov[l1][l2][r1][r2][l2prime][l1prime][r2prime][r1prime] = cov[l1][l2][r1][r2][l1prime][l2prime][r1prime][r2prime]
                                                        cov[l2][l1][r2][r1][l2prime][l1prime][r2prime][r1prime] = cov[l1][l2][r1][r2][l1prime][l2prime][r1prime][r2prime] # by symmetry of covariance matrix
                                               
    return const_1*cov

# ----------------------------------------------------
# Map 10-dimensional Covariance Matrix to a 2D tensor
# ----------------------------------------------------
# Remap 10d Tensor to 2d matrix
"""
def flatten_10d_2d_mock_covar_bd(mock_covar_10d_active, bin_dict, num_of_ells, offset_r1, offset_r2):
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
                        # THIS LINE NEEDS TO BE CHANGED
                        covar_2d[index, index_prime] =  mock_covar_6d_active[ell, ellprime, bin_dict[i][0]-offset_r1, bin_dict[i][1]-offset_r2, bin_dict[j][0]-offset_r1, bin_dict[j][1]-offset_r2]
                    else:
                        covar_2d[index, index_prime] = covar_2d[index_prime, index]
                    index_prime += 1
            index += 1
    return covar_2d
"""

covariance = covariance_matrix(f2, f3)

np.save('/Users/anya/Desktop/3PCF/Results/covariance/aniso_covariance.npy', covariance)                    
              
