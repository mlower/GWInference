## Modified version of the Inference example of: https://git.ligo.org/rory-smith/emcee_examples/blob/master/mpi4py/two_detectors/tc_marg_12D_PE.py
##
## Marcus E. Lower (2017)

import numpy as np
from scipy.misc import logsumexp
from scipy.interpolate import interp1d
import lal
import lalsimulation
import emcee
from emcee import PTSampler
from emcee.utils import MPIPool

import os, sys
import time

sys.path.append('../../../..')
import MonashGWTools.waveforms as wv
import MonashGWTools.tools as tools

def Ndiv2(n):
    i = 0
    while n%2==0:
        n /= 2
        i += 1
    return i

def FFT_size(arr):
    '''
    Checking how many times the length of arrays associated with g(t) can be divided by 2. 
    Appends zeros until it can be divided by 2 at least 3 times.   
    This optimises the array length for taking the FFT of the array.
    '''
    lgDiv2 = Ndiv2(np.size(arr))

    while lgDiv2 <= 2:
        arr = np.append(arr,[0.])
        lgDiv2 = Ndiv2(np.size(arr))
    else:
        pass
    return arr

def make_arr_samesize(arr1,arr2):
    '''
    Makes two arrays the same size by appending zeroes to the end of the 2nd one.
    '''
    while len(arr2) < 8192:#len(arr1):
        arr2 = np.append(arr2,[0.])
    else:
        pass
    return arr2

def h_of_t(fmin, e_min, deltaT, dist, m1, m2):
    '''
    Generates an inspiral waveform using the EccentricTD approximant
    '''
    # preset parameters: [refrence phase, reference freq., inclination, amp/phase order]
    phiRef = 0.
    fRef = fmin
    iota = 0
    ampO = -1
    phiO = -1

    # generate waveform:
    H = lalsimulation.SimInspiralEccentricTDPNGenerator(phiRef, deltaT, m1, m2, fRef, fmin, dist, iota, e_min, ampO, phiO)

    # retrieve plus & cross polarized strain
    hplus = H[0].data.data
    hcross = H[1].data.data
    
    # Tukey window to avoid edge effects:
    hplus = tools.window_Tukey(hplus,alpha=0.001)
    hcross = tools.window_Tukey(hcross,alpha=0.001)
    
    return hplus, hcross

def htilde_of_f(hplus, hcross, deltaT):
    '''
    Returns the FD of input TD waveform
    '''
    
    hfplus,f = tools.nfft(hplus, Fs)
    hfcross,f = tools.nfft(hcross, Fs)

    return hfplus, hfcross, f

#------------------------------------------------#
## Load in data from file:
data_H1TD = np.column_stack( np.loadtxt("./injections/injection.dat") )
#data_H1TD = np.column_stack( np.loadtxt("./data/data-dumpH1-timeData.dat") )
data_H1TD[1,:] = tools.window_Tukey(data_H1TD[1,:],alpha=0.2)
#PSD_fileH1 = np.column_stack( np.loadtxt("./data/data-dumpH1-PSD.dat") )
PSD_fileH1 = np.column_stack( np.loadtxt("../../../../MonashGWTools/NoiseCurves/aLIGO_ZERO_DET_high_P_asd.txt") )

## Setting up sampling parameters:
ntemps = 8 
nwalkers = 500
ndim = 7
nsteps = 1000

## Waveform parameters:
deltaT = 1./1024.
wFs = 1./deltaT
fmax = wFs/2.
fmin = 20.

#------------------------------------------------#                                                                               
def LALInferenceCubeToFlatPrior(r, x1, x2):
    return x1 + r * ( x2 - x1 );

def logPrior(x):
    '''
    Generates the log of the prior distribution
    '''

    m1 = x[0]
    m2 = x[1]
    e_min = x[2]
    dist = x[3]
    RA = x[4]
    DEC = x[5]
    psi = x[6]

    if m1 >= m1_min and m1 <= m1_max and m2 <= m2_max and m2 >= m2_min and e_min <= 10**(ecc_max) and e_min >= 10**(ecc_min)  and dist <= dist_max and dist >= dist_min and RA <= angle_max and RA >= angle_min and DEC <= angle_max and DEC >= angle_min and psi <= angle_max and psi >= angle_min and (m1 - m2) >= 0:
        
        eta = (m1*m2)/((m1+m2)**2.)
        logprior = 0#np.log(((m1+m2)*(m1+m2))/((m1-m2)*pow(eta,3.0/5.0)) )
        #print('logP = ',logprior)
        return logprior
    else:
        return -np.inf

def logL(x, data_H1TD, PSD_fileH1, fmin, fmax, deltaT):
    '''
    Generates the log likelihood
    '''

    m1 = x[0]
    m2 = x[1]
    e_min = x[2]
    dist = x[3]
    RA = x[4]
    DEC = x[5]
    psi = x[6]

    wFs = 1./deltaT
    dFs = 1./(data_H1TD[0,1] - data_H1TD[0,0])
 
    if m1 >= m1_min and m1 <= m1_max and m2 <= m2_max and m2 >= m2_min and e_min <= 10**(ecc_max) and e_min >= 10**(ecc_min)  and dist <= dist_max and dist >= dist_min and RA <= angle_max and RA >= angle_min and DEC <= angle_max and DEC >= angle_min and psi <= angle_max and psi >= angle_min :

        # masses + distance in SI units:
        m1 *= lal.lal.MSUN_SI
        m2 *= lal.lal.MSUN_SI
        dist *= 1e6*lal.lal.PC_SI
        
        # generate the waveform:
        hp, hc = h_of_t(fmin, e_min, deltaT, dist, m1, m2)
        
        # Hardcoded to be the time at the start of the data segment for G268556
        epoch = 1000000003.060546875#1000000000#1167559934.62
        
        epoch_GPS = lal.lal.LIGOTimeGPS(epoch)
        gmst = lal.GreenwichMeanSiderealTime(epoch_GPS)
        
        # Strain at the detector:
        fplus, fcross = wv.AntennaResponse(RA, DEC, psi, epoch, ifo='H1')        
        h_t = (fplus * hp) + (fcross * hc)
        
        # Ensure len(h_t) =/= prime number [appends empty elements to ensure this]:
        h_t = make_arr_samesize(data_H1TD[1,:],h_t)
        
        # Make data array the same size as the template by appending zeroes:
        h_H1TD = make_arr_samesize(h_t,data_H1TD[1,:])
        
        # Converting the TD waveform to be FD:
        htilde, hfreq = tools.nfft(h_t, wFs)
        deltaF = hfreq[1] - hfreq[0]
        fmin_index = int(fmin / deltaF)
        htilde = htilde[int(fmin/deltaF):]
        
        H1FD, FDff = tools.nfft(h_H1TD, dFs)
        H1FD = H1FD[int(fmin/deltaF):]
        
        # Interpolate PSD to template frequency array:
        PSD_interp_funct = interp1d(PSD_fileH1[0,:], PSD_fileH1[1,:], bounds_error=False, fill_value=np.inf)
        PSD = PSD_interp_funct(hfreq)
        PSD = PSD[int(fmin/deltaF):]
        
        ##-- Likelihood --###
        logL = -0.5 * (4*deltaF * np.vdot(H1FD - htilde, (H1FD - htilde)/PSD)).real
        #print('logL = ',logL)
        return logL
        
    else:
        return -np.inf

## Min/max of the parameters to be searched over:
m1_min, m1_max = 20., 60.
m2_min, m2_max = 15., 40.
ecc_min, ecc_max = np.log10(1.e-8), np.log10(0.8)
angle_min, angle_max = 0., np.pi*2.
dist_min, dist_max = 50., 2500.
antenna_min, antenna_max = 0, 1

## Setting parameters:
m1 = np.random.uniform(low=m1_min, high=m1_max, size=(ntemps, nwalkers, 1))
m2 =  np.random.uniform(low=m2_min, high=m2_max, size=(ntemps, nwalkers, 1))
e_min = 10**(np.random.uniform(low=ecc_min, high=ecc_max, size=(ntemps, nwalkers, 1)))
dist = np.random.uniform(low=dist_min, high=dist_max, size=(ntemps, nwalkers, 1))

RA = np.random.uniform(low=angle_min, high=angle_max, size=(ntemps, nwalkers, 1))
DEC = np.random.uniform(low=angle_min, high=angle_max, size=(ntemps, nwalkers, 1))
psi = np.random.uniform(low=angle_min, high=angle_max, size=(ntemps, nwalkers, 1))

## m2 cannot be greater than m1, set m2 = m1:
for i in range(0,ntemps):
    for k in range(0,nwalkers):
        if m1[i,k,0] <= m2[i,k,0]:
            m2[i,k,0] = 0.99*m1[i,k,0]
        else:
            pass

## entering parameters:
p0 = np.array([m1, m2, e_min, dist, RA, DEC, psi])
p0 = np.reshape(p0, (ndim,ntemps, nwalkers))
p0 = np.swapaxes(p0, 2,1)
p0 = np.swapaxes(p0, 0,2)

data_H1FD, FDff = tools.nfft(data_H1TD[1,:],wFs)
deltaF = data_H1FD[1] - data_H1FD[0]
PSD = PSD_fileH1[1]
PSD_interp_funct = interp1d(PSD_fileH1[0,:], PSD_fileH1[1,:], bounds_error=False, fill_value=np.inf)
PSD = PSD_interp_funct(FDff)

## Setting up the sampler:
betas = np.logspace(0, -ntemps, ntemps, base=10)

pool = MPIPool(loadbalance=True)

if not pool.is_master():
    pool.wait()
    sys.exit(0)
sampler = PTSampler(ntemps, nwalkers, ndim, logL, logPrior, loglargs=[data_H1TD, PSD_fileH1, fmin, fmax, deltaT], a=10., betas=betas, pool=pool)

## Running the sampler:
t1 = time.time()
print('sampling underway...')
(pos, lnprob, rstate) = sampler.run_mcmc(p0, nsteps)
#all_nans = np.isnan(lnprob)
#lnprob[all_nans] = -15
t2=time.time()
pool.close()
print('posterior = ',pos)
print('lnprob = ',lnprob)
print('rstate = ',rstate)

## Getting the evidence and Bayes factor:
(lnZ_pt, dlnZ_pt) = sampler.thermodynamic_integration_log_evidence(fburnin=0.5)
print "lnZ_pt = {} +/- {}".format(lnZ_pt, dlnZ_pt)

BF = lnZ_pt + 0.5*4*deltaF*np.vdot(data_H1FD,data_H1FD/PSD)
print "BF = {} ".format(BF)
print t2-t1

## Plotting the posterior distributions:
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

## Plotting the posterior distributions:

param_list = ['m1', 'm2', 'e', 'dist']
true_values = [35, 30, 0.2, 400]

for i in range(4):
    plt.figure()
    #pl.hist(sampler.flatchain[:,i][::int(sampler.acor[i])], 100, color="k", histtype="step")
    plt.hist(sampler.flatchain[i,:,i], 100, color=["k"], histtype="step")
    plt.axvline(true_values[i])
    plt.title("%s"%(param_list[i]))
    plt.savefig("test/%s.png"%param_list[i],dpi=200)

print('shape of sampler.chain = ',np.shape(sampler.chain))    

## Plotting walker:
plt.clf()
res=plt.plot(sampler.chain[0,:,:,1].T, '-', color='k', alpha=0.3)
plt.axhline(30, color='blue')
plt.savefig('test/walker.png',dpi=200)
    
## Making corner plot:
import corner
print('making corner plots...')
samples = sampler.chain[0]
samples = samples[:, 100:, :].reshape(-1, ndim)
fig = corner.corner(samples,labels=['m1', 'm2', 'e', 'dist','RA','DEC','psi'])
fig.savefig("test/triangle.png")

np.savetxt('evidence.txt', np.c_[lnZ_pt, dlnZ_pt, BF])
np.savetxt('posteriors.txt', np.c_[pos])

print('done!')
