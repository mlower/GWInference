## Modified version of the Inference example of: https://git.ligo.org/rory-smith/emcee_examples/blob/master/mpi4py/two_detectors/tc_marg_12D_PE.py
##
## Marcus E. Lower (2017)

import numpy as np
from scipy.misc import logsumexp
from scipy.interpolate import interp1d
import lal
import lalsimulation as lalsim
import emcee
from emcee import PTSampler
from emcee.utils import MPIPool

import os, sys
import time
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('../../../..')
import MonashGWTools.waveforms as wv
import MonashGWTools.tools as tools

def mc_eta_to_m1m2(mc, eta):
  # note m1 >= m2
    if eta <= 0.25 and eta > 0.:
        root = np.sqrt(0.25-eta)
        fraction = (0.5+root) / (0.5-root)
        m1 = mc * (pow(1+1.0/fraction,0.2) / pow(1.0/fraction,0.6))
        m2 = mc * (pow(1+fraction,0.2) / pow(fraction,0.6))
        return m1, m2
    else:
        return 1., 500.

def gen_waveform(deltaF, m1, m2, fmin, fmax, iota, dist, e_min):

    fref = 20.
    m1 *= lal.lal.MSUN_SI
    m2 *= lal.lal.MSUN_SI
    dist = 1e6*lal.lal.PC_SI*dist
    phaseO = 1
    phiRef = 0.
    
    meanPerAno = 0.0
    longAscNodes = 0.0
    approx = lalsim.GetApproximantFromString("EccentricFD")
    
    WFdict = lal.CreateDict()
    
    H = lalsim.SimInspiralChooseFDWaveform(m1, m2, 0, 0, 0, 0, 0, 0, dist, iota, phiRef, longAscNodes, e_min, meanPerAno, deltaF, fmin, fmax, fref, WFdict, approx)
    
    hplus = H[0].data.data
    hcross = H[1].data.data
    
    return hplus, hcross

def detector_strain(h_p, h_c, RA, DEC, psi, epoch, deltaF):
    
    tgps = lal.LIGOTimeGPS(epoch)
    gmst = lal.GreenwichMeanSiderealTime(tgps)
    
    # Antenna response:
    fplus, fcross = wv.AntennaResponse(RA, DEC, psi, epoch, ifo='H1')
    diff = lal.LALDetectorIndexLHODIFF
    
    timedelay = lal.TimeDelayFromEarthCenter(lal.CachedDetectors[diff].location, RA, DEC, tgps)
    timeshift = epoch + timedelay
    
    #Calculate the observed strain at the detector, properly shifting the waveform from geocenter to detector frame.
    h = np.zeros_like(h_p, dtype=complex)
    pit = np.pi*timeshift
    if timeshift != 0.0:
        shift = complex(1.0, 0)
        dshift = complex(-2.*np.sin(pit*deltaF)*np.sin(pit*deltaF), -2.*np.sin(pit*deltaF)*np.cos(pit*deltaF))
        for i in xrange(0,h_p.size):
            h[i] = shift*(fplus*h_p[i] + fcross*h_c[i])
            shift += shift*dshift
    else: h = (fplus*h_p) + (fcroh_c)
        
    return h

def make_arr_samesize(arr1,arr2):
    '''
    Makes two arrays the same size by appending zeroes to the end of the 2nd one.
    '''
    
    len1 = np.size(arr1)
    len2 = np.size(arr2)
    
    if len1 < len2:
        diff = int(len2 - len1)
        arr1 = np.append(arr1,np.zeros(diff))
        len1 = np.size(arr1)

    elif len2 < len1:
        diff = int(len1 - len2)
        arr2 = np.append(arr2,np.zeros(diff))
        len1 = np.size(arr2)

    return arr1, arr2                
                                                                             
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
    iota = x[4]
    RA = x[5]
    DEC = x[6]

    if m1 >= m1_min and m1 <= m1_max and m2 <= m2_max and m2 >= m2_min and e_min <= ecc_max and e_min >= ecc_min and dist <= dist_max and dist >= dist_min and iota <= angle_max and iota >= angle_min and RA <= angle_max and RA >= angle_min and DEC <= angle_max and DEC >= angle_min and m1 > m2 :
        
        eta = (m1*m2)/((m1+m2)**2.)
        
        logprior = np.log(((m1+m2)*(m1+m2))/((m1-m2)*pow(eta,3.0/5.0)) )

        return logprior
    else:
        return -np.inf

def logL(x, data, PSD, fmin, fmax, deltaF):
    '''
    Generates the log likelihood
    '''
    m1 = x[0]
    m2 = x[1]
    e_min = x[2]
    dist = x[3]
    iota = x[4]
    RA = x[5]
    DEC = x[6]

    if m1 >= m1_min and m1 <= m1_max and m2 <= m2_max and m2 >= m2_min and e_min <= ecc_max and e_min >= ecc_min and dist <= dist_max and dist >= dist_min and iota <= angle_max and iota >= angle_min and RA <= angle_max and RA >= angle_min and DEC <= angle_max and DEC >= angle_min and m1 > m2 :
                  
        # generate the waveform:
        e_min = 10**(e_min)
        hp, hc = gen_waveform(deltaF, m1, m2, fmin, fmax, iota, dist, e_min)
               
        # Start time of data + strain in detector frame:
        epoch = 1000000008
        htilde = detector_strain(hp, hc, RA, DEC, 0., epoch, deltaF)

        # make len(htilde) = len(data) by appending zeroes
        htilde, data = make_arr_samesize(htilde, data)
        
        ##-- Likelihood --###
        dh =  deltaF*4*data.conjugate()*htilde / PSD
        hh = deltaF*4.*np.sum( htilde.conjugate()*htilde/PSD ).real
        dd = deltaF*4.*np.sum( data.conjugate()*data/PSD ).real
        scores = len(data)*( -0.5* ( -2*np.fft.irfft(dh)) )
        
        # log(Likelihood):
        logL = logsumexp( scores ).real - 0.5*( hh + dd ) + np.log(1./(2.*fmax))
        return logL
        
    else:
        return -np.inf

#------------------------------------------------#
## Setting up sampling parameters:
ntemps = 16
nwalkers = 500
ndim = 7
nsteps = 2000

## Waveform parameters:
fmax = 512.
fmin = 20.
Fs = 2*fmax
deltaF = 1./8.

## Load in data from file:

data_file = np.load('injections/injectionFFT.npy')
data = data_file[:,1]
data = data[:int(fmax/deltaF)+1]
freq = data_file[:,0].real[:int(fmax/deltaF)+1]

PSD_file = np.loadtxt("../../../../MonashGWTools/NoiseCurves/aLIGO_ZERO_DET_high_P_psd.txt")
PSD = PSD_file[:,1][:int(fmax/deltaF)+1]
PSD_interp_func = interp1d(PSD_file[:,0], PSD_file[:,1], bounds_error=False, fill_value=np.inf)
PSD = PSD_interp_func(freq)
#------------------------------------------------#  
## Minimum + maximum thresholds:
m1_min, m1_max = 5, 50
m2_min, m2_max = 5, 50
ecc_min, ecc_max = np.log10(0.1),np.log10(0.5)
angle_min, angle_max = 0., np.pi*2.
dist_min, dist_max = 50, 3000.
    
m1 = np.random.uniform(low=(m1_min+5), high=m1_max, size=(ntemps, nwalkers, 1))
m2 =  np.random.uniform(low=m2_min, high=m2_max, size=(ntemps, nwalkers, 1))

e_min = np.random.uniform(low=ecc_min, high=ecc_max, size=(ntemps, nwalkers, 1))
    
dist = np.random.uniform(low=dist_min, high=dist_max, size=(ntemps, nwalkers, 1))
iota = np.random.uniform(low=angle_min, high=0.5*angle_max, size=(ntemps, nwalkers, 1))
RA = np.random.uniform(low=angle_min, high=angle_max, size=(ntemps, nwalkers, 1))
DEC = np.random.uniform(low=angle_min, high=angle_max, size=(ntemps, nwalkers, 1))      

## Ensure m1 > m2:
for i in range(0,ntemps):
    for k in range(0,nwalkers):
        if m2[i,k,0] >= m1[i,k,0]:
            m2[i,k,0] = np.random.uniform(low=m2_min,high=(0.9*m1[i,k,0]),size=1)
        else:
            pass

## Setting initial walker positions:
p0 = np.array([m1, m2, e_min, dist, iota, RA, DEC])
p0 = np.reshape(p0, (ndim,ntemps, nwalkers))
p0 = np.swapaxes(p0, 2,1)
p0 = np.swapaxes(p0, 0,2)

## Setting up the sampler:
betas = np.logspace(0, -ntemps, ntemps, base=10)
pool = MPIPool(loadbalance=True)

if not pool.is_master():
    pool.wait()
    sys.exit(0)
sampler = PTSampler(ntemps, nwalkers, ndim, logL, logPrior, loglargs=[data, PSD, fmin, fmax, deltaF], a=10., betas=betas, pool=pool)

## Running the sampler:
t1 = time.time()
print('sampling underway...')
(pos, lnprob, rstate) = sampler.run_mcmc(p0, nsteps)

t2=time.time()
pool.close()
print(np.shape(lnprob))
print('sampler = ',sampler)

## Getting the evidence and Bayes factor:
(lnZ_pt, dlnZ_pt) = sampler.thermodynamic_integration_log_evidence(fburnin=0.5)
print "lnZ_pt = {} +/- {}".format(lnZ_pt, dlnZ_pt)

BF = lnZ_pt + 0.5*4*deltaF*np.vdot(data,data/PSD).real
print "BF = {} ".format(BF)
print t2-t1

## Plotting the posterior distributions:
param_list = ['m1', 'm2', 'log$_{10}$e', 'distance']
true_values = [35., 30., np.log10(0.1), 410.]

for i in range(4):
    plt.figure()
    plt.hist(sampler.flatchain[i,:,i], 100, color=["k"], histtype="step")
    plt.axvline(true_values[i])
    plt.title("%s"%(param_list[i]))
    plt.xlim(min(sampler.flatchain[i,:,i]),max(sampler.flatchain[i,:,i]))
    plt.savefig("posteriors/%s.png"%param_list[i],dpi=200)

print('shape of sampler.chain = ',np.shape(sampler.chain))    

## Plotting walker:
plt.clf()
res=plt.plot(sampler.chain[0,:,:,0].T, '-', color='k', alpha=0.3)
plt.axhline(35)
plt.xlim(0,2000)
plt.ylim(25,45)
plt.xlabel('Steps',fontsize=14)
plt.ylabel('m1',fontsize=14)
plt.tight_layout()
plt.savefig('posteriors/walker.png',dpi=300)

## Making corner plot:

import corner
print('making corner plots...')
truths=[35.,30.,np.log10(0.4),440.,(0.*np.pi/180),(90*np.pi/180.),(90.*np.pi/180.)]
samples = sampler.chain[0]
samples = samples[:, 100:, :].reshape(-1, ndim)
fig = corner.corner(samples,labels=['m1', 'm2', 'log$_{10}$e', 'dist', 'iota', 'RA', 'DEC'],show_titles=True,quantiles=[0.16, 0.5, 0.84], truths=truths)
fig.savefig("posteriors/triangle.png")

np.savetxt('posteriorsevidence.txt', np.c_[lnZ_pt, dlnZ_pt, BF])

with open('samples/posteriors_DO_NOT_REMOVE_.pickle', 'wb') as handle:
    pickle.dump(pos, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(np.shape(lnprob))
print('sampler = ',sampler)
print('done!')