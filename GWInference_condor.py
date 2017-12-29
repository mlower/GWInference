##

import numpy as np
from scipy.misc import logsumexp
from scipy.interpolate import interp1d
import lal
import lalsimulation as lalsim
import emcee
from emcee import PTSampler
import GenWaveform as wv

import os, sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import corner

sys.path.append('../')
import MonashGWTools.tools as tools

#-----------------------------------------#

## Minimum + maximum thresholds:
m1_min = 1.0; m1_max = 100.0
m2_min = 1.0; m2_max = 100.0
dist_min = 10; dist_max = 1e4

ecc_min = np.log10(1.e-4); ecc_max = np.log10(0.5)
angle_min = 0.0; angle_max = np.pi*2.0

#thresh = [[m1_min, m1_max],
#         [m2_min, m2_max],
#         [dist_min, dist_max],
#         [ecc_min, ecc_max],
#         [angle_min, angle_max]]

#-----------------------------------------#

def gen_waveform(deltaF, m1, m2, fmin, fmax, iota, dist, e_min):

    fref = 20.
    m1 *= lal.lal.MSUN_SI
    m2 *= lal.lal.MSUN_SI
    dist = 1e6*lal.lal.PC_SI*dist
    S1 = [0,0,0]
    S2 = [0,0,0]
    phaseO = 1
    phiRef = np.pi
    
    meanPerAno = 0.0
    longAscNodes = 0.0
    approx = lalsim.GetApproximantFromString("EccentricFD")
    
    WFdict = lal.CreateDict()
    
    H = lalsim.SimInspiralChooseFDWaveform(m1, m2, S1[0], S1[1], S1[2], S2[0], S2[1], S2[2], dist, iota, phiRef, longAscNodes, e_min, meanPerAno, deltaF, fmin, fmax, fref, WFdict, approx)
    
    hplus = H[0].data.data
    hcross = H[1].data.data
    
    return hplus, hcross

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

def uniformPrior(var, varMin, varMax):
    ## uniform prior
    return 10**(varMin + var * (varMax - varMin))


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

    if m1 >= m1_min and m1 <= m1_max and m2 <= m2_max and m2 >= m2_min and ((e_min <= ecc_max and e_min >= ecc_min) or e_min == 0.0) and dist <= dist_max and dist >= dist_min and iota <= angle_max and iota >= angle_min and RA <= angle_max and RA >= angle_min and DEC <= angle_max and DEC >= angle_min and m1 > m2 :
        
        eta = (m1*m2)/((m1+m2)**2.)
        
        ##-- Flat Prior --###
        logprior = 1.#np.log(((m1+m2)*(m1+m2))/((m1-m2)*pow(eta,3.0/5.0)) )
        return logprior
    
    else:
        return -np.inf

def logL(x, dataH1, dataL1, PSD, fmin, fmax, deltaF):
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
    psi = 0.0
    S1=[0,0,0]
    S2=[0,0,0]
    phi = np.pi
    fRef = fmin
    
    if m1 >= m1_min and m1 <= m1_max and m2 <= m2_max and m2 >= m2_min and ((e_min <= ecc_max and e_min >= ecc_min) or e_min == 0.0) and dist <= dist_max and dist >= dist_min and iota <= angle_max and iota >= angle_min and RA <= angle_max and RA >= angle_min and DEC <= angle_max and DEC >= angle_min and m1 > m2 :
                  
        # Generate the waveform:
        tc = 1000000008
        e_min = 10**(e_min)        
        hp, hc = wv.GenFDWaveform(fmin, fmax, deltaF, dist, m1, m2, S1, S2, e_min, fRef, iota, phi, waveform='EccentricFD')
        htildeH,f_arrH = wv.DetectorStrain(hp, hc, fmax, deltaF, RA, DEC, psi, tc, ifo='H1')       
        
        ##-- Likelihood --###
        
        logL = -0.5 * (4*deltaF*np.vdot(dataH1 - htildeH, (dataH1 - htildeH)/PSD)).real
        print(logL)
        return logL

def run_sampler(dataH1, dataL1, PSD, fmin, fmax, deltaF, ntemps, ndim, nsteps, nwalkers, job, ecc=True):
    '''
    Setting parameters: 
    '''
    m1_min, m1_max = 5, 50
    m2_min, m2_max = 5, 50

    angle_min, angle_max = 0., np.pi*2.
    dist_min, dist_max = 50, 3000.
    
    m1 = np.random.uniform(low=(m1_min+5), high=m1_max, size=(ntemps, nwalkers, 1))
    m2 =  np.random.uniform(low=m2_min, high=m2_max, size=(ntemps, nwalkers, 1))
    
    if ecc == True:
        ecc_min, ecc_max = np.log10(1.e-4),np.log10(0.5)
        e_min = np.random.uniform(low=ecc_min, high=ecc_max, size=(ntemps, nwalkers, 1))
    else:
        e_min = np.zeros((ntemps, nwalkers, 1))
    
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
    sampler = PTSampler(ntemps, nwalkers, ndim, logL, logPrior, loglargs=[dataH1, dataL1, PSD, fmin, fmax, deltaF], threads=16, a=10., betas=betas)

    ## Running the sampler:
    print 'sampling underway...'
    (pos, lnprob, rstate) = sampler.run_mcmc(p0, nsteps)
    
    ## Get log evidence & log Bayes factor
    lnZ, dlnZ = get_Evidence(sampler, pos, lnprob, rstate)
    
    ## make corner plots:
    
    if ecc == True:
        print "making corner plots..."
        make_triangles(sampler, job, ndim)
    
    #return sampler, pos, lnprob, rstate
    return lnZ, dlnZ

def get_Evidence(sampler, pos, lnprob, rstate):
    '''
    Getting the evidence and Bayes factor:
    '''
    (lnZ_pt, dlnZ_pt) = sampler.thermodynamic_integration_log_evidence(fburnin=0.5)
    
    return lnZ_pt, dlnZ_pt

def make_triangles(sampler, job, ndim):
    ## Making corner plots:
    truths=[35.,30.,np.log10(0.1),220.,(90.*np.pi/180),(90*np.pi/180.),(90.*np.pi/180.)]
    samples = sampler.chain[0]
    samples = samples[:, 100:, :].reshape(-1, ndim)
    fig = corner.corner(samples,labels=['m1', 'm2', 'log$_{10}$e', 'dist', 'iota', 'RA', 'DEC'],show_titles=True,quantiles=[0.16, 0.5, 0.84], truths=truths)
    fig.savefig("posteriors/triangle_"+str(job.filename)+".png")
