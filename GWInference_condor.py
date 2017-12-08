##

import numpy as np
from scipy.misc import logsumexp
from scipy.interpolate import interp1d
import lal
import lalsimulation as lalsim
import emcee
from emcee import PTSampler

import os, sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import corner

sys.path.append('../')
import MonashGWTools.waveforms as wv
import MonashGWTools.tools as tools

#-----------------------------------------#

## Setting parameters ##
## Minimum + maximum thresholds:
m1_min, m1_max = 5, 50
m2_min, m2_max = 5, 50
ecc_min, ecc_max = np.log10(1.e-4),np.log10(0.5)
angle_min, angle_max = 0., np.pi*2.
dist_min, dist_max = 50, 3000.

#-----------------------------------------#

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

def detector_strain(h_p, h_c, RA, DEC, psi, epoch, deltaF, ifo):
    
    epoch_GPS = lal.LIGOTimeGPS(epoch)
    gmst = lal.GreenwichMeanSiderealTime(epoch_GPS)
    
    # Antenna response:
    fplus, fcross = wv.AntennaResponse(RA, DEC, psi, epoch, ifo)
    diff = lal.LALDetectorIndexLHODIFF
    
    timedelay = lal.TimeDelayFromEarthCenter(lal.CachedDetectors[diff].location, RA, DEC, epoch_GPS)
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
        
    return h, timedelay

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
        
        ##-- Flat Prior --###
        logprior = 1.#np.log(((m1+m2)*(m1+m2))/((m1-m2)*pow(eta,3.0/5.0)) )
        return logprior
    
    elif m1 >= m1_min and m1 <= m1_max and m2 <= m2_max and m2 >= m2_min and e_min == 0.0 and dist <= dist_max and dist >= dist_min and iota <= angle_max and iota >= angle_min and RA <= angle_max and RA >= angle_min and DEC <= angle_max and DEC >= angle_min and m1 > m2 :
        
        eta = (m1*m2)/((m1+m2)**2.)
        
        ##-- Flat Prior --###
        logprior = 1.#np.log(((m1+m2)*(m1+m2))/((m1-m2)*pow(eta,3.0/5.0)) )
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
                  
        # Generate the waveform:
        e_min = 10**(e_min)
        hp, hc = gen_waveform(deltaF, m1, m2, fmin, fmax, iota, dist, e_min)
               
        # Start time of data + strain in detector frame:
        epoch = 1000000008
        
        # Strain at Hanford:
        htildeH, timedelay_H = detector_strain(hp, hc, RA, DEC, 0., epoch, deltaF, ifo='H1')
        
        # Strain at Livingston
        htildeL, timedelay_L = detector_strain(hp, hc, RA, DEC, 0., epoch, deltaF, ifo='L1')

        # Make len(htilde) = len(data) by appending zeroes
        htildeH, dataH = make_arr_samesize(htildeH, data)
        htildeL, dataL = make_arr_samesize(htildeL, data)
        
        # Adjust Fourier spectra for time-delay b/w H1 & L1:
        fseries = np.linspace(0, fmax, int(fmax/deltaF)+1)
        htildeH = np.exp(1j*np.pi*2*fseries*timedelay_H)
        htildeL = np.exp(1j*np.pi*2*fseries*timedelay_L)
        ##-- Likelihood --###
        
        dh_H = deltaF*4*data.conjugate()*htildeH / PSD
        hh_H = deltaF*4.*np.sum( htildeH.conjugate()*htildeH/PSD ).real
        dd_H = deltaF*4.*np.sum( dataH.conjugate()*dataH/PSD ).real
        
        dh_L = deltaF*4*data.conjugate()*htildeL / PSD
        hh_L = deltaF*4.*np.sum( htildeL.conjugate()*htildeL/PSD ).real
        dd_L = deltaF*4.*np.sum( dataL.conjugate()*dataL/PSD ).real
        
        scores_H = len(dataH)*( -0.5* ( -2*np.fft.irfft(dh_H)) )
        scores_L = len(dataL)*( -0.5* ( -2*np.fft.irfft(dh_L)) )
        
        # log(Likelihood):
        logL = logsumexp(scores_H + scores_L).real - 0.5*(hh_H + hh_L + dd_H + dd_L) + np.log(2./(2.*fmax)) 
        # -0.5 * ( 4*deltaF*np.vdot(data - htilde, (data - htilde)/PSD) - 4*deltaF*np.vdot(data, data/PSD) ).real
        return logL
 
    elif m1 >= m1_min and m1 <= m1_max and m2 <= m2_max and m2 >= m2_min and e_min == 0.0 and dist <= dist_max and dist >= dist_min and iota <= angle_max and iota >= angle_min and RA <= angle_max and RA >= angle_min and DEC <= angle_max and DEC >= angle_min and m1 > m2 :    
        
        # Generate the waveform:
        hp, hc = gen_waveform(deltaF, m1, m2, fmin, fmax, iota, dist, e_min)
               
        # Start time of data + strain in detector frame:
        epoch = 1000000008
        
        # Strain at Hanford:
        htildeH, timedelay_H = detector_strain(hp, hc, RA, DEC, 0., epoch, deltaF, ifo='H1')
        
        # Strain at Livingston
        htildeL, timedelay_L = detector_strain(hp, hc, RA, DEC, 0., epoch, deltaF, ifo='L1')

        # Make len(htilde) = len(data) by appending zeroes
        htildeH, dataH = make_arr_samesize(htildeH, data)
        htildeL, dataL = make_arr_samesize(htildeL, data)
        
        # Adjust Fourier spectra for time-delay b/w H1 & L1:
        fseries = np.linspace(0, fmax, int(fmax/deltaF)+1)
        htildeH = np.exp(1j*np.pi*2*fseries*timedelay_H)
        htildeL = np.exp(1j*np.pi*2*fseries*timedelay_L)
        ##-- Likelihood --###
        
        dh_H = deltaF*4*data.conjugate()*htildeH / PSD
        hh_H = deltaF*4.*np.sum( htildeH.conjugate()*htildeH/PSD ).real
        dd_H = deltaF*4.*np.sum( dataH.conjugate()*dataH/PSD ).real
        
        dh_L = deltaF*4*data.conjugate()*htildeL / PSD
        hh_L = deltaF*4.*np.sum( htildeL.conjugate()*htildeL/PSD ).real
        dd_L = deltaF*4.*np.sum( dataL.conjugate()*dataL/PSD ).real
        
        scores_H = len(dataH)*( -0.5* ( -2*np.fft.irfft(dh_H)) )
        scores_L = len(dataL)*( -0.5* ( -2*np.fft.irfft(dh_L)) )
        
        # log(Likelihood):
        logL = logsumexp(scores_H + scores_L).real - 0.5*(hh_H + hh_L + dd_H + dd_L) + np.log(2./(2.*fmax)) 
        # -0.5 * ( 4*deltaF*np.vdot(data - htilde, (data - htilde)/PSD) - 4*deltaF*np.vdot(data, data/PSD) ).real
        return logL
    
    else:
        print('logL params out of range')
        return -np.inf

def run_sampler(data, PSD, fmin, fmax, deltaF, ntemps, ndim, nsteps, nwalkers, job, ecc=True):
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
    sampler = PTSampler(ntemps, nwalkers, ndim, logL, logPrior, loglargs=[data, PSD, fmin, fmax, deltaF], threads=16, a=10., betas=betas)

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
