## Marcus E. Lower (2017)

import numpy as np
import os, sys, time
import GWInference_condor as gwi
from scipy.interpolate import interp1d
import argparse

import pickle
#------------------------------------------------#
## Get job number:
parser = argparse.ArgumentParser(description='Setting name of output file.')
parser.add_argument('-f','--file',type=str,required=True,dest='filename',help='filename output')
job = parser.parse_args()

## Setting up sampling parameters:
ntemps = 8
nwalkers = 200
ndim = 7
nsteps = 1000

## Frequency parameters:
fmax = 512.
fmin = 20.
Fs = 2*fmax
deltaF = 1./8.

## Load in data from file:
data_file = np.load('Injection_'+str(job.filename)+'/injectionFFT_'+str(job.filename)+'.npy')
dataH1 = data_file[:,1]
dataH1 = dataH1[:int(fmax/deltaF)+1]
dataL1 = data_file[:,2]
dataL1 = dataL1[:int(fmax/deltaF)+1]
freq = data_file[:,0].real[:int(fmax/deltaF)+1]

# Interpolating PSD:
PSD_file = np.loadtxt("../MonashGWTools/NoiseCurves/aLIGO_ZERO_DET_high_P_psd.txt")
PSD = PSD_file[:,1][:int(fmax/deltaF)+1]
PSD_interp_func = interp1d(PSD_file[:,0], PSD_file[:,1], bounds_error=False, fill_value=np.inf)
PSD = PSD_interp_func(freq)
#------------------------------------------------#

## Running sampler + get log evidence & log Bayes factor:
t1 = time.time()

lnZe_pt, dlnZe_pt = gwi.run_sampler(dataH1, dataL1, PSD, fmin, fmax, deltaF, ntemps, ndim, nsteps, nwalkers, job, ecc=True)
print("finished sampling with e > 0")
t2 = time.time()
print("time taken = ",t2 - t1)

lnZ0_pt, dlnZ0_pt = gwi.run_sampler(dataH1, dataL1, PSD, fmin, fmax, deltaF, ntemps, ndim, nsteps, nwalkers, job, ecc=False)
print("finished sampling with e = 0")
t3=time.time()
print("time taken = ",t3 - t2)
    
print("lnZe_pt = {} +/- {}".format(lnZe_pt, dlnZe_pt))
print("lnZ0_pt = {} +/- {}".format(lnZ0_pt, dlnZ0_pt))

lnBF = lnZe_pt - lnZ0_pt
BF = np.exp(lnBF)
test = (np.exp(lnZe_pt))/(np.exp(lnZ0_pt))

print("lnBF = {} ".format(lnBF))
print("BF = {} ".format(BF))

np.savetxt('samples/BayesFactor/logEvidence_and_logBF_'+str(job.filename)+'.txt',np.c_[lnZe_pt, dlnZe_pt, lnZ0_pt, dlnZ0_pt, BF])

print("finished job "+str(job.filename)+"!")
