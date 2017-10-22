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
data_file = np.load('waveformInjection_'+str(job.filename)+'/injectionFFT_'+str(job.filename)+'.npy')
data = data_file[:,1]
data = data[:int(fmax/deltaF)+1]
freq = data_file[:,0].real[:int(fmax/deltaF)+1]

PSD_file = np.loadtxt("../../../../MonashGWTools/NoiseCurves/aLIGO_ZERO_DET_high_P_psd.txt")
PSD = PSD_file[:,1][:int(fmax/deltaF)+1]
PSD_interp_func = interp1d(PSD_file[:,0], PSD_file[:,1], bounds_error=False, fill_value=np.inf)
PSD = PSD_interp_func(freq)
#------------------------------------------------#

## Running sampler + get log evidence & log Bayes factor:
t1 = time.time()

sampler_e, pos, lnprob, rstate = gwi.run_sampler(data, PSD, fmin, fmax, deltaF, ntemps, ndim, nsteps, nwalkers, ecc=True)
print 'finished sampling with e > 0'
sampler_0, pos, lnprob, rstate = gwi.run_sampler(data, PSD, fmin, fmax, deltaF, ntemps, ndim, nsteps, nwalkers, ecc=False)
print 'finished sampling with e = 0'

t2=time.time()
print 'time taken = ',t2-t1

## save samples

#file = 'samples/samples_e_'+str(job.filename)
#with open(file, 'wb') as handle:
#    pickle.dump(sampler_e, handle, protocol=pickle.HIGHEST_PROTOCOL)

#file = 'samples/samples_0_'+str(job.filename)
#with open(file, 'wb') as handle:
#    pickle.dump(sampler_0, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
lnZe_pt, dlnZe_pt = gwi.get_Evidence(sampler_e, pos, lnprob, rstate)
print "lnZe_pt = {} +/- {}".format(lnZe_pt, dlnZe_pt)
lnZ0_pt, dlnZ0_pt = gwi.get_Evidence(sampler_0, pos, lnprob, rstate)
print "lnZ0_pt = {} +/- {}".format(lnZ0_pt, dlnZ0_pt)

lnBF = lnZe_pt - lnZ0_pt
BF = np.exp(lnBF)
test = (np.exp(lnZe_pt))/(np.exp(lnZ0_pt))

print "lnBF = {} ".format(lnBF)
print "BF = {} ".format(BF)
print ""
print'test = ',test

np.savetxt('samples/BayesFactor/logEvidence_and_logBF_'+str(job.filename)+'.txt',np.c_[lnZe_pt, dlnZe_pt, dlnZ0_pt, dlnZ0_pt, BF])

## make corner plots:
print "making corner plots..."
gwi.make_triangles(sampler_e, job, ndim)

print "finished job "+str(job.filename)+"!"
