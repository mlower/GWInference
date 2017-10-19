import numpy as np
import os, sys
import pickle

file = 'samples/samples_e_0'
with open(file, 'rb') as handle:
    sampler = pickle.load(handle)

#file = 'samples/samples_0_job0'
#with open(file, 'rb') as handle:
#    sample_0 = pickle.load(handle)

sample_e = sampler.flatchain
#print(np.shape(sample_e))
sample_e[:,:, = np.append(sample_e[0][:][:],sample_e[0][:][:])

#print(np.shape(sample_e))
print(sample_e[4])
quit()

ntemps = 16
nwalkers = 20
ndim = 8
nsteps = 2000

import corner
print('making corner plots...')
truths=[35.,30.,np.log10(0.1),300.,(10.*np.pi/180),(20.9375*np.pi/180.),(45.*np.pi/180.),0.]
samples = sampler.chain[0]
samples = samples[:, 100:, :].reshape(-1, ndim)
fig = corner.corner(samples,labels=['m1', 'm2', 'log$_{10}$e', 'dist', 'iota', 'RA', 'DEC', '$\phi_{ref}$'],show_titles=True,quantiles=[0.16, 0.5, 0.84], truths=truths)
fig.savefig("posteriors/triangle.png")
