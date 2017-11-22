##
import numpy as np
import os, sys, argparse

## get job number:
parser = argparse.ArgumentParser(description='Setting name of output file.')
parser.add_argument('-f','--file',type=str,required=True,dest='filename',help='filename output')
job = parser.parse_args()

## import parameters:
m1,m2,ecc,dist,RA,DEC = np.loadtxt('Injection_'+str(job.filename)+'/true_parameters',unpack=True)
lnZe_pt, dlnZe_pt, dlnZ0_pt, dlnZ0_pt, BF = np.loadtxt('samples/BayesFactor/logEvidence_and_logBF_'+str(job.filename)+'.txt')

lnBF = np.log(BF)

np.savetxt('output/results_job_'+str(job.filename)+'.txt',np.c_[m1,m2,ecc,dist,RA,DEC,lnZe_pt,dlnZe_pt,dlnZ0_pt,dlnZ0_pt,BF,lnBF])

print("Saved parameters")