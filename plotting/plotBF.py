## Plotting ln(Bayes Factor):
import numpy as np
import os, sys, argparse, glob
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Setting name of output file.')
parser.add_argument('-f','--file',type=str,required=True,dest='filename',help='filename output')
job = parser.parse_args()

read_files = glob.glob("../output/results_job_*.txt")

with open('../output/'+str(job.filename)+'.dat', "wb") as outfile:
    for f in read_files:
        with open(f, "rb") as infile:
            outfile.write(infile.read())

data = np.loadtxt('../output/'+str(job.filename)+'.dat')
data = data[data[:,2].argsort()]
Ecc = data[:,2]
logEcc = np.log10(Ecc)
lnBF = data[:,-1]

print(data[:,-1])

plt.clf()
plt.tick_params(axis='both',which='both',direction='in',bottom='on',top='on',left='on',right='on')
plt.plot(logEcc,lnBF,ls='None',marker='^',color='b',markerfacecolor='none',label='GW150914')
plt.plot(logEcc,lnBF,color='b')
plt.axhline(8,color='k',ls='--',lw=1)
#plt.xlim(-3.1,0.1)
plt.ylim(-8,40)
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.ylabel(r'ln($\mathcal{B}$)',fontsize=16)
plt.xlabel('log eccentricity at 10 Hz',fontsize=16)
plt.tight_layout()
plt.savefig('lnBFvsEcc.png',dpi=200)

