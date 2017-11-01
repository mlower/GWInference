## Plotting ln(Bayes Factor):
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data1 = np.loadtxt('../output/GW150914.dat')
Ecc1 = data1[:,2]
logEcc1 = np.log10(Ecc1)
lnBF1 = data1[:,-1]

data2 = np.loadtxt('../output/LVT151012.dat')
Ecc2 = data2[:,2]
logEcc2 = np.log10(Ecc2)
lnBF2 = data2[:,-1]

data3 = np.loadtxt('../output/GW151226.dat')
Ecc3 = data3[:,2]
logEcc3 = np.log10(Ecc3)
lnBF3 = data3[:,-1]

data4 = np.loadtxt('../output/GW170104.dat')
Ecc4 = data4[:,2]
logEcc4 = np.log10(Ecc4)
lnBF4 = data4[:,-1]

plt.clf()
plt.tick_params(axis='both',which='both',direction='in',bottom='on',top='on',left='on',right='on')
#plt.rcParams['text.usetex'] = True
#plt.rcParams['text.latex.unicode'] = True
plt.plot(logEcc1,lnBF1,ls='None',marker='^',color='b',markerfacecolor='none')
plt.plot(logEcc2,lnBF2,ls='None',marker='s',color='r',markerfacecolor='none')
plt.plot(logEcc3,lnBF3,ls='None',marker='o',color='g',markerfacecolor='none')
plt.plot(logEcc4,lnBF4,ls='None',marker='h',color='m',markerfacecolor='none')
plt.axhline(8,color='k',ls='--',lw=1)
plt.xlim(-3.1,0.1)
#plt.ylim(-5,10)
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.ylabel(r'ln($\mathcal{B}$)',fontsize=16)
plt.xlabel('log Eccentricity at 10 Hz',fontsize=16)
plt.tight_layout()
plt.savefig('lnBFvsEcc.png',dpi=200)

