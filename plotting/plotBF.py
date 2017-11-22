## Plotting ln(Bayes Factor):
import numpy as np
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data1 = np.loadtxt('../output/GW150914.dat')
data1 = data1[data1[:,2].argsort()]



Ecc1 = data1[:,2]
logEcc1 = np.log10(Ecc1)
lnBF1 = data1[:,-1]

lnBF1p = lnBF1 + (data1[:,2] - data1[:,3])
lnBF1m = lnBF1 - (data1[:,2] - data1[:,3])

xx1 = np.linspace(-4,-0.2,3000)
interpfunc = interp1d(logEcc1, lnBF1, bounds_error=False, fill_value=np.inf)
yy1 = interpfunc(xx1)

# data2 = np.loadtxt('../output/LVT151012.dat')
# data2 = data2[data2[:,2].argsort()]
# Ecc2 = data2[:,2]
# logEcc2 = np.log10(Ecc2)
# lnBF2 = data2[:,-1]

# data3 = np.loadtxt('../output/GW151226.dat')
# data3 = data3[data3[:,2].argsort()]
# Ecc3 = data3[:,2]
# logEcc3 = np.log10(Ecc3)
# lnBF3 = data3[:,-1]

# data4 = np.loadtxt('../output/GW170104.dat')
# data4 = data4[data4[:,2].argsort()]
# Ecc4 = data4[:,2]
# logEcc4 = np.log10(Ecc4)
# lnBF4 = data4[:,-1]

# xx4 = np.linspace(-4,-0.2,3000)
# interpfunc = interp1d(logEcc4, lnBF4, bounds_error=False, fill_value=np.inf)
# yy4 = interpfunc(xx4)

plt.clf()
plt.tick_params(axis='both',which='both',direction='in',bottom='on',top='on',left='on',right='on')
#plt.rcParams['text.usetex'] = True
#plt.rcParams['text.latex.unicode'] = True
#plt.plot(logEcc1,lnBF1,ls='None',marker='^',color='b',markerfacecolor='none',label='GW150914')
#plt.plot(xx1,yy1,color='b')
plt.plot(logEcc1,lnBF1,color='b')
plt.fill_between(logEcc1, lnBF1p, lnBF1m, color='b', alpha='0.5')
#plt.plot(logEcc2,lnBF2,ls='None',marker='s',color='r',markerfacecolor='none')
#plt.plot(logEcc3,lnBF3,ls='None',marker='o',color='g',markerfacecolor='none')
#plt.plot(logEcc4,lnBF4,ls='None',marker='h',color='m',markerfacecolor='none',label='GW151226')
#plt.plot(xx4,yy4,color='m')
plt.axhline(8,color='k',ls='--',lw=1)
plt.xlim(-3.1,0.1)
#plt.ylim(-5,10)
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.ylabel(r'ln($\mathcal{B}$)',fontsize=16)
plt.xlabel('log Eccentricity at 10 Hz',fontsize=16)
plt.tight_layout()
plt.savefig('lnBFvsEcc.png',dpi=200)

