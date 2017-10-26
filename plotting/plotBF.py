## Plotting ln(Bayes Factor):
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dat15 = np.loadtxt('../samples/BayesFactor/logEvidence_and_logBF_15.txt')
true15 = np.loadtxt('../waveformInjection_15/true_parameters')
dat17 = np.loadtxt('../samples/BayesFactor/logEvidence_and_logBF_17.txt')
true17 = np.loadtxt('../waveformInjection_17/true_parameters')
dat20 = np.loadtxt('../samples/BayesFactor/logEvidence_and_logBF_20.txt')
true20 = np.loadtxt('../waveformInjection_20/true_parameters')
dat21 = np.loadtxt('../samples/BayesFactor/logEvidence_and_logBF_21.txt')
true21 = np.loadtxt('../waveformInjection_21/true_parameters')
dat25 = np.loadtxt('../samples/BayesFactor/logEvidence_and_logBF_25.txt')
true25 = np.loadtxt('../waveformInjection_25/true_parameters')
dat29 = np.loadtxt('../samples/BayesFactor/logEvidence_and_logBF_29.txt')
true29 = np.loadtxt('../waveformInjection_29/true_parameters')
dat31 = np.loadtxt('../samples/BayesFactor/logEvidence_and_logBF_31.txt')
true31 = np.loadtxt('../waveformInjection_31/true_parameters')
dat34 = np.loadtxt('../samples/BayesFactor/logEvidence_and_logBF_34.txt')
true34 = np.loadtxt('../waveformInjection_34/true_parameters')
dat36 = np.loadtxt('../samples/BayesFactor/logEvidence_and_logBF_36.txt')
true36 = np.loadtxt('../waveformInjection_36/true_parameters')
dat37 = np.loadtxt('../samples/BayesFactor/logEvidence_and_logBF_37.txt')
true37 = np.loadtxt('../waveformInjection_37/true_parameters')
dat63 = np.loadtxt('../samples/BayesFactor/logEvidence_and_logBF_63.txt')
true63 = np.loadtxt('../waveformInjection_63/true_parameters')
dat86 = np.loadtxt('../samples/BayesFactor/logEvidence_and_logBF_86.txt')
true86 = np.loadtxt('../waveformInjection_86/true_parameters')
dat88 = np.loadtxt('../samples/BayesFactor/logEvidence_and_logBF_88.txt')
true88 = np.loadtxt('../waveformInjection_88/true_parameters')
dat89 = np.loadtxt('../samples/BayesFactor/logEvidence_and_logBF_89.txt')
true89 = np.loadtxt('../waveformInjection_89/true_parameters')
dat91 = np.loadtxt('../samples/BayesFactor/logEvidence_and_logBF_91.txt')
true91 = np.loadtxt('../waveformInjection_91/true_parameters')
dat92 = np.loadtxt('../samples/BayesFactor/logEvidence_and_logBF_92.txt')
true92 = np.loadtxt('../waveformInjection_92/true_parameters')

BF = [dat15[-1],dat17[-1],dat20[-1],dat21[-1],dat25[-1],dat29[-1],dat31[-1],dat34[-1],dat36[-1],dat37[-1],dat63[-1],dat86[-1],dat88[-1],dat89[-1],dat91[-1],dat92[-1]]
lnBF = np.log(BF)

Ecc = [true15[2],true17[2],true20[2],true21[2],true25[2],true29[2],true31[2],true34[2],true36[2],true37[2],true63[2],true86[2],true88[2],true89[2],true91[2],true92[2]]
logEcc = np.log10(Ecc)

plt.clf()
plt.tick_params(axis='both',which='both',direction='in',bottom='on',top='on',left='on',right='on')
#plt.rcParams['text.usetex'] = True
#plt.rcParams['text.latex.unicode'] = True
plt.plot(logEcc,lnBF,ls='None',marker='^',color='b')
plt.axhline(8,color='k',ls='--',lw=1)
plt.xlim(-7,0)
#plt.ylim(-5,10)
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.ylabel(r'ln($\mathcal{B}$)',fontsize=16)
plt.xlabel('log Eccentricity at 10 Hz',fontsize=16)
plt.tight_layout()
plt.savefig('lnBFvsEcc.png',dpi=200)

