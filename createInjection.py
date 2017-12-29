## Script to create eccentric injection waveforms:
import numpy as np
import sys, os, argparse
import lalsimulation as lalsim
import lal
import argparse

sys.path.append('../')

import MonashGWTools.waveforms as wv
import MonashGWTools.tools as tools

def make_waveform(m1, m2, eccen, dist, iota, job):
    '''
    Generates an eccentric timesries using LALSimulation
    '''
    #cmde = ["/bin/lalsim-inspiral -a EccentricFD -F -O -1 -u 0 -f 10 -r 20 -e "+str(eccen)+" -R 1024. -m1 "+str(m1)+" -m2 "+str(m2)+" -i "+str(iota)+" -d "+str(dist)+" > Injection_"+job+"/signal.dat"]
    cmde = ["/bin/lalsim-inspiral -a EccentricFD -F -O -1 -u 0 -f 10 -r 20 -e 0.0 -R 1024. -m1 "+str(m1)+" -m\
2 "+str(m2)+" -i "+str(iota)+" -d "+str(dist)+" > Injection_"+job+"/signal.dat"]
    cmde = ''.join(cmde)
    os.system(cmde)
    
    
def detector_strain(h_p, h_c, RA, DEC, psi, epoch, deltaF, ifo):
    
    epoch_GPS = lal.LIGOTimeGPS(epoch)
    gmst = lal.GreenwichMeanSiderealTime(epoch_GPS)
    
    if ifo == 'H1':
        ifo_cache = lal.LALDetectorIndexLHODIFF
        diff = lal.LALDetectorIndexLHODIFF
    else:
        ifo_cache = lal.LALDetectorIndexLLODIFF
        diff = lal.LALDetectorIndexLLODIFF
    
    IFO_cached = lal.CachedDetectors[ifo_cache]
    timedelay = lal.TimeDelayFromEarthCenter(lal.CachedDetectors[diff].location, RA, DEC, epoch_GPS)
    
    Fp, Fc = lal.ComputeDetAMResponse(IFO_cached.response, RA, DEC, psi, gmst)
    
    hf = Fp * h_p + Fc * h_c
    
    return hf

parser = argparse.ArgumentParser(description='Setting name of output file.')
parser.add_argument('-f','--file',type=str,required=True,dest='filename',help='filename output')
inj = parser.parse_args()

## Creates folder for injection data:
os.system("mkdir Injection_"+str(inj.filename))

## Import binary parameters:
m1, m2, dist, iota, RA, DEC, psi, epoch = np.loadtxt('inj_params.txt',unpack=True)

## Randomly samples log(eccentricity):
min_e,max_e = np.log10(1.e-3),np.log10(0.1)
eccen = round(10**(np.random.uniform(low=min_e,high=max_e,size=1)),10)

## Generates the data:
make_waveform(m1, m2, eccen, dist, iota, str(inj.filename))

## Processing and saving injection data:
data = np.loadtxt('Injection_'+str(inj.filename)+'/signal.dat')
deltaF = data[1,0] - data[0,0]

data_p = data[:,1] + 1j*data[:,2]
data_c = data[:,3] + 1j*data[:,4]
data_LHO = detector_strain(data_p, data_c, RA, DEC, psi, epoch, deltaF, ifo='H1')
data_LLO = detector_strain(data_p, data_c, RA, DEC, psi, epoch, deltaF, ifo='L1')

output = 'Injection_'+str(inj.filename)+'/injectionFFT_'+str(inj.filename)+'.npy'
np.save(output,np.c_[data[:,0],data_LHO,data_LLO])
np.savetxt('Injection_'+str(inj.filename)+'/true_parameters',np.c_[m1,m2,eccen,dist,RA,DEC])

print("Injection made")


