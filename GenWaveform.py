import numpy as np
import lal
import lalsimulation as lalsim

def AntennaResponse(RA, DEC, psi, tc, ifo):
    """
    Taken from AntennaPattern.py

    RA, DEC
    psi: polarisation angle
    tc: coalescence time (GPS)
    IFO: either 'H1', 'L1' or 'V1'
    
    Return: Antenna response functions, F_+ and F_x, and LALDetectorIndex (diff)
    """
    tgps = lal.LIGOTimeGPS(tc)
    gmst = lal.GreenwichMeanSiderealTime(tgps)

    if ifo == 'H1':
        diff = lal.LALDetectorIndexLHODIFF
    elif ifo == 'L1':
        diff = lal.LALDetectorIndexLLODIFF
    elif ifo == 'V1':
        diff = lal.LALDetectorIndexVIRGODIFF
    else:
        raise ValueError('detector not recognized: ' + ifo)

    fplus, fcross = lal.ComputeDetAMResponse(lal.CachedDetectors[diff].response, RA, DEC, psi, gmst)

    return fplus, fcross, diff


def GenFDWaveform(fmin, fmax, deltaF, dist, m1, m2, S1, S2, e_min, fRef, iota, phi, waveform='IMRPhenomPv2'):
    """
    Returns the plus and cross polarizations of h(f)

    fmin, fmax: minimum and maximum frequency to consider
    dist: distance to merger in Mpc
    m1, m2: component masses in solar masses
    S1,S2: spin vectors, e.g., S1=[0,0.3,0.2] with sqrt(S1[0] + S1[1] + S1[2]) <= 1
    fRef: reference frequency at which spins are calculated
    iota: inclination
    phi: orbital phase at fRef
    waveform: name of waveform to use, default 'IMRPhenomPv2'

    Returns: plus and cross polarizations
    """
    # do not allow frequencies less than 10 Hz as this leads to gibberish
    if fmin < 10: raise ValueError('fmin must be 10 Hz or greater')

    #Distance and masses have to be in SI units.
    dist *= 1e6*lal.lal.PC_SI
    m1 *= lal.lal.MSUN_SI
    m2 *= lal.lal.MSUN_SI

    # extra arguments for new waveform interface
    longAscNodes = 0.0
    meanPerAno = 0.0
    
    if waveform == 'EccentricFD':
        e_min = e_min
    else:
        e_min = 0.0

    WFdict = lal.CreateDict()

    # Get waveform approximant
    approx = lalsim.GetApproximantFromString(waveform)

    #Plus and cross polarisation waveforms in the frequency domain.
    hplus,hcross = lalsim.SimInspiralChooseFDWaveform(m1, m2, S1[0], S1[1], S1[2], S2[0], S2[1], S2[2], dist, iota, phi, longAscNodes, e_min, meanPerAno, deltaF, fmin, fmax, fRef, WFdict, approx)

    h_p = hplus.data.data
    h_c = hcross.data.data

    return h_p, h_c

def DetectorStrain(h_p, h_c, fmax, deltaF, RA, DEC, psi, tc, ifo):
    """
    Returns h(f) in the detector frame.
    
    h_p, h_c: plus/cross polarised strain
    RA, DEC: sky location
    psi: polarisation angle
    tc: coalesence time
    ifo: detector ('H1', 'L1', 'V1')
    
    Returns: strain in the detector and associated frequencies
    """
    Fp,Fc,diff = AntennaResponse(RA, DEC, psi, tc, ifo)
    tgps = lal.LIGOTimeGPS(tc)
    timedelay = lal.TimeDelayFromEarthCenter(lal.CachedDetectors[diff].location, RA, DEC, tgps)
    timeshift = tc + timedelay
    
    h_t = Fp * h_p + Fc * h_c

    f_arr = np.linspace(0, fmax, int(fmax/deltaF) + 1)
    h_t *= np.exp(1j*np.pi*2*f_arr*timedelay)

    
    return h_t, f_arr
