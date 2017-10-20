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
    
    Return: Antenna response functions, F_+ and F_x
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

    return fplus, fcross

def FD(fmin, fmax, deltaF, dist, m1, m2, S1, S2, fRef, iota, RA, DEC, psi, phi, tc, ifo, waveform='IMRPhenomPv2'):
    """
    Adapted from Rory Smith's original code (by Colm Talbot), taken from:
    /home/rory.smith/projects/stochasticCBC/pythonWaveformExamples/generateIMRPhenomPv2.py

    fmin, fmax: minimum and maximum frequency to consider
    dist: distance to merger in Mpc
    m1, m2: component masses in solar masses
    S1,S2: spin vectors, e.g., S1=[0,0.3,0.2] with sqrt(S1[0] + S1[1] + S1[2]) <= 1
    fRef: reference frequency at which spins are calculated
    iota: inclination
    RA, DEC: **are these the same as longitude, latitude
    psi: polarization angle
    phi: orbital phase at fRef
    tc: coalescence time at geocenter (a GPS time)
    ifo: either 'H1', 'L1' or 'V1'
    waveform: name of waveform to use, default 'IMRPhenomPv2'
        allowed waveforms: IMRPhenomP, IMRPhenomPv2, SEOBNRv1_ROM_DoubleSpin, SEOBNRv1_ROM_EffectiveSpin

    Returns: Fourier transform of the waveform strain in ifo including the timeshift from geocenter (beginning at f=0 Hz)
    """
    # do not allow frequencies less than 10 Hz as this leads to gibberish
    if fmin < 10: raise ValueError('fmin must be 10 Hz or greater')

    tgps = lal.LIGOTimeGPS(tc)
    gmst = lal.GreenwichMeanSiderealTime(tgps)

    #Distance and masses have to be in SI units.
    dist *= 1e6*lal.lal.PC_SI
    m1 *= lal.lal.MSUN_SI
    m2 *= lal.lal.MSUN_SI

    # extra arguments for new waveform interface
    longAscNodes = 0.0
    eccentricity = 0.0
    meanPerAno = 0.0

    WFdict = lal.CreateDict()
    
    #lalsim.SimInspiralWaveformParamsInsertFrameAxis(WFdict, LAL_SIM_INSPIRAL_FRAME_AXIS_VIEW)
    #lalsim.SimInspiralWaveformParamsInsertPNSpinOrder(WFdict, -1)
    #lalsim.SimInspiralWaveformParamsInsertPNTidalOrder(WFdict, -1)
    #lalsim.SimInspiralWaveformParamsInsertPNPhaseOrder(WFdict, -1)
    #lalsim.SimInspiralWaveformParamsInsertPNAmplitudeOrder(WFdict, -1)
    #lalsim.SimInspiralWaveformParamsInsertTidalLambda1(WFdict, 0.0)
    #lalsim.SimInspiralWaveformParamsInsertTidalLambda2(WFdict, 0.0)

    # Get waveform approximant
    approx = lalsim.GetApproximantFromString(waveform)
    
    #Plus and cross polarisation waveforms in the frequency domain.
    hplus,hcross = lalsim.SimInspiralChooseFDWaveform(m1, m2, S1[0], S1[1], S1[2], S2[0], S2[1], S2[2], dist, iota, phi, longAscNodes, eccentricity, meanPerAno, deltaF, fmin, fmax, fRef, WFdict, approx)

    # Start the waveform at frequency fmin
    # EHT (20/12/16): commented out: ALWAYS use full frequency array to
    # avoid problems with irfft
    #h_p = hplus.data.data[(fmin/hplus.deltaF):]
    #h_c = hcross.data.data[(fmin/hcross.deltaF):]
    h_p = hplus.data.data
    h_c = hcross.data.data

    #The coalescence (more precisely, the maximum of the waveform) occurs at 'tc' at the geocenter. Calculate time delay between geocenter and the chosen detector.
    if ifo == 'H1':
        diff = lal.LALDetectorIndexLHODIFF
    elif ifo == 'L1':
        diff = lal.LALDetectorIndexLLODIFF
    elif ifo == 'V1':
        diff = lal.LALDetectorIndexVIRGODIFF
    else:
        raise ValueError('detector not recognized: ' + ifo)

    timedelay = lal.TimeDelayFromEarthCenter(lal.CachedDetectors[diff].location, RA, DEC, tgps)
    timeshift = tc + timedelay
    #timeshift=0.0 #TEST.

    #Obtain antenna pattern factors for the detector for a source at location RA,DEC at the precise instant of coalescence (and assume these factors do not change significantly during the entire chirp).
    fplus, fcross = AntennaResponse(RA, DEC, psi, tc, ifo)

    #Calculate the observed strain at the detector, properly shifting the waveform from geocenter to detector frame.
    h = np.zeros_like(h_p, dtype=complex)
    pit = np.pi*timeshift
    if timeshift != 0.0:
        shift = complex(1.0, 0)
        dshift = complex(-2.*np.sin(pit*deltaF)*np.sin(pit*deltaF), -2.*np.sin(pit*deltaF)*np.cos(pit*deltaF))
        for i in xrange(0,h_p.size):
            h[i] = shift*(fplus*h_p[i] + fcross*h_c[i])
            shift += shift*dshift
    else: h = (fplus*h_p) + (fcross*h_c)

    # create frequency array up to and including fmax
    f = np.linspace(0, fmax, (fmax+deltaF)/deltaF)

    # zero spectral content below fmin to avoid unphysical numbers
    h[0:int(fmin/deltaF)] = 0

    return h, f

def TD(fmin, deltaT, dist, m1, m2, S1, S2, fRef, iota, RA, DEC, psi, phi, tc, ifo, waveform='IMRPhenomPv2'):
    """
    Adapted from Rory Smith's original code (by Colm Talbot), taken from:
    /home/rory.smith/projects/stochasticCBC/pythonWaveformExamples/generateIMRPhenomPv2.py

    fmin, fmax: minimum and maximum frequency to consider
    dist: distance to merger in Mpc
    m1, m2: component masses in solar masses
    S1,S2: spin vectors, e.g., S1=[0,0.3,0.2] with sqrt(S1[0] + S1[1] + S1[2]) <= 1
    fRef: reference frequency at which spins are calculated
    iota: inclination
    RA, DEC: **are these the same as longitude, latitude
    psi: polarization angle
    phi: orbital phase at fRef
    tc: coalescence time at geocenter (a GPS time)
    ifo: either 'H1', 'L1' or 'V1'
    waveform: name of waveform to use, default 'IMRPhenomPv2'
        allowed waveforms: IMRPhenomPv2, SEOBNRv1, SEOBNRv3

    Returns: Time domain waveform strain in ifo, timing needs to be though about.
    """
    # do not allow frequencies less than 10 Hz as this leads to gibberish
    if fmin < 10: raise ValueError('fmin must be 10 Hz or greater')

    tgps = lal.LIGOTimeGPS(tc)
    gmst = lal.GreenwichMeanSiderealTime(tgps)

    #Distance and masses have to be in SI units.
    dist *= 1e6*lal.lal.PC_SI
    m1 *= lal.lal.MSUN_SI
    m2 *= lal.lal.MSUN_SI

    # extra arguments for new waveform interface
    longAscNodes = 0.0
    eccentricity = 0.0
    meanPerAno = 0.0

    WFdict = lal.CreateDict()
    
    #lalsim.SimInspiralWaveformParamsInsertFrameAxis(WFdict, LAL_SIM_INSPIRAL_FRAME_AXIS_VIEW)
    #lalsim.SimInspiralWaveformParamsInsertPNSpinOrder(WFdict, -1)
    #lalsim.SimInspiralWaveformParamsInsertPNTidalOrder(WFdict, -1)
    #lalsim.SimInspiralWaveformParamsInsertPNPhaseOrder(WFdict, -1)
    #lalsim.SimInspiralWaveformParamsInsertPNAmplitudeOrder(WFdict, -1)
    #lalsim.SimInspiralWaveformParamsInsertTidalLambda1(WFdict, 0.0)
    #lalsim.SimInspiralWaveformParamsInsertTidalLambda2(WFdict, 0.0)

    # Get waveform approximant
    approx = lalsim.GetApproximantFromString(waveform)
    
    #Plus and cross polarisation waveforms in the frequency domain.
    hplus,hcross = lalsim.SimInspiralChooseTDWaveform(m1, m2, S1[0], S1[1], S1[2], 
                                                      S2[0], S2[1], S2[2], dist, iota, phi, 
                                                      longAscNodes, eccentricity, meanPerAno, 
                                                      deltaT, fmin, fRef, WFdict, approx)

    # Start the waveform at frequency fmin
    # EHT (20/12/16): commented out: ALWAYS use full frequency array to
    # avoid problems with irfft
    #h_p = hplus.data.data[(fmin/hplus.deltaF):]
    #h_c = hcross.data.data[(fmin/hcross.deltaF):]
    h_p = hplus.data.data
    h_c = hcross.data.data

    #The coalescence (more precisely, the maximum of the waveform) occurs at 'tc' at the geocenter. Calculate time delay between geocenter and the chosen detector.
    if ifo == 'H1':
        diff = lal.LALDetectorIndexLHODIFF
    elif ifo == 'L1':
        diff = lal.LALDetectorIndexLLODIFF
    elif ifo == 'V1':
        diff = lal.LALDetectorIndexVIRGODIFF
    else:
        raise ValueError('detector not recognized: ' + ifo)

    timedelay = lal.TimeDelayFromEarthCenter(lal.CachedDetectors[diff].location, RA, DEC, tgps)
    timeshift = tc + timedelay

    #Obtain antenna pattern factors for the detector for a source at location RA,DEC at the precise instant of coalescence (and assume these factors do not change significantly during the entire chirp).
    fplus, fcross = AntennaResponse(RA, DEC, psi, tc, ifo)

    #Calculate the observed strain at the detector, properly shifting the waveform from geocenter to detector frame.  For time delay, this is just a projection with no time dependance?
    h = (fplus*h_p) + (fcross*h_c)

    # create time array at correct time for detector
    t = np.linspace(0, deltaT*len(h), len(h))
    t -= t[h==max(h)]
    t += timeshift
    
    return h, t

def surrogate_waveform(m1, m2, dist, iota, psi, h5file, lm=[(2,2), ], times = 'NAN'):
    '''
    Paul Lasky

    Calculates a BBH waveform using the surrogate models of Field et al. (2014)

    returns time, hp, hx
    where hp and hx have shape (len(lm), len(times)) 

    *** important: must have gwsurrogate installed:
    pip install gwsurrogate
    *** also important: must have an hdf5 file called SpEC_q1_10_NoSpin_nu5thDegPoly_exclude_2_0.h5

    m1, m2: masses in solar masses
    dist: distance to binary in Mpc
    iota: inclination angle
    psi: polarization angle
    h5file: location of hdf5 file
    lm: spherical harmonic modes. e.g., for lm = 22 and 33 modes, call lm = [(2,2), (3,3)]
    times: if you want to specify the time array to evaluate the modes, pass an array of times to this variable

    '''
    import gwsurrogate as gws

    spec = gws.EvaluateSurrogate(h5file, ell_m = lm)

    if m1 < m2:
        qq = float(m2)/float(m1)
    else:
        qq = float(m1)/float(m2)

    mtot = float(m1+m2)
    
    hp = []
    hx = []
    if times is not 'NAN':
        time = []

    for ii in range(len(lm)):
        if times is not 'NAN':
            _, time, hp_tmp, hx_tmp = spec(q=qq, M=mtot, dist=dist, theta=iota, phi=psi,
                                           ell=[lm[ii][0]], m=[lm[ii][1]],
                                           samples=times, samples_units='mks',
                                           mode_sum=False, fake_neg_modes=False)
        else:
            _, time, hp_tmp, hx_tmp = spec(q=qq, M=mtot, dist=dist, theta=iota, phi=psi,
                                           ell=[lm[ii][0]], m=[lm[ii][1]],
                                           mode_sum=False, fake_neg_modes=False)

        hp.append(hp_tmp)
        hx.append(hx_tmp)
        
    return time, hp, hx