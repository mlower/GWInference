

'''
Paul Lasky
Handy tools
'''
import numpy as np
import harmonics
from scipy.interpolate import interp1d
from scipy.signal import hann
import constants as cc
from scipy.integrate import dblquad

def m12_to_mc(m1, m2):
    # convert m1 and m2 to chirp mass
    return (m1*m2)**(3./5.) / (m1 + m2)**(1./5.)

def m12_to_symratio(m1, m2):
    # convert m1 and m2 to symmetric mass ratio
    return m1 * m2 / (m1 + m2)**2

def mc_eta_to_m12(mc, eta):
    """
    Convert chirp mass and symmetric mass ratio to component masses.

    Input: mc - chirp mass
    eta - symmetric mass ratio
    Return: m1, m2 - primary and secondary masses, m1>m2
    """
    m1 = mc/eta**0.6*(1+(1-4*eta)**0.5)/2
    m2 = mc/eta**0.6*(1-(1-4*eta)**0.5)/2
    return m1, m2

def m_sol_to_geo(mm):
    # convert from solar masses to geometric units
    return mm / cc.kg * cc.GG / cc.cc**2

def m_geo_to_sol(mm):
    # convert from geometric units to solar masses
    return mm * cc.kg / cc.GG * cc.cc**2

def time_s_to_geo(time):
    # convert time from seconds to geometric units
    return time * cc.cc

def time_geo_to_s(time):
    # convert time from seconds to geometric units
    return time / cc.cc

def freq_Hz_to_geo(freq):
    # convert freq from Hz to geometric units
    return freq / cc.cc

def freq_geo_to_Hz(freq):
    # convert freq from geometric units to Hz
    return freq * cc.cc

def dist_Mpc_to_geo(dist):
    # convert distance from Mpc to geometric units (i.e., metres)
    return dist * cc.Mpc

def h_tot(hp, Fp, hx, Fx):
    # calculate h_total from plus and cross polarizations and antenna pattern
    return Fp * hp + Fx * hx


def nfft(ht, Fs):
    '''
    performs an FFT while keeping track of the frequency bins
    assumes input time series is real (positive frequencies only)

    ht = time series
    Fs = sampling frequency

    returns
    hf = single-sided FFT of ft normalised to units of strain / sqrt(Hz)
    f = frequencies associated with hf
    '''
    # add one zero padding if time series does not have even number of sampling times
    if np.mod(len(ht), 2) == 1:
        ht = np.append(ht, 0)
    LL = len(ht)
    # frequency range
    ff = Fs / 2 * np.linspace(0, 1, LL/2+1)

    # calculate FFT
    # rfft computes the fft for real inputs
    hf = np.fft.rfft(ht)

    # normalise to units of strain / sqrt(Hz)
    hf = hf / Fs

    return hf, ff

def infft(hf, Fs):
    '''
    inverse FFT for use in conjunction with nfft
    eric.thrane@ligo.org
    input:
    hf = single-side FFT calculated by fft_eht
    Fs = sampling frequency
    output:
    h = time series
    '''
    # use irfft to work with positive frequencies only
    h = np.fft.irfft(hf)
    # undo LAL/Lasky normalisation
    h = h*Fs

    return h


def inner_product(aa, bb, freq, PSD):
    '''
    Calculate the inner product defined in the matched filter statistic

    arguments:
    aai, bb: single-sided Fourier transform, created, e.g., by the nfft function above
    freq: an array of frequencies associated with aa, bb, also returned by nfft
    PSD: an Nx2 array describing the noise power spectral density

    Returns:
    The matched filter inner product for aa and bb
    '''
    # interpolate the PSD to the freq grid
    PSD_interp_func = interp1d(PSD[:, 0], PSD[:, 1], bounds_error=False, fill_value=np.inf)
    PSD_interp = PSD_interp_func(freq)

    # caluclate the inner product
    integrand = np.conj(aa) * bb / PSD_interp

    df = freq[1] - freq[0]
    integral = np.sum(integrand) * df

    product = 4. * np.real(integral)

    return product


def snr_exp(aa, freq, PSD):
    '''
    Calculates the expectation value for the optimal matched filter SNR

    arguments:
    aa: single-sided Fourier transform, created, e.g., by the nfft function above
    freq: an array of frequencies associated with aa, also returned by nfft
    PSD: an Nx2 array describing the noise power spectral density

    Returns:
    (The expectation value of) the matched filter SNR for aa
    '''
    return np.sqrt(inner_product(aa, aa, freq, PSD))

def snr_matchedfilter(hf, muf, f, Sh):
    '''
    eric.thrane@ligo.org
    calculate matched filter SNR for template muf given data hf
    '''
    snr = inner_product(hf, muf, f, Sh) / np.sqrt(inner_product(muf, muf, f, Sh))
    return snr

def window_Hann(ht):
    '''
    Apply the Hann window to a time series.

    arguments:
    ht = times series h(t)

    returns Hann-windowed function using the native scipy Hann window
    '''

    window = hann(len(ht))
    ht *= window

    return ht

def window_Tukey(ht, alpha=0.5):
    '''
    Apply the Tukey window to a time series.

    arguments:
    ht = times series h(t)
    a = alpha, a parameter related to the width of the window

    returns:
    Tukey-windowed time series

    Note: native scipy Tukey requires v0.16 which is not available on ldas-pcdev2
          content taken from https://github.com/scipy/scipy/blob/v0.19.1/scipy/signal/windows.py#L819-L899
    '''
    M = len(ht)
    if alpha <=0:
        return np.ones(M)
    elif alpha >=1:
        return window_Hann(ht)

    n = np.arange(0, M)
    n = np.arange(0, M)
    width = int(np.floor(alpha*(M-1)/2.0))
    n1 = n[0:width+1]
    n2 = n[width+1:M-width-1]
    n3 = n[M-width-1:]

    w1 = 0.5 * (1 + np.cos(np.pi * (-1 + 2.0*n1/alpha/(M-1))))
    w2 = np.ones(n2.shape)
    w3 = 0.5 * (1 + np.cos(np.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))

    window = np.concatenate((w1, w2, w3))
    ht *= window

    return ht

def cal_memory(t_arr,hp,hx,l,m,r,inc,pol,theta0,phi0):
    '''
    Calculate memory from waveform using Favata 2009 Eq.12.

    arguments:
    t_arr: time array
    hp: the plus polarization of the wave associated with the time array
    hx: the cross polarization of the wave associated with the time array
    l, m: mode of the wave
    r: distance from the GW source to the observer, in MPC
    inc: inclination angle in radians (for the observer of the memory)
    pol: polarization angle in radians  (for the observer of the memory)
    theta0, phi0: angles in radians, describing the observer direction associated with the

    returns:
    hmem: the memory waveform, with t_arr as the time scale
    '''
    r = dist_Mpc_to_geo(r) # distance between source and detecter in MPC
    s = -2 #spin weight
    const = r*cc.cc**3./4./np.pi/cc.GG # constant terms
    const *= cc.GG/(cc.cc**4) #unit conversion (from geometric units)
    div = np.absolute(harmonics.sYlm(s,l,m,theta0,phi0))**2
    const /= div
    Nx = np.sin(inc)*np.cos(pol)
    Ny = np.sin(inc)*np.sin(pol)
    Nz = np.cos(inc)

    # helper function
    def angle_integrand_p(theta,phi):
        har = harmonics.sYlm(s,l,m,theta,phi)
        nx = np.sin(theta)*np.cos(phi)
        ny = np.sin(theta)*np.sin(phi)
        nz = np.cos(theta)
        return (np.cos(theta)**2)/(1-Nx*nx-Ny*ny-Nz*nz)*np.sin(theta)*np.absolute(har)**2

    # helper function
    def angle_integrand_x(theta,phi):
        har = harmonics.sYlm(s,l,m,theta,phi)
        nx = np.sin(theta)*np.cos(phi)
        ny = np.sin(theta)*np.sin(phi)
        nz = np.cos(theta)
        return (np.sin(theta)*np.cos(theta)*np.sin(phi))/(1-Nx*nx-Ny*ny-Nz*nz)*np.sin(theta)*np.absolute(har)**2

    dt = t_arr[1]-t_arr[0] # sampling distance
    # time derivatives of h
    hp_d = np.gradient(hp,dt)
    hx_d = np.gradient(hx,dt)

    # integrate over theta (0 to pi), phi (0 to 2pi).
    angle_integral_p,err_angle_p = dblquad(angle_integrand_p,0,2*np.pi,lambda theta:0, lambda theta:np.pi)
    angle_integral_x,err_angle_x = dblquad(angle_integrand_x,0,2*np.pi,lambda theta:0, lambda theta:np.pi)

    hmem_p=[0]
    hmem_x=[0]
    t_integrand = hp_d**2+hx_d**2

    # integrate over t
    for i in range(len(t_arr)-1):
        temp = np.trapz(t_integrand[:i+2],x=t_arr[:i+2])
        hmem_p.append(const*angle_integral_p*temp)
        hmem_x.append(const*angle_integral_x*temp)

    return hmem_p,hmem_x

def normed_inner_product(Fh, Fg, freq, PSD):
    '''
    Calculates the normed inner product between two waveforms.
    marcus.lower@ligo.org

    arguments:
    Fh, Fg = FFT of waveforms you want to get the overlap of (found using nFFT)
    freq, = array of frequencies associated with the FFT waveforms
    PSD = array containing power spectral density of detector noise

    returns:
    Normed inner product between the two waveforms
    '''
    ## Taking the inner products:
    ip_hg = inner_product(Fh, Fg, freq, PSD)
    ip_hh = inner_product(Fh, Fh, freq, PSD)
    ip_gg = inner_product(Fg, Fg, freq, PSD)

    ## Calculating overlap between waveforms:
    norm_ip = ip_hg/((ip_hh*ip_gg)**(1./2.))
    return norm_ip

def nextpow2(i):
    """
    Find 2^n that is equal to or greater than.
    """
    n = 1
    while n < i: n *= 2
    return n

def gaussian_noise(Sh, sampleRate, duration):
    '''
    adapted from gaussian_noise.m in matapps and pycbc gaussian_noise
    eric.thrane@ligo.org
    borrows from code in readTimeSeries2) creates a NxT array of h(t)
    given a noise power spectrum Sh
    sampleRate = sample rate
    duration = signal duration
    Sh = noise power spectral density
    Nikhil: Modified normalisations to work with python
    '''

    # calculate N = nuber of samples
    N = duration * sampleRate
    N = int(np.round(N))

    # prepare for FFT
    numFreqs = (N-1)//2
    deltaF = 1./duration
    # python: start from DC
    #Changed linspace start point from 0 -> 1 to remove double zeros at the start
    f = deltaF*np.linspace(1, numFreqs, numFreqs)

    # next power of 2 from length of y
    amp_values = Sh[:,1]
    f_transfer1 = Sh[:,0]
    Pf1_interp_func = interp1d(Sh[:, 0], Sh[:, 1], bounds_error=False, fill_value=np.inf)
    Pf1 = Pf1_interp_func(f)
    # python: remove infinities
    if sum(np.isinf(Pf1)) > 0:
        #print("Warning: extrapolating outside of noise curve, P=%1.1e/Hz" % max(Pf1[~np.isinf(Pf1)]))
        Pf1[np.isinf(Pf1)] = max(Pf1[~np.isinf(Pf1)])
    #
    deltaT = 1./sampleRate
    #departed from the matlab code because of different FFT normalisation
    #Consistent with pycbc code
    #norm1 = np.sqrt(N/(2*deltaT)) * np.sqrt(Pf1)
    norm1 = 0.5*(Pf1/deltaF)**0.5
    re1 = np.random.normal(0,norm1,int(numFreqs))
    im1 = np.random.normal(0,norm1,int(numFreqs))
    z1  = re1 + 1j*im1

    # freq domain solution for htilde1, htilde2 in terms of z1, z2
    htilde1 = z1
    # convolve data with instrument transfer function
    otilde1 = htilde1*1.
    # set DC and Nyquist = 0
    # python: we are working entirely with positive frequencies
    if ( np.mod(N,2)==0 ):
        otilde1 = np.concatenate(([0], otilde1, [0]))
        f = np.concatenate(([0], f, [sampleRate/2.]))
    else:
        # no Nyquist frequency when N=odd
        otilde1 = np.concatenate(([0], otilde1))
        f = np.concatenate(([0], f))

    # normalise for positive frequencies and units of strain/rHz
    hf = otilde1
    # python: transpose for use with infft
    hf = np.transpose(hf)
    f = np.transpose(f)

    # python: return Fourier transform, not time series
    return hf,f
