
from telescope_lib import *
from HI_galaxy_lib import *

c_light = 29979245800    # cm/s
HI_freq = 1420405751.768 # Hz

def E(z, q0, j0):
    '''
    Returns the re-scaled Hubble parameter (dimensionless)

    z  = observed redshift
    q0 = deceleration parameter
    j0 = jerk parameter
    '''

    a = (q0 + 1) * z
    b = 1/2 * (j0 - q0**2) * z**2
    return 1 + a + b

def Z1(z, q0, j0):
    '''
    Returns the first time derivative of the redshift (dimensionless)

    z  = observed redshift
    q0 = deceleration parameter
    j0 = jerk parameter
    '''

    return 1 + z - E(z, q0, j0)

def delta_v_func(z, p, t_exp):
    '''
    Returns the spectroscopic velocity shift [cm/s]

    z     = observed redshift
    p     = array of H0, q0 and j0 values
    t_exp = total experiment time (time span between observations) [yrs]
    '''

    H0, q0, j0 = p
    H0_s       = H0 / 3.08567758128e19 # km/s/Mpc to 1/s
    return c_light * H0_s * t_exp*365*24*3600 * Z1(z, q0, j0) / (1 + z)


def relvelo(obsfreq):
    '''
    Returns the relativistic velocity [cm/s]
    (equation 10-77 and 10-78a Kraus, Radio Astronomy)

    obsfreq = observed frequency [Hz]
    '''

    z     = HI_freq/obsfreq -1.
    m     = (z + 1)**2
    rvelo = c_light * (m - 1)/(m + 1)
    return rvelo

    
def Dnu2dv(Dnu,z):
    '''
    Returns the channel width in velocity [cm/s]

    Dnu = channel width in frequency [Hz]
    '''
    
    lower_freq = HI_freq/(z+1) - Dnu/2
    upper_freq = HI_freq/(z+1) + Dnu/2
    upper_v    = relvelo(lower_freq)
    lower_v    = relvelo(upper_freq)
    dv = (upper_v - lower_v) 
    return dv


def sigma_v_func(z, t_obs, N_ant, Dnu, S_area, fwhm, delta_z):
    '''
    Returns the uncertainty sigma_v in the cosmological velocity drift [cm/s]
    
    z       = observed redshift
    t_obs   = observation time (per pointing) [s]
    N_ant   = total number of antennas
    Dnu     = channel width [Hz]
    S_area  = observed survey area [sq deg]
    fwhm    = HI line width [cm/s]
    delta_z = delta z for dN/dz integration
    '''

    # determine dispersion of the HI line
    # sigma_HI or p4.
    #
    # Note this factor is based on the definition
    # of the Gaussian function in the Minin and Kamalabadi
    # and can not be used in general !
    #
    p4            = fwhm / (2*np.sqrt(np.log(2))) 

    # determine the channel width in velocity
    #
    dv            = Dnu2dv(Dnu,z)

    # Here we determine the scaling factor for
    # detecting a galaxy at a single channel, so 
    # essentially determine the galaxies we can work
    # with. Note that is not related to the detection
    # which we assume to stack.
    #
    # HI profile at 20 percent level see Obreschow et al. 2009
    #
    w20_HI_galaxy = 3.6 * p4 * 1/np.sqrt(2)                         
    #
    # Essentially this factor scales the Dnu to a single
    # channel.
    #
    Single_channel_detection  = w20_HI_galaxy/dv
    #
    # If Dnu is so large that a HI profile is just confined in a
    # single channel.
    # 
    if Single_channel_detection > 1:
        Single_channel_detection_factor  = 1/np.sqrt(Single_channel_detection)
    else:
        Single_channel_detection_factor  = 1

    # Determine the number of HI galaxies using the 
    # single channel detection
    #
    rms                                 = S_rms_func(t_obs, N_ant, Dnu) * Single_channel_detection_factor
    N_galaxies, Minimum_peak_Amplitude  = N_func(z, rms, S_area, delta_z)

    # Error on the line center parameter of a pure Gaussian will 
    # be used to estimate the accuray of determine the
    #
    # redshift_drift
    #
    # Based on the equation 14 of Minin and Kamalabadi (2009)
    # and assuming that the sigma in the equation reduces by
    # 1/sqrt(number_of_staked_spectra)
    #
    # Caution the Minimum_peak_Amplitude is a powerfull factor 
    # here and we use the boundaries of defined in N_func and
    # assuming that all, even the weakest line will
    # contribute to reduce the rms. For the future this needs to
    # be evaluated properly.
    #
    sigma_v = rms/np.sqrt(N_galaxies) * (Minimum_peak_Amplitude**2 * np.sqrt(np.pi) / (p4*dv*np.sqrt(2)))**-.5 

    return sigma_v

